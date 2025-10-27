# recommender/online/online_inference_pipeline.py
# ======================================================================
# ONLINE INFERENCE PIPELINE — PROD READY (seen-filter + quotas + fallback)
# Khớp offline artifacts (ranker_*.txt/pkl, latest.version) và load FAISS
# ======================================================================

from __future__ import annotations

import time
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import pickle
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.getLogger("lightgbm").setLevel(logging.WARNING)

# Project root
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Artifacts + FE
from recommender.offline.artifact_manager import ArtifactManager, get_latest_version_dir
from recommender.common.feature_engineer import FeatureEngineer

# Recall
from recommender.online.recall import FollowingRecall, CFRecall, ContentRecall, TrendingRecall
from recommender.online.recall.covisit import CovisitRecall

# Ranking / Rerank
from recommender.online.ranking import MLRanker, Reranker

# Telemetry
from recommender.online.telemetry import AsyncDBLogger

# Optional Redis
try:
    import redis as _redis
    REDIS_AVAILABLE = True
except Exception:
    _redis = None
    REDIS_AVAILABLE = False
    logger.warning("redis not available")

# Optional DB
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Optional FAISS
try:
    import faiss  # type: ignore
    FAISS_OK = True
except Exception:
    FAISS_OK = False


# -------------------------- Data classes ---------------------------
# @dataclass
# class RecallDiag:
#     following: int = 0
#     cf: int = 0
#     content: int = 0
#     covisit: int = 0
#     trending: int = 0
#     total: int = 0
#     notes: Dict[str, str] = None
class RecallDiag:
    def __init__(self, notes: Dict[str, str] | None = None):
        self.following = 0
        self.cf = 0
        self.content = 0
        self.covisit = 0
        self.trending = 0
        self.total = 0
        self.notes = notes or {}
        # bổ sung chẩn đoán chi tiết
        self.reasons: Dict[str, List[str]] = {
            "following": [], "cf": [], "content": [], "covisit": [], "trending": []
        }
        self.status_filtered = 0   # số bài bị loại vì status != 10

# ------------------------ Helpers (artifacts) -----------------------
def _load_ranker_artifacts(base_dir: str):
    """
    Load ranker artifacts theo format:
      models/<version>/
        - ranker_model.txt
        - ranker_scaler.pkl
        - ranker_feature_cols.pkl
        - meta.json (optional)
    Tự động resolve latest.version hoặc symlink latest/
    """
    import json, pickle
    from lightgbm import Booster

    base = Path(base_dir)
    vdir = get_latest_version_dir(str(base))
    if vdir is None:
        latest_file = base / "latest.version"
        if latest_file.exists():
            ver = latest_file.read_text(encoding="utf-8").strip()
            vdir = base / ver
        elif (base / "latest").exists():
            try:
                vdir = (base / "latest").resolve(strict=True)
            except Exception:
                vdir = None
    if vdir is None:
        raise FileNotFoundError(f"Cannot resolve latest model version under {base_dir}")

    f_model = vdir / "ranker_model.txt"
    f_scaler = vdir / "ranker_scaler.pkl"
    f_cols  = vdir / "ranker_feature_cols.pkl"
    f_meta  = vdir / "meta.json"

    if not f_model.exists():
        raise FileNotFoundError(f"Missing model file: {f_model}")
    model = Booster(model_file=str(f_model))

    scaler = None
    if f_scaler.exists():
        with f_scaler.open("rb") as f:
            scaler = pickle.load(f)

    feature_cols = None
    if f_cols.exists():
        with f_cols.open("rb") as f:
            feature_cols = pickle.load(f)

    meta = {}
    if f_meta.exists():
        try:
            meta = json.loads(f_meta.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    return model, scaler, feature_cols, meta, vdir


# ========================== Main Pipeline ===========================
class OnlineInferencePipeline:
    """
    Flow: Recall (multi-channel, quotas, parallel) → Seen-dedupe → Ranking → Re-ranking → Mark Seen → Output
    + Fallback nếu recall rỗng (trending → popular → random recent)
    """

    def __init__(
        self,
        config_path: str = "configs/config_online.yaml",
        models_dir: str = "models",
        data_dir: str = "dataset",
        use_redis: bool = True,
    ):
        self.models_dir = Path(models_dir)
        
        self.data_dir = Path(data_dir)

        logger.info("\n" + "=" * 70)
        logger.info("INITIALIZING ONLINE INFERENCE PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Using config: {config_path}")

        # ----------------- CONFIG -----------------
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        self.recall_config   = (self.config.get("recall") or {})
        self.ranking_config  = (self.config.get("ranking") or {})
        self.reranking_config= (self.config.get("reranking") or {})
        self.performance_cfg = (self.config.get("performance") or {})
        self.fallback_cfg    = (self.config.get("fallback") or {})
        self.redis_cfg       = (self.config.get("redis") or {})
        self.telemetry_cfg   = (self.config.get("telemetry") or {})

        # Sanity logs
        tr_cfg = (self.recall_config.get("channels") or {}).get("trending", {}) or {}
        logger.info(
            "Config sanity: trending_window_hours=%s | following.recent_hours=%s",
            tr_cfg.get("trending_window_hours"),
            (self.recall_config.get("channels") or {}).get("following",{}).get("recent_hours")
        )

        # TTLs
        ttl_cfg = (self.redis_cfg.get("ttl") or {})
        self.ttl_seen = int(ttl_cfg.get("seen_posts", 7 * 24 * 3600))  # default 7d

        # ----------------- REDIS ------------------
        self.redis: Optional[_redis.Redis] = None
        if use_redis and REDIS_AVAILABLE:
            try:
                url = self.redis_cfg.get("url")
                if url:
                    self.redis = _redis.from_url(url, decode_responses=True)
                else:
                    self.redis = _redis.Redis(
                        host=self.redis_cfg.get("host", "localhost"),
                        port=self.redis_cfg.get("port", 6379),
                        db=self.redis_cfg.get("db", 0),
                        decode_responses=True,
                        socket_timeout=self.redis_cfg.get("socket_timeout", 5),
                        max_connections=self.redis_cfg.get("max_connections", 50),
                    )
                self.redis.ping()
                logger.info("✅ Redis connected")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis = None
        else:
            logger.warning("Redis disabled or not available")

        # ----------------- BACKEND DB (optional) --------------
        self.db_engine = None
        self.db_session_factory = None
        try:
            db = (self.config.get("database") or self.config.get("backend_db") or {}) or {}
            db_url = db.get("url")
            if db_url:
                self.db_engine = create_engine(
                    db_url,
                    pool_size=db.get("pool_size", 20),
                    max_overflow=db.get("max_overflow", 40),
                    pool_recycle=db.get("pool_recycle", 1800),
                    pool_pre_ping=db.get("pool_pre_ping", True),
                    future=True,
                )
                self.db_session_factory = sessionmaker(bind=self.db_engine, autocommit=False, autoflush=False, future=True)
                with self.db_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("✅ Backend DB connected")
        except Exception as e:
            logger.warning(f"Backend DB connection failed: {e}")

        # ----------------- ARTIFACTS (ranker) -----------------
        logger.info("\n📦 Loading model artifacts...")
        self.artifact_mgr = ArtifactManager(artifacts_base_dir=str(self.models_dir))
        req_version = (self.config.get("models") or {}).get("version", "latest")
        if req_version == "latest":
            vdir = self.artifact_mgr.get_latest_version_dir()
            if not vdir:
                raise RuntimeError("No model version found in models/ (missing latest.version)")
            self.current_version = vdir.name
        else:
            self.current_version = req_version

        self.ranking_model, self.ranking_scaler, self.ranking_feature_cols, self.metadata, self._version_dir = \
            _load_ranker_artifacts(base_dir=str(self.models_dir))
        logger.info("✅ Ranker artifacts loaded")

        # ----------------- DATA (CSV/dev) ---------------------
        logger.info("\n📊 Loading data...")
        from recommender.common.data_loader import load_data
        raw = load_data(str(self.data_dir)) or {}
        self.data = self._normalize_data_keys(raw)
        logger.info("✅ Data loaded")

        # ----------------- Optional embeddings / FAISS --------
        self.embeddings: Dict[str, Dict[int, np.ndarray]] = {"post": {}, "user": {}}
        self.faiss_index = None
        self.faiss_post_ids: List[int] = []
        self._load_optional_embeddings_file()   # post_embeddings.pkl nếu có
        self._load_faiss_artifacts()            # faiss_index.bin + mapping

        # ----------------- CF model placeholder ---------------
        self.cf_model = None  # nếu bạn có CF offline thì nạp ở đây

        # ----------------- Stats / Following ------------------
        self.user_stats: Dict = {}
        self.author_stats: Dict = {}
        self.following_dict: Dict[int, set] = {}

        # ----------------- Feature Engineer -------------------
        self.feature_engineer = FeatureEngineer(
            data=self.data,
            user_stats=self.user_stats,
            author_stats=self.author_stats,
            following=self.following_dict,
            embeddings=self.embeddings,
        )

        # ----------------- Recalls ----------------------------
        channels_cfg = (self.recall_config.get("channels") or {})
        self.following_recall = FollowingRecall(
            redis_client=self.redis, data=self.data,
            following_dict=self.following_dict,
            config=channels_cfg.get("following", {}) or {}
        )
        self.cf_recall = CFRecall(
            redis_client=self.redis, data=self.data,
            cf_model=self.cf_model,
            config=channels_cfg.get("collaborative_filtering", {}) or {}
        )
        self.content_recall = ContentRecall(
            redis_client=self.redis, data=self.data,
            embeddings=self.embeddings,
            faiss_index=self.faiss_index, faiss_post_ids=self.faiss_post_ids,
            config=channels_cfg.get("content_based", {}) or {}
        )
        self.trending_recall = TrendingRecall(
            redis_client=self.redis, data=self.data,
            config=channels_cfg.get("trending", {}) or {}
        )
        self.covisit_recall = CovisitRecall(self.redis, k_per_anchor=25, max_anchors=5)

        # ----------------- Ranker / Reranker ------------------
        self.ranker = MLRanker(
            model=self.ranking_model,
            scaler=self.ranking_scaler,
            feature_cols=self.ranking_feature_cols,
            feature_engineer=self.feature_engineer,
            config=self.ranking_config,
        )
        rr = self.reranking_config
        if "freshness_enabled" in rr:
            rr.setdefault("freshness", {})["enabled"] = bool(rr.pop("freshness_enabled"))
        self.reranker = Reranker(config=rr)

        # ----------------- Telemetry (async) ------------------
        self.telemetry = AsyncDBLogger(self.telemetry_cfg, redis_client=self.redis)

        # ----------------- Final logs -------------------------
        if self.faiss_index is not None:
            logger.info("✅ FAISS ready: %d vectors", getattr(self.faiss_index, "ntotal", 0))
        else:
            logger.warning("⚠️  FAISS not loaded. Content recall will fallback to in-memory or return empty.\n"
                           "    Run: python -m recommender.offline.post_embeddings --mode full")

        logger.info("\n✅ ONLINE PIPELINE READY!")
        logger.info("=" * 70 + "\n")

    # ====================== Public API ======================
    def generate_feed(self, user_id: int, limit: int = 50, exclude_seen: Optional[List[int]] = None, mark_seen: bool = False) -> List[Dict]:
        st = time.time()
        stage_ms = {"recall": 0, "ranking": 0, "rerank": 0}
        reasons: List[str] = []
        last_diag: Optional[RecallDiag] = None

        try:
            # ---------- Recall ----------
            t1 = time.time()
            target = int(self.recall_config.get("target_count", 1000))
            candidates, diag = self._recall_candidates(user_id, target_count=target)
            last_diag = diag
            stage_ms["recall"] = int((time.time() - t1) * 1000)

            # ---------- Seen-dedupe ----------
            seen = set(exclude_seen or [])
            seen |= self._get_seen_posts(user_id)
            if seen:
                before = len(candidates)
                candidates = [pid for pid in candidates if pid not in seen]
                if before and before != len(candidates):
                    reasons.append(f"seen_filtered:{before - len(candidates)}")

            # ---------- Fallback ----------
            if not candidates:
                fb = (self.fallback_cfg.get("personalization_failure") or {}).get("strategy", "trending")
                fallback = self._fallback_candidates(user_id, fb, k=max(limit * 3, 100))
                reasons.append(f"fallback:{fb}")
                candidates = fallback

            if not candidates:
                logger.warning("No candidates for user %s even after fallback", user_id)
                self._log_feed_telemetry(user_id, reasons, stage_ms, st, last_diag, final_count=0)
                return []

            # ---------- Ranking ----------
            t2 = time.time()
            ranked_df = self.ranker.rank(user_id, candidates)
            stage_ms["ranking"] = int((time.time() - t2) * 1000)
            if ranked_df.empty:
                reasons.append("ranking_empty")
                self._log_feed_telemetry(user_id, reasons, stage_ms, st, last_diag, final_count=0)
                return []

            top_k = int(self.ranking_config.get("top_k", 100))
            ranked_df = ranked_df.head(top_k)

            # ---------- Re-ranking ----------
            t3 = time.time()
            post_meta = self._build_post_metadata(ranked_df["post_id"].tolist())
            final_feed = self.reranker.rerank(ranked_df=ranked_df, post_metadata=post_meta, limit=limit)
            stage_ms["rerank"] = int((time.time() - t3) * 1000)

            # ---------- Mark Seen ----------
            if mark_seen: # Không auto mark-seen khi gọi feed
                self._mark_seen_posts(user_id, [x["post_id"] for x in final_feed])

            # ---------- Telemetry ----------
            self._log_feed_telemetry(user_id, reasons, stage_ms, st, last_diag, final_count=len(final_feed))

            return final_feed

        except Exception as e:
            logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
            return []

    # ====================== Internals ========================
    def _normalize_data_keys(self, raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if raw is None:
            return {}
        data = dict(raw)
        if "users" not in data and "user" in data: data["users"] = data["user"]
        if "posts" not in data and "post" in data: data["posts"] = data["post"]
        if "friendships" not in data and "friendship" in data: data["friendships"] = data["friendship"]
        if "post_hashtags" not in data and "post_hashtag" in data: data["post_hashtags"] = data["post_hashtag"]
        return data

    # ---------- FAISS & Embedding loaders ----------
    def _load_optional_embeddings_file(self):
        """
        Nạp models/post_embeddings.pkl (nếu có) để dùng cho một số feature/diag.
        Không bắt buộc vì ContentRecall sẽ ưu tiên dùng FAISS.
        """
        print(self.models_dir)
        pkl = self.models_dir / "post_embeddings.pkl"
        if pkl.exists():
            try:
                with pkl.open("rb") as f:
                    obj = pickle.load(f)  # type: ignore
                # chấp nhận 2 format: {post_id: vec} hoặc {'post':{...}, 'user':{...}}
                if isinstance(obj, dict) and "post" in obj:
                    self.embeddings.update(obj)
                else:
                    self.embeddings["post"] = obj  # type: ignore
                logger.info("Optional embeddings file loaded: %s", pkl)
            except Exception as e:
                logger.warning("Failed to load optional embeddings file: %s", e)

    def _load_faiss_artifacts(self):
        """
        Nạp FAISS index + mapping post_ids từ models/.
        """
        if not FAISS_OK:
            logger.warning("faiss not installed; content-based recall will be limited.")
            return
        print(self.models_dir)
        idx_path = self.models_dir / "faiss_index.bin"
        map_path = self.models_dir / "faiss_post_ids.pkl"
        if not (idx_path.exists() and map_path.exists()):
            logger.warning("FAISS artifacts not found in %s (missing faiss_index.bin or faiss_post_ids.pkl)", self.models_dir)
            return
        try:
            self.faiss_index = faiss.read_index(str(idx_path))  # type: ignore
            with map_path.open("rb") as f:
                self.faiss_post_ids = list(pickle.load(f))  # type: ignore
            # Sanity
            if hasattr(self.faiss_index, "ntotal"):
                if int(self.faiss_index.ntotal) != len(self.faiss_post_ids):
                    logger.warning("FAISS vectors (%s) != mapping size (%s)", int(self.faiss_index.ntotal), len(self.faiss_post_ids))
            logger.info("FAISS loaded: %d vectors", getattr(self.faiss_index, "ntotal", 0))
        except Exception as e:
            logger.warning("Failed to load FAISS artifacts: %s", e)
            self.faiss_index = None
            self.faiss_post_ids = []

    # ---------------- Recall orchestration -------------------

    # ---- NEW: recall following via Redis ZSET + DB backfill ----
    def _recall_following_redis(self, user_id: int, k: int) -> list[int]:
        out, seen = [], set()
        channels_cfg = (self.recall_config.get("channels") or {}).get("following", {}) or {}
        recent_hours = int(channels_cfg.get("recent_hours", 24 * 30))  # nới 30 ngày cho giai đoạn đầu
        max_fetch    = int(channels_cfg.get("max_fetch", 5000))

        # 1) Ưu tiên từ ZSET following:{user_id}:posts (được webhook đẩy)
        try:
            if self.redis:
                now = int(time.time())
                min_ts = now - recent_hours * 3600
                key = f"following:{user_id}:posts"
                rows = self.redis.zrevrangebyscore(key, max=now, min=min_ts, start=0, num=max_fetch) or []
                for s in rows:
                    pid = int(s)
                    if pid not in seen:
                        out.append(pid); seen.add(pid)
                    if len(out) >= k:
                        return out
        except Exception:
            pass

        # 2) Backfill từ danh sách author mà user follow (following_dict/DB)
        try:
            authors = list((self.following_dict or {}).get(user_id, []))
            if authors and "posts" in self.data:
                posts_df = self.data["posts"]
                sub = posts_df[posts_df["UserId"].isin(authors)].copy()
                if "CreateDate" in sub.columns:
                    dt = pd.to_datetime(sub["CreateDate"], errors="coerce", utc=True)
                    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=recent_hours)
                    sub = sub[dt >= cutoff]
                if "CreateDate" in sub.columns:
                    sub = sub.sort_values("CreateDate", ascending=False)
                for pid in sub["Id"].dropna().astype(int).tolist():
                    if pid not in seen:
                        out.append(pid); seen.add(pid)
                    if len(out) >= k:
                        break
        except Exception:
            pass

        return out[:k]



    # def _recall_candidates(self, user_id: int, target_count: int = 1000) -> Tuple[List[int], RecallDiag]:
    #     from concurrent.futures import ThreadPoolExecutor, as_completed

    #     channels_cfg = self.recall_config.get("channels", {}) or {}
    #     timeout_ms = int(self.recall_config.get("timeout_ms", 500))
    #     parallel = bool((self.performance_cfg.get("parallel") or {}).get("recall_channels", True))

    #     # quotas
    #     q_follow  = int((channels_cfg.get("following", {}) or {}).get("count", 400))
    #     q_cf      = int((channels_cfg.get("collaborative_filtering", {}) or {}).get("count", 300))
    #     q_content = int((channels_cfg.get("content_based", {}) or {}).get("count", 200))
    #     q_trend   = int((channels_cfg.get("trending", {}) or {}).get("count", 100))
    #     q_covisit = int((channels_cfg.get("covisit", {}) or {}).get("count", 150))

    #     diag = RecallDiag(notes={})
    #     results: Dict[str, List[int]] = {"following": [], "cf": [], "content": [], "trending": [], "covisit": []}

    #     def _run(name: str, fn, k: int) -> Tuple[str, List[int]]:
    #         try:
    #             return name, fn(k=k)
    #         except Exception as e:
    #             logger.debug(f"{name} recall error: {e}")
    #             return name, []

    #     calls = [
    #         ("following", lambda kk: self._recall_following_redis(user_id, kk), q_follow),
    #         ("cf",        self.cf_recall.recall,        q_cf),
    #         ("content",   self.content_recall.recall,   q_content),
    #         ("covisit",   self.covisit_recall.recall,   q_covisit),
    #         ("trending",  self.trending_recall.recall,  q_trend),
    #     ]
    #     print(f"{calls} recall channels")
    #     if parallel:
    #         with ThreadPoolExecutor(max_workers=5) as ex:
    #             futs = {ex.submit(_run, n, f, k): n for (n, f, k) in calls}
    #             for ft in as_completed(futs, timeout=max(timeout_ms/1000.0, 0.05) + 5.0):
    #                 try:
    #                     n, posts = ft.result(timeout=max(timeout_ms/1000.0, 0.05))
    #                     results[n] = posts or []
    #                 except Exception:
    #                     pass
    #     else:
    #         for n, f, k in calls:
    #             name, posts = _run(n, f, k)
    #             results[name] = posts or []

    #     diag.following = len(results["following"])
    #     diag.cf        = len(results["cf"])
    #     diag.content   = len(results["content"])
    #     diag.covisit   = len(results["covisit"])
    #     diag.trending  = len(results["trending"])

    #     # unique preserve order
    #     all_candidates: List[int] = []
    #     for n in ["following", "cf", "content", "covisit", "trending"]:
    #         for pid in results[n]:
    #             if pid not in all_candidates:
    #                 all_candidates.append(pid)

    #     print(f"All candidates: {all_candidates}")
    #     unique_candidates = all_candidates[:target_count]
    #     diag.total = len(unique_candidates)

    #     logger.info(
    #         "RECALL SUMMARY | user=%s | following=%s cf=%s content=%s covisit=%s trending=%s | total=%s",
    #         user_id, diag.following, diag.cf, diag.content, diag.covisit, diag.trending, diag.total,
    #     )

    #     # Notes
    #     if diag.following == 0 and self._user_following_count(user_id) == 0:
    #         diag.notes["following"] = "User has no following"
    #     if diag.trending == 0:
    #         tw = (channels_cfg.get("trending", {}) or {}).get("trending_window_hours", 6)
    #         if self._trending_empty_last_hours(tw):
    #             diag.notes["trending"] = "No posts within trending window"
    #     if diag.content == 0 and not self._has_user_embedding(user_id):
    #         diag.notes["content"] = "User embedding missing (cold start)"
    #     if self.faiss_index is None:
    #         diag.notes["content"] = (diag.notes.get("content","") + " | FAISS missing").strip(" |")

    #     return unique_candidates, diag

    def _recall_candidates(self, user_id: int, target_count: int = 1000) -> Tuple[List[int], RecallDiag]:
        channels_cfg = self.recall_config.get("channels", {}) or {}
        timeout_ms = int(self.recall_config.get("timeout_ms", 500))
        parallel = bool((self.performance_cfg.get("parallel") or {}).get("recall_channels", True))

        # quotas
        q_follow  = int((channels_cfg.get("following", {}) or {}).get("count", 400))
        q_cf      = int((channels_cfg.get("collaborative_filtering", {}) or {}).get("count", 300))
        q_content = int((channels_cfg.get("content_based", {}) or {}).get("count", 200))
        q_trend   = int((channels_cfg.get("trending", {}) or {}).get("count", 100))
        q_covisit = int((channels_cfg.get("covisit", {}) or {}).get("count", 150))

        diag = RecallDiag(notes={})
        results: Dict[str, List[int]] = {"following": [], "cf": [], "content": [], "trending": [], "covisit": []}
        reasons: Dict[str, List[str]] = {"following": [], "cf": [], "content": [], "trending": [], "covisit": []}

        def _normalize_result(name: str, ret) -> Tuple[List[int], List[str]]:
            """
            Chuẩn hoá giá trị trả về của mỗi recall channel:
                - Nếu hàm trả về List[int] -> (posts, [])
                - Nếu trả về Tuple[List[int], List[str]] -> (posts, reasons)
            """
            if ret is None:
                return [], []
            if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[0], list):
                return (ret[0] or []), (ret[1] or [])
            if isinstance(ret, list):
                return ret, []
            # bất thường
            return [], [f"UNEXPECTED_RETURN_TYPE:{type(ret)}"]

        def _run(name: str, fn, k: int) -> Tuple[str, List[int], List[str]]:
            try:
                ret = fn(k=k)
                posts, _reasons = _normalize_result(name, ret)
                return name, posts, _reasons
            except Exception as e:
                logger.exception("%s recall error", name)
                return name, [], [f"EXCEPTION:{e!r}"]

        calls = [
            ("following", lambda kk: self._recall_following_redis(user_id, kk), q_follow),
            ("cf",        self.cf_recall.recall,        q_cf),
            ("content",   self.content_recall.recall,   q_content),
            ("covisit",   self.covisit_recall.recall,   q_covisit),
            ("trending",  self.trending_recall.recall,  q_trend),
        ]
        logger.info("%s recall channels", [(n, f, k) for (n, f, k) in calls])
        from concurrent.futures import ThreadPoolExecutor, as_completed
        started = time.time()
        if parallel:
            with ThreadPoolExecutor(max_workers=5) as ex:
                futs = {ex.submit(_run, n, f, k): n for (n, f, k) in calls}
                for ft in as_completed(futs, timeout=max(timeout_ms/1000.0, 0.05) + 5.0):
                    try:
                        n, posts, rsns = ft.result(timeout=max(timeout_ms/1000.0, 0.05))
                        results[n] = posts or []
                        reasons[n] = rsns or []
                    except Exception as e:
                        # đã exception ở _run, tuy nhiên vẫn catch ở đây cho chắc
                        logger.debug("as_completed result error: %r", e)
        else:
            for n, f, k in calls:
                name, posts, rsns = _run(n, f, k)
                results[name] = posts or []
                reasons[name] = rsns or []

        # Ghi số lượng trước khi lọc status
        diag.following = len(results["following"])
        diag.cf        = len(results["cf"])
        diag.content   = len(results["content"])
        diag.covisit   = len(results["covisit"])
        diag.trending  = len(results["trending"])

        # Hợp nhất + preserve order
        all_candidates: List[int] = []
        for n in ["following", "cf", "content", "covisit", "trending"]:
            for pid in results[n]:
                if pid not in all_candidates:
                    all_candidates.append(pid)

        # LỌC STATUS=10 Ở ĐÂY (yêu cầu của anh) — KHÔNG LÀM SEEN FILTER ở service
        # Dùng redis hash/key: post:{pid}:status -> int
        def _status_ok(pid: int) -> bool:
            try:
                st = self.redis.hget(f"post:{pid}", "status")
                # fallback: có repo dùng key "post:{pid}:meta" -> HGET ... status
                if st is None:
                    st = self.redis.get(f"post:{pid}:status")
                if st is None:
                    return False  # thiếu status -> loại
                return int(st) == 10
            except Exception as e:
                logger.debug("status check error pid=%s e=%r", pid, e)
                return False

        before_status = len(all_candidates)
        filtered_candidates = [pid for pid in all_candidates if _status_ok(pid)]
        diag.status_filtered = before_status - len(filtered_candidates)

        unique_candidates = filtered_candidates[:target_count]
        diag.total = len(unique_candidates)

        # Ghi reasons chi tiết
        diag.reasons = reasons

        # Notes bổ sung
        if diag.following == 0 and self._user_following_count(user_id) == 0:
            diag.notes["following"] = "User has no following"
        if diag.trending == 0:
            tw = (channels_cfg.get("trending", {}) or {}).get("trending_window_hours", 6)
            if self._trending_empty_last_hours(tw):
                diag.notes["trending"] = "No posts within trending window"
        if diag.content == 0 and not self._has_user_embedding(user_id):
            prev = diag.notes.get("content", "")
            diag.notes["content"] = (prev + " | User embedding missing (cold start)").strip(" |")
        if self.faiss_index is None:
            prev = diag.notes.get("content", "")
            diag.notes["content"] = (prev + " | FAISS missing").strip(" |")

        # Health check Covisit graph
        try:
            covisit_info = self.covisit_recall.health_stats()
            if covisit_info.get("empty", False):
                diag.notes["covisit"] = "Covisit graph empty"
        except Exception as e:
            diag.notes["covisit"] = f"Covisit health error: {e!r}"

        took = int((time.time() - started) * 1000)
        logger.info(
            "RECALL SUMMARY | user=%s | following=%s cf=%s content=%s covisit=%s trending=%s | total=%s | status.filtered=%s | took_ms=%s",
            user_id, diag.following, diag.cf, diag.content, diag.covisit, diag.trending, diag.total, diag.status_filtered, took
        )
        logger.info("RECALL REASONS | %s", {k: v for k, v in diag.reasons.items() if v})
        logger.info("RECALL NOTES   | %s", diag.notes)

        # Debug print khi cần
        logger.debug("All candidates (pre-status): %s", all_candidates)
        logger.debug("Candidates (status=10 only): %s", unique_candidates)

        return unique_candidates, diag

    # -------------------------- Fallback --------------------------
    def _fallback_candidates(self, user_id: int, strategy: str, k: int) -> List[int]:
        strategy = (strategy or "trending").lower()
        if strategy == "trending":
            posts = self.trending_recall.recall(k=k)
            if posts:
                return posts
        if strategy == "popular":
            try:
                r = self.redis
                keys = r.scan_iter(match="post:*:eng1h", count=1000) if r else []
                scores = []
                for key in keys or []:
                    h = r.hgetall(key) if r else {}
                    sc = sum(float(v or 0) for v in h.values()) if h else 0.0
                    if sc > 0:
                        pid = int(key.split(":")[1])
                        scores.append((sc, pid))
                scores.sort(reverse=True)
                return [pid for _, pid in scores[:k]]
            except Exception:
                pass
        # random recent if nothing else
        try:
            posts_df = self.data.get("posts")
            if posts_df is not None and not posts_df.empty:
                sub = posts_df.copy()
                if "CreateDate" in sub.columns:
                    dt = pd.to_datetime(sub["CreateDate"], errors="coerce", utc=True)
                    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=7)
                    sub = sub[dt >= cutoff]
                pids = sub["Id"].dropna().astype(int).tolist()
                np.random.shuffle(pids)
                return pids[:k]
        except Exception:
            pass
        return []

    # ---------------------- Seen tracking -----------------------
    def _seen_key(self, user_id: int) -> str:
        return f"user:{user_id}:seen_posts"

    def _get_seen_posts(self, user_id: int) -> set:
        if not self.redis:
            return set()
        try:
            rows = self.redis.zrevrange(self._seen_key(user_id), 0, 2000)
            return set(int(x) for x in rows) if rows else set()
        except Exception:
            return set()

    def _mark_seen_posts(self, user_id: int, post_ids: List[int]):
        if not (self.redis and post_ids):
            return
        try:
            now = int(time.time())
            key = self._seen_key(user_id)
            pipe = self.redis.pipeline()
            for pid in post_ids:
                pipe.zadd(key, {str(int(pid)): now})
            pipe.expire(key, self.ttl_seen)
            pipe.execute()
        except Exception:
            pass

    # ---------------------- Post metadata -----------------------
    def _safe_text(self, v) -> str:
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        return str(v).strip()

    def _build_post_metadata(self, post_ids: List[int]) -> Dict[int, Dict]:
        meta: Dict[int, Dict] = {}

        # 1) CSV/dev
        try:
            post_df: pd.DataFrame = self.data["posts"]
            sub = post_df[post_df["Id"].isin(post_ids)][["Id", "UserId", "Status", "CreateDate", "Content"]].copy()
            for _, r in sub.iterrows():
                pid = int(r["Id"])
                content = self._safe_text(r.get("Content"))
                title = ""
                content_hash = None
                if content or title:
                    import hashlib
                    content_hash = hashlib.sha1(f"{title.lower()}|{content.lower()}".encode("utf-8")).hexdigest()
                meta[pid] = {
                    "author_id": int(r.get("UserId")) if not pd.isna(r.get("UserId")) else None,
                    "status": int(r.get("Status")) if not pd.isna(r.get("Status")) else None,
                    "created_at": pd.to_datetime(r.get("CreateDate"), errors="coerce", utc=True),
                    "title": title,
                    "content": content,
                    "content_hash": content_hash,
                }
        except Exception:
            pass

        # 2) DB fallback
        missing = [pid for pid in post_ids if pid not in meta]
        if missing and self.db_session_factory:
            with self.db_session_factory() as s:
                rows = s.execute(
                    text("SELECT Id, UserId, Status, CreateDate, Content FROM Post WHERE Id IN :ids"),
                    {"ids": tuple(missing)},
                ).all()
                for row in rows:
                    pid, uid, st, cd, ct = row
                    content = self._safe_text(ct)
                    title = ""
                    content_hash = None
                    if content or title:
                        import hashlib
                        content_hash = hashlib.sha1(f"{title.lower()}|{content.lower()}".encode("utf-8")).hexdigest()
                    meta[int(pid)] = {
                        "author_id": int(uid) if uid is not None else None,
                        "status": int(st) if st is not None else None,
                        "created_at": pd.to_datetime(cd, errors="coerce", utc=True),
                        "title": title,
                        "content": content,
                        "content_hash": content_hash,
                    }
        return meta

    # -------------------- Diag utils ------------------
    def _user_following_count(self, user_id: int) -> int:
        try:
            flw = (self.following_dict or {}).get(user_id)
            return len(flw) if flw else 0
        except Exception:
            return 0

    def _has_user_embedding(self, user_id: int) -> bool:
        try:
            return user_id in (self.embeddings.get("user", {}) or {})
        except Exception:
            return False

    def _trending_empty_last_hours(self, hours: int) -> bool:
        try:
            post_df = self.data.get("posts")
            if post_df is None or post_df.empty or "CreateDate" not in post_df.columns:
                return True
            dt = pd.to_datetime(post_df["CreateDate"], errors="coerce", utc=True)
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=hours)
            return bool((dt >= cutoff).sum() == 0)
        except Exception:
            return True

    # ------------------------ Telemetry ------------------------
    def _log_feed_telemetry(self, user_id: int, reasons: List[str], stage_ms: Dict[str, int],
                            st: float, diag: Optional[RecallDiag], final_count: int):
        try:
            if not self.telemetry or not getattr(self.telemetry, "enabled", False):
                return
            data = {
                "user_id": user_id,
                "timestamp": time.time(),
                "recall": {
                    "following": (diag.following if diag else 0),
                    "cf": (diag.cf if diag else 0),
                    "content": (diag.content if diag else 0),
                    "trending": (diag.trending if diag else 0),
                    "total": (diag.total if diag else 0),
                },
                "final_count": final_count,
                "reasons": reasons,
                "latency_ms": int((time.time() - st) * 1000),
                "stage_ms": stage_ms,
            }
            self.telemetry.log_feed_request(data)
        except Exception:
            pass
