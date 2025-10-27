

"""
Post Embeddings + FAISS (MySQL)
===============================

- Đọc Post/Hashtag/Reaction/View từ MySQL (schema như ảnh bạn cung cấp).
- Sinh embedding bằng SentenceTransformers (GPU nếu có).
- Lưu vào bảng MySQL `post_embeddings` (LONGBLOB).
- Build FAISS index (cosine qua IndexFlatIP + normalize).

Usage:
  python -m recommender.offline.post_embeddings --mode full
  python -m recommender.offline.post_embeddings --mode incremental --since-days 7
"""

from __future__ import annotations

import os
import sys
import math
import json
import time
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 3rd libs
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.pool import NullPool

try:
    import faiss
    FAISS_OK = True
except Exception:
    FAISS_OK = False

try:
    import torch
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except Exception:
    ST_OK = False

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("post_embeddings")

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# DB helpers
# --------------------------------------------------------------------------------------
def make_engine_from_env() -> Engine:
    url = "mysql+pymysql://way_root:YmhNWpppahN92AtJotFDoHnCoW38keDp@14.225.220.56:15479/wayjet_system"
    if not url:
        raise RuntimeError("Missing BACKEND_DB_URL (e.g. mysql+pymysql://user:pass@host:3306/dbname)")
    # dùng NullPool để script batch không giữ connection lâu
    engine = create_engine(url, poolclass=NullPool, future=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


def _is_mysql(engine: Engine) -> bool:
    try:
        name = engine.dialect.name.lower()
        return "mysql" in name
    except Exception:
        return False


def ensure_post_embeddings_table(engine: Engine):
    """
    Tạo bảng lưu embeddings cho MySQL. Không dùng cú pháp Postgres.
    """
    ddl_mysql = """
    CREATE TABLE IF NOT EXISTS post_embeddings (
      PostId BIGINT PRIMARY KEY,
      Embedding LONGBLOB NOT NULL,
      UpdatedAt TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    with engine.begin() as conn:
        if _is_mysql(engine):
            conn.exec_driver_sql(ddl_mysql)
        else:
            # Fallback generic (SQLite, v.v.) – không dùng ON UPDATE
            ddl_generic = """
            CREATE TABLE IF NOT EXISTS post_embeddings (
              PostId INTEGER PRIMARY KEY,
              Embedding BLOB NOT NULL,
              UpdatedAt TEXT NOT NULL
            );
            """
            conn.exec_driver_sql(ddl_generic)


# --------------------------------------------------------------------------------------
# Fetch data from your schema
# --------------------------------------------------------------------------------------
def fetch_posts(engine: Engine, mode: str = "incremental", since_days: int = 7) -> pd.DataFrame:
    """
    Đọc Post + hashtags (nếu có) theo schema của bạn:
      - Post(Id, UserId, Content, Status, CreateDate, ...)
      - PostHashtag(PostId, HashtagId, Status, ...)
      - (Optional) Hashtag(Id, Name) — nếu không có, dùng HashtagId làm tag.
    Chỉ lấy Status hợp lệ (10) và Content không rỗng.
    """
    where_time = ""
    params = {}
    if mode in ("incremental",) and since_days is not None:
        where_time = "AND p.CreateDate >= DATE_SUB(NOW(), INTERVAL :days DAY)"
        params["days"] = int(since_days)

    # Kiểm tra có bảng Hashtag(Name) hay không
    has_hashtag_name = False
    with engine.connect() as conn:
        try:
            conn.execute(text("SELECT 1 FROM Hashtag LIMIT 1"))
            # thử cột Name
            conn.execute(text("SELECT Name FROM Hashtag LIMIT 1"))
            has_hashtag_name = True
        except Exception:
            has_hashtag_name = False

    # Query MySQL: group_concat hashtags
    if has_hashtag_name:
        tag_expr = "GROUP_CONCAT(DISTINCT CONCAT('#', h.Name) SEPARATOR ' ')"
        join_tag = """
            LEFT JOIN PostHashtag ph ON ph.PostId = p.Id AND (ph.Status IS NULL OR ph.Status = 10)
            LEFT JOIN Hashtag h ON h.Id = ph.HashtagId
        """
    else:
        tag_expr = "GROUP_CONCAT(DISTINCT CONCAT('#', ph.HashtagId) SEPARATOR ' ')"
        join_tag = "LEFT JOIN PostHashtag ph ON ph.PostId = p.Id AND (ph.Status IS NULL OR ph.Status = 10)"

    sql = f"""
    SELECT
      p.Id           AS PostId,
      p.UserId       AS AuthorId,
      p.Content      AS Content,
      p.Status       AS Status,
      p.CreateDate   AS CreateDate,
      {tag_expr}     AS Hashtags
    FROM Post p
    {join_tag}
    WHERE
      (p.Status IS NULL OR p.Status = 10)
      AND p.Content IS NOT NULL AND LENGTH(TRIM(p.Content)) > 0
      {where_time}
    GROUP BY p.Id, p.UserId, p.Content, p.Status, p.CreateDate
    ORDER BY p.CreateDate DESC
    """

    with engine.connect() as conn:
        try:
            rows = conn.execute(text(sql), params).mappings().all()
        except ProgrammingError as e:
            logger.error("SQL error when reading posts: %s", e)
            raise

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No posts returned from DB.")
        return df

    # Chuẩn hóa cột/kiểu
    df["CreateDate"] = pd.to_datetime(df["CreateDate"], errors="coerce", utc=True)
    df["Hashtags"] = df["Hashtags"].fillna("")

    # Loại những bài quá cũ nếu chạy full mà DB quá lớn? (giữ nguyên cho bạn chủ động)
    logger.info("Fetched %s posts from DB.", len(df))
    return df


# --------------------------------------------------------------------------------------
# Embedding runner
# --------------------------------------------------------------------------------------
class MySQLEmbedder:
    def __init__(
        self,
        engine: Engine,
        model_name: str = None,
        batch_size: int = 512,
    ):
        if not ST_OK:
            raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")

        self.engine = engine
        self.batch = int(batch_size)
        self.model_name = model_name or os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Init embedder | model=%s | device=%s", self.model_name, device)
        self.model = SentenceTransformer(self.model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

        ensure_post_embeddings_table(self.engine)

    # --------------- Text prep ----------------
    @staticmethod
    def _prep_text(content: str, hashtags: str) -> str:
        content = (content or "").strip()
        if hashtags and hashtags.strip():
            return f"{content} [HASHTAGS: {hashtags.strip()}]"
        return content or "empty post"

    # --------------- Already embedded? -------
    def _get_existing_ids(self) -> set[int]:
        sql = "SELECT PostId FROM post_embeddings"
        with self.engine.connect() as conn:
            try:
                rows = conn.execute(text(sql)).fetchall()
            except Exception:
                return set()
        return set(int(r[0]) for r in rows)

    # --------------- Save embeddings ---------
    def _save_embeddings(self, post_ids: List[int], embs: np.ndarray):
        rows = []
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        # store as float32 bytes
        for pid, vec in zip(post_ids, embs):
            rows.append({"PostId": int(pid), "Embedding": vec.astype(np.float32).tobytes(), "UpdatedAt": now})

        if _is_mysql(self.engine):
            # MySQL upsert
            sql = text("""
                INSERT INTO post_embeddings (PostId, Embedding, UpdatedAt)
                VALUES (:PostId, :Embedding, :UpdatedAt)
                ON DUPLICATE KEY UPDATE
                    Embedding = VALUES(Embedding),
                    UpdatedAt = VALUES(UpdatedAt)
            """)
        else:
            # Generic replace (delete+insert)
            sql_del = text("DELETE FROM post_embeddings WHERE PostId = :pid")
            sql_ins = text("INSERT INTO post_embeddings (PostId, Embedding, UpdatedAt) VALUES (:PostId, :Embedding, :UpdatedAt)")
            with self.engine.begin() as conn:
                for r in rows:
                    conn.execute(sql_del, {"pid": r["PostId"]})
                    conn.execute(sql_ins, r)
            logger.info("Saved %d embeddings (generic).", len(rows))
            return

        # batch insert
        with self.engine.begin() as conn:
            for i in range(0, len(rows), 1000):
                conn.execute(sql, rows[i:i+1000])
        logger.info("Saved %d embeddings to MySQL.", len(rows))

    # --------------- Build FAISS ------------
    def build_faiss(self):
        if not FAISS_OK:
            logger.warning("faiss not available. Skip index build.")
            return
        sql = "SELECT PostId, Embedding FROM post_embeddings"
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()

        if not rows:
            logger.warning("No vectors to build index.")
            return

        post_ids: List[int] = []
        vecs: List[np.ndarray] = []
        for pid, blob in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            post_ids.append(int(pid))
            vecs.append(v)

        X = np.vstack(vecs)
        # normalize for cosine
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X.astype(np.float32))

        faiss.write_index(index, str(MODELS_DIR / "faiss_index.bin"))
        with open(MODELS_DIR / "faiss_post_ids.pkl", "wb") as f:
            pickle.dump(post_ids, f)

        logger.info("FAISS built: %s vectors -> %s", index.ntotal, MODELS_DIR / "faiss_index.bin")

    # --------------- Run --------------------
    def run(self, mode: str = "incremental", since_days: int = 7, save: bool = True):
        """
        mode = full | incremental
        """
        posts = fetch_posts(self.engine, mode=mode, since_days=since_days)
        if posts.empty:
            logger.info("No posts to process.")
            return

        existing = self._get_existing_ids()
        # nếu full → chỉ bỏ qua id đã có; incremental → đã filter theo thời gian rồi, vẫn skip id đã có.
        posts = posts[~posts["PostId"].astype(int).isin(existing)].reset_index(drop=True)
        if posts.empty:
            logger.info("All posts already embedded (nothing new).")
            self.build_faiss()
            return

        logger.info("To-embed: %d", len(posts))

        texts: List[str] = []
        ids: List[int] = []
        for _, r in posts.iterrows():
            texts.append(self._prep_text(str(r["Content"]), str(r.get("Hashtags", ""))))
            ids.append(int(r["PostId"]))

        # batch encode
        all_vecs: List[np.ndarray] = []
        total = len(texts)
        bs = self.batch
        n_batches = math.ceil(total / bs)
        for i in range(n_batches):
            s, e = i * bs, min((i + 1) * bs, total)
            t0 = time.time()
            vec = self.model.encode(
                texts[s:e],
                batch_size=bs,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_vecs.append(vec)
            logger.info("Encoded batch %d/%d (%d items) in %.1fs", i+1, n_batches, e - s, time.time()-t0)

        X = np.vstack(all_vecs)
        if save:
            self._save_embeddings(ids, X)
        self.build_faiss()
        logger.info("Done. Embedded %d posts.", len(ids))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["full", "incremental"], default="incremental")
    p.add_argument("--since-days", type=int, default=7, help="Only used for incremental")
    p.add_argument("--batch-size", type=int, default=int(os.getenv("EMB_BATCH", "512")))
    p.add_argument("--model", type=str, default=os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    return p.parse_args()


def main():
    args = _parse_args()
    engine = make_engine_from_env()
    emb = MySQLEmbedder(engine=engine, model_name=args.model, batch_size=args.batch_size)
    emb.run(mode=args.mode, since_days=args.since_days, save=True)


if __name__ == "__main__":
    main()
