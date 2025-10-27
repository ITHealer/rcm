
# recommender/common/data_loading.py
"""
Data Loading from MySQL & CSV (dev fallback)
===========================================

- Gộp interactions từ nhiều bảng MySQL:
  * PostView      -> action='view'      (ViewDate | CreateDate)
  * PostReaction  -> join ReactionType  -> action in {like, love, laugh, wow, sad, angry, care}
  * Comment       -> action='comment'
- Lọc theo window [since, until] (UTC), chunking LIMIT/OFFSET
- Trả kèm side tables: users, posts, friendships, post_hashtags, reaction_types, comments
- Chuẩn hoá schema interactions: user_id, post_id, action, created_at (UTC)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------- Utilities -------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _ensure_datetime_utc(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, utc=True, errors="coerce")
    else:
        if series.dt.tz is None:
            series = series.dt.tz_localize("UTC")
        else:
            series = series.dt.tz_convert("UTC")
    return series

def _clip_window(df: pd.DataFrame, col: str, since: datetime, until: datetime) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    d = df.copy()
    d[col] = _ensure_datetime_utc(d[col])
    m = (d[col] >= since) & (d[col] <= until)
    return d.loc[m].reset_index(drop=True)

def _read_sql_chunked(
    conn: Any, base_sql: str, base_params: Tuple, chunk_size: int, normalize_ts_cols: List[str]
) -> pd.DataFrame:
    """Chunking kiểu LIMIT/OFFSET (MySQL/SQLAlchemy compatible)."""
    limit = int(chunk_size)
    offset = 0
    chunks: List[pd.DataFrame] = []
    total = 0
    while True:
        sql = f"{base_sql} LIMIT %s OFFSET %s"
        params = (*base_params, limit, offset)
        df = pd.read_sql_query(sql, conn, params=params)
        n = len(df)
        if n == 0:
            break
        for c in normalize_ts_cols:
            if c in df.columns:
                df[c] = _ensure_datetime_utc(df[c])
        chunks.append(df)
        total += n
        offset += n
        logger.info(f"  -> loaded {n:,} rows (total {total:,})")
        if n < limit:
            break
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame()


# ----------------------------- DataLoader ------------------------------------

class DataLoader:
    def __init__(
        self,
        db_connection: Any = None,
        config: Optional[Dict] = None,
        data_dir: str = "dataset",
    ):
        self.conn = db_connection
        self.config = config or {}
        self.data_dir = Path(self.config.get("csv_dir", data_dir) or data_dir)
        self.chunk_size = int(self.config.get("chunk_size", 200_000))
        self.lookback_days = int(self.config.get("lookback_days", 14))

        # table names
        t = self.config.get("tables", {}) or {}
        self.tbl_view = t.get("post_view", "PostView")
        self.tbl_react = t.get("post_reaction", "PostReaction")
        self.tbl_reaction_type = t.get("reaction_type", "ReactionType")
        self.tbl_user = t.get("user", "User")
        self.tbl_post = t.get("post", "Post")
        self.tbl_hashtag = t.get("post_hashtag", "PostHashtag")
        self.tbl_friendship = t.get("friendship", "Friendship")
        self.tbl_comment = t.get("comment", "Comment")

        # ReactionType → action mapping (Code/Name). MẶC ĐỊNH KHỚP ẢNH DB: like/love/laugh/wow/sad/angry/care
        self.code_map = {
            **{"like": "like", "love": "love", "laugh": "laugh", "wow": "wow", "sad": "sad", "angry": "angry", "care": "care"},
            **(self.config.get("reaction_code_map") or {}),
        }
        self.name_map = {
            **{"Like": "like", "Love": "love", "Laugh": "laugh", "Wow": "wow", "Sad": "sad", "Angry": "angry", "Care": "care"},
            **(self.config.get("reaction_name_map") or {}),
        }

        # csv files
        self.csv_files = self.config.get("csv_files", {}) or {}

        logger.info("DataLoader initialized:")
        logger.info(f"  mode = {'DB' if self.conn is not None else 'CSV'}")
        logger.info(f"  lookback_days = {self.lookback_days} | chunk_size = {self.chunk_size:,}")

    # --------------------- public high-level APIs -----------------------------

    def load_training_bundle(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        use_csv: Optional[bool] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Trả về đầy đủ frames:
          - interactions (user_id, post_id, action, created_at)
          - views, reactions, comments, reaction_types, users, posts, friendships, post_hashtags
        """
        use_csv_mode = (use_csv if use_csv is not None else (self.conn is None))
        since, until = self._resolve_window(since, until)

        logger.info("\n" + "=" * 70)
        logger.info("LOADING TRAINING BUNDLE")
        logger.info("=" * 70)
        logger.info(f"Source: {'CSV' if use_csv_mode else 'DB'}; Window: {since.isoformat()} -> {until.isoformat()}")

        if use_csv_mode:
            frames = self._load_bundle_from_csv(since, until)
        else:
            frames = self._load_bundle_from_db(since, until)

        # Hợp nhất interactions
        inter = self._build_interactions(
            views=frames["views"], reactions=frames["reactions"],
            comments=frames["comments"], reaction_types=frames["reaction_types"]
        )
        inter = _clip_window(inter, "created_at", since, until)

        frames["interactions"] = inter
        self._log_interactions(inter)

        return frames

    # ------------------------ internal: window --------------------------------

    def _resolve_window(self, since: Optional[datetime], until: Optional[datetime]) -> Tuple[datetime, datetime]:
        now = _now_utc()
        u = (until or now).astimezone(timezone.utc)
        s = (since or (u - timedelta(days=self.lookback_days))).astimezone(timezone.utc)
        return s, u

    # ---------------------- internal: DB loaders ------------------------------

    def _load_bundle_from_db(self, since: datetime, until: datetime) -> Dict[str, pd.DataFrame]:
        return {
            "views": self._load_views_db(since, until),
            "reactions": self._load_reactions_db(since, until),
            "reaction_types": self._load_reaction_types_db(),
            "comments": self._load_comments_db(since, until),
            "users": self._load_users_db(),
            "posts": self._load_posts_db(),
            "friendships": self._load_friendships_db(),
            "post_hashtags": self._load_post_hashtags_db(),
        }

    def _load_views_db(self, since: datetime, until: datetime) -> pd.DataFrame:
        logger.info("DB: loading PostView ...")
        base = (
            f"SELECT Id, UserId, PostId, ViewDate, CreateDate, Status "
            f"FROM {self.tbl_view} "
            f"WHERE (ViewDate >= %s OR (ViewDate IS NULL AND CreateDate >= %s)) "
            f"AND (ViewDate <= %s OR (ViewDate IS NULL AND CreateDate <= %s)) "
            f"ORDER BY COALESCE(ViewDate, CreateDate) ASC"
        )
        df = _read_sql_chunked(self.conn, base, (since, since, until, until), self.chunk_size, ["ViewDate", "CreateDate"])
        if df.empty:
            return pd.DataFrame(columns=["user_id", "post_id", "action", "created_at"])
        df.rename(columns={"UserId": "user_id", "PostId": "post_id"}, inplace=True)
        df["created_at"] = df["ViewDate"].fillna(df["CreateDate"])
        df["action"] = "view"
        return df[["user_id", "post_id", "action", "created_at"]]

    def _load_reactions_db(self, since: datetime, until: datetime) -> pd.DataFrame:
        logger.info("DB: loading PostReaction ...")
        base = (
            f"SELECT Id, PostId, UserId, ReactionTypeId, CreateDate, Status "
            f"FROM {self.tbl_react} "
            f"WHERE CreateDate >= %s AND CreateDate <= %s "
            f"ORDER BY CreateDate ASC"
        )
        df = _read_sql_chunked(self.conn, base, (since, until), self.chunk_size, ["CreateDate"])
        if df.empty:
            return pd.DataFrame(columns=["user_id", "post_id", "reaction_type_id", "created_at"])
        df.rename(columns={"UserId": "user_id", "PostId": "post_id", "CreateDate": "created_at", "ReactionTypeId": "ReactionTypeId"}, inplace=True)
        return df[["user_id", "post_id", "ReactionTypeId", "created_at"]]

    def _load_reaction_types_db(self) -> pd.DataFrame:
        logger.info("DB: loading ReactionType ...")
        sql = f"SELECT Id AS ReactionTypeId, Code, Name FROM {self.tbl_reaction_type}"
        return pd.read_sql_query(sql, self.conn)

    def _load_comments_db(self, since: datetime, until: datetime) -> pd.DataFrame:
        logger.info("DB: loading Comment ...")
        base = (
            f"SELECT Id, PostId, UserId, CreateDate, Status "
            f"FROM {self.tbl_comment} "
            f"WHERE CreateDate >= %s AND CreateDate <= %s "
            f"ORDER BY CreateDate ASC"
        )
        df = _read_sql_chunked(self.conn, base, (since, until), self.chunk_size, ["CreateDate"])
        if df.empty:
            return pd.DataFrame(columns=["user_id", "post_id", "action", "created_at"])
        df.rename(columns={"UserId": "user_id", "PostId": "post_id", "CreateDate": "created_at"}, inplace=True)
        df["action"] = "comment"
        return df[["user_id", "post_id", "action", "created_at"]]

    def _load_users_db(self) -> pd.DataFrame:
        logger.info("DB: loading User ...")
        sql = f"SELECT Id, CreateDate FROM {self.tbl_user}"
        df = pd.read_sql_query(sql, self.conn)
        if "CreateDate" in df.columns:
            df["CreateDate"] = _ensure_datetime_utc(df["CreateDate"])
        return df

    def _load_posts_db(self) -> pd.DataFrame:
        logger.info("DB: loading Post ...")
        sql = f"SELECT Id, UserId, CreateDate, IsRepost, IsPin FROM {self.tbl_post}"
        df = pd.read_sql_query(sql, self.conn)
        if "CreateDate" in df.columns:
            df["CreateDate"] = _ensure_datetime_utc(df["CreateDate"])
        return df

    def _load_post_hashtags_db(self) -> pd.DataFrame:
        logger.info("DB: loading PostHashtag ...")
        sql = f"SELECT Id, PostId, HashtagId, CreateDate FROM {self.tbl_hashtag}"
        return pd.read_sql_query(sql, self.conn)

    def _load_friendships_db(self) -> pd.DataFrame:
        logger.info("DB: loading Friendship ...")
        sql = f"SELECT Id, UserId, FriendId, CreateDate FROM {self.tbl_friendship}"
        return pd.read_sql_query(sql, self.conn)

    # ---------------------- internal: CSV loaders -----------------------------

    def _load_bundle_from_csv(self, since: datetime, until: datetime) -> Dict[str, pd.DataFrame]:
        logger.info("CSV: loading all sources ...")

        def _csv(name: str, default: str) -> Path:
            return self.data_dir / self.config.get("csv_files", {}).get(name, default)

        def _read_csv(path: Path, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
            if not path.exists():
                logger.warning(f"CSV not found: {path}")
                return pd.DataFrame()
            return pd.read_csv(path, parse_dates=parse_dates)

        views = _read_csv(_csv("post_view", "PostView.csv"), parse_dates=["ViewDate", "CreateDate"])
        reactions = _read_csv(_csv("post_reaction", "PostReaction.csv"), parse_dates=["CreateDate"])
        reaction_types = _read_csv(_csv("reaction_type", "ReactionType.csv"))
        comments = _read_csv(_csv("comment", "Comment.csv"), parse_dates=["CreateDate"])
        users = _read_csv(_csv("user", "User.csv"), parse_dates=["CreateDate"])
        posts = _read_csv(_csv("post", "Post.csv"), parse_dates=["CreateDate"])
        post_hashtags = _read_csv(_csv("post_hashtag", "PostHashtag.csv"), parse_dates=["CreateDate"])
        friendships = _read_csv(_csv("friendship", "Friendship.csv"), parse_dates=["CreateDate"])

        # Normalize to UTC
        for df, col in [
            (views, "ViewDate"), (views, "CreateDate"),
            (reactions, "CreateDate"), (comments, "CreateDate"),
            (users, "CreateDate"), (posts, "CreateDate"),
        ]:
            if not df.empty and col in df.columns:
                df[col] = _ensure_datetime_utc(df[col])

        # Filter by window for raw sources
        if not views.empty:
            views = views[(views["ViewDate"].fillna(views["CreateDate"]) >= since) &
                          (views["ViewDate"].fillna(views["CreateDate"]) <= until)]
        if not reactions.empty:
            reactions = reactions[(reactions["CreateDate"] >= since) & (reactions["CreateDate"] <= until)]
        if not comments.empty:
            comments = comments[(comments["CreateDate"] >= since) & (comments["CreateDate"] <= until)]

        return {
            "views": views.reset_index(drop=True),
            "reactions": reactions.reset_index(drop=True),
            "reaction_types": reaction_types.reset_index(drop=True),
            "comments": comments.reset_index(drop=True),
            "users": users.reset_index(drop=True),
            "posts": posts.reset_index(drop=True),
            "friendships": friendships.reset_index(drop=True),
            "post_hashtags": post_hashtags.reset_index(drop=True),
        }

    # ------------------------ build interactions -----------------------------

    def _build_interactions(
        self,
        views: pd.DataFrame,
        reactions: pd.DataFrame,
        comments: pd.DataFrame,
        reaction_types: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build unified interactions dataframe.
        Hỗ trợ cả 2 trường hợp:
        - views/comments đã chuẩn hoá (user_id, post_id, action, created_at)
        - hoặc raw (UserId, PostId, ViewDate/CreateDate,…)
        """
        parts: List[pd.DataFrame] = []

        # ---------- VIEWS ----------
        if views is not None and not views.empty:
            # Case A: đã chuẩn hoá
            if {"user_id", "post_id", "created_at"}.issubset(views.columns):
                v = views.copy()
                if "action" not in v.columns:
                    v["action"] = "view"
                v["created_at"] = _ensure_datetime_utc(v["created_at"])
                parts.append(v[["user_id", "post_id", "action", "created_at"]])
            # Case B: raw
            elif {"UserId", "PostId"}.issubset(views.columns) and (
                "ViewDate" in views.columns or "CreateDate" in views.columns
            ):
                v = views.rename(columns={"UserId": "user_id", "PostId": "post_id"}).copy()
                if "ViewDate" in v.columns and "CreateDate" in v.columns:
                    base_ts = v["ViewDate"].fillna(v["CreateDate"])
                elif "ViewDate" in v.columns:
                    base_ts = v["ViewDate"]
                else:
                    base_ts = v["CreateDate"]
                v["created_at"] = _ensure_datetime_utc(base_ts)
                v["action"] = "view"
                parts.append(v[["user_id", "post_id", "action", "created_at"]])
            else:
                logger.warning("PostView columns not recognized. Skipping views in interactions build.")

        # ---------- REACTIONS ----------
        if reactions is not None and not reactions.empty:
            r = reactions.rename(
                columns={"UserId": "user_id", "PostId": "post_id", "CreateDate": "created_at"}
            ).copy()

            # Join ReactionType nếu có
            if reaction_types is not None and not reaction_types.empty:
                # reaction_types: ReactionTypeId, Code, Name (đã đúng theo query DB)
                rt = reaction_types.copy()
                r = r.merge(rt[["ReactionTypeId", "Code", "Name"]], on="ReactionTypeId", how="left")

            # Lấy code/name an toàn, KHÔNG dùng 'or' với Series
            if "Code" in r.columns:
                codes = r["Code"].astype(str).str.lower()
            else:
                codes = pd.Series([None] * len(r), index=r.index)

            if "Name" in r.columns:
                names = r["Name"].astype(str)
            else:
                names = pd.Series([None] * len(r), index=r.index)

            # Map Code/Name -> action với fallback 'like'
            def _map_action(code_val: Optional[str], name_val: Optional[str]) -> str:
                act = None
                if code_val is not None and pd.notna(code_val):
                    act = self.code_map.get(str(code_val).lower())
                if act is None and name_val is not None and pd.notna(name_val):
                    act = self.name_map.get(str(name_val))
                return act or "like"

            r["action"] = [ _map_action(c, n) for c, n in zip(codes, names) ]
            r["created_at"] = _ensure_datetime_utc(r["created_at"])
            parts.append(r[["user_id", "post_id", "action", "created_at"]])

        # ---------- COMMENTS ----------
        if comments is not None and not comments.empty:
            # Case A: đã chuẩn hoá
            if {"user_id", "post_id", "created_at"}.issubset(comments.columns):
                c = comments.copy()
                if "action" not in c.columns:
                    c["action"] = "comment"
                c["created_at"] = _ensure_datetime_utc(c["created_at"])
                parts.append(c[["user_id", "post_id", "action", "created_at"]])
            # Case B: raw
            elif {"UserId", "PostId", "CreateDate"}.issubset(comments.columns):
                c = comments.rename(
                    columns={"UserId": "user_id", "PostId": "post_id", "CreateDate": "created_at"}
                ).copy()
                c["action"] = "comment"
                c["created_at"] = _ensure_datetime_utc(c["created_at"])
                parts.append(c[["user_id", "post_id", "action", "created_at"]])
            else:
                logger.warning("Comment columns not recognized. Skipping comments in interactions build.")

        # ---------- CONCAT ----------
        if not parts:
            return pd.DataFrame(columns=["user_id", "post_id", "action", "created_at"])

        inter = pd.concat(parts, ignore_index=True)
        inter["created_at"] = _ensure_datetime_utc(inter["created_at"])
        inter = inter.dropna(subset=["user_id", "post_id", "created_at"])
        inter["user_id"] = inter["user_id"].astype("int64", errors="ignore")
        inter["post_id"] = inter["post_id"].astype("int64", errors="ignore")
        return inter



    # ---------------------------- logging ------------------------------------

    @staticmethod
    def _log_interactions(df: pd.DataFrame) -> None:
        if df.empty:
            logger.info("Loaded 0 interactions.")
            return
        logger.info(
            f"✅ Interactions: {len(df):,} | users={df['user_id'].nunique():,} | posts={df['post_id'].nunique():,}"
        )
        logger.info(f"   Range: {df['created_at'].min()} -> {df['created_at'].max()}")
