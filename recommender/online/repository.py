# recommender/online/repository.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .cache import RedisClient, RedisConfig


class OnlineRepository:
    """
    Repository lớp đọc dữ liệu cho ONLINE:
    - ƯU TIÊN Redis (realtime interactions, following)
    - Fallback MySQL khi cache không có / thiếu
    """
    def __init__(self, db_url: str, redis_cfg: RedisConfig | None = None):
        self.engine: Engine = create_engine(db_url, pool_pre_ping=True, future=True)
        self.redis = RedisClient(redis_cfg) if redis_cfg else RedisClient(None)

    # ---------------------- realtime interactions -----------------------------
    def get_user_recent_interactions(self, user_id: int, days: int = 30) -> pd.DataFrame:
        since = datetime.now(timezone.utc) - timedelta(days=days)

        # 1) Redis first
        df_cache = self.redis.get_recent_interactions(user_id, since)
        # 2) DB fallback (và hợp nhất)
        sql = text("""
            SELECT PostId AS post_id, 'view' AS action, COALESCE(ViewDate, CreateDate) AS created_at
            FROM PostView WHERE UserId = :uid AND COALESCE(ViewDate, CreateDate) >= :since
            UNION ALL
            SELECT pr.PostId AS post_id, COALESCE(rt.Code, rt.Name) AS action, pr.CreateDate AS created_at
            FROM PostReaction pr
            LEFT JOIN ReactionType rt ON rt.Id = pr.ReactionTypeId
            WHERE pr.UserId = :uid AND pr.CreateDate >= :since
            UNION ALL
            SELECT c.PostId AS post_id, 'comment' AS action, c.CreateDate AS created_at
            FROM Comment c
            WHERE c.UserId = :uid AND c.CreateDate >= :since
        """)
        with self.engine.begin() as cx:
            df_db = pd.read_sql(sql, cx, params={"uid": user_id, "since": since})

        # Normalize
        for d in (df_cache, df_db):
            if not d.empty:
                d["action"] = d["action"].astype(str).str.lower()
                d["created_at"] = pd.to_datetime(d["created_at"], utc=True, errors="coerce")

        if df_cache.empty:
            return df_db

        if df_db.empty:
            return df_cache

        # merge: ưu tiên bản ghi mới nhất, loại trùng theo (post_id, action, created_at)
        all_df = pd.concat([df_db, df_cache], ignore_index=True)
        all_df = all_df.drop_duplicates(subset=["post_id", "action", "created_at"], keep="last")
        return all_df

    # ---------------------- following ----------------------------------------
    def get_following_ids(self, user_id: int) -> List[int]:
        ids = self.redis.get_following_ids(user_id)
        if ids:
            return ids
        # fallback DB
        sql = text("SELECT FriendId FROM Friendship WHERE UserId = :uid")
        with self.engine.begin() as cx:
            rows = cx.execute(sql, {"uid": user_id}).fetchall()
        return [int(r[0]) for r in rows]

    # ---------------------- posts & side data ---------------------------------
    def get_posts_by_authors(self, author_ids: List[int], since_days: int = 30, limit: int = 2000) -> pd.DataFrame:
        if not author_ids:
            return pd.DataFrame(columns=["Id", "UserId", "CreateDate", "IsRepost", "IsPin"])
        since = datetime.now(timezone.utc) - timedelta(days=since_days)
        sql = text("""
            SELECT Id, UserId, CreateDate, IsRepost, IsPin
            FROM Post
            WHERE UserId IN :aids AND CreateDate >= :since
            ORDER BY CreateDate DESC
            LIMIT :limit
        """)
        with self.engine.begin() as cx:
            df = pd.read_sql(sql, cx, params={"aids": tuple(author_ids), "since": since, "limit": limit})
        df["CreateDate"] = pd.to_datetime(df["CreateDate"], utc=True, errors="coerce")
        return df

    def get_popular_posts(self, days: int = 7, top_k: int = 1000) -> pd.DataFrame:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        sql = text("""
        WITH inter AS (
            SELECT PostId, COALESCE(ViewDate, CreateDate) AS ts FROM PostView WHERE COALESCE(ViewDate, CreateDate) >= :since
            UNION ALL
            SELECT PostId, CreateDate AS ts FROM PostReaction WHERE CreateDate >= :since
            UNION ALL
            SELECT PostId, CreateDate AS ts FROM Comment WHERE CreateDate >= :since
        )
        SELECT p.Id, p.UserId, p.CreateDate, p.IsRepost, p.IsPin, COUNT(*) AS inter_cnt
        FROM inter i
        JOIN Post p ON p.Id = i.PostId
        GROUP BY p.Id, p.UserId, p.CreateDate, p.IsRepost, p.IsPin
        ORDER BY inter_cnt DESC
        LIMIT :k
        """)
        with self.engine.begin() as cx:
            df = pd.read_sql(sql, cx, params={"since": since, "k": top_k})
        df["CreateDate"] = pd.to_datetime(df["CreateDate"], utc=True, errors="coerce")
        return df

    def get_hashtags_by_posts(self, post_ids: List[int]) -> pd.DataFrame:
        if not post_ids:
            return pd.DataFrame(columns=["PostId", "HashtagId"])
        sql = text("SELECT PostId, HashtagId FROM PostHashtag WHERE PostId IN :pids")
        with self.engine.begin() as cx:
            return pd.read_sql(sql, cx, params={"pids": tuple(post_ids)})

    def get_posts(self, post_ids: List[int]) -> pd.DataFrame:
        if not post_ids:
            return pd.DataFrame(columns=["Id", "UserId", "CreateDate", "IsRepost", "IsPin"])
        sql = text("SELECT Id, UserId, CreateDate, IsRepost, IsPin FROM Post WHERE Id IN :ids")
        with self.engine.begin() as cx:
            df = pd.read_sql(sql, cx, params={"ids": tuple(post_ids)})
        df["CreateDate"] = pd.to_datetime(df["CreateDate"], utc=True, errors="coerce")
        return df

    def get_user_created(self, user_id: int):
        q = text("SELECT CreateDate FROM User WHERE Id = :uid")
        with self.engine.begin() as cx:
            res = pd.read_sql(q, cx, params={"uid": user_id})
        if res.empty:
            return None
        return pd.to_datetime(res["CreateDate"].iloc[0], utc=True, errors="coerce")
