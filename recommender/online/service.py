# recommender/online/service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .artifacts import load_ranker_artifacts
from .repository import OnlineRepository
from .features import OnlineFeatureService, FEATURE_COLS
from .cache import RedisConfig


@dataclass
class FeedRequest:
    user_id: int
    limit: int = 50
    since_days_following: int = 14
    popular_days: int = 7
    mix_ratio_following: float = 0.7  # 70% following, 30% popular


class OnlineRecommenderService:
    def __init__(self, db_url: str, artifacts_dir: str = "models", redis_url: str | None = None):
        rcfg = RedisConfig(url=redis_url) if redis_url else None
        self.repo = OnlineRepository(db_url, redis_cfg=rcfg)
        self.feature_service = OnlineFeatureService(self.repo)
        self.model, self.scaler, self.feature_cols, self.meta, self.version_dir = load_ranker_artifacts(artifacts_dir)
        self.input_cols = self.feature_cols  # align to offline
  
    # ---------------------------- FEED ---------------------------------------
    def _candidate_posts(self, user_id: int, req: FeedRequest) -> pd.DataFrame:
        following = self.repo.get_following_ids(user_id)
        from_following = self.repo.get_posts_by_authors(following, since_days=req.since_days_following, limit=2000)
        popular = self.repo.get_popular_posts(days=req.popular_days, top_k=1000)

        # Mix
        if from_following.empty and popular.empty:
            return pd.DataFrame(columns=["Id","UserId","CreateDate","IsRepost","IsPin"])

        # Simple mix: ratio on count
        k1 = int(req.limit * req.mix_ratio_following)
        k2 = req.limit - k1
        a = from_following.head(max(k1, 0)) if not from_following.empty else pd.DataFrame()
        b = popular.head(max(k2, 0)) if not popular.empty else pd.DataFrame()
        cand = pd.concat([a, b], ignore_index=True)
        # Nếu thiếu hụt, bổ sung từ nguồn còn lại
        if len(cand) < req.limit:
            if len(a) < k1 and not popular.empty:
                extra = popular.iloc[k2: k2 + (req.limit - len(cand))]
                cand = pd.concat([cand, extra], ignore_index=True)
            elif len(b) < k2 and not from_following.empty:
                extra = from_following.iloc[k1: k1 + (req.limit - len(cand))]
                cand = pd.concat([cand, extra], ignore_index=True)
        cand = cand.drop_duplicates(subset=["Id"])
        return cand

    def _post_hashtag_count(self, post_ids: List[int]) -> Dict[int, int]:
        df = self.repo.get_hashtags_by_posts(post_ids)
        if df.empty:
            return {}
        return df.groupby("PostId").size().to_dict()

    def _follower_count_by_author(self, author_ids: List[int]) -> Dict[int, int]:
        # đếm follower từ bảng Friendship (FriendId = author)
        if not author_ids:
            return {}
        # tải một lần toàn bộ follower count
        q = "SELECT FriendId AS author_id, COUNT(*) AS cnt FROM Friendship WHERE FriendId IN :ids GROUP BY FriendId"
        with self.repo.engine.begin() as cx:
            tmp = pd.read_sql(q, cx, params={"ids": tuple(author_ids)})
        return dict(zip(tmp["author_id"].astype(int), tmp["cnt"].astype(int)))

    def _user_created_at(self, user_id: int):
        q = "SELECT CreateDate FROM User WHERE Id = :uid"
        with self.repo.engine.begin() as cx:
            res = pd.read_sql(q, cx, params={"uid": user_id})
        if res.empty:
            return None
        return pd.to_datetime(res["CreateDate"].iloc[0], utc=True, errors="coerce")

    def recommend_feed(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        req = FeedRequest(user_id=user_id, limit=limit)

        # Candidates
        cand = self._candidate_posts(user_id, req)
        if cand.empty:
            return []

        # Side info
        user_recent = self.repo.get_user_recent_interactions(user_id, days=30)
        post_ids = cand["Id"].astype(int).tolist()
        author_ids = cand["UserId"].dropna().astype(int).unique().tolist()
        phc = self._post_hashtag_count(post_ids)
        fcnt = self._follower_count_by_author(author_ids)
        created_at = self._user_created_at(user_id)
        following_ids = self.repo.get_following_ids(user_id)

        # Features
        feats = self.feature_service.build_row_features(
            user_id=user_id,
            candidate_posts=cand,
            user_recent=user_recent,
            following_ids=following_ids,
            post_hashtag_count=phc,
            follower_count_by_author=fcnt,
            user_created_at=created_at,
        )

        # Align columns to offline feature order
        X = feats[self.input_cols].fillna(0.0).to_numpy()
        if self.scaler:
            X = self.scaler.transform(X)

        # Predict
        if hasattr(self.model, "predict"):
            scores = self.model.predict(X)
        else:
            # Booster (LightGBM)
            scores = self.model.predict(X)

        out = []
        for i, row in feats.iterrows():
            out.append({
                "post_id": int(row["post_id"]),
                "author_id": int(row["author_id"]),
                "score": float(scores[i]),
            })
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:limit]

    # ------------------------ FRIEND RECOMMENDATION ---------------------------
    def recommend_friends(self, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Gợi ý bạn bè đơn giản:
        - Friend-of-friend (FoF)
        - Tần suất tương tác gần đây (view/reaction/comment cùng bài)
        - Loại trừ người đã follow
        """
        following = set(self.repo.get_following_ids(user_id))
        # FoF
        fof = set()
        for fid in list(following)[:500]:
            fof.update(self.repo.get_following_ids(fid))
        fof -= following
        fof.discard(user_id)

        if not fof:
            return []

        # Scoring: số lượng common-follow + số lần “đồng tương tác” bài viết gần đây
        scores = {uid: 0.0 for uid in fof}
        # common-follow
        for uid in fof:
            f2 = set(self.repo.get_following_ids(uid))
            common = len(f2 & following)
            scores[uid] += 1.0 * common

        # co-interaction: user và candidate cùng tương tác các post trong 30 ngày
        me = self.repo.get_user_recent_interactions(user_id, days=30)
        if not me.empty:
            me_posts = set(me["post_id"].dropna().astype(int).tolist())
            for uid in list(fof)[:1000]:
                df = self.repo.get_user_recent_interactions(uid, days=30)
                if df.empty:
                    continue
                overlap = len(me_posts & set(df["post_id"].dropna().astype(int).tolist()))
                scores[uid] += 0.5 * overlap

        ranked = sorted([{"user_id": int(u), "score": float(s)} for u, s in scores.items()],
                        key=lambda x: x["score"], reverse=True)
        return ranked[:limit]
