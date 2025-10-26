# recommender/online/features.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .repository import OnlineRepository


FEATURE_COLS = [
    "user_total_interactions", "user_positive_rate", "user_account_age_days",
    "post_total_interactions", "post_positive_rate", "post_age_hours",
    "post_is_repost", "post_is_pin", "post_hashtag_count",
    "author_id_known", "author_total_interactions", "author_positive_rate",
    "author_follower_count", "is_following_author",
    "user_post_cosine",  # giữ chỗ cho future embeddings
]


class OnlineFeatureService:
    def __init__(self, repo: OnlineRepository):
        self.repo = repo

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    # ---------- aggregate stats ----------
    def _user_stats(self, interactions: pd.DataFrame) -> Dict[int, Dict]:
        if interactions.empty:
            return {}
        d = {}
        by_u = interactions.groupby("user_id")
        for uid, g in by_u:
            cnt = len(g)
            pos = (g["action"].isin(["like","love","laugh","wow","sad","angry","care","comment","share","save"])).mean()
            d[int(uid)] = {"total_interactions": float(cnt), "positive_rate": float(pos)}
        return d

    def _author_stats(self, labeled_inter: pd.DataFrame) -> Dict[int, Dict]:
        d = {}
        if labeled_inter.empty or "author_id" not in labeled_inter.columns:
            return d
        by_a = labeled_inter.groupby("author_id")
        for aid, g in by_a:
            if pd.isna(aid):
                continue
            cnt = len(g)
            pos = g["label"].mean()
            d[int(aid)] = {"total_interactions": float(cnt), "positive_rate": float(pos)}
        return d

    def build_row_features(
        self,
        user_id: int,
        candidate_posts: pd.DataFrame,  # columns: Id, UserId, CreateDate, IsRepost, IsPin
        user_recent: pd.DataFrame,      # user interactions (post_id, action, created_at)
        following_ids: List[int],
        post_hashtag_count: Dict[int, int],
        follower_count_by_author: Dict[int, int],
        user_created_at: pd.Timestamp | None,
    ) -> pd.DataFrame:
        """
        build features cho tập (user_id, post_id) theo cùng schema offline.
        """
        now = self._now()
        # user-level
        if user_recent is None or user_recent.empty:
            usr_total = 0.0; usr_pos = 0.0
        else:
            usr_total = float(len(user_recent))
            usr_pos = float((user_recent["action"].isin(
                ["like","love","laugh","wow","sad","angry","care","comment","share","save"]
            )).mean())

        acc_age_days = 0.0
        if user_created_at is not None and not pd.isna(user_created_at):
            acc_age_days = max((now - user_created_at).total_seconds() / 86400.0, 0.0)

        # assemble features per post
        rows = []
        for _, p in candidate_posts.iterrows():
            pid = int(p["Id"])
            author_id = int(p["UserId"]) if pd.notna(p["UserId"]) else -1
            post_age_h = 0.0
            if pd.notna(p["CreateDate"]):
                post_age_h = max((now - pd.to_datetime(p["CreateDate"], utc=True)).total_seconds()/3600.0, 0.0)

            rows.append({
                "user_total_interactions": usr_total,
                "user_positive_rate": usr_pos,
                "user_account_age_days": acc_age_days,

                # chưa có post-level historical → để 0 (cùng cách offline lúc vector hoá)
                "post_total_interactions": 0.0,
                "post_positive_rate": 0.0,

                "post_age_hours": post_age_h,
                "post_is_repost": float(p.get("IsRepost", 0) or 0),
                "post_is_pin": float(p.get("IsPin", 0) or 0),
                "post_hashtag_count": float(post_hashtag_count.get(pid, 0)),

                "author_id_known": 1.0 if author_id != -1 else 0.0,
                "author_total_interactions": float(0.0),  # sẽ fill sau nếu có
                "author_positive_rate": float(0.0),       # sẽ fill sau nếu có
                "author_follower_count": float(follower_count_by_author.get(author_id, 0)),
                "is_following_author": 1.0 if author_id in following_ids else 0.0,

                "user_post_cosine": 0.0,  # placeholder
                "post_id": pid,
                "author_id": author_id,
            })
        df = pd.DataFrame(rows)

        # fill author stats nếu có label với interactions của user theo author
        if not user_recent.empty:
            tmp = user_recent.copy()
            # gắn author_id vào user_recent để ước lượng positive rate theo author
            # cần posts của những post user từng tương tác để map author
            posts_recent = self.repo.get_posts(tmp["post_id"].dropna().astype(int).tolist())
            if not posts_recent.empty:
                posts_recent = posts_recent.rename(columns={"Id":"post_id","UserId":"author_id"})
                tmp = tmp.merge(posts_recent[["post_id","author_id"]], on="post_id", how="left")
                tmp["label"] = tmp["action"].isin(
                    ["like","love","laugh","wow","sad","angry","care","comment","share","save"]
                ).astype(int)
                ast = self._author_stats(tmp)
                if ast:
                    df["author_total_interactions"] = df["author_id"].map(lambda a: ast.get(int(a),{}).get("total_interactions",0.0))
                    df["author_positive_rate"] = df["author_id"].map(lambda a: ast.get(int(a),{}).get("positive_rate",0.0))

        # ensure all feature cols exist
        for c in FEATURE_COLS:
            if c not in df.columns:
                df[c] = 0.0

        return df[FEATURE_COLS + ["post_id", "author_id"]]
