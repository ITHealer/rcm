# recommender/online/recall/covisit.py
from __future__ import annotations
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class CovisitRecall:
    """
    Recall theo lối 'Users who viewed X also viewed Y' dựa trên Redis:
    - Lấy recent_items của user
    - Với mỗi item, lấy ZREVRANGE covisit:<item> top N
    - Hợp nhất & dedup
    """
    def __init__(self, redis_client, k_per_anchor: int = 25, max_anchors: int = 5):
        self.redis = redis_client
        self.k_per_anchor = k_per_anchor
        self.max_anchors = max_anchors

    def recall(self, user_id: int, k: int = 200) -> List[int]:
        if self.redis is None:
            return []
        anchors = self.redis.lrange(f"user:{user_id}:recent_items", 0, self.max_anchors - 1) or []
        anchors = [int(x) for x in anchors if x is not None]
        if not anchors:
            return []
        agg = []
        seen = set()
        for a in anchors:
            key = f"covisit:{a}"
            pairs = self.redis.zrevrange(key, 0, self.k_per_anchor - 1) or []
            for pid in pairs:
                try:
                    pid = int(pid)
                    if pid not in seen:
                        seen.add(pid)
                        agg.append(pid)
                        if len(agg) >= k:
                            return agg
                except Exception:
                    continue
        return agg[:k]
