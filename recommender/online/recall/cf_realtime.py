from __future__ import annotations
from typing import Dict, List
import logging
logger = logging.getLogger(__name__)

class CFRealtimeRecall:
    def __init__(self, redis_client, config: Dict):
        self.redis = redis_client
        self.enabled = True
        self.k_per_seed = int(config.get("k_per_seed", 50))
        self.max_recent = int(config.get("max_recent", 20))
        self.weight_decay = float(config.get("weight_decay", 0.95))

    def recall(self, user_id: int, k: int = 300) -> List[int]:
        if not self.redis: return []
        try:
            seeds = self.redis.lrange(f"user:{user_id}:recent_items", 0, self.max_recent-1) or []
            if not seeds: return []
            seeds = [int(x) for x in seeds]
            scores: Dict[int, float] = {}
            for r, seed in enumerate(seeds):
                decay = self.weight_decay ** r
                cands = self.redis.zrevrange(f"covisit:{seed}", 0, self.k_per_seed-1, withscores=True) or []
                for cid, w in cands:
                    try:
                        cid = int(cid); scores[cid] = scores.get(cid, 0.0) + float(w) * decay
                    except: pass
            return [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]
        except Exception as e:
            logger.debug(f"CFRealtime recall fail: {e}")
            return []
