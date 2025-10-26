# recommender/online/cache.py
from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List
import pandas as pd

try:
    import redis  # pip install redis
except Exception:
    redis = None

@dataclass
class RedisCfg:
    url: str = "redis://127.0.0.1:6379/0"
    # recent interactions – hỗ trợ LIST hoặc ZSET:
    recent_tpl: str = "wayjet:inter:user:{uid}"   # LPUSH json OR ZADD score=json
    following_tpl: str = "wayjet:following:{uid}" # SET of friend ids
    max_recent: int = 500

class RedisClient:
    def __init__(self, cfg: RedisCfg | None):
        self.cfg = cfg
        self.ok = bool(cfg and cfg.url and redis is not None)
        self.r = redis.from_url(cfg.url, decode_responses=True) if self.ok else None

    def _k_recent(self, uid: int) -> str: return self.cfg.recent_tpl.format(uid=uid)
    def _k_follow(self, uid: int) -> str: return self.cfg.following_tpl.format(uid=uid)

    def get_recent(self, uid: int, since: datetime) -> pd.DataFrame:
        if not self.ok: 
            return pd.DataFrame(columns=["post_id","action","created_at"])
        k = self._k_recent(uid)
        t = self.r.type(k)
        out = []
        try:
            if t == "list":
                items = self.r.lrange(k, 0, self.cfg.max_recent-1) or []
                for s in items:
                    try:
                        obj = json.loads(s)
                        ts = pd.to_datetime(obj.get("ts") or obj.get("created_at"), utc=True, errors="coerce")
                        if ts is pd.NaT or ts < since: 
                            continue
                        out.append({
                            "post_id": int(obj.get("post_id", 0)),
                            "action": str(obj.get("action","view")).lower(),
                            "created_at": ts,
                        })
                    except Exception:
                        continue
            elif t == "zset":
                items = self.r.zrangebyscore(k, since.timestamp(), "+inf", start=0, num=self.cfg.max_recent) or []
                for s in items:
                    try:
                        obj = json.loads(s)
                        ts = pd.to_datetime(obj.get("ts") or obj.get("created_at"), utc=True, errors="coerce")
                        out.append({
                            "post_id": int(obj.get("post_id", 0)),
                            "action": str(obj.get("action","view")).lower(),
                            "created_at": ts,
                        })
                    except Exception:
                        continue
        except Exception:
            return pd.DataFrame(columns=["post_id","action","created_at"])
        if not out:
            return pd.DataFrame(columns=["post_id","action","created_at"])
        df = pd.DataFrame(out)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        return df

    def get_following_ids(self, uid: int) -> List[int]:
        if not self.ok: 
            return []
        k = self._k_follow(uid)
        try:
            if self.r.exists(k):
                return [int(x) for x in (self.r.smembers(k) or []) if str(x).isdigit()]
        except Exception:
            pass
        return []
