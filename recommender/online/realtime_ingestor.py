# recommender/online/realtime_ingestor.py
from __future__ import annotations
import logging, time, threading, random
from typing import Dict, Any, Optional, List
from sqlalchemy import text
import pandas as pd

logger = logging.getLogger(__name__)

class BackendIngestor:
    def __init__(self, cfg: Dict, mysql_engine, redis_client, pipeline=None):
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.mysql_engine = mysql_engine
        self.redis = redis_client
        self.pipeline = pipeline

        self.poll_interval = int(self.cfg.get("poll_interval_seconds", 15))
        self.max_rows = int(self.cfg.get("max_rows_per_tick", 2000))
        self.backoff_on_error = int(self.cfg.get("backoff_on_error_seconds", 5))
        self.wm_key = self.cfg.get("watermark_key", "ingest:backend:last_id")

        mcfg = self.cfg.get("mysql", {}) or {}
        self.tbl_inter = mcfg.get("interactions_table", "PostReaction")
        self.tbl_posts = mcfg.get("posts_table", "Post")

        self.col_id   = mcfg.get("id_column", "Id")
        self.col_ts   = mcfg.get("time_column", "CreateDate")
        self.col_uid  = mcfg.get("user_id_column", "UserId")
        self.col_pid  = mcfg.get("post_id_column", "PostId")
        self.col_rtid = mcfg.get("reaction_type_column", "ReactionTypeId")  # <-- dùng RT
        self.col_stat = mcfg.get("status_column", "Status")
        self.col_post_author = mcfg.get("post_author_column", "UserId")

        self.action_weights = self.cfg.get("action_weights", {}) or {}
        self.embed_sample = float(self.cfg.get("update_embedding_sample_rate", 0.2))

        self._stop = False
        self._thr = None

    def start(self):
        if not self.enabled:
            logger.info("Realtime ingestor disabled.")
            return
        if self.mysql_engine is None or self.redis is None:
            logger.warning("Realtime ingestor missing dependencies (mysql_engine/redis). Disabled.")
            return
        self._thr = threading.Thread(target=self._loop, name="BackendIngestor", daemon=True)
        self._thr.start()
        logger.info("✅ Realtime ingestor started (interval=%ss, max_rows=%s)", self.poll_interval, self.max_rows)

    def stop(self):
        self._stop = True

    def _get_last_id(self) -> int:
        try:
            v = self.redis.get(self.wm_key)
            return int(v) if v else 0
        except Exception:
            return 0

    def _set_last_id(self, last_id: int):
        try:
            self.redis.set(self.wm_key, str(last_id))
        except Exception:
            pass

    def _map_reaction_to_action(self, reaction_type_id: int) -> str:
        """
        Map ReactionTypeId -> action string dùng cho counters/embedding.
        1 like, 2 love, 3 laugh, 4 wow, 5 sad, 6 angry, 7 care (theo bảng bạn chụp).
        Tùy business, mình gom 1,2,3,4,7 -> 'like'; 5,6 -> 'dislike' (hoặc 'view').
        """
        if reaction_type_id in (1, 2, 3, 4, 7):
            return "like"
        elif reaction_type_id in (5, 6):
            return "dislike"   # hoặc "view" nếu không muốn ảnh hưởng embedding
        return "view"
    
    def _loop(self):
        last_id = self._get_last_id()
        while not self._stop:
            try:
                # Lấy batch mới, lọc status hợp lệ ở cả reaction và post
                # JOIN Post để lấy author_id
                sql = f"""
                    SELECT pr.{self.col_id}   AS id,
                           pr.{self.col_uid}  AS user_id,
                           pr.{self.col_pid}  AS post_id,
                           pr.{self.col_rtid} AS reaction_type_id,
                           pr.{self.col_ts}   AS created_at,
                           p.{self.col_post_author} AS author_id
                    FROM {self.tbl_inter} pr
                    LEFT JOIN {self.tbl_posts} p ON p.Id = pr.{self.col_pid}
                    WHERE pr.{self.col_id} > :last_id
                      AND (pr.{self.col_stat} = 10 OR pr.{self.col_stat} IS NULL)
                      AND (p.Status = 10 OR p.Status IS NULL)
                    ORDER BY pr.{self.col_id} ASC
                    LIMIT :lim
                """

                with self.mysql_engine.connect() as conn:
                    rs = conn.execute(text(sql), {"last_id": last_id, "lim": self.max_rows})
                    rows = rs.fetchall()

                if not rows:
                    time.sleep(self.poll_interval)
                    continue

                max_seen = last_id
                for r in rows:
                    rid = int(r.id)
                    uid = int(r.user_id) if r.user_id is not None else None
                    pid = int(r.post_id) if r.post_id is not None else None
                    aid = int(r.author_id) if r.author_id is not None else None
                    rtid = int(r.reaction_type_id) if r.reaction_type_id is not None else None

                    action = self._map_reaction_to_action(rtid) if rtid is not None else "view"

                    if pid and uid:
                        self._apply_counters(uid, pid, aid, action)
                        if self.pipeline is not None and action in self.action_weights:
                            import random
                            if random.random() < self.embed_sample:
                                try:
                                    self.pipeline.update_user_embedding_realtime(uid, pid, action)
                                except Exception as e:
                                    logger.debug("embedding rt update fail: %s", e)

                    if rid > max_seen:
                        max_seen = rid

                self._set_last_id(max_seen)
                last_id = max_seen
                time.sleep(0.05 if len(rows) >= self.max_rows else self.poll_interval)

            except Exception as e:
                logger.warning("ingestor tick error: %s", e)
                time.sleep(self.backoff_on_error)
                
    def _apply_counters(self, user_id: int, post_id: int, author_id: Optional[int], action: str):
        """
        Tăng counters / covisit trong Redis, các key khớp phần bạn đã dùng
        """
        try:
            pipe = self.redis.pipeline(transaction=False)
            # user (24h)
            pipe.hincrby(f"user:{user_id}:engagement_24h", action, 1)
            pipe.expire(f"user:{user_id}:engagement_24h", 24*3600)

            # post (1h)
            pipe.hincrby(f"post:{post_id}:engagement_1h", action, 1)
            pipe.expire(f"post:{post_id}:engagement_1h", 3600)

            # author (1h)
            if author_id:
                pipe.hincrby(f"author:{author_id}:engagement_1h", action, 1)
                pipe.expire(f"author:{author_id}:engagement_1h", 3600)

            # recent items để làm covisit
            pipe.lpush(f"user:{user_id}:recent_items", post_id)
            pipe.ltrim(f"user:{user_id}:recent_items", 0, 99)
            pipe.expire(f"user:{user_id}:recent_items", 24*3600)

            # covisit (rất nhẹ: với 3 item gần nhất)
            recent = self.redis.lrange(f"user:{user_id}:recent_items", 0, 2)
            for other in recent:
                try:
                    other = int(other)
                    if other != post_id:
                        pipe.zincrby(f"covisit:{other}", 1.0, post_id)
                        pipe.expire(f"covisit:{other}", 7*24*3600)
                except Exception:
                    pass

            pipe.execute()
        except Exception as e:
            logger.debug("redis counters fail: %s", e)
