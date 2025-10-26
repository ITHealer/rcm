# # recommender/online/realtime_ingestor.py
# from __future__ import annotations
# import logging, time, threading, random
# from typing import Dict, Any, Optional, List
# from sqlalchemy import text
# import pandas as pd

# logger = logging.getLogger(__name__)

# class BackendIngestor:
#     def __init__(self, cfg: Dict, mysql_engine, redis_client, pipeline=None):
#         self.cfg = cfg or {}
#         self.enabled = bool(self.cfg.get("enabled", False))
#         self.mysql_engine = mysql_engine
#         self.redis = redis_client
#         self.pipeline = pipeline

#         self.poll_interval = int(self.cfg.get("poll_interval_seconds", 15))
#         self.max_rows = int(self.cfg.get("max_rows_per_tick", 2000))
#         self.backoff_on_error = int(self.cfg.get("backoff_on_error_seconds", 5))
#         self.wm_key = self.cfg.get("watermark_key", "ingest:backend:last_id")

#         mcfg = self.cfg.get("mysql", {}) or {}
#         self.tbl_inter = mcfg.get("interactions_table", "PostReaction")
#         self.tbl_posts = mcfg.get("posts_table", "Post")

#         self.col_id   = mcfg.get("id_column", "Id")
#         self.col_ts   = mcfg.get("time_column", "CreateDate")
#         self.col_uid  = mcfg.get("user_id_column", "UserId")
#         self.col_pid  = mcfg.get("post_id_column", "PostId")
#         self.col_rtid = mcfg.get("reaction_type_column", "ReactionTypeId")  # <-- dùng RT
#         self.col_stat = mcfg.get("status_column", "Status")
#         self.col_post_author = mcfg.get("post_author_column", "UserId")

#         self.action_weights = self.cfg.get("action_weights", {}) or {}
#         self.embed_sample = float(self.cfg.get("update_embedding_sample_rate", 0.2))

#         self._stop = False
#         self._thr = None

#     def start(self):
#         if not self.enabled:
#             logger.info("Realtime ingestor disabled.")
#             return
#         if self.mysql_engine is None or self.redis is None:
#             logger.warning("Realtime ingestor missing dependencies (mysql_engine/redis). Disabled.")
#             return
#         self._thr = threading.Thread(target=self._loop, name="BackendIngestor", daemon=True)
#         self._thr.start()
#         logger.info("✅ Realtime ingestor started (interval=%ss, max_rows=%s)", self.poll_interval, self.max_rows)

#     def stop(self):
#         self._stop = True

#     def _get_last_id(self) -> int:
#         try:
#             v = self.redis.get(self.wm_key)
#             return int(v) if v else 0
#         except Exception:
#             return 0

#     def _set_last_id(self, last_id: int):
#         try:
#             self.redis.set(self.wm_key, str(last_id))
#         except Exception:
#             pass

#     def _map_reaction_to_action(self, reaction_type_id: int) -> str:
#         """
#         Map ReactionTypeId -> action string dùng cho counters/embedding.
#         1 like, 2 love, 3 laugh, 4 wow, 5 sad, 6 angry, 7 care (theo bảng bạn chụp).
#         Tùy business, mình gom 1,2,3,4,7 -> 'like'; 5,6 -> 'dislike' (hoặc 'view').
#         """
#         if reaction_type_id in (1, 2, 3, 4, 7):
#             return "like"
#         elif reaction_type_id in (5, 6):
#             return "dislike"   # hoặc "view" nếu không muốn ảnh hưởng embedding
#         return "view"
    
#     def _loop(self):
#         last_id = self._get_last_id()
#         while not self._stop:
#             try:
#                 # Lấy batch mới, lọc status hợp lệ ở cả reaction và post
#                 # JOIN Post để lấy author_id
#                 sql = f"""
#                     SELECT pr.{self.col_id}   AS id,
#                            pr.{self.col_uid}  AS user_id,
#                            pr.{self.col_pid}  AS post_id,
#                            pr.{self.col_rtid} AS reaction_type_id,
#                            pr.{self.col_ts}   AS created_at,
#                            p.{self.col_post_author} AS author_id
#                     FROM {self.tbl_inter} pr
#                     LEFT JOIN {self.tbl_posts} p ON p.Id = pr.{self.col_pid}
#                     WHERE pr.{self.col_id} > :last_id
#                       AND (pr.{self.col_stat} = 10 OR pr.{self.col_stat} IS NULL)
#                       AND (p.Status = 10 OR p.Status IS NULL)
#                     ORDER BY pr.{self.col_id} ASC
#                     LIMIT :lim
#                 """

#                 with self.mysql_engine.connect() as conn:
#                     rs = conn.execute(text(sql), {"last_id": last_id, "lim": self.max_rows})
#                     rows = rs.fetchall()

#                 if not rows:
#                     time.sleep(self.poll_interval)
#                     continue

#                 max_seen = last_id
#                 for r in rows:
#                     rid = int(r.id)
#                     uid = int(r.user_id) if r.user_id is not None else None
#                     pid = int(r.post_id) if r.post_id is not None else None
#                     aid = int(r.author_id) if r.author_id is not None else None
#                     rtid = int(r.reaction_type_id) if r.reaction_type_id is not None else None

#                     action = self._map_reaction_to_action(rtid) if rtid is not None else "view"

#                     if pid and uid:
#                         self._apply_counters(uid, pid, aid, action)
#                         if self.pipeline is not None and action in self.action_weights:
#                             import random
#                             if random.random() < self.embed_sample:
#                                 try:
#                                     self.pipeline.update_user_embedding_realtime(uid, pid, action)
#                                 except Exception as e:
#                                     logger.debug("embedding rt update fail: %s", e)

#                     if rid > max_seen:
#                         max_seen = rid

#                 self._set_last_id(max_seen)
#                 last_id = max_seen
#                 time.sleep(0.05 if len(rows) >= self.max_rows else self.poll_interval)

#             except Exception as e:
#                 logger.warning("ingestor tick error: %s", e)
#                 time.sleep(self.backoff_on_error)
                
#     def _apply_counters(self, user_id: int, post_id: int, author_id: Optional[int], action: str):
#         """
#         Tăng counters / covisit trong Redis, các key khớp phần bạn đã dùng
#         """
#         try:
#             pipe = self.redis.pipeline(transaction=False)
#             # user (24h)
#             pipe.hincrby(f"user:{user_id}:engagement_24h", action, 1)
#             pipe.expire(f"user:{user_id}:engagement_24h", 24*3600)

#             # post (1h)
#             pipe.hincrby(f"post:{post_id}:engagement_1h", action, 1)
#             pipe.expire(f"post:{post_id}:engagement_1h", 3600)

#             # author (1h)
#             if author_id:
#                 pipe.hincrby(f"author:{author_id}:engagement_1h", action, 1)
#                 pipe.expire(f"author:{author_id}:engagement_1h", 3600)

#             # recent items để làm covisit
#             pipe.lpush(f"user:{user_id}:recent_items", post_id)
#             pipe.ltrim(f"user:{user_id}:recent_items", 0, 99)
#             pipe.expire(f"user:{user_id}:recent_items", 24*3600)

#             # covisit (rất nhẹ: với 3 item gần nhất)
#             recent = self.redis.lrange(f"user:{user_id}:recent_items", 0, 2)
#             for other in recent:
#                 try:
#                     other = int(other)
#                     if other != post_id:
#                         pipe.zincrby(f"covisit:{other}", 1.0, post_id)
#                         pipe.expire(f"covisit:{other}", 7*24*3600)
#                 except Exception:
#                     pass

#             pipe.execute()
#         except Exception as e:
#             logger.debug("redis counters fail: %s", e)



"""
BackendIngestor
---------------
- Poll backend DB (PostReaction, Comment, PostView) định kỳ và đẩy tín hiệu realtime vào Redis
- Tự phát hiện dialect (postgresql vs mysql) và quote tên schema/bảng phù hợp
- Lưu checkpoint last_id vào Redis (nếu có), fallback in-memory
"""

from __future__ import annotations

import threading
import time
import logging
from typing import Optional, Dict, Any, Tuple, List

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def _dialect(engine: Optional[Engine]) -> str:
    try:
        return engine.url.get_dialect().name if engine is not None else "unknown"
    except Exception:
        return "unknown"


def _q(schema: str, table: str, dialect: str) -> str:
    """
    Tạo fully-qualified name cho bảng với schema.
    - PostgreSQL: wayjet_system."PostReaction"
    - MySQL     : wayjet_system.PostReaction (hoặc chỉ PostReaction nếu không dùng schema)
    """
    if not schema:
        # no schema
        if dialect == "postgresql":
            return f'"{table}"'
        return table

    if dialect == "postgresql":
        return f'{schema}."{table}"'
    return f"{schema}.{table}"


def _load_last_id(redis, key: str, default: int = 0) -> int:
    if redis is None:
        return default
    try:
        val = redis.get(key)
        return int(val) if val is not None else default
    except Exception:
        return default


def _save_last_id(redis, key: str, value: int):
    if redis is None:
        return
    try:
        redis.set(key, int(value))
    except Exception:
        pass


class BackendIngestor:
    def __init__(
        self,
        cfg: Dict[str, Any],
        mysql_engine: Optional[Engine],   # có thể là Postgres engine (tên tham số cũ)
        redis_client,
        pipeline,
    ):
        self.cfg = cfg or {}
        self.engine = mysql_engine
        self.redis = redis_client
        self.pipeline = pipeline

        self.interval = int(self.cfg.get("interval_seconds", 15))
        self.max_rows = int(self.cfg.get("max_rows", 2000))
        self.schema = self.cfg.get("schema", "wayjet_system")

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        # in-memory fallback
        self._mem_last_ids = {
            "reaction": 0,
            "comment": 0,
            "view": 0,
        }

    # ------------- public -------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="BackendIngestor", daemon=True)
        self._thread.start()
        logger.info("✅ Realtime ingestor started (interval=%ss, max_rows=%s)", self.interval, self.max_rows)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------- internals ----------
    def _loop(self):
        while not self._stop.is_set():
            st = time.time()
            try:
                self._tick()
            except Exception as e:
                logger.warning("ingestor tick error: %s", e, exc_info=True)
            dt = time.time() - st
            sleep = max(self.interval - dt, 0.2)
            time.sleep(sleep)

    def _tick(self):
        if self.engine is None:
            # không có DB → không làm gì
            return

        dialect = _dialect(self.engine)
        # Lấy last_id từ Redis nếu có
        last_react_id = self._get_last_id("reaction")
        last_comment_id = self._get_last_id("comment")
        last_view_id = self._get_last_id("view")

        with self.engine.connect() as conn:
            # --------- PostReaction ----------
            try:
                rows = self._fetch_reactions(conn, dialect, last_react_id)
                if rows:
                    new_last = last_react_id
                    for r in rows:
                        rid = int(r["id"])
                        user_id = int(r["user_id"])
                        post_id = int(r["post_id"])
                        author_id = int(r["author_id"]) if r["author_id"] is not None else None
                        action = self._map_reaction_type(r.get("reaction_type_id"))
                        ts = r.get("created_at")

                        self._mirror_realtime(user_id, post_id, author_id, action)
                        if rid > new_last:
                            new_last = rid
                    self._set_last_id("reaction", new_last)
            except Exception as e:
                logger.debug("fetch reactions failed: %s", e, exc_info=True)

            # --------- Comment --------------
            try:
                rows = self._fetch_comments(conn, dialect, last_comment_id)
                if rows:
                    new_last = last_comment_id
                    for r in rows:
                        cid = int(r["id"])
                        user_id = int(r["user_id"])
                        post_id = int(r["post_id"])
                        author_id = int(r["author_id"]) if r["author_id"] is not None else None
                        action = "comment"
                        self._mirror_realtime(user_id, post_id, author_id, action)
                        if cid > new_last:
                            new_last = cid
                    self._set_last_id("comment", new_last)
            except Exception as e:
                logger.debug("fetch comments failed: %s", e, exc_info=True)

            # --------- PostView --------------
            try:
                rows = self._fetch_views(conn, dialect, last_view_id)
                if rows:
                    new_last = last_view_id
                    for r in rows:
                        vid = int(r["id"])
                        user_id = int(r["user_id"])
                        post_id = int(r["post_id"])
                        author_id = int(r["author_id"]) if r["author_id"] is not None else None
                        action = "view"
                        self._mirror_realtime(user_id, post_id, author_id, action)
                        if vid > new_last:
                            new_last = vid
                    self._set_last_id("view", new_last)
            except Exception as e:
                logger.debug("fetch views failed: %s", e, exc_info=True)

    # -------- fetchers (SQL by dialect) --------
    def _fetch_reactions(self, conn, dialect: str, last_id: int) -> List[Dict[str, Any]]:
        """
        SELECT PostReaction + join Post.UserId → author_id
        """
        pr = _q(self.schema, "PostReaction", dialect)
        p = _q(self.schema, "Post", dialect)
        sql = f"""
            SELECT pr.Id AS id,
                   pr.UserId AS user_id,
                   pr.PostId AS post_id,
                   pr.ReactionTypeId AS reaction_type_id,
                   pr.CreateDate AS created_at,
                   p.UserId AS author_id
            FROM {pr} pr
            LEFT JOIN {p} p ON p.Id = pr.PostId
            WHERE pr.Id > :last_id
              AND (pr.Status = 10 OR pr.Status IS NULL)
              AND (p.Status = 10 OR p.Status IS NULL)
            ORDER BY pr.Id ASC
            LIMIT :lim
        """
        rs = conn.execute(text(sql), {"last_id": int(last_id), "lim": int(self.max_rows)})
        cols = rs.keys()
        return [dict(zip(cols, row)) for row in rs.fetchall()]

    def _fetch_comments(self, conn, dialect: str, last_id: int) -> List[Dict[str, Any]]:
        c = _q(self.schema, "Comment", dialect)
        p = _q(self.schema, "Post", dialect)
        sql = f"""
            SELECT c.Id AS id,
                   c.UserId AS user_id,
                   c.PostId AS post_id,
                   c.CreateDate AS created_at,
                   p.UserId AS author_id
            FROM {c} c
            LEFT JOIN {p} p ON p.Id = c.PostId
            WHERE c.Id > :last_id
              AND (c.Status = 10 OR c.Status IS NULL)
              AND (p.Status = 10 OR p.Status IS NULL)
            ORDER BY c.Id ASC
            LIMIT :lim
        """
        rs = conn.execute(text(sql), {"last_id": int(last_id), "lim": int(self.max_rows)})
        cols = rs.keys()
        return [dict(zip(cols, row)) for row in rs.fetchall()]

    def _fetch_views(self, conn, dialect: str, last_id: int) -> List[Dict[str, Any]]:
        v = _q(self.schema, "PostView", dialect)
        p = _q(self.schema, "Post", dialect)
        sql = f"""
            SELECT v.Id AS id,
                   v.UserId AS user_id,
                   v.PostId AS post_id,
                   v.CreateDate AS created_at,
                   p.UserId AS author_id
            FROM {v} v
            LEFT JOIN {p} p ON p.Id = v.PostId
            WHERE v.Id > :last_id
              AND (v.Status = 10 OR v.Status IS NULL)
              AND (p.Status = 10 OR p.Status IS NULL)
            ORDER BY v.Id ASC
            LIMIT :lim
        """
        rs = conn.execute(text(sql), {"last_id": int(last_id), "lim": int(self.max_rows)})
        cols = rs.keys()
        return [dict(zip(cols, row)) for row in rs.fetchall()]

    # -------- realtime mirrors ----------
    def _mirror_realtime(self, user_id: int, post_id: int, author_id: Optional[int], action: str):
        """
        Đẩy các counter nhẹ vào Redis (nếu có), giống logic trong /interaction
        """
        r = self.redis
        if r is None:
            return
        try:
            # user 24h counters
            r.hincrby(f"user:{user_id}:eng24h", action, 1)
            r.expire(f"user:{user_id}:eng24h", 24 * 3600)

            # post 1h counters
            r.hincrby(f"post:{post_id}:eng1h", action, 1)
            r.expire(f"post:{post_id}:eng1h", 3600)

            # author 1h counters
            if author_id:
                r.hincrby(f"author:{author_id}:eng1h", action, 1)
                r.expire(f"author:{author_id}:eng1h", 3600)

            # co-visit nhẹ (giống /interaction): recent_items → covisit
            r.lpush(f"user:{user_id}:recent_items", post_id)
            r.ltrim(f"user:{user_id}:recent_items", 0, 99)
            r.expire(f"user:{user_id}:recent_items", 7 * 24 * 3600)

            recent = r.lrange(f"user:{user_id}:recent_items", 0, 50) or []
            for rid in recent:
                try:
                    rid = int(rid)
                    if rid == post_id:
                        continue
                    r.zincrby(f"covisit:{rid}", 1.0, post_id)
                    r.zincrby(f"covisit:{post_id}", 1.0, rid)
                    r.expire(f"covisit:{rid}", 7 * 24 * 3600)
                    r.expire(f"covisit:{post_id}", 7 * 24 * 3600)
                except Exception:
                    pass
        except Exception as e:
            logger.debug("mirror realtime failed: %s", e)

    # -------- last_id helpers ----------
    def _key(self, typ: str) -> str:
        return f"ingest:last_id:{typ}"

    def _get_last_id(self, typ: str) -> int:
        # ưu tiên Redis
        rid = _load_last_id(self.redis, self._key(typ), self._mem_last_ids.get(typ, 0))
        self._mem_last_ids[typ] = rid
        return rid

    def _set_last_id(self, typ: str, value: int):
        self._mem_last_ids[typ] = int(value)
        _save_last_id(self.redis, self._key(typ), int(value))

    # -------- mapping reaction type ----------
    def _map_reaction_type(self, reaction_type_id: Optional[int]) -> str:
        """
        Map ReactionTypeId -> action string
        Theo mapping (ảnh bạn gửi):
          1 like, 2 love, 3 laugh, 4 wow, 5 sad, 6 angry, 7 care, 8 save, 9 share
        Trả về key action phù hợp counters / embedding update.
        """
        m = {
            1: "like",
            2: "love",
            3: "laugh",
            4: "wow",
            5: "sad",
            6: "angry",
            7: "care",
            8: "save",
            9: "share",
        }
        return m.get(int(reaction_type_id) if reaction_type_id is not None else -1, "like")
