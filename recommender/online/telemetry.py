# from __future__ import annotations
# import json, threading, time, logging, queue
# from typing import Dict, Any, Optional, List
# from datetime import datetime
# from sqlalchemy import text
# from recommender.common.db import get_session_factory

# logger = logging.getLogger(__name__)

# class AsyncDBLogger:
#     """
#     Ghi log “không chặn request”:
#     - Producer: push event vào queue (memory hoặc Redis)
#     - Consumer: background thread flush batch vào MySQL
#     """
#     def __init__(self, config: Dict, redis_client=None):
#         self.enabled = bool(config.get("enabled", False))
#         if not self.enabled:
#             self.mode = "off"
#             return
#         self.mode = config.get("mode", "queue")
#         self.batch_size = int(config.get("batch_size", 200))
#         self.flush_interval = float(config.get("flush_interval_ms", 1000)) / 1000.0
#         self.max_queue = int(config.get("max_queue", 100000))
#         self.drop_on_overflow = bool(config.get("drop_on_overflow", True))

#         # DB
#         mysql_cfg = config.get("mysql", {})
#         self.mysql_enabled = bool(mysql_cfg.get("enabled", False))
#         self._SessionFactory = None
#         if self.mysql_enabled:
#             self._SessionFactory = get_session_factory(
#                 mysql_cfg.get("url"),
#                 pool_size=mysql_cfg.get("pool_size", 20),
#                 max_overflow=mysql_cfg.get("max_overflow", 40),
#                 pool_recycle=mysql_cfg.get("pool_recycle", 1800),
#                 pool_pre_ping=mysql_cfg.get("pool_pre_ping", True),
#             )

#         # Redis queue (optional)
#         self.redis = redis_client if config.get("redis_queue", {}).get("enabled", False) else None
#         self.redis_key = (config.get("redis_queue", {}) or {}).get("key", "telemetry:events")
#         self.redis_maxlen = (config.get("redis_queue", {}) or {}).get("maxlen", 200000)

#         # Local queue fallback
#         self.q: queue.Queue = queue.Queue(maxsize=self.max_queue)

#         self._stop = False
#         self._thread = threading.Thread(target=self._run_consumer, name="AsyncDBLogger", daemon=True)
#         self._thread.start()

#     # ---------- Public API ----------
#     def log_feed_request(self, payload: Dict[str, Any]):
#         self._enqueue({"type": "feed_request", "ts": time.time(), "data": payload})

#     def log_interaction(self, payload: Dict[str, Any]):
#         self._enqueue({"type": "interaction", "ts": time.time(), "data": payload})

#     def log_rerank_audit(self, payload: Dict[str, Any]):
#         self._enqueue({"type": "rerank_audit", "ts": time.time(), "data": payload})

#     def shutdown(self):
#         self._stop = True
#         self._thread.join(timeout=2)

#     # ---------- Internal ----------
#     def _enqueue(self, item: Dict[str, Any]):
#         if self.mode == "off" or not self.enabled:
#             return
#         # Prefer Redis as write-ahead (không block)
#         if self.redis is not None:
#             try:
#                 self.redis.xadd(self.redis_key, {"event": json.dumps(item)}, maxlen=self.redis_maxlen, approximate=True)
#                 return
#             except Exception as e:
#                 logger.warning(f"Telemetry redis enqueue failed: {e}")
#         # Fallback to local queue
#         try:
#             self.q.put_nowait(item)
#         except queue.Full:
#             if not self.drop_on_overflow:
#                 self.q.put(item)  # block (không khuyến nghị)
#             else:
#                 logger.warning("Telemetry queue overflow — drop event to protect server")

#     def _run_consumer(self):
#         last_flush = time.time()
#         batch: List[Dict[str, Any]] = []

#         while not self._stop:
#             try:
#                 # Pick events from Redis first (drain quickly)
#                 if self.redis is not None:
#                     try:
#                         # read 500 at a time
#                         msgs = self.redis.xread({self.redis_key: "0-0"}, count=500, block=1)
#                         if msgs:
#                             _, entries = msgs[0]
#                             for _id, fields in entries:
#                                 ev = json.loads(fields.get(b"event", fields.get("event", "{}")))
#                                 batch.append(ev)
#                                 # delete processed entry
#                                 self.redis.xdel(self.redis_key, _id)
#                     except Exception as e:
#                         logger.debug(f"Telemetry redis read err: {e}")

#                 # Drain local queue
#                 while len(batch) < self.batch_size:
#                     try:
#                         batch.append(self.q.get_nowait())
#                     except queue.Empty:
#                         break

#                 now = time.time()
#                 if (len(batch) >= self.batch_size) or (now - last_flush >= self.flush_interval):
#                     if batch:
#                         self._flush_batch(batch)
#                         batch.clear()
#                         last_flush = now

#                 time.sleep(0.01)
#             except Exception as e:
#                 logger.warning(f"Telemetry consumer loop error: {e}")
#                 time.sleep(0.2)

#     def _flush_batch(self, batch: List[Dict[str, Any]]):
#         if not self.mysql_enabled or self._SessionFactory is None:
#             # Dev: chỉ log
#             logger.debug(f"(dev) telemetry dropped batch(n={len(batch)})")
#             return

#         # Split by type
#         feed_rows, inter_rows, audit_rows = [], [], []
#         for ev in batch:
#             t = ev.get("type")
#             d = ev.get("data", {})
#             if t == "feed_request":
#                 feed_rows.append(d)
#             elif t == "interaction":
#                 inter_rows.append(d)
#             elif t == "rerank_audit":
#                 audit_rows.append(d)

#         with self._SessionFactory() as session:
#             if feed_rows:
#                 session.execute(text("""
#                     INSERT INTO feed_requests
#                     (user_id, requested_at, recall_following, recall_cf, recall_content, recall_trending, recall_total,
#                      final_count, reasons, latency_ms, recall_ms, ranking_ms, rerank_ms)
#                     VALUES
#                     (:user_id, :requested_at, :rf, :rcf, :rct, :rt, :rtot, :final_count, :reasons, :lat, :r_ms, :m_ms, :b_ms)
#                 """), [
#                     {
#                         "user_id": r["user_id"],
#                         "requested_at": datetime.utcfromtimestamp(r["timestamp"]),
#                         "rf": r["recall"]["following"],
#                         "rcf": r["recall"]["cf"],
#                         "rct": r["recall"]["content"],
#                         "rt": r["recall"]["trending"],
#                         "rtot": r["recall"]["total"],
#                         "final_count": r["final_count"],
#                         "reasons": json.dumps(r.get("reasons") or []),
#                         "lat": int(r["latency_ms"]),
#                         "r_ms": int(r["stage_ms"]["recall"]),
#                         "m_ms": int(r["stage_ms"]["ranking"]),
#                         "b_ms": int(r["stage_ms"]["rerank"]),
#                     } for r in feed_rows
#                 ])

#             if inter_rows:
#                 session.execute(text("""
#                     INSERT INTO user_interactions
#                     (user_id, post_id, author_id, action, created_at, session_id, meta)
#                     VALUES
#                     (:user_id, :post_id, :author_id, :action, :created_at, :session_id, :meta)
#                 """), [
#                     {
#                         "user_id": r["user_id"], "post_id": r["post_id"], "author_id": r.get("author_id"),
#                         "action": r["action"], "created_at": datetime.utcfromtimestamp(r["timestamp"]),
#                         "session_id": r.get("session_id"), "meta": json.dumps(r.get("meta") or {})
#                     } for r in inter_rows
#                 ])

#             if audit_rows:
#                 session.execute(text("""
#                     INSERT INTO rerank_audit
#                     (user_id, created_at, pre_count, pre_authors, pre_top_author_share,
#                      fresh_boosted, q_removed_status, q_removed_min_score,
#                      dedup_removed, div_removed_consecutive, div_removed_cap,
#                      post_count, post_authors, post_top_author_share)
#                     VALUES
#                     (:user_id, :created_at, :pre_count, :pre_authors, :pre_share,
#                      :fresh, :q_rs, :q_rmin,
#                      :dedup, :div_consec, :div_cap,
#                      :post_count, :post_authors, :post_share)
#                 """), [
#                     {
#                         "user_id": r["user_id"], "created_at": datetime.utcfromtimestamp(r["timestamp"]),
#                         "pre_count": r["pre"]["count"], "pre_authors": r["pre"]["authors"],
#                         "pre_share": r["pre"]["top_author_share"], "fresh": r["freshness"]["boosted"],
#                         "q_rs": r["quality"]["removed_status"], "q_rmin": r["quality"]["removed_min_score"],
#                         "dedup": r["dedup"]["removed"], "div_consec": r["diversity"]["removed_consecutive"],
#                         "div_cap": r["diversity"]["removed_cap"], "post_count": r["post"]["count"],
#                         "post_authors": r["post"]["authors"], "post_share": r["post"]["top_author_share"]
#                     } for r in audit_rows
#                 ])

#             session.commit()


# recommender/online/telemetry.py
from __future__ import annotations
import json, time, threading, queue, logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy import text
from recommender.common.db import get_session_factory

logger = logging.getLogger(__name__)

class AsyncDBLogger:
    """
    Non-blocking telemetry logger:
      - Enqueue events (feed / rerank_audit / interaction)
      - Background thread flushes to Postgres (or any SQLAlchemy URL)
      - Optional Redis stream as write-ahead queue
    """

    def __init__(self, cfg: Dict, redis_client=None):
        self.enabled = bool(cfg.get("enabled", False))
        if not self.enabled:
            self.mode = "off"
            return

        self.mode = cfg.get("mode", "queue")            # queue | direct | off
        self.batch_size = int(cfg.get("batch_size", 200))
        self.flush_interval = float(cfg.get("flush_interval_ms", 1000)) / 1000.0
        self.max_queue = int(cfg.get("max_queue", 100000))
        self.drop_on_overflow = bool(cfg.get("drop_on_overflow", True))

        # "mysql" key is generic DSN holder (có thể là postgres URL)
        mcfg = cfg.get("mysql", {})
        self.mysql_enabled = bool(mcfg.get("enabled", False))
        self._Session = None
        if self.mysql_enabled:
            self._Session = get_session_factory(
                mcfg.get("url"),
                pool_size=mcfg.get("pool_size", 20),
                max_overflow=mcfg.get("max_overflow", 40),
                pool_recycle=mcfg.get("pool_recycle", 1800),
                pool_pre_ping=mcfg.get("pool_pre_ping", True),
            )

        # Redis stream (optional)
        rcfg = cfg.get("redis_queue", {}) or {}
        self.redis = redis_client if rcfg.get("enabled") else None
        self.redis_key = rcfg.get("key", "telemetry:events")
        self.redis_maxlen = rcfg.get("maxlen", 200000)

        # Local in-memory queue
        self.q: "queue.Queue[Dict]" = queue.Queue(maxsize=self.max_queue)

        self._stop = False
        threading.Thread(target=self._consumer, name="TelemetryConsumer", daemon=True).start()

    # ------------- Public logging APIs -------------

    def log_feed_request(self, data: Dict[str, Any]) -> None:
        """Enqueue feed request event."""
        self._enqueue({"t": "feed", "ts": time.time(), "d": data})

    def log_rerank_audit(self, data: Dict[str, Any]) -> None:
        """Enqueue rerank audit event."""
        self._enqueue({"t": "audit", "ts": time.time(), "d": data})

    def log_interaction(self, data: Dict[str, Any]) -> None:
        """Enqueue user interaction event (optional but used by /interaction)."""
        self._enqueue({"t": "interaction", "ts": time.time(), "d": data})

    def ping(self) -> bool:
        """Return True if DB is reachable, False otherwise."""
        if not self.mysql_enabled or self._Session is None:
            return False
        try:
            with self._Session() as s:
                s.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def shutdown(self):
        self._stop = True

    # ------------- Internal helpers ----------------

    def _enqueue(self, ev: Dict[str, Any]):
        if not self.enabled or self.mode == "off":
            return
        # Try Redis stream first
        if self.redis is not None:
            try:
                self.redis.xadd(self.redis_key, {"e": json.dumps(ev)}, maxlen=self.redis_maxlen, approximate=True)
                return
            except Exception as e:
                logger.warning(f"telemetry redis enqueue fail: {e}")
        # Fallback: local queue
        try:
            self.q.put_nowait(ev)
        except queue.Full:
            if self.drop_on_overflow:
                logger.warning("telemetry queue overflow; drop event")
            else:
                self.q.put(ev)

    def _consumer(self):
        last = time.time()
        batch: List[Dict] = []
        while not self._stop:
            try:
                # Drain Redis stream
                if self.redis is not None:
                    try:
                        msgs = self.redis.xread({self.redis_key: "0-0"}, count=500, block=1) or []
                        for _, entries in msgs:
                            for msg_id, fields in entries:
                                payload = fields.get(b"e", fields.get("e", "{}"))
                                ev = json.loads(payload)
                                batch.append(ev)
                                self.redis.xdel(self.redis_key, msg_id)
                    except Exception:
                        pass

                # Drain local queue
                while len(batch) < self.batch_size:
                    try:
                        batch.append(self.q.get_nowait())
                    except queue.Empty:
                        break

                now = time.time()
                if batch and (len(batch) >= self.batch_size or now - last >= self.flush_interval):
                    self._flush(batch)
                    batch.clear()
                    last = now

                time.sleep(0.01)
            except Exception as e:
                logger.debug(f"telemetry consumer err: {e}")
                time.sleep(0.1)

    def _flush(self, batch: List[Dict]):
        """Flush a mixed batch to DB (feed / audit / interaction)."""
        if not self.mysql_enabled or self._Session is None:
            # Dev mode: drop silently
            logger.debug(f"(dev) drop telemetry batch n={len(batch)}")
            return

        feed_rows: List[Dict] = []
        audit_rows: List[Dict] = []
        inter_rows: List[Dict] = []

        for ev in batch:
            t = ev.get("t")
            if t == "feed":
                feed_rows.append(ev["d"])
            elif t == "audit":
                audit_rows.append(ev["d"])
            elif t == "interaction":
                inter_rows.append(ev["d"])

        with self._Session() as s:
            if feed_rows:
                s.execute(text("""
                    INSERT INTO ai_logs.feed_request_log
                      (user_id, requested_at, recall_following, recall_cf, recall_content, recall_trending,
                       recall_total, final_count, reasons, latency_ms, recall_ms, ranking_ms, rerank_ms)
                    VALUES
                      (:uid, to_timestamp(:at), :rf, :rcf, :rct, :rt, :tot, :cnt, CAST(:reasons AS JSONB),
                       :lat, :rms, :mms, :bms)
                """), [
                    dict(
                        uid=r["user_id"],
                        at=r["timestamp"],
                        rf=r["recall"]["following"], rcf=r["recall"]["cf"], rct=r["recall"]["content"],
                        rt=r["recall"]["trending"], tot=r["recall"]["total"],
                        cnt=r["final_count"], reasons=json.dumps(r.get("reasons") or []),
                        lat=int(r["latency_ms"]), rms=int(r["stage_ms"]["recall"]),
                        mms=int(r["stage_ms"]["ranking"]), bms=int(r["stage_ms"]["rerank"])
                    )
                    for r in feed_rows
                ])

            if audit_rows:
                s.execute(text("""
                    INSERT INTO ai_logs.rerank_audit_log
                      (user_id, created_at, pre_count, pre_authors, pre_top_author_share, fresh_boosted,
                       q_removed_status, q_removed_min_score, dedup_removed, div_removed_consecutive,
                       div_removed_cap, post_count, post_authors, post_top_author_share)
                    VALUES
                      (:uid, to_timestamp(:at), :pc, :pa, :ps, :fb, :qrs, :qrm, :dd, :drc, :drcap, :oc, :oa, :os)
                """), [
                    dict(
                        uid=r["user_id"], at=r["timestamp"],
                        pc=r["pre"]["count"], pa=r["pre"]["authors"], ps=r["pre"]["top_author_share"],
                        fb=r["freshness"]["boosted"], qrs=r["quality"]["removed_status"],
                        qrm=r["quality"]["removed_min_score"], dd=r["dedup"]["removed"],
                        drc=r["diversity"]["removed_consecutive"], drcap=r["diversity"]["removed_cap"],
                        oc=r["post"]["count"], oa=r["post"]["authors"], os=r["post"]["top_author_share"]
                    )
                    for r in audit_rows
                ])

            if inter_rows:
                s.execute(text("""
                    INSERT INTO ai_logs.user_interaction_log
                      (user_id, post_id, author_id, action, created_at, session_id, meta)
                    VALUES
                      (:uid, :pid, :aid, :act, to_timestamp(:at), :sid, CAST(:meta AS JSONB))
                """), [
                    dict(
                        uid=r["user_id"],
                        pid=r["post_id"],
                        aid=r.get("author_id"),
                        act=r["action"],
                        at=r["timestamp"],
                        sid=r.get("session_id"),
                        meta=json.dumps(r.get("meta") or {})
                    )
                    for r in inter_rows
                ])

            s.commit()
