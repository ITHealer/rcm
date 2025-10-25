# recommender/online/realtime_jobs.py
from __future__ import annotations
import logging, threading, time
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

class RealTimeJobs:
    """
    Các job nền siêu nhẹ dùng Redis:
    - Refresh trending pool (mỗi 5’): score từ counter 1h và tuổi bài
    - Refresh post features nóng (mỗi 15’)
    Không đụng MySQL, đọc counters đã đổ vào Redis.
    """

    def __init__(self, redis_client, interval_trending=300, interval_post_feat=900):
        self.redis = redis_client
        self.int_trend = int(interval_trending)
        self.int_post = int(interval_post_feat)
        self._stop = False
        self._thr1 = None
        self._thr2 = None

    def start(self):
        if self.redis is None:
            logger.info("RealTimeJobs disabled (no Redis).")
            return
        self._thr1 = threading.Thread(target=self._loop_trending, name="TrendJob", daemon=True)
        self._thr2 = threading.Thread(target=self._loop_post_features, name="PostFeatJob", daemon=True)
        self._thr1.start(); self._thr2.start()
        logger.info("✅ RealTimeJobs started (trending=%ss, post_feat=%ss)", self.int_trend, self.int_post)

    def stop(self):
        self._stop = True

    # -----------------------------
    # Trending
    # -----------------------------
    def _loop_trending(self):
        while not self._stop:
            try:
                # Giả sử ta giữ một set các post “active” gần đây:
                # key 'active_posts' (zset score=created_ts) được cập nhật bởi ingestor/handlers
                now = int(time.time())
                six_hours_ago = now - 6 * 3600
                # lấy ~10k posts gần đây (nếu có)
                posts = self.redis.zrevrangebyscore("active_posts", now, six_hours_ago, start=0, num=10000) or []
                scores = {}
                pipe = self.redis.pipeline(transaction=False)
                for pid in posts:
                    pid = int(pid)
                    # engagement 1h
                    h = self.redis.hgetall(f"post:{pid}:engagement_1h") or {}
                    like = float(h.get(b'like', 0) or 0)
                    cmt  = float(h.get(b'comment', 0) or 0)
                    share= float(h.get(b'share', 0) or 0)
                    # tuổi bài (giờ)
                    created = self.redis.hget(f"post:{pid}:meta", "created_at") or None
                    if created:
                        try:
                            # created lưu epoch int
                            age_h = max((now - int(created)) / 3600.0, 0.1)
                        except Exception:
                            age_h = 1.0
                    else:
                        age_h = 1.0
                    score = (like + 2*cmt + 3*share) / (age_h ** 1.5)
                    if score > 0:
                        scores[pid] = score

                # reset pool trending
                self.redis.delete("trending:global:6h")
                if scores:
                    self.redis.zadd("trending:global:6h", scores)
                    self.redis.expire("trending:global:6h", 6*3600)
                logger.debug("Trending refreshed: %s posts", len(scores))
            except Exception as e:
                logger.debug("trend job error: %s", e)
            time.sleep(self.int_trend)

    # -----------------------------
    # Post features nóng (CTR, velocity…)
    # -----------------------------
    def _loop_post_features(self):
        while not self._stop:
            try:
                # dùng cùng nguồn 'active_posts'
                now = int(time.time())
                six_hours_ago = now - 6 * 3600
                posts = self.redis.zrevrangebyscore("active_posts", now, six_hours_ago, start=0, num=10000) or []
                p = self.redis.pipeline(transaction=False)
                for pid in posts:
                    pid = int(pid)
                    feat_key = f"post:{pid}:features"
                    h = self.redis.hgetall(f"post:{pid}:engagement_1h") or {}
                    v = float(h.get(b'view', 0) or 0)
                    l = float(h.get(b'like', 0) or 0) + float(h.get(b'love', 0) or 0)
                    c = float(h.get(b'comment', 0) or 0)
                    s = float(h.get(b'share', 0) or 0)
                    ctr_1h = (l + c + s) / v if v > 0 else 0.0
                    p.hset(feat_key, mapping={
                        "ctr_1h": ctr_1h,
                        "velocity": "high" if ctr_1h >= 0.2 else ("normal" if ctr_1h >= 0.05 else "low"),
                        "updated_at": datetime.utcnow().isoformat()
                    })
                    p.expire(feat_key, 24*3600)
                p.execute()
                logger.debug("Post features refreshed: %s posts", len(posts))
            except Exception as e:
                logger.debug("post feat job error: %s", e)
            time.sleep(self.int_post)
