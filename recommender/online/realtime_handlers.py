# recommender/online/realtime_handlers.py
from __future__ import annotations
import logging, time
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RealtimeHandlers:
    """
    Các thao tác cập nhật Redis realtime khi:
    - User tương tác: like/comment/share/save/view
    - Tác giả đăng bài: fanout following feed
    - Cập nhật profile/interest đơn giản
    NOTE: Chỉ Redis – không đụng MySQL path /feed
    """

    def __init__(self, redis_client, action_weights: Optional[Dict[str, float]] = None):
        self.redis = redis_client
        self.action_weights = action_weights or {
            "view": 0.2, "like": 1.0, "comment": 1.5, "share": 2.0, "save": 1.2
        }

    # -----------------------------
    # USER INTERACTION
    # -----------------------------
    def on_user_interaction(self, user_id: int, post_id: int, author_id: Optional[int],
                            action: str, post_topic: Optional[str] = None, ts: Optional[float] = None):
        """
        Ghi nhận tương tác realtime vào Redis (counter/covisit/profile).
        Được gọi từ /interaction (push) hoặc ingestor (pull).
        """
        if self.redis is None:
            return
        ts = ts or time.time()
        key_user_24h = f"user:{user_id}:engagement_24h"
        key_post_1h  = f"post:{post_id}:engagement_1h"
        key_author_1h = f"author:{author_id}:engagement_1h" if author_id else None
        key_recent_items = f"user:{user_id}:recent_items"
        pipe = self.redis.pipeline(transaction=False)

        # 1) Counters (user / post / author)
        pipe.hincrby(key_user_24h, action, 1)
        pipe.expire(key_user_24h, 24*3600)

        pipe.hincrby(key_post_1h, action, 1)
        pipe.expire(key_post_1h, 3600)

        if key_author_1h:
            pipe.hincrby(key_author_1h, action, 1)
            pipe.expire(key_author_1h, 3600)

        # 2) Recent items + covisit
        pipe.lpush(key_recent_items, post_id)
        pipe.ltrim(key_recent_items, 0, 99)
        pipe.expire(key_recent_items, 24*3600)

        # Covisit nhẹ (3 item gần nhất)
        recent = self.redis.lrange(key_recent_items, 0, 2)
        for other in recent:
            try:
                other = int(other)
                if other != post_id:
                    pipe.zincrby(f"covisit:{other}", 1.0, post_id)
                    pipe.expire(f"covisit:{other}", 7*24*3600)
            except Exception:
                pass

        # 3) User profile interests (nếu có topic)
        if post_topic:
            try:
                pipe.hincrbyfloat(f"user:{user_id}:interests", post_topic, self.action_weights.get(action, 0.5))
                pipe.expire(f"user:{user_id}:interests", 7*24*3600)
                pipe.hset(f"user:{user_id}:profile", "last_active", datetime.utcnow().isoformat())
                pipe.expire(f"user:{user_id}:profile", 24*3600)
            except Exception:
                pass

        pipe.execute()

    # -----------------------------
    # FANOUT FOLLOWING FEED
    # -----------------------------
    def on_author_create_post(self, author_id: int, post_id: int, post_timestamp: Optional[float] = None,
                              followers: Optional[list] = None):
        """
        Khi tác giả đăng bài, fanout vào sorted-set của follower.
        followers: danh sách follower_id (đưa vào từ service của bạn / cache)
        """
        if self.redis is None or not followers:
            return
        score = int(post_timestamp or time.time())
        p = self.redis.pipeline(transaction=False)
        for fid in followers:
            key = f"following:{fid}:posts"
            p.zadd(key, {post_id: score})
            p.expire(key, 48*3600)
        p.execute()
        logger.debug(f"Fanout post {post_id} -> {len(followers)} followers")
