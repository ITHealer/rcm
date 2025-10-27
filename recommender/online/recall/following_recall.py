"""
FOLLOWING RECALL CHANNEL
========================
Get posts from users that target user follows

Strategy:
- Query followed users
- Get their recent posts (last 48h)
- Sort by recency
- Cache results

Target: 400 posts
Latency: < 20ms
"""

from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
import json

from .base_recall import BaseRecall

logger = logging.getLogger(__name__)


class FollowingRecall(BaseRecall):
    """
    Recall posts from followed users
    """
    
    def __init__(
        self,
        redis_client=None,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        following_dict: Optional[Dict[int, List[int]]] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize following recall
        
        Args:
            redis_client: Redis connection
            data: Data dictionary with 'post' and 'friendship' DataFrames
            following_dict: Precomputed following relationships
            config: Configuration
        """
        super().__init__(redis_client, config)
        self.r = redis_client
        self.data = data or {}
        self.following_dict = following_dict or {}
        self.r = redis_client
        # Configuration
        self.recent_hours = self.config.get('recent_hours', 48)
        self.cache_ttl = self.config.get('cache_ttl', 1800)  # 30 minutes
    
    def recall(self, user_id: int, k: int = 400) -> List[int]:
        """
        Recall posts from followed users
        
        Args:
            user_id: Target user ID
            k: Number of posts to return
            
        Returns:
            List of post IDs (sorted by recency)
        """
        if not self.r:
            return []
        
        # Try cache first
        cached = self._get_from_cache_json(user_id)
        if cached is not None:
            return cached[:k]
        
        # Get followed users
        followed_users = self.following_dict.get(user_id, [])
        
        if not followed_users:
            return []
        
        # Get posts from followed users
        candidate_posts = []
        
        if 'post' in self.data:
            posts_df = self.data['post']
            
            # Filter: posts by followed users + recent
            cutoff_time = datetime.now() - timedelta(hours=self.recent_hours)
            
            for author_id in followed_users:
                author_posts = posts_df[
                    (posts_df['UserId'] == author_id) &
                    (pd.to_datetime(posts_df['CreateDate']) >= cutoff_time)
                ]
                
                for _, post in author_posts.iterrows():
                    candidate_posts.append({
                        'post_id': int(post['Id']),
                        'created_at': pd.to_datetime(post['CreateDate']),
                        'author_id': int(post['UserId'])
                    })
        
        # Sort by recency (newest first)
        candidate_posts.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Extract post IDs
        post_ids = [p['post_id'] for p in candidate_posts]
        
        # Cache result
        self._set_to_cache_json(user_id, post_ids)
        
        return post_ids[:k]
    
    def _get_from_cache_json(self, user_id: int) -> Optional[List[int]]:
        """Get following posts from Redis cache"""
        if self.redis is None:
            return None
        
        key = f"following:{user_id}:posts"
        
        try:
            value = self.redis.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        return None
    
    def _set_to_cache_json(self, user_id: int, post_ids: List[int]):
        """Cache following posts to Redis"""
        if self.redis is None:
            return
        
        key = f"following:{user_id}:posts"
        
        try:
            value = json.dumps(post_ids)
            self.redis.setex(key, self.cache_ttl, value)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")