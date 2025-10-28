"""
Following Recall Channel
=======================
Get recent posts from users that this user follows

Strategy:
- Get following list from cache/DB
- Get their recent posts (48 hours)
- Sort by recency
- Cache results

Target: 400 posts
Latency: < 15ms
"""

from typing import List, Dict, Optional, Set
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import time

from .base_recall import BaseRecall

logger = logging.getLogger(__name__)


class FollowingRecall(BaseRecall):
    """
    Following feed recall channel
    "Posts from people you follow"
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
            data: Data dictionary with 'posts' DataFrame
            following_dict: Pre-computed following relationships {user_id: [author_ids]}
            config: Configuration
        """
        super().__init__(redis_client, config)
        
        self.data = data or {}
        self.following_dict = following_dict or {}
        
        # Configuration
        self.recent_hours = self.config.get('recent_hours', 48)
        self.cache_ttl = self.config.get('cache_ttl', 1800)  # 30 minutes
        
        # Build author->posts index for fast lookup
        self._build_author_posts_index()
    
    def _build_author_posts_index(self):
        """
        Build index: author_id -> list of recent posts
        For fast lookup during recall
        """
        self.author_posts = {}
        
        if 'posts' not in self.data:
            logger.warning("No posts data available for following recall")
            return
        
        posts = self.data['posts'].copy()
        
        # Filter recent posts only
        if 'CreateDate' in posts.columns:
            posts['created_at'] = pd.to_datetime(posts['CreateDate'], errors='coerce')
            cutoff = pd.Timestamp.now() - pd.Timedelta(hours=self.recent_hours)
            posts = posts[posts['created_at'] >= cutoff]
        
        # Group by author
        for author_id, group in posts.groupby('UserId'):
            # Sort by recency
            if 'created_at' in group.columns:
                group = group.sort_values('created_at', ascending=False)
            
            self.author_posts[int(author_id)] = group['Id'].astype(int).tolist()
        
        logger.info(f"Built author-posts index: {len(self.author_posts)} authors, "
                   f"{sum(len(p) for p in self.author_posts.values())} recent posts")
    
    def recall(self, user_id: int, k: int = 400) -> List[int]:
        """
        Recall posts from followed users
        
        Process:
        1. Get following list
        2. Get recent posts from those authors
        3. Sort by recency
        4. Return top k
        
        Args:
            user_id: User ID
            k: Number of candidates to return
            
        Returns:
            List of post IDs (most recent first)
        """
        start_time = time.time()
        
        # Try cache first
        cache_key = f"following_recall:{user_id}:{k}"
        cached = self._cache_get(cache_key)
        
        if cached:
            self.metrics['cache_hits'] += 1
            self.metrics['total_recalls'] += 1
            candidates = json.loads(cached)
            self.metrics['total_candidates'] += len(candidates)
            self.metrics['total_time_ms'] += (time.time() - start_time) * 1000
            return candidates
        
        self.metrics['cache_misses'] += 1
        
        # Get following list
        following = self._get_following_list(user_id)
        
        if not following:
            logger.debug(f"User {user_id} has no following")
            self.metrics['total_recalls'] += 1
            self.metrics['total_time_ms'] += (time.time() - start_time) * 1000
            return []
        
        # Collect posts from followed authors
        candidates = []
        seen = set()
        
        for author_id in following:
            author_posts = self.author_posts.get(author_id, [])
            
            for post_id in author_posts:
                if post_id not in seen:
                    candidates.append(post_id)
                    seen.add(post_id)
                
                if len(candidates) >= k:
                    break
            
            if len(candidates) >= k:
                break
        
        # Update metrics
        self.metrics['total_recalls'] += 1
        self.metrics['total_candidates'] += len(candidates)
        self.metrics['total_time_ms'] += (time.time() - start_time) * 1000
        
        # Cache result
        self._cache_set(cache_key, json.dumps(candidates), self.cache_ttl)
        
        logger.debug(f"Following recall for user {user_id}: {len(candidates)} posts from {len(following)} authors")
        
        return candidates[:k]
    
    def _get_following_list(self, user_id: int) -> List[int]:
        """
        Get list of users that this user follows
        
        Priority:
        1. Redis cache
        2. Pre-computed following_dict
        3. Database query (if data available)
        
        Args:
            user_id: User ID
            
        Returns:
            List of author IDs
        """
        # Try Redis cache first
        if self.redis:
            try:
                cache_key = f"user:following:{user_id}"
                cached = self.redis.smembers(cache_key)
                
                if cached:
                    return [int(uid) for uid in cached]
            except Exception as e:
                logger.debug(f"Redis get following error: {e}")
        
        # Try pre-computed dict
        if user_id in self.following_dict:
            return self.following_dict[user_id]
        
        # Try database (if friendships data available)
        if 'friendships' in self.data:
            friendships = self.data['friendships']
            following = friendships[friendships['UserId'] == user_id]['FriendId'].tolist()
            return [int(fid) for fid in following]
        
        return []
    
    def refresh_index(self):
        """Refresh the author-posts index (call periodically)"""
        logger.info("Refreshing following recall index...")
        self._build_author_posts_index()
        logger.info("Following recall index refreshed")