"""
Trending Recall Channel
======================
Get trending/popular posts based on engagement

Strategy:
- Calculate engagement score (likes + 2×comments + 3×shares)
- Apply time decay (recent posts get boost)
- Cache in Redis sorted set
- Refresh periodically (e.g., every 5 minutes)

Target: 100 posts
Latency: < 5ms (cached)
"""

from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import numpy as np
import time

from .base_recall import BaseRecall

logger = logging.getLogger(__name__)


class TrendingRecall(BaseRecall):
    """
    Trending posts recall
    "What's hot right now"
    """
    
    def __init__(
        self,
        redis_client=None,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize trending recall
        
        Args:
            redis_client: Redis connection
            data: Data dictionary with 'posts' and 'postreaction' DataFrames
            config: Configuration
        """
        super().__init__(redis_client, config)
        
        self.data = data or {}
        
        # Configuration
        self.trending_window_hours = self.config.get('trending_window_hours', 24)
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        self.time_decay_factor = self.config.get('time_decay_factor', 0.8)
        
        # Engagement weights
        self.like_weight = self.config.get('like_weight', 1.0)
        self.comment_weight = self.config.get('comment_weight', 2.0)
        self.share_weight = self.config.get('share_weight', 3.0)
        self.save_weight = self.config.get('save_weight', 1.5)
        
        # Compute trending posts
        self._compute_trending()
    
    def _compute_trending(self):
        """
        Compute trending posts with engagement scores
        
        Formula:
        score = (likes + 2×comments + 3×shares) × time_decay
        time_decay = (1 / (1 + age_hours)) ^ decay_factor
        """
        self.trending_posts = []
        
        if 'posts' not in self.data or 'postreaction' not in self.data:
            logger.warning("Missing data for trending computation")
            return
        
        posts = self.data['posts'].copy()
        reactions = self.data['postreaction'].copy()
        
        # Filter recent posts only
        if 'CreateDate' in posts.columns:
            posts['created_at'] = pd.to_datetime(posts['CreateDate'], errors='coerce')
            cutoff = pd.Timestamp.now() - pd.Timedelta(hours=self.trending_window_hours)
            posts = posts[posts['created_at'] >= cutoff]
        
        if posts.empty:
            logger.warning(f"No posts in trending window ({self.trending_window_hours}h)")
            return
        
        # Count reactions by type for each post
        reaction_counts = reactions.groupby(['PostId', 'ReactionTypeId']).size().unstack(fill_value=0)
        
        # Calculate engagement scores
        trending_scores = []
        now = pd.Timestamp.now()
        
        for _, post in posts.iterrows():
            post_id = int(post['Id'])
            created_at = post.get('created_at', pd.Timestamp.now())
            
            # Get reaction counts
            likes = reaction_counts.loc[post_id, 1] if post_id in reaction_counts.index and 1 in reaction_counts.columns else 0
            comments = reaction_counts.loc[post_id, 2] if post_id in reaction_counts.index and 2 in reaction_counts.columns else 0
            shares = reaction_counts.loc[post_id, 3] if post_id in reaction_counts.index and 3 in reaction_counts.columns else 0
            saves = reaction_counts.loc[post_id, 5] if post_id in reaction_counts.index and 5 in reaction_counts.columns else 0
            
            # Engagement score
            engagement = (
                likes * self.like_weight +
                comments * self.comment_weight +
                shares * self.share_weight +
                saves * self.save_weight
            )
            
            # Time decay
            age_hours = (now - created_at).total_seconds() / 3600
            time_decay = (1 / (1 + age_hours)) ** self.time_decay_factor
            
            # Final score
            score = engagement * time_decay
            
            trending_scores.append({
                'post_id': post_id,
                'score': score,
                'engagement': engagement,
                'age_hours': age_hours
            })
        
        # Sort by score
        trending_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Store top posts
        self.trending_posts = [item['post_id'] for item in trending_scores]
        
        logger.info(f"Computed trending: {len(self.trending_posts)} posts, "
                   f"top score: {trending_scores[0]['score']:.2f if trending_scores else 0}")
    
    def recall(self, user_id: int = None, k: int = 100) -> List[int]:
        """
        Recall trending posts
        
        Note: user_id is ignored for trending (same for all users)
        
        Args:
            user_id: User ID (ignored)
            k: Number of candidates to return
            
        Returns:
            List of trending post IDs
        """
        start_time = time.time()
        
        # Try cache first (global trending, not user-specific)
        cache_key = "trending_posts"
        
        if self.redis:
            try:
                cached = self.redis.zrevrange(cache_key, 0, k - 1)
                
                if cached:
                    self.metrics['cache_hits'] += 1
                    self.metrics['total_recalls'] += 1
                    candidates = [int(post_id) for post_id in cached]
                    self.metrics['total_candidates'] += len(candidates)
                    self.metrics['total_time_ms'] += (time.time() - start_time) * 1000
                    return candidates
            except Exception as e:
                logger.debug(f"Redis get trending error: {e}")
        
        self.metrics['cache_misses'] += 1
        
        # Return precomputed trending
        candidates = self.trending_posts[:k]
        
        # Update metrics
        self.metrics['total_recalls'] += 1
        self.metrics['total_candidates'] += len(candidates)
        self.metrics['total_time_ms'] += (time.time() - start_time) * 1000
        
        logger.debug(f"Trending recall: {len(candidates)} posts")
        
        return candidates
    
    def refresh_cache(self):
        """
        Refresh trending cache in Redis
        
        Should be called periodically (e.g., every 5 minutes via cron)
        """
        if not self.redis:
            logger.warning("Redis not available, cannot refresh cache")
            return
        
        logger.info("Refreshing trending cache...")
        
        # Recompute trending
        self._compute_trending()
        
        # Store in Redis sorted set
        if self.trending_posts:
            cache_key = "trending_posts"
            
            try:
                # Clear old data
                self.redis.delete(cache_key)
                
                # Add new data with scores
                mapping = {
                    post_id: len(self.trending_posts) - idx  # Higher score = higher rank
                    for idx, post_id in enumerate(self.trending_posts)
                }
                
                self.redis.zadd(cache_key, mapping)
                
                # Set expiry
                self.redis.expire(cache_key, self.cache_ttl * 2)
                
                logger.info(f"Cached {len(self.trending_posts)} trending posts in Redis")
                
            except Exception as e:
                logger.error(f"Failed to cache trending: {e}")
        else:
            logger.warning("No trending posts to cache")


def compute_and_cache_trending(
    data: Dict,
    redis_client=None,
    config: Optional[Dict] = None
) -> List[int]:
    """
    Convenience function to compute and cache trending posts
    
    Use this in a cron job (e.g., every 5 minutes):
    ```python
    # cron_trending.py
    from app.recall.trending_recall import compute_and_cache_trending
    
    trending = compute_and_cache_trending(data, redis_client)
    print(f"Cached {len(trending)} trending posts")
    ```
    
    Args:
        data: Data dictionary
        redis_client: Redis connection
        config: Configuration
        
    Returns:
        List of trending post IDs
    """
    recall = TrendingRecall(
        redis_client=redis_client,
        data=data,
        config=config
    )
    
    recall.refresh_cache()
    
    # Also return for inspection
    return recall.recall(k=1000)