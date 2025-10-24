"""
TRENDING RECALL CHANNEL
=======================
Get trending/viral posts (recent popular content)

Strategy:
- Get recent posts (last 6 hours)
- Calculate trending score: engagement_rate × recency_boost × velocity
- Cache with short TTL (5 minutes)
- Global trending (not personalized)

Target: 100 posts
Latency: < 10ms (cached)
"""

from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import numpy as np

from .base_recall import BaseRecall

logger = logging.getLogger(__name__)


class TrendingRecall(BaseRecall):
    """
    Trending/viral posts recall
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
            data: Data dictionary with 'post' and 'postreaction' DataFrames
            config: Configuration
        """
        super().__init__(redis_client, config)
        
        self.data = data or {}
        
        # Configuration
        self.trending_window_hours = self.config.get('trending_window_hours', 6)
        self.min_engagement_count = self.config.get('min_engagement_count', 5)
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes (short!)
        
        # Weights for trending score
        self.view_weight = self.config.get('view_weight', 1.0)
        self.like_weight = self.config.get('like_weight', 2.0)
        self.comment_weight = self.config.get('comment_weight', 3.0)
        self.share_weight = self.config.get('share_weight', 5.0)
        
        logger.info(f"TrendingRecall initialized (window: {self.trending_window_hours}h)")
    
    def recall(self, user_id: int = None, k: int = 100) -> List[int]:
        """
        Recall trending posts
        
        Note: user_id is ignored (trending is global, not personalized)
        
        Process:
        1. Check cache (5 min TTL)
        2. If miss, compute trending:
           - Get recent posts (last 6h)
           - Calculate trending score
           - Rank by score
        3. Cache and return top K
        
        Args:
            user_id: Ignored (trending is global)
            k: Number of posts to return
            
        Returns:
            List of post IDs (sorted by trending score)
        """
        # Try cache first (trending is global, so no user_id in key)
        cached = self._get_from_cache_json()
        if cached is not None:
            return cached[:k]
        
        # Compute trending posts
        trending_posts = self._compute_trending()
        
        # Cache result
        self._set_to_cache_json(trending_posts)
        
        return trending_posts[:k]
    
    def _compute_trending(self) -> List[int]:
        """
        Compute trending posts
        
        Trending score formula:
        score = (views + 2×likes + 3×comments + 5×shares) × recency_boost × velocity
        
        Where:
        - recency_boost: 1.0 for posts < 1h old, decay exponentially
        - velocity: engagement_count / hours_since_post
        
        Returns:
            List of post IDs sorted by trending score
        """
        if 'post' not in self.data or 'postreaction' not in self.data:
            logger.warning("Missing data for trending computation")
            return []
        
        posts_df = self.data['post']
        reactions_df = self.data['postreaction']
        
        # Filter recent posts
        cutoff_time = datetime.now() - timedelta(hours=self.trending_window_hours)
        
        recent_posts = posts_df[
            pd.to_datetime(posts_df['CreateDate']) >= cutoff_time
        ].copy()
        
        if recent_posts.empty:
            logger.warning("No recent posts for trending")
            return []
        
        # Calculate engagement for each post
        post_scores = {}
        
        for _, post in recent_posts.iterrows():
            post_id = int(post['Id'])
            created_at = pd.to_datetime(post['CreateDate'])
            
            # Get reactions for this post
            post_reactions = reactions_df[reactions_df['PostId'] == post_id]
            
            if len(post_reactions) < self.min_engagement_count:
                continue  # Skip low-engagement posts
            
            # Count engagement types
            views = len(post_reactions[post_reactions['ReactionTypeId'] == 4])
            likes = len(post_reactions[post_reactions['ReactionTypeId'] == 1])
            comments = len(post_reactions[post_reactions['ReactionTypeId'] == 2])
            shares = len(post_reactions[post_reactions['ReactionTypeId'] == 3])
            
            # Calculate engagement score
            engagement_score = (
                views * self.view_weight +
                likes * self.like_weight +
                comments * self.comment_weight +
                shares * self.share_weight
            )
            
            # Calculate recency boost (exponential decay)
            hours_ago = (datetime.now() - created_at).total_seconds() / 3600
            recency_boost = np.exp(-hours_ago / (self.trending_window_hours / 2))
            
            # Calculate velocity (engagement per hour)
            velocity = (views + likes + comments + shares) / (hours_ago + 0.1)  # +0.1 to avoid /0
            
            # Final trending score
            trending_score = engagement_score * recency_boost * velocity
            
            post_scores[post_id] = trending_score
        
        # Sort by score
        ranked = sorted(post_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Extract post IDs
        post_ids = [post_id for post_id, _ in ranked]
        
        logger.debug(f"Computed trending: {len(post_ids)} posts")
        
        return post_ids
    
    def _get_from_cache_json(self) -> Optional[List[int]]:
        """
        Get trending posts from Redis cache
        
        Note: Global cache (no user_id)
        """
        if self.redis is None:
            return None
        
        # Use current hour as key part (so cache refreshes every hour)
        current_hour = datetime.now().strftime('%Y%m%d_%H')
        key = f"trending:posts:{current_hour}"
        
        try:
            value = self.redis.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.debug(f"Cache get error: {e}")
        
        return None
    
    def _set_to_cache_json(self, post_ids: List[int]):
        """
        Cache trending posts to Redis
        
        Note: Global cache (no user_id)
        """
        if self.redis is None:
            return
        
        # Use current hour as key part
        current_hour = datetime.now().strftime('%Y%m%d_%H')
        key = f"trending:posts:{current_hour}"
        
        try:
            value = json.dumps(post_ids)
            self.redis.setex(key, self.cache_ttl, value)
            logger.debug(f"Cached {len(post_ids)} trending posts")
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def refresh_cache(self):
        """
        Manually refresh trending cache
        
        Useful for scheduled tasks (e.g., cron job every 5 minutes)
        """
        logger.info("Refreshing trending cache...")
        trending_posts = self._compute_trending()
        self._set_to_cache_json(trending_posts)
        logger.info(f"Cached {len(trending_posts)} trending posts")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
    from recommender.online.recall.trending_recall import compute_and_cache_trending
    
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


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Test trending recall"""
    from recommender.common.data_loader import load_data
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    print("Loading data...")
    data = load_data('dataset')
    
    # Initialize recall
    print("Initializing trending recall...")
    trending_recall = TrendingRecall(
        redis_client=None,
        data=data
    )
    
    # Test recall
    print("\nComputing trending posts...")
    candidates = trending_recall.recall(k=100)
    
    print(f"\nFound {len(candidates)} trending posts")
    print(f"Top 10: {candidates[:10]}")
    
    # Print metrics
    metrics = trending_recall.get_metrics()
    print(f"\nMetrics: {metrics}")
    
    # Test refresh
    print("\nTesting cache refresh...")
    trending_recall.refresh_cache()