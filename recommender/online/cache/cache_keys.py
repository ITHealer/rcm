"""
CACHE KEY NAMING CONVENTIONS
=============================
Centralized cache key management for Redis

Benefits:
- Consistent naming
- Easy to update
- Type-safe keys
- Documentation
"""

from typing import Optional
from datetime import datetime


class CacheKeys:
    """
    Centralized cache key generator
    
    All Redis keys should be generated through this class
    to ensure consistency and avoid typos
    """
    
    # ========================================================================
    # EMBEDDINGS
    # ========================================================================
    
    @staticmethod
    def user_embedding(user_id: int) -> str:
        """
        User embedding key
        
        Format: user:{user_id}:embedding
        TTL: 7 days
        Type: Binary (384 floats)
        """
        return f"user:{user_id}:embedding"
    
    @staticmethod
    def post_embedding(post_id: int) -> str:
        """
        Post embedding key
        
        Format: post:{post_id}:embedding
        TTL: 30 days
        Type: Binary (384 floats)
        """
        return f"post:{post_id}:embedding"
    
    # ========================================================================
    # CF SIMILARITIES
    # ========================================================================
    
    @staticmethod
    def cf_user_similar(user_id: int) -> str:
        """
        Similar users for CF
        
        Format: cf:user:{user_id}:similar
        TTL: 1 day
        Type: JSON array of {user_id, similarity}
        """
        return f"cf:user:{user_id}:similar"
    
    @staticmethod
    def cf_post_similar(post_id: int) -> str:
        """
        Similar posts for CF
        
        Format: cf:post:{post_id}:similar
        TTL: 1 day
        Type: JSON array of {post_id, similarity}
        """
        return f"cf:post:{post_id}:similar"
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    @staticmethod
    def user_stats(user_id: int) -> str:
        """
        User statistics
        
        Format: stats:user:{user_id}
        TTL: 1 day
        Type: JSON object
        """
        return f"stats:user:{user_id}"
    
    @staticmethod
    def author_stats(author_id: int) -> str:
        """
        Author statistics
        
        Format: stats:author:{author_id}
        TTL: 1 day
        Type: JSON object
        """
        return f"stats:author:{author_id}"
    
    @staticmethod
    def post_stats(post_id: int) -> str:
        """
        Post statistics
        
        Format: stats:post:{post_id}
        TTL: 1 hour
        Type: JSON object
        """
        return f"stats:post:{post_id}"
    
    # ========================================================================
    # RECALL CHANNELS
    # ========================================================================
    
    @staticmethod
    def following_posts(user_id: int) -> str:
        """
        Following feed posts
        
        Format: following:{user_id}:posts
        TTL: 30 minutes
        Type: JSON array of post_ids
        """
        return f"following:{user_id}:posts"
    
    @staticmethod
    def following_list(user_id: int) -> str:
        """
        List of users being followed
        
        Format: following:{user_id}:list
        TTL: 1 hour
        Type: JSON array of user_ids
        """
        return f"following:{user_id}:list"
    
    @staticmethod
    def cf_posts(user_id: int) -> str:
        """
        CF recall posts
        
        Format: cf:user:{user_id}:posts
        TTL: 1 hour
        Type: JSON array of post_ids
        """
        return f"cf:user:{user_id}:posts"
    
    @staticmethod
    def content_posts(user_id: int) -> str:
        """
        Content-based recall posts
        
        Format: content:user:{user_id}:posts
        TTL: 1 hour
        Type: JSON array of post_ids
        """
        return f"content:user:{user_id}:posts"
    
    @staticmethod
    def trending_posts(hour: Optional[str] = None) -> str:
        """
        Trending posts (global, not user-specific)
        
        Format: trending:posts:{hour}
        TTL: 5 minutes
        Type: JSON array of post_ids
        
        Args:
            hour: Optional hour string (YYYYMMDD_HH), defaults to current
        """
        if hour is None:
            hour = datetime.now().strftime('%Y%m%d_%H')
        return f"trending:posts:{hour}"
    
    # ========================================================================
    # FEED CACHE
    # ========================================================================
    
    @staticmethod
    def user_feed(user_id: int) -> str:
        """
        Cached user feed
        
        Format: feed:{user_id}
        TTL: 5 minutes
        Type: JSON object with posts and metadata
        """
        return f"feed:{user_id}"
    
    @staticmethod
    def seen_posts(user_id: int) -> str:
        """
        Posts user has seen (for deduplication)
        
        Format: user:{user_id}:seen
        TTL: 1 day
        Type: Set of post_ids
        """
        return f"user:{user_id}:seen"
    
    # ========================================================================
    # USER INTERACTIONS
    # ========================================================================
    
    @staticmethod
    def user_recent_interactions(user_id: int) -> str:
        """
        Recent user interactions (for real-time embedding update)
        
        Format: user:{user_id}:interactions:recent
        TTL: 1 hour
        Type: JSON array of {post_id, action, timestamp}
        """
        return f"user:{user_id}:interactions:recent"
    
    @staticmethod
    def user_interaction_queue(user_id: int) -> str:
        """
        Queue for processing user interactions
        
        Format: queue:interactions:{user_id}
        TTL: None (processed then deleted)
        Type: Redis List
        """
        return f"queue:interactions:{user_id}"
    
    # ========================================================================
    # FRIEND RECOMMENDATIONS
    # ========================================================================
    
    @staticmethod
    def friend_recommendations(user_id: int) -> str:
        """
        Friend recommendations for user
        
        Format: friends:recs:{user_id}
        TTL: 1 hour
        Type: JSON array of {user_id, score, reason}
        """
        return f"friends:recs:{user_id}"
    
    # ========================================================================
    # MODEL METADATA
    # ========================================================================
    
    @staticmethod
    def model_version() -> str:
        """
        Current model version
        
        Format: model:version
        TTL: None
        Type: String (e.g., "v20251024_120504")
        """
        return "model:version"
    
    @staticmethod
    def model_metadata(version: str) -> str:
        """
        Model metadata for specific version
        
        Format: model:{version}:metadata
        TTL: 30 days
        Type: JSON object
        """
        return f"model:{version}:metadata"
    
    # ========================================================================
    # LOCKS & FLAGS
    # ========================================================================
    
    @staticmethod
    def update_lock(resource: str) -> str:
        """
        Lock for preventing concurrent updates
        
        Format: lock:{resource}
        TTL: 5 minutes
        Type: String (timestamp)
        """
        return f"lock:{resource}"
    
    @staticmethod
    def cache_warming_flag() -> str:
        """
        Flag indicating cache warming in progress
        
        Format: flag:cache_warming
        TTL: 1 hour
        Type: Boolean
        """
        return "flag:cache_warming"
    
    # ========================================================================
    # PATTERN MATCHING
    # ========================================================================
    
    @staticmethod
    def pattern_user_embeddings() -> str:
        """Pattern to match all user embeddings"""
        return "user:*:embedding"
    
    @staticmethod
    def pattern_post_embeddings() -> str:
        """Pattern to match all post embeddings"""
        return "post:*:embedding"
    
    @staticmethod
    def pattern_user_feeds() -> str:
        """Pattern to match all user feeds"""
        return "feed:*"
    
    @staticmethod
    def pattern_trending() -> str:
        """Pattern to match all trending caches"""
        return "trending:posts:*"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_key_ttl(key: str) -> int:
    """
    Get recommended TTL for a key type
    
    Args:
        key: Redis key
        
    Returns:
        TTL in seconds
    """
    ttl_map = {
        'user:.*:embedding': 604800,      # 7 days
        'post:.*:embedding': 2592000,     # 30 days
        'cf:.*:similar': 86400,           # 1 day
        'stats:user:.*': 86400,           # 1 day
        'stats:author:.*': 86400,         # 1 day
        'stats:post:.*': 3600,            # 1 hour
        'following:.*:posts': 1800,       # 30 minutes
        'following:.*:list': 3600,        # 1 hour
        'cf:user:.*:posts': 3600,         # 1 hour
        'content:user:.*:posts': 3600,    # 1 hour
        'trending:posts:.*': 300,         # 5 minutes
        'feed:.*': 300,                   # 5 minutes
        'user:.*:seen': 86400,            # 1 day
        'friends:recs:.*': 3600,          # 1 hour
    }
    
    import re
    for pattern, ttl in ttl_map.items():
        if re.match(pattern, key):
            return ttl
    
    return 3600  # Default: 1 hour


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Test cache keys"""
    
    keys = CacheKeys()
    
    print("Embedding keys:")
    print(f"  {keys.user_embedding(123)}")
    print(f"  {keys.post_embedding(456)}")
    
    print("\nCF keys:")
    print(f"  {keys.cf_user_similar(123)}")
    print(f"  {keys.cf_post_similar(456)}")
    
    print("\nRecall keys:")
    print(f"  {keys.following_posts(123)}")
    print(f"  {keys.cf_posts(123)}")
    print(f"  {keys.content_posts(123)}")
    print(f"  {keys.trending_posts()}")
    
    print("\nFeed keys:")
    print(f"  {keys.user_feed(123)}")
    print(f"  {keys.seen_posts(123)}")
    
    print("\nTTLs:")
    print(f"  user:123:embedding -> {get_key_ttl('user:123:embedding')}s")
    print(f"  feed:123 -> {get_key_ttl('feed:123')}s")