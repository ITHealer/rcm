"""
Redis Cache Service for Recommendation System

This service provides caching layer for:
- User profiles
- Post features
- Following lists
- Trending posts
- Seen posts
- CF similarities

Team App will sync data from MySQL to Redis.
AI team only reads from Redis to avoid overwhelming MySQL database.
"""

import redis
from redis.connection import ConnectionPool
from typing import Optional, List, Dict, Set, Any, Tuple
import json
import pickle
import logging
import time
from datetime import datetime

from recommender.online.app.config import settings
from recommender.online.app.utils.logger import setup_logger

logger = setup_logger(__name__)


class CacheService:
    """
    Redis cache service for recommendation system
    
    Architecture:
    - Team App syncs data from MySQL -> Redis
    - AI team reads from Redis only
    - No direct MySQL queries to avoid overload
    """
    
    def __init__(self):
        """Initialize Redis connection pool"""
        self.pool = ConnectionPool(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
            socket_connect_timeout=settings.REDIS_SOCKET_TIMEOUT,
            decode_responses=False,  # Handle encoding ourselves
            max_connections=50,
            retry_on_timeout=True
        )
        self.redis: Optional[redis.Redis] = None
        self._connected = False
        
    def connect(self) -> bool:
        """
        Connect to Redis server
        
        Returns:
            bool: True if connected successfully
        """
        try:
            self.redis = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            self.redis.ping()
            
            self._connected = True
            logger.info(f"✅ Connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            
            # Log some stats
            info = self.redis.info()
            logger.info(f"Redis version: {info.get('redis_version', 'unknown')}")
            logger.info(f"Used memory: {info.get('used_memory_human', 'unknown')}")
            
            return True
            
        except Exception as e:
            self._connected = False
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    def close(self):
        """Close Redis connection"""
        if self.redis:
            try:
                self.redis.close()
                self._connected = False
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
    
    def health_check(self) -> bool:
        """
        Check if Redis is healthy
        
        Returns:
            bool: True if Redis is responding
        """
        if not self.redis:
            return False
        try:
            return self.redis.ping()
        except:
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to Redis"""
        return self._connected and self.health_check()
    
    # ==================== USER PROFILE ====================
    
    def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user profile from cache
        
        Data synced by Team App from MySQL
        
        Args:
            user_id: User ID
            
        Returns:
            Dict with user profile or None if not found
        """
        try:
            key = f"user:profile:{user_id}"
            data = self.redis.get(key)
            
            if data:
                logger.debug(f"Cache HIT: user profile {user_id}")
                return json.loads(data.decode('utf-8'))
            
            logger.debug(f"Cache MISS: user profile {user_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting user profile {user_id}: {e}")
            return None
    
    def set_user_profile(
        self, 
        user_id: int, 
        profile: Dict[str, Any],
        ttl: int = None
    ) -> bool:
        """
        Cache user profile
        
        Called by Team App when syncing from MySQL
        
        Args:
            user_id: User ID
            profile: User profile dict
            ttl: Time to live in seconds (None = use default)
            
        Returns:
            True if successful
        """
        try:
            key = f"user:profile:{user_id}"
            value = json.dumps(profile, ensure_ascii=False)
            
            if ttl is None:
                ttl = settings.USER_PROFILE_TTL
            
            self.redis.setex(key, ttl, value)
            logger.debug(f"Cached user profile {user_id} with TTL {ttl}s")
            return True
            
        except Exception as e:
            logger.error(f"Error caching user profile {user_id}: {e}")
            return False
    
    def delete_user_profile(self, user_id: int) -> bool:
        """
        Delete user profile from cache
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful
        """
        try:
            key = f"user:profile:{user_id}"
            self.redis.delete(key)
            logger.debug(f"Deleted user profile {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting user profile {user_id}: {e}")
            return False
    
    # ==================== POST FEATURES ====================
    
    def get_post_features(self, post_id: int) -> Optional[Dict[str, Any]]:
        """
        Get post features from cache
        
        Data synced by Team App from MySQL
        
        Args:
            post_id: Post ID
            
        Returns:
            Dict with post features or None if not found
        """
        try:
            key = f"post:features:{post_id}"
            data = self.redis.get(key)
            
            if data:
                logger.debug(f"Cache HIT: post features {post_id}")
                return pickle.loads(data)  # Use pickle for numpy arrays
            
            logger.debug(f"Cache MISS: post features {post_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting post features {post_id}: {e}")
            return None
    
    def set_post_features(
        self,
        post_id: int,
        features: Dict[str, Any],
        ttl: int = None
    ) -> bool:
        """
        Cache post features
        
        Called by Team App when syncing from MySQL
        
        Args:
            post_id: Post ID
            features: Post features dict (can contain numpy arrays)
            ttl: Time to live in seconds (None = use default)
            
        Returns:
            True if successful
        """
        try:
            key = f"post:features:{post_id}"
            value = pickle.dumps(features)  # Pickle handles numpy arrays
            
            if ttl is None:
                ttl = settings.POST_FEATURES_TTL
            
            self.redis.setex(key, ttl, value)
            logger.debug(f"Cached post features {post_id} with TTL {ttl}s")
            return True
            
        except Exception as e:
            logger.error(f"Error caching post features {post_id}: {e}")
            return False
    
    def get_posts_features_batch(
        self, 
        post_ids: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get multiple post features in batch (more efficient)
        
        Args:
            post_ids: List of post IDs
            
        Returns:
            Dict mapping post_id to features
        """
        if not post_ids:
            return {}
        
        try:
            keys = [f"post:features:{pid}" for pid in post_ids]
            
            # Use pipeline for batch get - much faster
            pipe = self.redis.pipeline()
            for key in keys:
                pipe.get(key)
            results = pipe.execute()
            
            # Parse results
            features_dict = {}
            for post_id, data in zip(post_ids, results):
                if data:
                    try:
                        features_dict[post_id] = pickle.loads(data)
                    except Exception as e:
                        logger.warning(f"Error unpickling post {post_id}: {e}")
            
            hit_rate = len(features_dict) / len(post_ids) * 100 if post_ids else 0
            logger.debug(
                f"Batch get {len(post_ids)} posts, "
                f"hit rate: {hit_rate:.1f}% ({len(features_dict)}/{len(post_ids)})"
            )
            
            return features_dict
            
        except Exception as e:
            logger.error(f"Error batch getting post features: {e}")
            return {}
    
    def set_posts_features_batch(
        self,
        posts_features: Dict[int, Dict[str, Any]],
        ttl: int = None
    ) -> bool:
        """
        Set multiple post features in batch
        
        Args:
            posts_features: Dict mapping post_id to features
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        if not posts_features:
            return True
        
        try:
            if ttl is None:
                ttl = settings.POST_FEATURES_TTL
            
            pipe = self.redis.pipeline()
            for post_id, features in posts_features.items():
                key = f"post:features:{post_id}"
                value = pickle.dumps(features)
                pipe.setex(key, ttl, value)
            
            pipe.execute()
            logger.debug(f"Batch cached {len(posts_features)} post features")
            return True
            
        except Exception as e:
            logger.error(f"Error batch setting post features: {e}")
            return False
    
    def delete_post_features(self, post_id: int) -> bool:
        """Delete post features from cache"""
        try:
            key = f"post:features:{post_id}"
            self.redis.delete(key)
            logger.debug(f"Deleted post features {post_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting post features {post_id}: {e}")
            return False
    
    # ==================== FOLLOWING ====================
    
    def get_following_list(self, user_id: int) -> Set[int]:
        """
        Get list of users that this user follows
        
        Data synced by Team App from MySQL
        
        Args:
            user_id: User ID
            
        Returns:
            Set of author IDs
        """
        try:
            key = f"user:following:{user_id}"
            members = self.redis.smembers(key)
            
            if members:
                author_ids = {int(m.decode('utf-8')) for m in members}
                logger.debug(f"Got {len(author_ids)} following for user {user_id}")
                return author_ids
            
            logger.debug(f"No following found for user {user_id}")
            return set()
            
        except Exception as e:
            logger.error(f"Error getting following list {user_id}: {e}")
            return set()
    
    def set_following_list(
        self,
        user_id: int,
        author_ids: Set[int],
        ttl: int = None
    ) -> bool:
        """
        Cache following list
        
        Called by Team App when syncing from MySQL
        
        Args:
            user_id: User ID
            author_ids: Set of author IDs that user follows
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            key = f"user:following:{user_id}"
            
            # Clear old data first
            self.redis.delete(key)
            
            # Add new data
            if author_ids:
                self.redis.sadd(key, *author_ids)
                
                if ttl is None:
                    ttl = settings.FOLLOWING_CACHE_TTL
                
                self.redis.expire(key, ttl)
                logger.debug(f"Cached {len(author_ids)} following for user {user_id}")
            else:
                logger.debug(f"User {user_id} follows nobody, cleared cache")
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching following list {user_id}: {e}")
            return False
    
    def add_following(self, user_id: int, author_id: int) -> bool:
        """
        Add a single author to user's following list
        
        Args:
            user_id: User ID
            author_id: Author ID to follow
            
        Returns:
            True if successful
        """
        try:
            key = f"user:following:{user_id}"
            self.redis.sadd(key, author_id)
            logger.debug(f"User {user_id} now follows {author_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding following: {e}")
            return False
    
    def remove_following(self, user_id: int, author_id: int) -> bool:
        """
        Remove an author from user's following list
        
        Args:
            user_id: User ID
            author_id: Author ID to unfollow
            
        Returns:
            True if successful
        """
        try:
            key = f"user:following:{user_id}"
            self.redis.srem(key, author_id)
            logger.debug(f"User {user_id} unfollowed {author_id}")
            return True
        except Exception as e:
            logger.error(f"Error removing following: {e}")
            return False
    
    def get_following_posts(
        self, 
        user_id: int, 
        limit: int = 500
    ) -> List[Tuple[int, float]]:
        """
        Get recent posts from users that this user follows
        
        Returns posts sorted by timestamp (most recent first)
        
        Args:
            user_id: User ID
            limit: Max number of posts to return
            
        Returns:
            List of (post_id, timestamp) tuples
        """
        try:
            key = f"user:following:{user_id}:posts"
            
            # Get from sorted set (score = timestamp)
            # zrevrange returns highest scores first (most recent)
            results = self.redis.zrevrange(key, 0, limit - 1, withscores=True)
            
            if results:
                posts = [(int(pid.decode('utf-8')), float(score)) 
                        for pid, score in results]
                logger.debug(f"Got {len(posts)} following posts for user {user_id}")
                return posts
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting following posts {user_id}: {e}")
            return []
    
    def add_following_post(
        self,
        user_id: int,
        post_id: int,
        timestamp: float = None,
        ttl: int = 172800  # 48 hours default
    ) -> bool:
        """
        Add a new post to user's following feed
        
        Called by Team App when a followed author creates new post
        
        Args:
            user_id: User ID
            post_id: Post ID to add
            timestamp: Post creation timestamp (None = now)
            ttl: Time to live in seconds (default 48h)
            
        Returns:
            True if successful
        """
        try:
            key = f"user:following:{user_id}:posts"
            
            if timestamp is None:
                timestamp = time.time()
            
            # Add to sorted set with timestamp as score
            self.redis.zadd(key, {post_id: timestamp})
            
            # Set expiry on the sorted set
            self.redis.expire(key, ttl)
            
            # Keep only recent posts (max 1000 to avoid memory bloat)
            self.redis.zremrangebyrank(key, 0, -1001)
            
            logger.debug(f"Added post {post_id} to user {user_id} following feed")
            return True
            
        except Exception as e:
            logger.error(f"Error adding following post: {e}")
            return False
    
    def remove_following_post(self, user_id: int, post_id: int) -> bool:
        """
        Remove a post from user's following feed
        
        Args:
            user_id: User ID
            post_id: Post ID to remove
            
        Returns:
            True if successful
        """
        try:
            key = f"user:following:{user_id}:posts"
            self.redis.zrem(key, post_id)
            logger.debug(f"Removed post {post_id} from user {user_id} following feed")
            return True
        except Exception as e:
            logger.error(f"Error removing following post: {e}")
            return False
    
    # ==================== TRENDING ====================
    
    def get_trending_posts(
        self,
        category: Optional[str] = None,
        limit: int = 200
    ) -> List[Tuple[int, float]]:
        """
        Get trending posts (global or by category)
        
        Data synced by Team App from MySQL based on engagement metrics
        
        Args:
            category: Category name (None for global)
            limit: Max number of posts
            
        Returns:
            List of (post_id, score) tuples sorted by score DESC
        """
        try:
            if category:
                key = f"trending:posts:{category}"
            else:
                key = "trending:posts:global"
            
            # Get from sorted set with scores
            results = self.redis.zrevrange(key, 0, limit - 1, withscores=True)
            
            if results:
                posts = [(int(pid.decode('utf-8')), float(score)) 
                        for pid, score in results]
                logger.debug(f"Got {len(posts)} trending posts (category: {category or 'global'})")
                return posts
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting trending posts: {e}")
            return []
    
    def set_trending_posts(
        self,
        posts_with_scores: List[Tuple[int, float]],
        category: Optional[str] = None,
        ttl: int = None
    ) -> bool:
        """
        Update trending posts cache
        
        Called by Team App when recalculating trending scores
        
        Args:
            posts_with_scores: List of (post_id, engagement_score)
            category: Category name (None for global)
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            if category:
                key = f"trending:posts:{category}"
            else:
                key = "trending:posts:global"
            
            # Clear old data
            self.redis.delete(key)
            
            # Add new data
            if posts_with_scores:
                mapping = {post_id: score for post_id, score in posts_with_scores}
                self.redis.zadd(key, mapping)
                
                if ttl is None:
                    ttl = settings.TRENDING_CACHE_TTL
                
                self.redis.expire(key, ttl)
                
                logger.info(
                    f"Updated trending posts (category: {category or 'global'}): "
                    f"{len(posts_with_scores)} items, TTL {ttl}s"
                )
            else:
                logger.warning(f"No trending posts to cache for category: {category or 'global'}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting trending posts: {e}")
            return False
    
    # ==================== SEEN POSTS ====================
    
    def mark_posts_as_seen(
        self,
        user_id: int,
        post_ids: List[int],
        ttl: int = 604800  # 7 days
    ) -> bool:
        """
        Mark posts as seen by user
        
        Args:
            user_id: User ID
            post_ids: List of post IDs to mark as seen
            ttl: Time to live in seconds (default 7 days)
            
        Returns:
            True if successful
        """
        if not post_ids:
            return True
        
        try:
            key = f"user:seen:{user_id}"
            
            self.redis.sadd(key, *post_ids)
            self.redis.expire(key, ttl)
            
            logger.debug(f"Marked {len(post_ids)} posts as seen for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error marking posts as seen: {e}")
            return False
    
    def get_seen_posts(self, user_id: int) -> Set[int]:
        """
        Get all posts seen by user
        
        Args:
            user_id: User ID
            
        Returns:
            Set of post IDs
        """
        try:
            key = f"user:seen:{user_id}"
            members = self.redis.smembers(key)
            
            if members:
                post_ids = {int(m.decode('utf-8')) for m in members}
                logger.debug(f"User {user_id} has seen {len(post_ids)} posts")
                return post_ids
            
            return set()
            
        except Exception as e:
            logger.error(f"Error getting seen posts: {e}")
            return set()
    
    def clear_seen_posts(self, user_id: int) -> bool:
        """
        Clear all seen posts for user
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful
        """
        try:
            key = f"user:seen:{user_id}"
            self.redis.delete(key)
            logger.debug(f"Cleared seen posts for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing seen posts: {e}")
            return False
    
    # ==================== CF SIMILARITIES ====================
    
    def get_similar_users(
        self,
        user_id: int,
        limit: int = 100
    ) -> List[Tuple[int, float]]:
        """
        Get users similar to this user (for CF recall)
        
        Data computed offline and synced by Team App
        
        Args:
            user_id: User ID
            limit: Max number of similar users
            
        Returns:
            List of (similar_user_id, similarity_score) tuples
        """
        try:
            key = f"cf:similar:{user_id}"
            results = self.redis.zrevrange(key, 0, limit - 1, withscores=True)
            
            if results:
                similar_users = [(int(uid.decode('utf-8')), float(score)) 
                                for uid, score in results]
                logger.debug(f"Got {len(similar_users)} similar users for {user_id}")
                return similar_users
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting similar users: {e}")
            return []
    
    def set_similar_users(
        self,
        user_id: int,
        similar_users: List[Tuple[int, float]],
        ttl: int = 86400  # 24 hours
    ) -> bool:
        """
        Cache CF user similarities
        
        Called by Team App when syncing offline-computed similarities
        
        Args:
            user_id: User ID
            similar_users: List of (similar_user_id, similarity_score)
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            key = f"cf:similar:{user_id}"
            
            self.redis.delete(key)
            
            if similar_users:
                mapping = {uid: score for uid, score in similar_users}
                self.redis.zadd(key, mapping)
                self.redis.expire(key, ttl)
                logger.debug(f"Cached {len(similar_users)} similar users for {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting similar users: {e}")
            return False
    
    # ==================== USER INTERACTIONS ====================
    
    def get_user_liked_posts(self, user_id: int) -> Set[int]:
        """
        Get posts liked by user
        
        Args:
            user_id: User ID
            
        Returns:
            Set of post IDs
        """
        try:
            key = f"user:likes:{user_id}"
            members = self.redis.smembers(key)
            
            if members:
                return {int(m.decode('utf-8')) for m in members}
            return set()
            
        except Exception as e:
            logger.error(f"Error getting user liked posts: {e}")
            return set()
    
    def set_user_liked_posts(
        self,
        user_id: int,
        post_ids: Set[int],
        ttl: int = 86400
    ) -> bool:
        """
        Set posts liked by user
        
        Called by Team App when syncing from MySQL
        
        Args:
            user_id: User ID
            post_ids: Set of liked post IDs
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            key = f"user:likes:{user_id}"
            
            self.redis.delete(key)
            
            if post_ids:
                self.redis.sadd(key, *post_ids)
                self.redis.expire(key, ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting user liked posts: {e}")
            return False
    
    # ==================== UTILITIES ====================
    
    def delete_key(self, key: str) -> bool:
        """
        Delete a key from cache
        
        Args:
            key: Redis key to delete
            
        Returns:
            True if successful
        """
        try:
            self.redis.delete(key)
            logger.debug(f"Deleted key: {key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False
    
    def delete_keys_by_pattern(self, pattern: str) -> int:
        """
        Delete multiple keys matching pattern
        
        Args:
            pattern: Redis key pattern (e.g., "user:profile:*")
            
        Returns:
            Number of keys deleted
        """
        try:
            keys = self.redis.keys(pattern)
            if keys:
                count = self.redis.delete(*keys)
                logger.info(f"Deleted {count} keys matching pattern: {pattern}")
                return count
            return 0
        except Exception as e:
            logger.error(f"Error deleting keys by pattern {pattern}: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dict with cache stats
        """
        try:
            info = self.redis.info()
            
            stats = {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0"),
                "used_memory_peak_human": info.get("used_memory_peak_human", "0"),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                ),
                "evicted_keys": info.get("evicted_keys", 0),
                "expired_keys": info.get("expired_keys", 0)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    @staticmethod
    def _calculate_hit_rate(hits: int, misses: int) -> float:
        """
        Calculate cache hit rate percentage
        
        Args:
            hits: Number of cache hits
            misses: Number of cache misses
            
        Returns:
            Hit rate as percentage (0-100)
        """
        total = hits + misses
        if total == 0:
            return 0.0
        return round((hits / total) * 100, 2)
    
    def flush_db(self) -> bool:
        """
        DANGER: Flush entire database
        Only use for testing/development
        
        Returns:
            True if successful
        """
        try:
            self.redis.flushdb()
            logger.warning("⚠️ Flushed entire Redis database!")
            return True
        except Exception as e:
            logger.error(f"Error flushing database: {e}")
            return False


# Singleton instance
cache_service = CacheService()