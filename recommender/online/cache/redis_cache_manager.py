"""
REDIS CACHE MANAGER
===================
Manage 7 layers of caching for online inference

Layers:
1. Embeddings (user, post)
2. CF Similarities
3. User/Author Stats
4. Following Feed
5. Trending Posts
6. User Interactions
7. Model Metadata

Update: After offline training + Real-time for interactions
"""

import os
import sys
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Try imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️  redis not available. Install: pip install redis")


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RedisCacheManager:
    """
    Manage Redis cache for online inference
    
    7 Layers:
    1. Embeddings
    2. CF Similarities  
    3. User/Author Stats
    4. Following Feed
    5. Trending
    6. User Interactions
    7. Model Metadata
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6381,
        db: int = 0,
        password: Optional[str] = None
    ):
        """
        Initialize Redis cache manager
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis required. Install: pip install redis")
        
        logger.info("Connecting to Redis...")
        logger.info(f"  Host: {host}:{port}")
        logger.info(f"  Database: {db}")
        
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # Binary data
        )
        
        # Test connection
        try:
            self.redis.ping()
            logger.info("✅ Connected to Redis successfully")
        except redis.ConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
        
        # TTL constants (seconds)
        self.TTL = {
            'embedding_user': 7 * 24 * 3600,      # 7 days
            'embedding_post': 30 * 24 * 3600,     # 30 days
            'cf_similarity': 7 * 24 * 3600,       # 7 days
            'user_stats': 24 * 3600,              # 24 hours
            'author_stats': 24 * 3600,            # 24 hours
            'following_feed': 48 * 3600,          # 48 hours
            'trending_1h': 1 * 3600,              # 1 hour
            'trending_6h': 6 * 3600,              # 6 hours
            'trending_24h': 24 * 3600,            # 24 hours
            'user_interactions': 30 * 24 * 3600, # 30 days
            'model_metadata': None                # No expiry
        }
    
    # ════════════════════════════════════════════════════════════════════
    # LAYER 1: EMBEDDINGS
    # ════════════════════════════════════════════════════════════════════
    
    def set_user_embedding(self, user_id: int, embedding: np.ndarray):
        """Cache user embedding"""
        key = f"user:{user_id}:embedding"
        value = embedding.tobytes()
        self.redis.setex(key, self.TTL['embedding_user'], value)
    
    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Get user embedding from cache"""
        key = f"user:{user_id}:embedding"
        value = self.redis.get(key)
        
        if value:
            return np.frombuffer(value, dtype=np.float32)
        return None
    
    def set_post_embedding(self, post_id: int, embedding: np.ndarray):
        """Cache post embedding"""
        key = f"post:{post_id}:embedding"
        value = embedding.tobytes()
        self.redis.setex(key, self.TTL['embedding_post'], value)
    
    def get_post_embedding(self, post_id: int) -> Optional[np.ndarray]:
        """Get post embedding from cache"""
        key = f"post:{post_id}:embedding"
        value = self.redis.get(key)
        
        if value:
            return np.frombuffer(value, dtype=np.float32)
        return None
    
    def set_embeddings_batch(
        self,
        embeddings_dict: Dict[str, Dict[int, np.ndarray]]
    ):
        """
        Batch set embeddings
        
        Args:
            embeddings_dict: {'user': {user_id: emb}, 'post': {post_id: emb}}
        """
        logger.info("\n📦 Caching embeddings to Redis...")
        
        # User embeddings
        if 'user' in embeddings_dict:
            user_embs = embeddings_dict['user']
            logger.info(f"   Caching {len(user_embs):,} user embeddings...")
            
            pipe = self.redis.pipeline()
            for user_id, emb in user_embs.items():
                key = f"user:{user_id}:embedding"
                value = emb.tobytes()
                pipe.setex(key, self.TTL['embedding_user'], value)
            pipe.execute()
        
        # Post embeddings
        if 'post' in embeddings_dict:
            post_embs = embeddings_dict['post']
            logger.info(f"   Caching {len(post_embs):,} post embeddings...")
            
            pipe = self.redis.pipeline()
            for post_id, emb in post_embs.items():
                key = f"post:{post_id}:embedding"
                value = emb.tobytes()
                pipe.setex(key, self.TTL['embedding_post'], value)
            pipe.execute()
        
        logger.info("   ✅ Embeddings cached")
    
    # ════════════════════════════════════════════════════════════════════
    # LAYER 2: CF SIMILARITIES
    # ════════════════════════════════════════════════════════════════════
    
    def set_user_similar_users(
        self,
        user_id: int,
        similar_users: List[Tuple[int, float]]
    ):
        """Cache similar users for a user"""
        key = f"cf:user:{user_id}:similar"
        value = json.dumps(similar_users)
        self.redis.setex(key, self.TTL['cf_similarity'], value)
    
    def get_user_similar_users(
        self,
        user_id: int
    ) -> Optional[List[Tuple[int, float]]]:
        """Get similar users from cache"""
        key = f"cf:user:{user_id}:similar"
        value = self.redis.get(key)
        
        if value:
            return json.loads(value)
        return None
    
    def set_post_similar_posts(
        self,
        post_id: int,
        similar_posts: List[Tuple[int, float]]
    ):
        """Cache similar posts for a post"""
        key = f"cf:post:{post_id}:similar"
        value = json.dumps(similar_posts)
        self.redis.setex(key, self.TTL['cf_similarity'], value)
    
    def get_post_similar_posts(
        self,
        post_id: int
    ) -> Optional[List[Tuple[int, float]]]:
        """Get similar posts from cache"""
        key = f"cf:post:{post_id}:similar"
        value = self.redis.get(key)
        
        if value:
            return json.loads(value)
        return None
    
    def set_cf_similarities_batch(self, cf_model: Dict):
        """
        Batch set CF similarities
        
        Args:
            cf_model: Dict with 'user_similarities' and 'item_similarities'
        """
        logger.info("\n📦 Caching CF similarities to Redis...")
        
        # User similarities
        if 'user_similarities' in cf_model:
            user_sims = cf_model['user_similarities']
            logger.info(f"   Caching {len(user_sims):,} user similarities...")
            
            pipe = self.redis.pipeline()
            for user_id, similar_users in user_sims.items():
                key = f"cf:user:{user_id}:similar"
                value = json.dumps(similar_users)
                pipe.setex(key, self.TTL['cf_similarity'], value)
            pipe.execute()
        
        # Item similarities
        if 'item_similarities' in cf_model:
            item_sims = cf_model['item_similarities']
            logger.info(f"   Caching {len(item_sims):,} item similarities...")
            
            pipe = self.redis.pipeline()
            for post_id, similar_posts in item_sims.items():
                key = f"cf:post:{post_id}:similar"
                value = json.dumps(similar_posts)
                pipe.setex(key, self.TTL['cf_similarity'], value)
            pipe.execute()
        
        logger.info("   ✅ CF similarities cached")
    
    # ════════════════════════════════════════════════════════════════════
    # LAYER 3: USER/AUTHOR STATS
    # ════════════════════════════════════════════════════════════════════
    
    def set_user_stats(self, user_id: int, stats: Dict):
        """Cache user statistics"""
        key = f"stats:user:{user_id}"
        value = json.dumps(stats, default=str)
        self.redis.setex(key, self.TTL['user_stats'], value)
    
    def get_user_stats(self, user_id: int) -> Optional[Dict]:
        """Get user stats from cache"""
        key = f"stats:user:{user_id}"
        value = self.redis.get(key)
        
        if value:
            return json.loads(value)
        return None
    
    def set_author_stats(self, author_id: int, stats: Dict):
        """Cache author statistics"""
        key = f"stats:author:{author_id}"
        value = json.dumps(stats, default=str)
        self.redis.setex(key, self.TTL['author_stats'], value)
    
    def get_author_stats(self, author_id: int) -> Optional[Dict]:
        """Get author stats from cache"""
        key = f"stats:author:{author_id}"
        value = self.redis.get(key)
        
        if value:
            return json.loads(value)
        return None
    
    def set_stats_batch(self, user_stats: Dict, author_stats: Dict):
        """Batch set statistics"""
        logger.info("\n📦 Caching statistics to Redis...")
        
        # User stats
        logger.info(f"   Caching {len(user_stats):,} user stats...")
        pipe = self.redis.pipeline()
        for user_id, stats in user_stats.items():
            key = f"stats:user:{user_id}"
            value = json.dumps(stats, default=str)
            pipe.setex(key, self.TTL['user_stats'], value)
        pipe.execute()
        
        # Author stats
        logger.info(f"   Caching {len(author_stats):,} author stats...")
        pipe = self.redis.pipeline()
        for author_id, stats in author_stats.items():
            key = f"stats:author:{author_id}"
            value = json.dumps(stats, default=str)
            pipe.setex(key, self.TTL['author_stats'], value)
        pipe.execute()
        
        logger.info("   ✅ Statistics cached")
    
    # ════════════════════════════════════════════════════════════════════
    # LAYER 4: FOLLOWING FEED
    # ════════════════════════════════════════════════════════════════════
    
    def add_to_following_feed(
        self,
        user_id: int,
        post_id: int,
        timestamp: Optional[float] = None
    ):
        """
        Add post to user's following feed
        
        Sorted set: following:{user_id}:posts
        Score: timestamp (for sorting by recency)
        """
        key = f"following:{user_id}:posts"
        score = timestamp or datetime.now().timestamp()
        
        self.redis.zadd(key, {post_id: score})
        self.redis.expire(key, self.TTL['following_feed'])
    
    def get_following_feed(
        self,
        user_id: int,
        limit: int = 400
    ) -> List[int]:
        """Get posts from following feed (sorted by recency)"""
        key = f"following:{user_id}:posts"
        
        # Get top K (newest first)
        post_ids = self.redis.zrevrange(key, 0, limit - 1)
        
        return [int(pid) for pid in post_ids]
    
    def remove_from_following_feed(self, user_id: int, post_id: int):
        """Remove post from following feed"""
        key = f"following:{user_id}:posts"
        self.redis.zrem(key, post_id)
    
    # ════════════════════════════════════════════════════════════════════
    # LAYER 5: TRENDING
    # ════════════════════════════════════════════════════════════════════
    
    def set_trending_posts(
        self,
        window: str,  # '1h', '6h', '24h'
        posts_with_scores: List[Tuple[int, float]]
    ):
        """
        Set trending posts for a time window
        
        Sorted set: trending:global:{window}
        Score: engagement score
        """
        key = f"trending:global:{window}"
        
        # Clear old data
        self.redis.delete(key)
        
        # Add new data
        mapping = {post_id: score for post_id, score in posts_with_scores}
        self.redis.zadd(key, mapping)
        
        # Set TTL
        ttl_key = f'trending_{window}'
        if ttl_key in self.TTL:
            self.redis.expire(key, self.TTL[ttl_key])
    
    def get_trending_posts(
        self,
        window: str = '6h',
        limit: int = 100
    ) -> List[int]:
        """Get trending posts"""
        key = f"trending:global:{window}"
        
        # Get top K (highest scores first)
        post_ids = self.redis.zrevrange(key, 0, limit - 1)
        
        return [int(pid) for pid in post_ids]
    
    # ════════════════════════════════════════════════════════════════════
    # LAYER 6: USER INTERACTIONS
    # ════════════════════════════════════════════════════════════════════
    
    def add_user_interaction(
        self,
        user_id: int,
        post_id: int,
        action: str  # 'liked', 'viewed', 'hidden'
    ):
        """Track user interaction"""
        key = f"user:{user_id}:{action}_posts"
        
        self.redis.sadd(key, post_id)
        self.redis.expire(key, self.TTL['user_interactions'])
    
    def get_user_interactions(
        self,
        user_id: int,
        action: str
    ) -> Set[int]:
        """Get user's interactions"""
        key = f"user:{user_id}:{action}_posts"
        
        post_ids = self.redis.smembers(key)
        
        return {int(pid) for pid in post_ids}
    
    def remove_user_interaction(
        self,
        user_id: int,
        post_id: int,
        action: str
    ):
        """Remove user interaction"""
        key = f"user:{user_id}:{action}_posts"
        self.redis.srem(key, post_id)
    
    # ════════════════════════════════════════════════════════════════════
    # LAYER 7: MODEL METADATA
    # ════════════════════════════════════════════════════════════════════
    
    def set_model_metadata(self, metadata: Dict):
        """Set model metadata (no TTL)"""
        key = "model:latest:metadata"
        value = json.dumps(metadata, default=str)
        self.redis.set(key, value)
    
    def get_model_metadata(self) -> Optional[Dict]:
        """Get model metadata"""
        key = "model:latest:metadata"
        value = self.redis.get(key)
        
        if value:
            return json.loads(value)
        return None
    
    def set_model_version(self, version: str):
        """Set current model version"""
        key = "model:latest:version"
        self.redis.set(key, version)
    
    def get_model_version(self) -> Optional[str]:
        """Get current model version"""
        key = "model:latest:version"
        value = self.redis.get(key)
        
        if value:
            return value.decode('utf-8') if isinstance(value, bytes) else value
        return None
    
    # ════════════════════════════════════════════════════════════════════
    # BATCH OPERATIONS
    # ════════════════════════════════════════════════════════════════════
    
    def update_all_from_artifacts(self, artifacts: Dict):
        """
        Update all cache layers from trained artifacts
        
        Args:
            artifacts: Output from ArtifactManager.load_artifacts()
        """
        logger.info("\n" + "="*70)
        logger.info("UPDATING REDIS CACHE FROM ARTIFACTS")
        logger.info("="*70)
        
        # Layer 1: Embeddings
        self.set_embeddings_batch(artifacts['embeddings'])
        
        # Layer 2: CF Similarities
        self.set_cf_similarities_batch(artifacts['cf_model'])
        
        # Layer 3: Stats
        self.set_stats_batch(
            artifacts['user_stats'],
            artifacts['author_stats']
        )
        
        # Layer 7: Metadata
        self.set_model_metadata(artifacts['metadata'])
        self.set_model_version(artifacts['metadata']['version'])
        
        logger.info("\n✅ REDIS CACHE UPDATED SUCCESSFULLY")
        logger.info("="*70 + "\n")
    
    # ════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ════════════════════════════════════════════════════════════════════
    
    def flush_all(self):
        """Flush all keys (USE WITH CAUTION!)"""
        logger.warning("⚠️  Flushing all Redis keys...")
        self.redis.flushdb()
        logger.info("✅ Redis flushed")
    
    def get_info(self) -> Dict:
        """Get Redis info"""
        info = self.redis.info()
        
        return {
            'connected_clients': info.get('connected_clients', 0),
            'used_memory_human': info.get('used_memory_human', '0'),
            'total_keys': self.redis.dbsize(),
            'uptime_in_seconds': info.get('uptime_in_seconds', 0)
        }
    
    def print_info(self):
        """Print Redis info"""
        info = self.get_info()
        
        print("\n" + "="*70)
        print("REDIS CACHE INFO")
        print("="*70)
        print(f"Connected Clients: {info['connected_clients']}")
        print(f"Used Memory: {info['used_memory_human']}")
        print(f"Total Keys: {info['total_keys']:,}")
        print(f"Uptime: {info['uptime_in_seconds']/3600:.1f} hours")
        print("="*70 + "\n")


# ════════════════════════════════════════════════════════════════════════
# MAIN TESTING
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Test Redis cache manager
    """
    # Initialize
    cache = RedisCacheManager()
    
    # Test embeddings
    print("\nTesting embeddings...")
    test_embedding = np.random.rand(384).astype(np.float32)
    cache.set_user_embedding(123, test_embedding)
    retrieved = cache.get_user_embedding(123)
    print(f"Embedding cached and retrieved: {np.allclose(test_embedding, retrieved)}")
    
    # Test CF similarities
    print("\nTesting CF similarities...")
    similar_users = [(456, 0.85), (789, 0.72)]
    cache.set_user_similar_users(123, similar_users)
    retrieved = cache.get_user_similar_users(123)
    print(f"Similar users: {retrieved}")
    
    # Test stats
    print("\nTesting stats...")
    stats = {'n_posts': 50, 'avg_engagement': 0.12}
    cache.set_user_stats(123, stats)
    retrieved = cache.get_user_stats(123)
    print(f"User stats: {retrieved}")
    
    # Print info
    cache.print_info()