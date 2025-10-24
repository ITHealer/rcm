"""
COLLABORATIVE FILTERING RECALL CHANNEL
======================================
Get posts liked by similar users

Strategy:
- Find similar users (from CF model)
- Get their recent liked posts (last 7 days)
- Score by similarity × recency
- Cache results

Target: 300 posts
Latency: < 20ms
"""

from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import numpy as np

from .base_recall import BaseRecall

logger = logging.getLogger(__name__)


class CFRecall(BaseRecall):
    """
    Collaborative Filtering based recall
    "Users similar to you also liked these posts"
    """
    
    def __init__(
        self,
        redis_client=None,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        cf_model: Optional[Dict] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize CF recall
        
        Args:
            redis_client: Redis connection
            data: Data dictionary with 'postreaction' DataFrame
            cf_model: CF similarities {'user_similarities': {user_id: [(sim_user, score)]}}
            config: Configuration
        """
        super().__init__(redis_client, config)
        
        self.data = data or {}
        self.cf_model = cf_model or {}
        
        # Configuration
        self.recent_days = self.config.get('recent_days', 7)
        self.top_k_similar_users = self.config.get('top_k_similar_users', 50)
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        
        # Precompute user-post interactions for fast lookup
        self._build_user_post_map()
    
    def _build_user_post_map(self):
        """
        Build mapping: user_id -> list of recent liked posts
        For fast lookup during recall
        """
        self.user_post_map = {}
        
        if 'postreaction' not in self.data:
            logger.warning("No postreaction data available for CF recall")
            return
        
        reactions = self.data['postreaction']
        
        # Filter: positive reactions only (like, comment, share, save)
        positive_reactions = reactions[
            reactions['ReactionTypeId'].isin([1, 2, 3, 5])
        ].copy()
        
        # Add timestamp
        if 'CreateDate' in positive_reactions.columns:
            positive_reactions['created_at'] = pd.to_datetime(positive_reactions['CreateDate'])
        
        # Group by user
        for user_id, group in positive_reactions.groupby('UserId'):
            # Sort by recency
            group = group.sort_values('created_at', ascending=False)
            
            # Store post IDs with timestamps
            self.user_post_map[int(user_id)] = [
                {
                    'post_id': int(row['PostId']),
                    'created_at': row['created_at']
                }
                for _, row in group.iterrows()
            ]
        
        logger.info(f"Built user-post map: {len(self.user_post_map)} users")
    
    def recall(self, user_id: int, k: int = 300) -> List[int]:
        """
        Recall posts via collaborative filtering
        
        Process:
        1. Get similar users (from CF model)
        2. Get posts they liked (recent 7 days)
        3. Score by: similarity × recency
        4. Return top K
        
        Args:
            user_id: Target user ID
            k: Number of posts to return
            
        Returns:
            List of post IDs (sorted by score)
        """
        # Try cache first
        cached = self._get_from_cache_json(user_id)
        if cached is not None:
            return cached[:k]
        
        # Get similar users from CF model
        similar_users = self._get_similar_users(user_id)
        
        if not similar_users:
            return []
        
        # Collect posts from similar users
        candidate_posts = {}
        cutoff_time = datetime.now() - timedelta(days=self.recent_days)
        
        for similar_user_id, similarity_score in similar_users:
            # Get posts liked by this similar user
            user_posts = self.user_post_map.get(similar_user_id, [])
            
            for post_info in user_posts:
                post_id = post_info['post_id']
                created_at = post_info['created_at']
                
                # Filter by recency
                if created_at < cutoff_time:
                    continue
                
                # Calculate score: similarity × recency
                days_ago = (datetime.now() - created_at).total_seconds() / 86400
                recency_weight = np.exp(-days_ago / self.recent_days)  # Exponential decay
                
                score = similarity_score * recency_weight
                
                # Aggregate scores (if multiple similar users liked same post)
                if post_id in candidate_posts:
                    candidate_posts[post_id] += score
                else:
                    candidate_posts[post_id] = score
        
        # Sort by score
        ranked_posts = sorted(
            candidate_posts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Extract post IDs
        post_ids = [post_id for post_id, _ in ranked_posts]
        
        # Cache result
        self._set_to_cache_json(user_id, post_ids)
        
        return post_ids[:k]
    
    def _get_similar_users(self, user_id: int) -> List[tuple]:
        """
        Get similar users from CF model
        
        Priority:
        1. Try Redis cache
        2. Try CF model
        3. Return empty
        
        Args:
            user_id: Target user ID
            
        Returns:
            List of (similar_user_id, similarity_score) tuples
        """
        # Try Redis cache first
        if self.redis is not None:
            try:
                key = f"cf:user:{user_id}:similar"
                cached = self.redis.get(key)
                
                if cached:
                    similar_users = json.loads(cached)
                    # Convert to list of tuples
                    return [(int(u['user_id']), float(u['similarity'])) 
                            for u in similar_users[:self.top_k_similar_users]]
            except Exception as e:
                logger.debug(f"Redis cache miss for CF user {user_id}: {e}")
        
        # Try CF model
        if 'user_similarities' in self.cf_model:
            user_sims = self.cf_model['user_similarities']
            
            if user_id in user_sims:
                similar_users = user_sims[user_id]
                
                # Convert to list of tuples if needed
                if isinstance(similar_users, list):
                    if len(similar_users) > 0:
                        if isinstance(similar_users[0], dict):
                            # Format: [{'user_id': 123, 'similarity': 0.85}]
                            return [(int(u['user_id']), float(u['similarity'])) 
                                    for u in similar_users[:self.top_k_similar_users]]
                        else:
                            # Format: [(123, 0.85)]
                            return similar_users[:self.top_k_similar_users]
        
        # No similar users found
        return []
    
    def _get_from_cache_json(self, user_id: int) -> Optional[List[int]]:
        """Get CF posts from Redis cache"""
        if self.redis is None:
            return None
        
        key = f"cf:user:{user_id}:posts"
        
        try:
            value = self.redis.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        return None
    
    def _set_to_cache_json(self, user_id: int, post_ids: List[int]):
        """Cache CF posts to Redis"""
        if self.redis is None:
            return
        
        key = f"cf:user:{user_id}:posts"
        
        try:
            value = json.dumps(post_ids)
            self.redis.setex(key, self.cache_ttl, value)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Test CF recall"""
    import pickle
    from recommender.common.data_loader import load_data
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    print("Loading data...")
    data = load_data('dataset')
    
    # Load CF model
    print("Loading CF model...")
    with open('models/latest/cf_model.pkl', 'rb') as f:
        cf_model = pickle.load(f)
    
    # Initialize recall
    print("Initializing CF recall...")
    cf_recall = CFRecall(
        redis_client=None,
        data=data,
        cf_model=cf_model
    )
    
    # Test recall
    print("\nTesting recall for user 1...")
    candidates = cf_recall.recall(user_id=1, k=300)
    
    print(f"\nRecalled {len(candidates)} candidates")
    print(f"Sample: {candidates[:10]}")
    
    # Print metrics
    metrics = cf_recall.get_metrics()
    print(f"\nMetrics: {metrics}")