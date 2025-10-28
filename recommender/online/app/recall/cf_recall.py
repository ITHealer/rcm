"""
Collaborative Filtering Recall Channel
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

from typing import List, Dict, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import numpy as np
import time

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
        
        reactions = self.data['postreaction'].copy()
        
        # Filter: positive reactions only (like=1, comment=2, share=3, save=5)
        positive_reactions = reactions[
            reactions['ReactionTypeId'].isin([1, 2, 3, 5])
        ].copy()
        
        # Add timestamp
        if 'CreateDate' in positive_reactions.columns:
            positive_reactions['created_at'] = pd.to_datetime(
                positive_reactions['CreateDate'], 
                errors='coerce'
            )
            
            # Filter recent only
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=self.recent_days)
            positive_reactions = positive_reactions[positive_reactions['created_at'] >= cutoff]
        
        # Group by user
        for user_id, group in positive_reactions.groupby('UserId'):
            # Sort by recency
            if 'created_at' in group.columns:
                group = group.sort_values('created_at', ascending=False)
            
            # Store post IDs with timestamps
            self.user_post_map[int(user_id)] = [
                {
                    'post_id': int(row['PostId']),
                    'created_at': row.get('created_at', pd.Timestamp.now()),
                    'reaction_type': int(row['ReactionTypeId'])
                }
                for _, row in group.iterrows()
            ]
        
        logger.info(f"Built user-post map: {len(self.user_post_map)} users, "
                   f"{sum(len(p) for p in self.user_post_map.values())} interactions")
    
    def recall(self, user_id: int, k: int = 300) -> List[int]:
        """
        Recall posts via collaborative filtering
        
        Process:
        1. Get similar users (from CF model)
        2. Get posts they liked (recent 7 days)
        3. Score by: similarity × recency × reaction_weight
        4. Return top k
        
        Args:
            user_id: User ID
            k: Number of candidates to return
            
        Returns:
            List of post IDs (sorted by CF score)
        """
        start_time = time.time()
        
        # Try cache first
        cache_key = f"cf_recall:{user_id}:{k}"
        cached = self._cache_get(cache_key)
        
        if cached:
            self.metrics['cache_hits'] += 1
            self.metrics['total_recalls'] += 1
            candidates = json.loads(cached)
            self.metrics['total_candidates'] += len(candidates)
            self.metrics['total_time_ms'] += (time.time() - start_time) * 1000
            return candidates
        
        self.metrics['cache_misses'] += 1
        
        # Get similar users
        similar_users = self._get_similar_users(user_id)
        
        if not similar_users:
            logger.debug(f"No similar users found for user {user_id}")
            self.metrics['total_recalls'] += 1
            self.metrics['total_time_ms'] += (time.time() - start_time) * 1000
            return []
        
        # Collect posts with scores
        post_scores = {}
        now = pd.Timestamp.now()
        
        # Reaction weights
        reaction_weights = {
            1: 1.0,   # like
            2: 1.5,   # comment (more engagement)
            3: 2.0,   # share (strong signal)
            5: 1.2    # save
        }
        
        for sim_user_id, similarity in similar_users[:self.top_k_similar_users]:
            user_posts = self.user_post_map.get(sim_user_id, [])
            
            for item in user_posts:
                post_id = item['post_id']
                created_at = item['created_at']
                reaction_type = item['reaction_type']
                
                # Recency decay (exponential)
                age_days = (now - created_at).total_seconds() / 86400
                recency_score = np.exp(-age_days / 3.0)  # decay over 3 days
                
                # Reaction weight
                reaction_weight = reaction_weights.get(reaction_type, 1.0)
                
                # Combined score
                score = similarity * recency_score * reaction_weight
                
                if post_id in post_scores:
                    post_scores[post_id] += score
                else:
                    post_scores[post_id] = score
        
        # Sort by score
        sorted_posts = sorted(post_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [post_id for post_id, score in sorted_posts[:k]]
        
        # Update metrics
        self.metrics['total_recalls'] += 1
        self.metrics['total_candidates'] += len(candidates)
        self.metrics['total_time_ms'] += (time.time() - start_time) * 1000
        
        # Cache result
        self._cache_set(cache_key, json.dumps(candidates), self.cache_ttl)
        
        logger.debug(f"CF recall for user {user_id}: {len(candidates)} posts from {len(similar_users)} similar users")
        
        return candidates
    
    def _get_similar_users(self, user_id: int) -> List[Tuple[int, float]]:
        """
        Get similar users with similarity scores
        
        Priority:
        1. Redis cache
        2. CF model
        
        Args:
            user_id: User ID
            
        Returns:
            List of (similar_user_id, similarity_score) tuples
        """
        # Try Redis cache first
        if self.redis:
            try:
                cache_key = f"cf:similar:{user_id}"
                cached = self.redis.zrevrange(cache_key, 0, -1, withscores=True)
                
                if cached:
                    return [(int(uid), float(score)) for uid, score in cached]
            except Exception as e:
                logger.debug(f"Redis get similar users error: {e}")
        
        # Try CF model
        if 'user_similarities' in self.cf_model:
            similarities = self.cf_model['user_similarities'].get(user_id, [])
            return similarities
        
        return []
    
    def refresh_index(self):
        """Refresh the user-post map (call periodically)"""
        logger.info("Refreshing CF recall index...")
        self._build_user_post_map()
        logger.info("CF recall index refreshed")