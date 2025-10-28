"""
Content-Based Recall Channel
============================
Get posts similar to user's interests (embedding-based)

Strategy:
- Get user embedding (aggregated from interactions)
- Search similar post embeddings (FAISS or brute-force)
- Filter by age (posts >= 6h old, so embeddings are ready)
- Cache results

Target: 200 posts
Latency: < 20ms
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import time

from .base_recall import BaseRecall

logger = logging.getLogger(__name__)

# Try FAISS import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, will use brute-force similarity")


class ContentRecall(BaseRecall):
    """
    Content-based recall using embeddings
    "Posts similar to your interests"
    """
    
    def __init__(
        self,
        redis_client=None,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        embeddings: Optional[Dict] = None,
        faiss_index=None,
        faiss_post_ids: Optional[List[int]] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize content recall
        
        Args:
            redis_client: Redis connection
            data: Data dictionary with 'posts' DataFrame
            embeddings: Dict with 'user' and 'post' embeddings
            faiss_index: Pre-built FAISS index
            faiss_post_ids: Post IDs corresponding to FAISS index
            config: Configuration
        """
        super().__init__(redis_client, config)
        
        self.data = data or {}
        self.embeddings = embeddings or {}
        self.faiss_index = faiss_index
        self.faiss_post_ids = faiss_post_ids or []
        
        # Configuration
        self.min_post_age_hours = self.config.get('min_post_age_hours', 6)
        self.max_post_age_days = self.config.get('max_post_age_days', 7)
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        self.use_faiss = self.config.get('use_faiss', True) and FAISS_AVAILABLE
        
        # Build post metadata for fast filtering
        self._build_post_metadata()
        
        logger.info(f"Content recall initialized: FAISS={self.use_faiss and self.faiss_index is not None}, "
                   f"Posts={len(self.post_metadata)}")
    
    def _build_post_metadata(self):
        """
        Build metadata for filtering posts by age
        """
        self.post_metadata = {}
        
        if 'posts' not in self.data:
            logger.warning("No posts data available for content recall")
            return
        
        posts = self.data['posts']
        
        if 'CreateDate' in posts.columns:
            for _, row in posts.iterrows():
                post_id = int(row['Id'])
                created_at = pd.to_datetime(row['CreateDate'], errors='coerce')
                
                self.post_metadata[post_id] = {
                    'created_at': created_at,
                    'author_id': int(row['UserId'])
                }
        
        logger.info(f"Built post metadata: {len(self.post_metadata)} posts")
    
    def recall(self, user_id: int, k: int = 200) -> List[int]:
        """
        Recall posts via content similarity
        
        Process:
        1. Get user embedding
        2. Search similar posts (FAISS or brute-force)
        3. Filter by age (6h - 7d)
        4. Return top k
        
        Args:
            user_id: User ID
            k: Number of candidates to return
            
        Returns:
            List of post IDs (sorted by similarity)
        """
        start_time = time.time()
        
        # Try cache first
        cache_key = f"content_recall:{user_id}:{k}"
        cached = self._cache_get(cache_key)
        
        if cached:
            self.metrics['cache_hits'] += 1
            self.metrics['total_recalls'] += 1
            candidates = json.loads(cached)
            self.metrics['total_candidates'] += len(candidates)
            self.metrics['total_time_ms'] += (time.time() - start_time) * 1000
            return candidates
        
        self.metrics['cache_misses'] += 1
        
        # Get user embedding
        user_embedding = self._get_user_embedding(user_id)
        
        if user_embedding is None:
            logger.debug(f"No embedding found for user {user_id}")
            self.metrics['total_recalls'] += 1
            self.metrics['total_time_ms'] += (time.time() - start_time) * 1000
            return []
        
        # Search similar posts
        if self.use_faiss and self.faiss_index is not None:
            candidates = self._search_faiss(user_embedding, k * 2)  # Get more for filtering
        else:
            candidates = self._search_bruteforce(user_embedding, k * 2)
        
        # Filter by age
        candidates = self._filter_by_age(candidates)
        
        # Limit to k
        candidates = candidates[:k]
        
        # Update metrics
        self.metrics['total_recalls'] += 1
        self.metrics['total_candidates'] += len(candidates)
        self.metrics['total_time_ms'] += (time.time() - start_time) * 1000
        
        # Cache result
        self._cache_set(cache_key, json.dumps(candidates), self.cache_ttl)
        
        logger.debug(f"Content recall for user {user_id}: {len(candidates)} similar posts")
        
        return candidates
    
    def _get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """
        Get user embedding
        
        Priority:
        1. Redis cache
        2. Embeddings dict
        
        Args:
            user_id: User ID
            
        Returns:
            User embedding vector or None
        """
        # Try Redis cache first
        if self.redis:
            try:
                cache_key = f"user:embedding:{user_id}"
                cached = self.redis.get(cache_key)
                
                if cached:
                    return np.frombuffer(cached, dtype=np.float32)
            except Exception as e:
                logger.debug(f"Redis get embedding error: {e}")
        
        # Try embeddings dict
        if 'user' in self.embeddings:
            user_embeddings = self.embeddings['user']
            
            if user_id in user_embeddings:
                return user_embeddings[user_id]
        
        return None
    
    def _search_faiss(self, query_embedding: np.ndarray, k: int) -> List[int]:
        """
        Search similar posts using FAISS
        
        Args:
            query_embedding: User embedding vector
            k: Number of results
            
        Returns:
            List of post IDs
        """
        try:
            # Reshape for FAISS
            query = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Search
            distances, indices = self.faiss_index.search(query, k)
            
            # Map indices to post IDs
            candidates = [self.faiss_post_ids[idx] for idx in indices[0] if idx < len(self.faiss_post_ids)]
            
            return candidates
            
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return []
    
    def _search_bruteforce(self, query_embedding: np.ndarray, k: int) -> List[int]:
        """
        Search similar posts using brute-force cosine similarity
        
        Args:
            query_embedding: User embedding vector
            k: Number of results
            
        Returns:
            List of post IDs
        """
        if 'post' not in self.embeddings:
            return []
        
        post_embeddings = self.embeddings['post']
        
        if not post_embeddings:
            return []
        
        # Compute cosine similarities
        similarities = {}
        
        for post_id, post_emb in post_embeddings.items():
            # Cosine similarity
            sim = np.dot(query_embedding, post_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(post_emb) + 1e-8
            )
            similarities[post_id] = float(sim)
        
        # Sort by similarity
        sorted_posts = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        candidates = [post_id for post_id, sim in sorted_posts[:k]]
        
        return candidates
    
    def _filter_by_age(self, post_ids: List[int]) -> List[int]:
        """
        Filter posts by age (6h - 7d old)
        
        Args:
            post_ids: List of post IDs
            
        Returns:
            Filtered list of post IDs
        """
        if not self.post_metadata:
            return post_ids
        
        now = pd.Timestamp.now()
        min_age = pd.Timedelta(hours=self.min_post_age_hours)
        max_age = pd.Timedelta(days=self.max_post_age_days)
        
        filtered = []
        
        for post_id in post_ids:
            metadata = self.post_metadata.get(post_id)
            
            if not metadata:
                continue
            
            created_at = metadata['created_at']
            
            if pd.isna(created_at):
                continue
            
            age = now - created_at
            
            # Filter: must be between min_age and max_age
            if min_age <= age <= max_age:
                filtered.append(post_id)
        
        return filtered
    
    def refresh_index(self):
        """Refresh post metadata (call periodically)"""
        logger.info("Refreshing content recall index...")
        self._build_post_metadata()
        logger.info("Content recall index refreshed")