# """
# CONTENT-BASED RECALL CHANNEL
# ============================
# Get posts similar to user's interests (embedding-based)

# Strategy:
# - Get user embedding (aggregated from interactions)
# - Search similar post embeddings (FAISS or brute-force)
# - Filter by age (posts >= 6h old, so embeddings are ready)
# - Cache results

# Target: 200 posts
# Latency: < 20ms
# """

# from typing import List, Dict, Optional
# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# import logging
# import json

# from .base_recall import BaseRecall

# logger = logging.getLogger(__name__)

# # Try FAISS import
# try:
#     import faiss
#     FAISS_AVAILABLE = True
# except ImportError:
#     FAISS_AVAILABLE = False
#     logger.warning("FAISS not available, will use brute-force similarity")


# class ContentRecall(BaseRecall):
#     """
#     Content-based recall using embeddings
#     "Posts similar to your interests"
#     """
    
#     def __init__(
#         self,
#         redis_client=None,
#         embeddings: Optional[Dict] = None,
#         faiss_index=None,
#         faiss_post_ids: Optional[List[int]] = None,
#         data: Optional[Dict[str, pd.DataFrame]] = None,
#         config: Optional[Dict] = None
#     ):
#         """
#         Initialize content recall
        
#         Args:
#             redis_client: Redis connection
#             embeddings: Dict with 'user' and 'post' embeddings
#             faiss_index: FAISS index for fast search
#             faiss_post_ids: Mapping from FAISS index to post_id
#             data: Data dictionary with 'post' DataFrame
#             config: Configuration
#         """
#         super().__init__(redis_client, config)
        
#         self.embeddings = embeddings or {'user': {}, 'post': {}}
#         self.faiss_index = faiss_index
#         self.faiss_post_ids = faiss_post_ids or []
#         self.data = data or {}
        
#         # Configuration
#         self.min_post_age_hours = self.config.get('min_post_age_hours', 6)
#         self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
#         self.use_faiss = FAISS_AVAILABLE and self.faiss_index is not None
        
#         logger.info(f"ContentRecall initialized (FAISS: {self.use_faiss})")
    
#     def recall(self, user_id: int, k: int = 200) -> List[int]:
#         """
#         Recall posts via content-based filtering
        
#         Process:
#         1. Get user embedding
#         2. Search similar post embeddings
#         3. Filter by age (>= 6h, so embeddings are ready)
#         4. Return top K
        
#         Args:
#             user_id: Target user ID
#             k: Number of posts to return
            
#         Returns:
#             List of post IDs (sorted by similarity)
#         """
#         # Try cache first
#         cached = self._get_from_cache_json(user_id)
#         if cached is not None:
#             return cached[:k]
        
#         # Get user embedding
#         user_emb = self._get_user_embedding(user_id)
        
#         if user_emb is None:
#             logger.debug(f"No embedding for user {user_id}")
#             return []
        
#         # Search similar posts
#         if self.use_faiss:
#             post_ids = self._faiss_search(user_emb, k=k*2)  # Over-fetch for filtering
#         else:
#             post_ids = self._brute_force_search(user_emb, k=k*2)
        
#         # Filter by age
#         filtered_posts = self._filter_by_age(post_ids)
        
#         # Cache result
#         self._set_to_cache_json(user_id, filtered_posts)
        
#         return filtered_posts[:k]
    
#     def _get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
#         """
#         Get user embedding
        
#         Priority:
#         1. Try Redis cache
#         2. Try embeddings dict
#         3. Return None
        
#         Args:
#             user_id: Target user ID
            
#         Returns:
#             User embedding (384-dim) or None
#         """
#         # Try Redis cache first
#         if self.redis is not None:
#             try:
#                 key = f"user:{user_id}:embedding"
#                 cached = self.redis.get(key)
                
#                 if cached:
#                     emb = np.frombuffer(cached, dtype=np.float32)
#                     return emb
#             except Exception as e:
#                 logger.debug(f"Redis cache miss for user {user_id}: {e}")
        
#         # Try embeddings dict
#         if 'user' in self.embeddings:
#             if user_id in self.embeddings['user']:
#                 return self.embeddings['user'][user_id]
        
#         return None
    
#     def _faiss_search(self, user_emb: np.ndarray, k: int) -> List[int]:
#         """
#         Fast similarity search using FAISS
        
#         Args:
#             user_emb: User embedding
#             k: Number of results
            
#         Returns:
#             List of post IDs
#         """
#         # Normalize embedding
#         user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
        
#         # Reshape for FAISS
#         query = user_emb.reshape(1, -1).astype(np.float32)
        
#         # Search
#         try:
#             distances, indices = self.faiss_index.search(query, k)
            
#             # Convert indices to post IDs
#             post_ids = []
#             for idx in indices[0]:
#                 if 0 <= idx < len(self.faiss_post_ids):
#                     post_ids.append(int(self.faiss_post_ids[idx]))
            
#             return post_ids
            
#         except Exception as e:
#             logger.error(f"FAISS search error: {e}")
#             return []
    
#     def _brute_force_search(self, user_emb: np.ndarray, k: int) -> List[int]:
#         """
#         Brute-force cosine similarity search
#         Fallback when FAISS not available
        
#         Args:
#             user_emb: User embedding
#             k: Number of results
            
#         Returns:
#             List of post IDs
#         """
#         if 'post' not in self.embeddings:
#             return []
        
#         post_embs = self.embeddings['post']
        
#         if not post_embs:
#             return []
        
#         # Normalize user embedding
#         user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
        
#         # Compute similarities
#         similarities = {}
        
#         for post_id, post_emb in post_embs.items():
#             # Normalize post embedding
#             post_emb = post_emb / (np.linalg.norm(post_emb) + 1e-8)
            
#             # Cosine similarity
#             sim = np.dot(user_emb, post_emb)
#             similarities[post_id] = float(sim)
        
#         # Sort by similarity
#         ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
#         # Extract post IDs
#         post_ids = [post_id for post_id, _ in ranked[:k]]
        
#         return post_ids
    
#     def _filter_by_age(self, post_ids: List[int]) -> List[int]:
#         """
#         Filter posts by age (>= 6h old)
        
#         Reason: Newer posts may not have embeddings yet
        
#         Args:
#             post_ids: List of post IDs
            
#         Returns:
#             Filtered list of post IDs
#         """
#         if 'post' not in self.data:
#             return post_ids  # Can't filter without data
        
#         posts_df = self.data['post']
        
#         # Calculate cutoff time
#         cutoff_time = datetime.now() - timedelta(hours=self.min_post_age_hours)
        
#         # Filter
#         filtered = []
        
#         for post_id in post_ids:
#             # Find post
#             post_row = posts_df[posts_df['Id'] == post_id]
            
#             if post_row.empty:
#                 continue
            
#             # Check age
#             created_at = pd.to_datetime(post_row.iloc[0]['CreateDate'])
            
#             if created_at <= cutoff_time:
#                 filtered.append(post_id)
        
#         return filtered
    
#     def _get_from_cache_json(self, user_id: int) -> Optional[List[int]]:
#         """Get content posts from Redis cache"""
#         if self.redis is None:
#             return None
        
#         key = f"content:user:{user_id}:posts"
        
#         try:
#             value = self.redis.get(key)
#             if value:
#                 return json.loads(value)
#         except Exception as e:
#             logger.warning(f"Cache get error: {e}")
        
#         return None
    
#     def _set_to_cache_json(self, user_id: int, post_ids: List[int]):
#         """Cache content posts to Redis"""
#         if self.redis is None:
#             return
        
#         key = f"content:user:{user_id}:posts"
        
#         try:
#             value = json.dumps(post_ids)
#             self.redis.setex(key, self.cache_ttl, value)
#         except Exception as e:
#             logger.warning(f"Cache set error: {e}")


# # ============================================================================
# # EXAMPLE USAGE
# # ============================================================================

# if __name__ == "__main__":
#     """Test content recall"""
#     import pickle
#     from recommender.common.data_loader import load_data
    
#     # Setup logging
#     logging.basicConfig(level=logging.INFO)
    
#     # Load data
#     print("Loading data...")
#     data = load_data('dataset')
    
#     # Load embeddings
#     print("Loading embeddings...")
#     with open('models/latest/embeddings.pkl', 'rb') as f:
#         embeddings = pickle.load(f)
    
#     # Load FAISS index
#     print("Loading FAISS index...")
#     if FAISS_AVAILABLE:
#         faiss_index = faiss.read_index('models/latest/faiss_index.bin')
#         with open('models/latest/faiss_post_ids.pkl', 'rb') as f:
#             faiss_post_ids = pickle.load(f)
#     else:
#         faiss_index = None
#         faiss_post_ids = None
    
#     # Initialize recall
#     print("Initializing content recall...")
#     content_recall = ContentRecall(
#         redis_client=None,
#         embeddings=embeddings,
#         faiss_index=faiss_index,
#         faiss_post_ids=faiss_post_ids,
#         data=data
#     )
    
#     # Test recall
#     print("\nTesting recall for user 1...")
#     candidates = content_recall.recall(user_id=1, k=200)
    
#     print(f"\nRecalled {len(candidates)} candidates")
#     print(f"Sample: {candidates[:10]}")
    
#     # Print metrics
#     metrics = content_recall.get_metrics()
#     print(f"\nMetrics: {metrics}")


"""
CONTENT-BASED RECALL CHANNEL
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
        embeddings: Optional[Dict] = None,
        faiss_index=None,
        faiss_post_ids: Optional[List[int]] = None,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize content recall
        
        Args:
            redis_client: Redis connection
            embeddings: Dict with 'user' and 'post' embeddings
            faiss_index: FAISS index for fast search
            faiss_post_ids: Mapping from FAISS index to post_id
            data: Data dictionary with 'post' DataFrame
            config: Configuration
        """
        super().__init__(redis_client, config)
        
        self.embeddings = embeddings or {'user': {}, 'post': {}}
        self.faiss_index = faiss_index
        
        # Fix: Handle numpy array properly
        if faiss_post_ids is None:
            self.faiss_post_ids = []
        elif isinstance(faiss_post_ids, np.ndarray):
            self.faiss_post_ids = faiss_post_ids
        else:
            self.faiss_post_ids = list(faiss_post_ids)
        
        self.data = data or {}
        
        # Configuration
        self.min_post_age_hours = self.config.get('min_post_age_hours', 6)
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        self.use_faiss = FAISS_AVAILABLE and self.faiss_index is not None
        
        logger.info(f"ContentRecall initialized (FAISS: {self.use_faiss})")
    
    def recall(self, user_id: int, k: int = 200) -> List[int]:
        """
        Recall posts via content-based filtering
        
        Process:
        1. Get user embedding
        2. Search similar post embeddings
        3. Filter by age (>= 6h, so embeddings are ready)
        4. Return top K
        
        Args:
            user_id: Target user ID
            k: Number of posts to return
            
        Returns:
            List of post IDs (sorted by similarity)
        """
        # Try cache first
        cached = self._get_from_cache_json(user_id)
        if cached is not None:
            return cached[:k]
        
        # Get user embedding
        user_emb = self._get_user_embedding(user_id)
        
        if user_emb is None:
            logger.debug(f"No embedding for user {user_id}")
            return []
        
        # Search similar posts
        if self.use_faiss:
            post_ids = self._faiss_search(user_emb, k=k*2)  # Over-fetch for filtering
        else:
            post_ids = self._brute_force_search(user_emb, k=k*2)
        
        # Filter by age
        filtered_posts = self._filter_by_age(post_ids)
        
        # Cache result
        self._set_to_cache_json(user_id, filtered_posts)
        
        return filtered_posts[:k]
    
    def _get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """
        Get user embedding
        
        Priority:
        1. Try Redis cache
        2. Try embeddings dict
        3. Return None
        
        Args:
            user_id: Target user ID
            
        Returns:
            User embedding (384-dim) or None
        """
        # Try Redis cache first
        if self.redis is not None:
            try:
                key = f"user:{user_id}:embedding"
                cached = self.redis.get(key)
                
                if cached:
                    emb = np.frombuffer(cached, dtype=np.float32)
                    return emb
            except Exception as e:
                logger.debug(f"Redis cache miss for user {user_id}: {e}")
        
        # Try embeddings dict
        if 'user' in self.embeddings:
            if user_id in self.embeddings['user']:
                return self.embeddings['user'][user_id]
        
        return None
    
    def _faiss_search(self, user_emb: np.ndarray, k: int) -> List[int]:
        """
        Fast similarity search using FAISS
        
        Args:
            user_emb: User embedding
            k: Number of results
            
        Returns:
            List of post IDs
        """
        # Normalize embedding
        user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
        
        # Reshape for FAISS
        query = user_emb.reshape(1, -1).astype(np.float32)
        
        # Search
        try:
            distances, indices = self.faiss_index.search(query, k)
            
            # Convert indices to post IDs
            post_ids = []
            for idx in indices[0]:
                if 0 <= idx < len(self.faiss_post_ids):
                    post_ids.append(int(self.faiss_post_ids[idx]))
            
            return post_ids
            
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return []
    
    def _brute_force_search(self, user_emb: np.ndarray, k: int) -> List[int]:
        """
        Brute-force cosine similarity search
        Fallback when FAISS not available
        
        Args:
            user_emb: User embedding
            k: Number of results
            
        Returns:
            List of post IDs
        """
        if 'post' not in self.embeddings:
            return []
        
        post_embs = self.embeddings['post']
        
        if not post_embs:
            return []
        
        # Normalize user embedding
        user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
        
        # Compute similarities
        similarities = {}
        
        for post_id, post_emb in post_embs.items():
            # Normalize post embedding
            post_emb = post_emb / (np.linalg.norm(post_emb) + 1e-8)
            
            # Cosine similarity
            sim = np.dot(user_emb, post_emb)
            similarities[post_id] = float(sim)
        
        # Sort by similarity
        ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Extract post IDs
        post_ids = [post_id for post_id, _ in ranked[:k]]
        
        return post_ids
    
    def _filter_by_age(self, post_ids: List[int]) -> List[int]:
        """
        Filter posts by age (>= 6h old)
        
        Reason: Newer posts may not have embeddings yet
        
        Args:
            post_ids: List of post IDs
            
        Returns:
            Filtered list of post IDs
        """
        if 'post' not in self.data:
            return post_ids  # Can't filter without data
        
        posts_df = self.data['post']
        
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(hours=self.min_post_age_hours)
        
        # Filter
        filtered = []
        
        for post_id in post_ids:
            # Find post
            post_row = posts_df[posts_df['Id'] == post_id]
            
            if post_row.empty:
                continue
            
            # Check age
            created_at = pd.to_datetime(post_row.iloc[0]['CreateDate'])
            
            if created_at <= cutoff_time:
                filtered.append(post_id)
        
        return filtered
    
    def _get_from_cache_json(self, user_id: int) -> Optional[List[int]]:
        """Get content posts from Redis cache"""
        if self.redis is None:
            return None
        
        key = f"content:user:{user_id}:posts"
        
        try:
            value = self.redis.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        return None
    
    def _set_to_cache_json(self, user_id: int, post_ids: List[int]):
        """Cache content posts to Redis"""
        if self.redis is None:
            return
        
        key = f"content:user:{user_id}:posts"
        
        try:
            value = json.dumps(post_ids)
            self.redis.setex(key, self.cache_ttl, value)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Test content recall"""
    import pickle
    from recommender.common.data_loader import load_data
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    print("Loading data...")
    data = load_data('dataset')
    
    # Load embeddings
    print("Loading embeddings...")
    with open('models/latest/embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    
    # Load FAISS index
    print("Loading FAISS index...")
    if FAISS_AVAILABLE:
        faiss_index = faiss.read_index('models/latest/faiss_index.bin')
        with open('models/latest/faiss_post_ids.pkl', 'rb') as f:
            faiss_post_ids = pickle.load(f)
    else:
        faiss_index = None
        faiss_post_ids = None
    
    # Initialize recall
    print("Initializing content recall...")
    content_recall = ContentRecall(
        redis_client=None,
        embeddings=embeddings,
        faiss_index=faiss_index,
        faiss_post_ids=faiss_post_ids,
        data=data
    )
    
    # Test recall
    print("\nTesting recall for user 1...")
    candidates = content_recall.recall(user_id=1, k=200)
    
    print(f"\nRecalled {len(candidates)} candidates")
    print(f"Sample: {candidates[:10]}")
    
    # Print metrics
    metrics = content_recall.get_metrics()
    print(f"\nMetrics: {metrics}")