# """
# Recommendation Service
# =====================
# Complete feed generation pipeline

# Flow:
# 1. Multi-channel Recall (~1000 candidates)
# 2. Feature Extraction
# 3. ML Ranking (score candidates)
# 4. Re-ranking (apply business rules)
# 5. Final feed (50 posts)

# Target latency: < 200ms end-to-end
# """

# from typing import List, Dict, Optional, Set
# import pandas as pd
# import logging
# import time
# from datetime import datetime

# from recommender.online.app.services.cache_service import cache_service
# from recommender.online.app.recall import FollowingRecall, CFRecall, ContentRecall, TrendingRecall
# from recommender.online.app.ranking import MLRanker, Reranker

# logger = logging.getLogger(__name__)


# class RecommendationService:
#     """
#     Main recommendation service orchestrating the entire pipeline
#     """
    
#     def __init__(
#         self,
#         data: Dict,
#         models: Dict,
#         embeddings: Dict,
#         faiss_index,
#         faiss_post_ids: List[int],
#         feature_engineer,
#         config: Dict
#     ):
#         """
#         Initialize recommendation service
        
#         Args:
#             data: Data dictionaries (posts, reactions, friendships, etc.)
#             models: Trained models (ranking_model, cf_model)
#             embeddings: User and post embeddings
#             faiss_index: FAISS index for content-based recall
#             faiss_post_ids: Post IDs for FAISS index
#             feature_engineer: FeatureEngineer instance
#             config: Configuration
#         """
#         self.data = data
#         self.models = models
#         self.embeddings = embeddings
#         self.faiss_index = faiss_index
#         self.faiss_post_ids = faiss_post_ids
#         self.feature_engineer = feature_engineer
#         self.config = config
        
#         # Extract configs
#         self.recall_config = config.get('recall', {})
#         self.ranking_config = config.get('ranking', {})
#         self.reranking_config = config.get('reranking', {})
        
#         # Initialize recall channels
#         self._init_recall_channels()
        
#         # Initialize ranking
#         self._init_ranking()
        
#         # Initialize reranking
#         self.reranker = Reranker(config=self.reranking_config)
        
#         # Metrics
#         self.metrics = {
#             'total_requests': 0,
#             'total_time_ms': 0,
#             'recall_time_ms': 0,
#             'ranking_time_ms': 0,
#             'reranking_time_ms': 0
#         }
        
#         logger.info("✅ RecommendationService initialized successfully")
    
#     def _init_recall_channels(self):
#         """Initialize all recall channels"""
#         channels_config = self.recall_config.get('channels', {})
        
#         # Build following dict
#         following_dict = {}
#         print(f"Building following dict...{self.data}")
#         if 'friendships' in self.data:
#             friendships = self.data['friendships']
#             for user_id, group in friendships.groupby('UserId'):
#                 following_dict[int(user_id)] = group['FriendId'].astype(int).tolist()
        
#         # Initialize channels
#         self.following_recall = FollowingRecall(
#             redis_client=cache_service.redis if cache_service.is_connected() else None,
#             data=self.data,
#             following_dict=following_dict,
#             config=channels_config.get('following', {})
#         )
        
#         self.cf_recall = CFRecall(
#             redis_client=cache_service.redis if cache_service.is_connected() else None,
#             data=self.data,
#             cf_model=self.models.get('cf_model'),
#             config=channels_config.get('collaborative_filtering', {})
#         )
        
#         self.content_recall = ContentRecall(
#             redis_client=cache_service.redis if cache_service.is_connected() else None,
#             data=self.data,
#             embeddings=self.embeddings,
#             faiss_index=self.faiss_index,
#             faiss_post_ids=self.faiss_post_ids,
#             config=channels_config.get('content_based', {})
#         )
        
#         self.trending_recall = TrendingRecall(
#             redis_client=cache_service.redis if cache_service.is_connected() else None,
#             data=self.data,
#             config=channels_config.get('trending', {})
#         )
        
#         logger.info("✅ Recall channels initialized")
    
#     def _init_ranking(self):
#         """Initialize ranking model"""
#         self.ranker = MLRanker(
#             model=self.models.get('ranking_model'),
#             scaler=self.models.get('scaler'),
#             feature_cols=self.models.get('feature_cols'),
#             feature_engineer=self.feature_engineer,
#             config=self.ranking_config
#         )
        
#         logger.info("✅ Ranker initialized")
    
#     def generate_feed(
#         self,
#         user_id: int,
#         limit: int = 50,
#         exclude_seen: Optional[List[int]] = None
#     ) -> Dict:
#         """
#         Generate personalized feed for user
        
#         Args:
#             user_id: User ID
#             limit: Number of posts to return
#             exclude_seen: List of post IDs to exclude (already seen)
            
#         Returns:
#             Dict with feed and metadata
#         """
#         start_time = time.time()
        
#         try:
#             # Step 1: Multi-channel Recall
#             t1 = time.time()
#             candidates = self._recall_candidates(user_id)
#             recall_time = (time.time() - t1) * 1000
            
#             logger.info(f"Recall: {len(candidates)} candidates for user {user_id}")
            
#             # Step 2: Filter seen posts
#             if exclude_seen:
#                 candidates = [pid for pid in candidates if pid not in exclude_seen]
            
#             # Also filter posts user has seen (from cache)
#             seen_posts = cache_service.get_seen_posts(user_id)
#             if seen_posts:
#                 candidates = [pid for pid in candidates if pid not in seen_posts]
            
#             logger.info(f"After filtering seen: {len(candidates)} candidates")
            
#             if not candidates:
#                 logger.warning(f"No candidates for user {user_id} after filtering")
#                 return {
#                     'user_id': user_id,
#                     'feed': [],
#                     'metadata': {
#                         'total_time_ms': (time.time() - start_time) * 1000,
#                         'recall_count': 0,
#                         'reason': 'no_candidates_after_filtering'
#                     }
#                 }
            
#             # Step 3: ML Ranking
#             t2 = time.time()
#             ranked_df = self.ranker.rank(user_id, candidates)
#             ranking_time = (time.time() - t2) * 1000
            
#             if ranked_df.empty:
#                 logger.warning(f"Ranking returned empty for user {user_id}")
#                 return {
#                     'user_id': user_id,
#                     'feed': [],
#                     'metadata': {
#                         'total_time_ms': (time.time() - start_time) * 1000,
#                         'recall_count': len(candidates),
#                         'reason': 'ranking_failed'
#                     }
#                 }
            
#             logger.info(f"Ranking: {len(ranked_df)} posts scored")
            
#             # Step 4: Build post metadata
#             post_metadata = self._build_post_metadata(ranked_df['post_id'].tolist())
            
#             # Step 5: Re-ranking
#             t3 = time.time()
#             final_feed = self.reranker.rerank(
#                 ranked_df=ranked_df,
#                 post_metadata=post_metadata,
#                 limit=limit
#             )
#             reranking_time = (time.time() - t3) * 1000
            
#             # Step 6: Mark as seen (if configured)
#             if self.config.get('mark_seen_after_generation', False):
#                 post_ids = [post['post_id'] for post in final_feed if not post.get('is_ad')]
#                 cache_service.mark_posts_as_seen(user_id, post_ids)
            
#             # Update metrics
#             total_time = (time.time() - start_time) * 1000
#             self.metrics['total_requests'] += 1
#             self.metrics['total_time_ms'] += total_time
#             self.metrics['recall_time_ms'] += recall_time
#             self.metrics['ranking_time_ms'] += ranking_time
#             self.metrics['reranking_time_ms'] += reranking_time
            
#             logger.info(f"✅ Feed generated for user {user_id}: {len(final_feed)} items in {total_time:.2f}ms")
            
#             return {
#                 'user_id': user_id,
#                 'feed': final_feed,
#                 'metadata': {
#                     'total_time_ms': round(total_time, 2),
#                     'recall_time_ms': round(recall_time, 2),
#                     'ranking_time_ms': round(ranking_time, 2),
#                     'reranking_time_ms': round(reranking_time, 2),
#                     'recall_count': len(candidates),
#                     'ranked_count': len(ranked_df),
#                     'final_count': len(final_feed),
#                     'timestamp': datetime.now().isoformat()
#                 }
#             }
            
#         except Exception as e:
#             logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
#             return {
#                 'user_id': user_id,
#                 'feed': [],
#                 'metadata': {
#                     'total_time_ms': (time.time() - start_time) * 1000,
#                     'error': str(e)
#                 }
#             }
    
#     def _recall_candidates(self, user_id: int) -> List[int]:
#         """
#         Multi-channel recall
        
#         Channels:
#         - Following: 400 posts
#         - CF: 300 posts
#         - Content: 200 posts
#         - Trending: 100 posts
        
#         Total: ~1000 candidates (deduplicated)
        
#         Args:
#             user_id: User ID
            
#         Returns:
#             List of candidate post IDs
#         """
#         channels_config = self.recall_config.get('channels', {})
        
#         # Get quotas
#         following_quota = channels_config.get('following', {}).get('count', 400)
#         cf_quota = channels_config.get('collaborative_filtering', {}).get('count', 300)
#         content_quota = channels_config.get('content_based', {}).get('count', 200)
#         trending_quota = channels_config.get('trending', {}).get('count', 100)
        
#         # Recall from each channel
#         following_posts = self.following_recall.recall(user_id, k=following_quota)
#         cf_posts = self.cf_recall.recall(user_id, k=cf_quota)
#         content_posts = self.content_recall.recall(user_id, k=content_quota)
#         trending_posts = self.trending_recall.recall(user_id, k=trending_quota)
        
#         logger.debug(f"Recall breakdown: following={len(following_posts)}, "
#                     f"cf={len(cf_posts)}, content={len(content_posts)}, "
#                     f"trending={len(trending_posts)}")
        
#         # Merge and deduplicate (preserve order)
#         candidates = []
#         seen = set()
        
#         # Priority: following > cf > content > trending
#         for post_id in following_posts + cf_posts + content_posts + trending_posts:
#             if post_id not in seen:
#                 candidates.append(post_id)
#                 seen.add(post_id)
        
#         return candidates
    
#     def _build_post_metadata(self, post_ids: List[int]) -> Dict[int, Dict]:
#         """
#         Build metadata for posts
        
#         Args:
#             post_ids: List of post IDs
            
#         Returns:
#             Dict mapping post_id to metadata
#         """
#         metadata = {}
        
#         if 'posts' not in self.data:
#             return metadata
        
#         posts = self.data['posts']
        
#         for post_id in post_ids:
#             post = posts[posts['Id'] == post_id]
            
#             if not post.empty:
#                 row = post.iloc[0]
                
#                 metadata[post_id] = {
#                     'author_id': int(row['UserId']),
#                     'category': row.get('CategoryId', None),
#                     'created_at': pd.to_datetime(row.get('CreateDate'), errors='coerce'),
#                     'title': row.get('Title', ''),
#                     'content': row.get('Content', '')
#                 }
        
#         return metadata
    
#     def get_metrics(self) -> Dict:
#         """Get service metrics"""
#         avg_time = (
#             self.metrics['total_time_ms'] / self.metrics['total_requests']
#             if self.metrics['total_requests'] > 0 else 0
#         )
        
#         return {
#             **self.metrics,
#             'avg_total_time_ms': round(avg_time, 2),
#             'recall_metrics': {
#                 'following': self.following_recall.get_metrics(),
#                 'cf': self.cf_recall.get_metrics(),
#                 'content': self.content_recall.get_metrics(),
#                 'trending': self.trending_recall.get_metrics()
#             },
#             'ranking_metrics': self.ranker.get_metrics(),
#             'reranking_metrics': self.reranker.get_metrics()
#         }

"""
Recommendation Service - COMPLETE FIX
======================================

Fixed issues:
1. ✅ Column name normalization (UserId vs userid)
2. ✅ Safe groupby with validation
3. ✅ Proper error handling
4. ✅ Null checks everywhere
"""

import logging
from typing import Dict, Set, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Recommendation Service with complete error handling
    """
    
    def __init__(
        self,
        cache_service,
        models: Dict,
        embeddings: Dict,
        faiss_index,
        faiss_post_ids: List[int],
        feature_engineer,
        data: Dict[str, pd.DataFrame]
    ):
        """
        Initialize Recommendation Service
        
        Args:
            cache_service: Redis cache service
            models: ML models dict
            embeddings: Post/user embeddings
            faiss_index: FAISS index for similarity search
            faiss_post_ids: Post IDs mapping for FAISS
            feature_engineer: Feature engineering service
            data: Data dictionary (posts, friendships, users)
        """
        self.cache_service = cache_service
        self.models = models
        self.embeddings = embeddings
        self.faiss_index = faiss_index
        self.faiss_post_ids = faiss_post_ids
        self.feature_engineer = feature_engineer
        self.data = data
        
        # Initialize recall channels
        self.following_dict: Dict[int, Set[int]] = {}
        self._init_recall_channels()
        
        logger.info("✅ RecommendationService initialized")
    
    # ==================== INITIALIZATION ====================
    
    def _init_recall_channels(self):
        """
        Initialize recall channels
        
        FIXED: Column name normalization and safe groupby
        """
        logger.info("🔧 Initializing recall channels...")
        
        # Build following dictionary from friendships
        friendships = self.data.get('friendships')
        
        if friendships is None:
            logger.warning("⚠️ No friendships data available (is None)")
            self.following_dict = {}
            return
        
        if friendships.empty:
            logger.warning("⚠️ Friendships data is empty")
            self.following_dict = {}
            return
        
        logger.info(f"📊 Processing {len(friendships):,} friendships...")
        logger.debug(f"   Original columns: {friendships.columns.tolist()}")
        
        # ========== COLUMN NORMALIZATION ==========
        
        # Mapping of all possible column name variations
        column_mapping = {
            # UserId variations
            'userid': 'UserId',
            'user_id': 'UserId',
            'UserID': 'UserId',
            'USERID': 'UserId',
            
            # FriendId variations
            'friendid': 'FriendId',
            'friend_id': 'FriendId',
            'FriendID': 'FriendId',
            'FRIENDID': 'FriendId',
            
            # Status variations
            'status': 'Status',
            'STATUS': 'Status'
        }
        
        # Create a copy to avoid modifying original data
        friendships = friendships.copy()
        
        # Build rename dictionary
        rename_dict = {}
        for col in friendships.columns:
            # Try exact match first
            if col in column_mapping:
                rename_dict[col] = column_mapping[col]
            # Try case-insensitive match
            elif col.lower() in column_mapping:
                rename_dict[col] = column_mapping[col.lower()]
        
        # Apply renaming
        if rename_dict:
            friendships.rename(columns=rename_dict, inplace=True)
            logger.debug(f"   Normalized columns: {rename_dict}")
            logger.debug(f"   After normalization: {friendships.columns.tolist()}")
        
        # ========== VALIDATION ==========
        
        required_columns = ['UserId', 'FriendId']
        missing_columns = [col for col in required_columns if col not in friendships.columns]
        
        if missing_columns:
            logger.error(f"❌ Missing required columns: {missing_columns}")
            logger.error(f"   Available columns: {friendships.columns.tolist()}")
            logger.error("   Cannot build following dictionary!")
            self.following_dict = {}
            return
        
        # ========== BUILD FOLLOWING DICT ==========
        
        try:
            # Group by UserId and collect FriendIds
            for user_id, group in friendships.groupby('UserId'):
                self.following_dict[int(user_id)] = set(group['FriendId'].astype(int).tolist())
            
            logger.info(f"✅ Built following dict: {len(self.following_dict):,} users")
            
            # Log some stats
            if self.following_dict:
                avg_friends = sum(len(friends) for friends in self.following_dict.values()) / len(self.following_dict)
                max_friends = max(len(friends) for friends in self.following_dict.values())
                logger.info(f"   Average friends per user: {avg_friends:.1f}")
                logger.info(f"   Max friends per user: {max_friends}")
            
        except KeyError as e:
            logger.error(f"❌ KeyError during groupby: {e}")
            logger.error(f"   Available columns: {friendships.columns.tolist()}")
            self.following_dict = {}
            
        except Exception as e:
            logger.error(f"❌ Failed to build following dict: {e}", exc_info=True)
            self.following_dict = {}
    
    # ==================== RECOMMENDATION METHODS ====================
    
    # def get_feed(
    #     self,
    #     user_id: int,
    #     limit: int = 50,
    #     exclude_seen: Optional[List[int]] = None
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Get personalized feed for user
        
    #     Args:
    #         user_id: User ID
    #         limit: Number of posts to return
    #         exclude_seen: Post IDs to exclude
        
    #     Returns:
    #         List of recommended posts
    #     """
    #     try:
    #         logger.info(f"📰 Getting feed for user {user_id}, limit={limit}")
            
    #         # 1. Recall from multiple channels
    #         candidates = self._recall_candidates(user_id, target=1000)
            
    #         if not candidates:
    #             logger.warning(f"No candidates for user {user_id}")
    #             return []
            
    #         logger.info(f"   Recalled {len(candidates)} candidates")
            
    #         # 2. Filter seen posts
    #         if exclude_seen:
    #             candidates = [c for c in candidates if c['post_id'] not in exclude_seen]
    #             logger.info(f"   After filtering seen: {len(candidates)} candidates")
            
    #         # 3. Rank candidates
    #         ranked = self._rank_candidates(user_id, candidates)
            
    #         # 4. Diversify and return top-K
    #         final_feed = ranked[:limit]
            
    #         logger.info(f"✅ Returning {len(final_feed)} posts for user {user_id}")
            
    #         return final_feed
            
    #     except Exception as e:
    #         logger.error(f"Error getting feed for user {user_id}: {e}", exc_info=True)
    #         return []
    def get_feed(self, user_id: int, limit: int = 50):
        """
        Get personalized feed for user
        
        Returns:
            dict: {
                "posts": List[dict],
                "count": int,
                "user_id": int
            }
        """
        candidates = self._recall_candidates(user_id)
        
        if not candidates:
            logger.warning(f"No candidates for user {user_id}")
            # ✅ FIX: Trả về dictionary thay vì empty list
            return {
                "posts": [],
                "count": 0,
                "user_id": user_id
            }
        
        # Ranking và trả về feed
        ranked_posts = self._rank_and_filter(candidates, limit)
        
        return {
            "posts": ranked_posts,
            "count": len(ranked_posts),
            "user_id": user_id
        }
        
    def _recall_candidates(
        self,
        user_id: int,
        target: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Recall candidates from multiple channels
        
        Channels:
        1. Following (posts from friends)
        2. CF (collaborative filtering)
        3. Content (similar posts)
        4. Trending (popular posts)
        """
        candidates = []
        
        # Channel 1: Following
        try:
            following_posts = self._recall_following(user_id, k=300)
            candidates.extend(following_posts)
            logger.debug(f"   Following recall: {len(following_posts)} posts")
        except Exception as e:
            logger.error(f"Following recall failed: {e}")
        
        # Channel 2: CF
        try:
            cf_posts = self._recall_cf(user_id, k=300)
            candidates.extend(cf_posts)
            logger.debug(f"   CF recall: {len(cf_posts)} posts")
        except Exception as e:
            logger.error(f"CF recall failed: {e}")
        
        # Channel 3: Content
        try:
            content_posts = self._recall_content(user_id, k=300)
            candidates.extend(content_posts)
            logger.debug(f"   Content recall: {len(content_posts)} posts")
        except Exception as e:
            logger.error(f"Content recall failed: {e}")
        
        # Channel 4: Trending
        try:
            trending_posts = self._recall_trending(k=200)
            candidates.extend(trending_posts)
            logger.debug(f"   Trending recall: {len(trending_posts)} posts")
        except Exception as e:
            logger.error(f"Trending recall failed: {e}")
        
        # Deduplicate by post_id
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate['post_id'] not in seen:
                seen.add(candidate['post_id'])
                unique_candidates.append(candidate)
        
        return unique_candidates[:target]
    
    def _recall_following(self, user_id: int, k: int = 300) -> List[Dict[str, Any]]:
        """Recall posts from users that this user follows"""
        following = self.following_dict.get(user_id, set())
        
        if not following:
            return []
        
        # Get posts from friends (simplified - in production use Redis/cache)
        posts = []
        
        # For now, return empty (implement with Redis)
        # In production: get recent posts from Redis sorted sets
        
        return posts
    
    def _recall_cf(self, user_id: int, k: int = 300) -> List[Dict[str, Any]]:
        """Collaborative filtering recall"""
        # Implement CF logic here
        return []
    
    def _recall_content(self, user_id: int, k: int = 300) -> List[Dict[str, Any]]:
        """Content-based recall using FAISS"""
        if self.faiss_index is None:
            return []
        
        # Get user embedding
        user_emb = self.embeddings.get('user', {}).get(user_id)
        
        if user_emb is None:
            return []
        
        # Search FAISS
        try:
            import faiss
            
            # Reshape for FAISS
            query = np.array([user_emb]).astype('float32')
            
            # Search
            distances, indices = self.faiss_index.search(query, k)
            
            # Map indices to post IDs
            posts = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.faiss_post_ids):
                    posts.append({
                        'post_id': self.faiss_post_ids[idx],
                        'score': float(dist),
                        'channel': 'content'
                    })
            
            return posts
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    def _recall_trending(self, k: int = 200) -> List[Dict[str, Any]]:
        """Recall trending posts"""
        # Implement trending logic (from Redis sorted set)
        return []
    
    def _rank_candidates(
        self,
        user_id: int,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates using ML model
        """
        if not candidates:
            return []
        
        # If no ranking model, return candidates sorted by score
        if self.models.get('ranking_model') is None:
            logger.warning("No ranking model, using recall scores")
            return sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)
        
        # Implement ranking with feature engineering
        # For now, return sorted by recall score
        return sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)
    
    # ==================== HELPER METHODS ====================
    
    def get_user_following(self, user_id: int) -> Set[int]:
        """Get set of user IDs that this user follows"""
        return self.following_dict.get(user_id, set())
    
    def is_following(self, user_id: int, target_user_id: int) -> bool:
        """Check if user follows target user"""
        return target_user_id in self.following_dict.get(user_id, set())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            'total_users_with_friends': len(self.following_dict),
            'total_embeddings': len(self.embeddings.get('post', {})),
            'faiss_vectors': self.faiss_index.ntotal if self.faiss_index else 0,
            'models_loaded': {
                'ranking': self.models.get('ranking_model') is not None,
                'cf': self.models.get('cf_model') is not None,
                'scaler': self.models.get('scaler') is not None
            }
        }


# ==================== HELPER FUNCTIONS ====================

def normalize_column_names(df: pd.DataFrame, table_name: str = "") -> pd.DataFrame:
    """
    Normalize column names to PascalCase
    
    Handles variations like: userid, user_id, UserID → UserId
    """
    if df.empty:
        return df
    
    column_mapping = {
        'userid': 'UserId',
        'user_id': 'UserId',
        'UserID': 'UserId',
        'USERID': 'UserId',
        
        'friendid': 'FriendId',
        'friend_id': 'FriendId',
        'FriendID': 'FriendId',
        'FRIENDID': 'FriendId',
        
        'postid': 'PostId',
        'post_id': 'PostId',
        'PostID': 'PostId',
        'POSTID': 'PostId',
    }
    
    df_normalized = df.copy()
    rename_dict = {}
    
    for col in df_normalized.columns:
        if col in column_mapping:
            rename_dict[col] = column_mapping[col]
        elif col.lower() in column_mapping:
            rename_dict[col] = column_mapping[col.lower()]
    
    if rename_dict:
        df_normalized.rename(columns=rename_dict, inplace=True)
        logger.debug(f"Normalized {len(rename_dict)} columns in {table_name}")
    
    return df_normalized    