"""
Recommendation Service
=====================
Complete feed generation pipeline

Flow:
1. Multi-channel Recall (~1000 candidates)
2. Feature Extraction
3. ML Ranking (score candidates)
4. Re-ranking (apply business rules)
5. Final feed (50 posts)

Target latency: < 200ms end-to-end
"""

from typing import List, Dict, Optional, Set
import pandas as pd
import logging
import time
from datetime import datetime

from recommender.online.app.services.cache_service import cache_service
from recommender.online.app.recall import FollowingRecall, CFRecall, ContentRecall, TrendingRecall
from recommender.online.app.ranking import MLRanker, Reranker

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Main recommendation service orchestrating the entire pipeline
    """
    
    def __init__(
        self,
        data: Dict,
        models: Dict,
        embeddings: Dict,
        faiss_index,
        faiss_post_ids: List[int],
        feature_engineer,
        config: Dict
    ):
        """
        Initialize recommendation service
        
        Args:
            data: Data dictionaries (posts, reactions, friendships, etc.)
            models: Trained models (ranking_model, cf_model)
            embeddings: User and post embeddings
            faiss_index: FAISS index for content-based recall
            faiss_post_ids: Post IDs for FAISS index
            feature_engineer: FeatureEngineer instance
            config: Configuration
        """
        self.data = data
        self.models = models
        self.embeddings = embeddings
        self.faiss_index = faiss_index
        self.faiss_post_ids = faiss_post_ids
        self.feature_engineer = feature_engineer
        self.config = config
        
        # Extract configs
        self.recall_config = config.get('recall', {})
        self.ranking_config = config.get('ranking', {})
        self.reranking_config = config.get('reranking', {})
        
        # Initialize recall channels
        self._init_recall_channels()
        
        # Initialize ranking
        self._init_ranking()
        
        # Initialize reranking
        self.reranker = Reranker(config=self.reranking_config)
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'total_time_ms': 0,
            'recall_time_ms': 0,
            'ranking_time_ms': 0,
            'reranking_time_ms': 0
        }
        
        logger.info("✅ RecommendationService initialized successfully")
    
    def _init_recall_channels(self):
        """Initialize all recall channels"""
        channels_config = self.recall_config.get('channels', {})
        
        # Build following dict
        following_dict = {}
        if 'friendships' in self.data:
            friendships = self.data['friendships']
            for user_id, group in friendships.groupby('UserId'):
                following_dict[int(user_id)] = group['FriendId'].astype(int).tolist()
        
        # Initialize channels
        self.following_recall = FollowingRecall(
            redis_client=cache_service.redis if cache_service.is_connected() else None,
            data=self.data,
            following_dict=following_dict,
            config=channels_config.get('following', {})
        )
        
        self.cf_recall = CFRecall(
            redis_client=cache_service.redis if cache_service.is_connected() else None,
            data=self.data,
            cf_model=self.models.get('cf_model'),
            config=channels_config.get('collaborative_filtering', {})
        )
        
        self.content_recall = ContentRecall(
            redis_client=cache_service.redis if cache_service.is_connected() else None,
            data=self.data,
            embeddings=self.embeddings,
            faiss_index=self.faiss_index,
            faiss_post_ids=self.faiss_post_ids,
            config=channels_config.get('content_based', {})
        )
        
        self.trending_recall = TrendingRecall(
            redis_client=cache_service.redis if cache_service.is_connected() else None,
            data=self.data,
            config=channels_config.get('trending', {})
        )
        
        logger.info("✅ Recall channels initialized")
    
    def _init_ranking(self):
        """Initialize ranking model"""
        self.ranker = MLRanker(
            model=self.models.get('ranking_model'),
            scaler=self.models.get('scaler'),
            feature_cols=self.models.get('feature_cols'),
            feature_engineer=self.feature_engineer,
            config=self.ranking_config
        )
        
        logger.info("✅ Ranker initialized")
    
    def generate_feed(
        self,
        user_id: int,
        limit: int = 50,
        exclude_seen: Optional[List[int]] = None
    ) -> Dict:
        """
        Generate personalized feed for user
        
        Args:
            user_id: User ID
            limit: Number of posts to return
            exclude_seen: List of post IDs to exclude (already seen)
            
        Returns:
            Dict with feed and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Multi-channel Recall
            t1 = time.time()
            candidates = self._recall_candidates(user_id)
            recall_time = (time.time() - t1) * 1000
            
            logger.info(f"Recall: {len(candidates)} candidates for user {user_id}")
            
            # Step 2: Filter seen posts
            if exclude_seen:
                candidates = [pid for pid in candidates if pid not in exclude_seen]
            
            # Also filter posts user has seen (from cache)
            seen_posts = cache_service.get_seen_posts(user_id)
            if seen_posts:
                candidates = [pid for pid in candidates if pid not in seen_posts]
            
            logger.info(f"After filtering seen: {len(candidates)} candidates")
            
            if not candidates:
                logger.warning(f"No candidates for user {user_id} after filtering")
                return {
                    'user_id': user_id,
                    'feed': [],
                    'metadata': {
                        'total_time_ms': (time.time() - start_time) * 1000,
                        'recall_count': 0,
                        'reason': 'no_candidates_after_filtering'
                    }
                }
            
            # Step 3: ML Ranking
            t2 = time.time()
            ranked_df = self.ranker.rank(user_id, candidates)
            ranking_time = (time.time() - t2) * 1000
            
            if ranked_df.empty:
                logger.warning(f"Ranking returned empty for user {user_id}")
                return {
                    'user_id': user_id,
                    'feed': [],
                    'metadata': {
                        'total_time_ms': (time.time() - start_time) * 1000,
                        'recall_count': len(candidates),
                        'reason': 'ranking_failed'
                    }
                }
            
            logger.info(f"Ranking: {len(ranked_df)} posts scored")
            
            # Step 4: Build post metadata
            post_metadata = self._build_post_metadata(ranked_df['post_id'].tolist())
            
            # Step 5: Re-ranking
            t3 = time.time()
            final_feed = self.reranker.rerank(
                ranked_df=ranked_df,
                post_metadata=post_metadata,
                limit=limit
            )
            reranking_time = (time.time() - t3) * 1000
            
            # Step 6: Mark as seen (if configured)
            if self.config.get('mark_seen_after_generation', False):
                post_ids = [post['post_id'] for post in final_feed if not post.get('is_ad')]
                cache_service.mark_posts_as_seen(user_id, post_ids)
            
            # Update metrics
            total_time = (time.time() - start_time) * 1000
            self.metrics['total_requests'] += 1
            self.metrics['total_time_ms'] += total_time
            self.metrics['recall_time_ms'] += recall_time
            self.metrics['ranking_time_ms'] += ranking_time
            self.metrics['reranking_time_ms'] += reranking_time
            
            logger.info(f"✅ Feed generated for user {user_id}: {len(final_feed)} items in {total_time:.2f}ms")
            
            return {
                'user_id': user_id,
                'feed': final_feed,
                'metadata': {
                    'total_time_ms': round(total_time, 2),
                    'recall_time_ms': round(recall_time, 2),
                    'ranking_time_ms': round(ranking_time, 2),
                    'reranking_time_ms': round(reranking_time, 2),
                    'recall_count': len(candidates),
                    'ranked_count': len(ranked_df),
                    'final_count': len(final_feed),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
            return {
                'user_id': user_id,
                'feed': [],
                'metadata': {
                    'total_time_ms': (time.time() - start_time) * 1000,
                    'error': str(e)
                }
            }
    
    def _recall_candidates(self, user_id: int) -> List[int]:
        """
        Multi-channel recall
        
        Channels:
        - Following: 400 posts
        - CF: 300 posts
        - Content: 200 posts
        - Trending: 100 posts
        
        Total: ~1000 candidates (deduplicated)
        
        Args:
            user_id: User ID
            
        Returns:
            List of candidate post IDs
        """
        channels_config = self.recall_config.get('channels', {})
        
        # Get quotas
        following_quota = channels_config.get('following', {}).get('count', 400)
        cf_quota = channels_config.get('collaborative_filtering', {}).get('count', 300)
        content_quota = channels_config.get('content_based', {}).get('count', 200)
        trending_quota = channels_config.get('trending', {}).get('count', 100)
        
        # Recall from each channel
        following_posts = self.following_recall.recall(user_id, k=following_quota)
        cf_posts = self.cf_recall.recall(user_id, k=cf_quota)
        content_posts = self.content_recall.recall(user_id, k=content_quota)
        trending_posts = self.trending_recall.recall(user_id, k=trending_quota)
        
        logger.debug(f"Recall breakdown: following={len(following_posts)}, "
                    f"cf={len(cf_posts)}, content={len(content_posts)}, "
                    f"trending={len(trending_posts)}")
        
        # Merge and deduplicate (preserve order)
        candidates = []
        seen = set()
        
        # Priority: following > cf > content > trending
        for post_id in following_posts + cf_posts + content_posts + trending_posts:
            if post_id not in seen:
                candidates.append(post_id)
                seen.add(post_id)
        
        return candidates
    
    def _build_post_metadata(self, post_ids: List[int]) -> Dict[int, Dict]:
        """
        Build metadata for posts
        
        Args:
            post_ids: List of post IDs
            
        Returns:
            Dict mapping post_id to metadata
        """
        metadata = {}
        
        if 'posts' not in self.data:
            return metadata
        
        posts = self.data['posts']
        
        for post_id in post_ids:
            post = posts[posts['Id'] == post_id]
            
            if not post.empty:
                row = post.iloc[0]
                
                metadata[post_id] = {
                    'author_id': int(row['UserId']),
                    'category': row.get('CategoryId', None),
                    'created_at': pd.to_datetime(row.get('CreateDate'), errors='coerce'),
                    'title': row.get('Title', ''),
                    'content': row.get('Content', '')
                }
        
        return metadata
    
    def get_metrics(self) -> Dict:
        """Get service metrics"""
        avg_time = (
            self.metrics['total_time_ms'] / self.metrics['total_requests']
            if self.metrics['total_requests'] > 0 else 0
        )
        
        return {
            **self.metrics,
            'avg_total_time_ms': round(avg_time, 2),
            'recall_metrics': {
                'following': self.following_recall.get_metrics(),
                'cf': self.cf_recall.get_metrics(),
                'content': self.content_recall.get_metrics(),
                'trending': self.trending_recall.get_metrics()
            },
            'ranking_metrics': self.ranker.get_metrics(),
            'reranking_metrics': self.reranker.get_metrics()
        }