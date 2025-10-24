"""
ONLINE INFERENCE PIPELINE - PRODUCTION READY
=============================================
Complete recommendation pipeline integrating all components

Architecture:
1. Load artifacts (models, embeddings, stats)
2. Initialize recall channels (Following, CF, Content, Trending)
3. Initialize ranker & reranker
4. Real-time user embedding updates
5. Feed generation with < 200ms latency

Flow: Recall → Feature Extraction → Ranking → Re-ranking → Output
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import logging
import yaml
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

# Core imports
from recommender.offline.artifact_manager import ArtifactManager
from recommender.common.feature_engineer import FeatureEngineer

# Recall channels
from recommender.online.recall import (
    FollowingRecall,
    CFRecall,
    ContentRecall,
    TrendingRecall
)

# Ranking
from recommender.online.ranking import MLRanker, Reranker

# Friend recommendation
from recommender.online.friend_recommendation import FriendRecommendation

# Try optional imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️  redis not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  lightgbm not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OnlineInferencePipeline:
    """
    Complete online inference pipeline
    
    Features:
    - Multi-channel recall (4 channels)
    - ML ranking with LightGBM
    - Business rules re-ranking
    - Real-time user embedding updates
    - Friend recommendations
    - Redis caching
    - < 200ms latency
    """
    
    def __init__(
        self,
        config_path: str = 'configs/config_online.yaml',
        models_dir: str = 'models',
        data_dir: str = 'dataset',
        use_redis: bool = True
    ):
        """
        Initialize online pipeline
        
        Args:
            config_path: Path to online config file
            models_dir: Directory with model artifacts
            data_dir: Directory with data (for development)
            use_redis: Whether to use Redis cache
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # ════════════════════════════════════════════════════════════════
        # STEP 1: LOAD CONFIGURATION
        # ════════════════════════════════════════════════════════════════
        
        logger.info("\n" + "="*70)
        logger.info("INITIALIZING ONLINE INFERENCE PIPELINE")
        logger.info("="*70)
        
        logger.info("\n📋 Step 1: Loading configuration...")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.recall_config = self.config.get('recall', {})
        self.ranking_config = self.config.get('ranking', {})
        self.reranking_config = self.config.get('reranking', {})
        
        logger.info("✅ Configuration loaded")
        
        # ════════════════════════════════════════════════════════════════
        # STEP 2: CONNECT TO REDIS
        # ════════════════════════════════════════════════════════════════
        
        logger.info("\n🔧 Step 2: Connecting to Redis...")
        
        self.redis = None
        if use_redis and REDIS_AVAILABLE:
            try:
                redis_config = self.config.get('redis', {})
                self.redis = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6380),
                    db=redis_config.get('db', 0),
                    decode_responses=False,
                    socket_timeout=redis_config.get('socket_timeout', 5),
                    max_connections=redis_config.get('max_connections', 50)
                )
                # Test connection
                self.redis.ping()
                logger.info("✅ Redis connected")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis = None
        else:
            logger.warning("Redis disabled or not available")
        
        # ════════════════════════════════════════════════════════════════
        # STEP 3: LOAD ARTIFACTS
        # ════════════════════════════════════════════════════════════════
        
        logger.info("\n📦 Step 3: Loading model artifacts...")
        
        self.artifact_mgr = ArtifactManager(base_dir=str(self.models_dir))
        
        # Get version
        model_version = self.config.get('models', {}).get('version', 'latest')
        if model_version == 'latest':
            self.current_version = self.artifact_mgr.get_latest_version()
        else:
            self.current_version = model_version
        
        logger.info(f"Loading version: {self.current_version}")
        
        # Load all artifacts
        artifacts = self.artifact_mgr.load_artifacts(self.current_version)
        
        # Extract components
        self.embeddings = artifacts['embeddings']
        self.faiss_index = artifacts.get('faiss_index')
        self.faiss_post_ids = artifacts.get('faiss_post_ids', [])
        self.cf_model = artifacts['cf_model']
        self.ranking_model = artifacts['ranking_model']
        self.ranking_scaler = artifacts['ranking_scaler']
        self.ranking_feature_cols = artifacts['ranking_feature_cols']
        self.user_stats = artifacts['user_stats']
        self.author_stats = artifacts['author_stats']
        self.following_dict = artifacts['following_dict']
        self.metadata = artifacts.get('metadata', {})
        
        logger.info("✅ Artifacts loaded successfully!")
        logger.info(f"   Post embeddings: {len(self.embeddings['post']):,}")
        logger.info(f"   User embeddings: {len(self.embeddings['user']):,}")
        logger.info(f"   Ranking features: {len(self.ranking_feature_cols)}")
        
        # ════════════════════════════════════════════════════════════════
        # STEP 4: LOAD DATA (for development)
        # ════════════════════════════════════════════════════════════════
        
        logger.info("\n📊 Step 4: Loading data...")
        
        from recommender.common.data_loader import load_data
        self.data = load_data(str(self.data_dir))
        
        logger.info("✅ Data loaded")
        
        # ════════════════════════════════════════════════════════════════
        # STEP 5: INITIALIZE FEATURE ENGINEER
        # ════════════════════════════════════════════════════════════════
        
        logger.info("\n🔧 Step 5: Initializing feature engineer...")
        
        self.feature_engineer = FeatureEngineer(
            data=self.data,
            user_stats=self.user_stats,
            author_stats=self.author_stats,
            following_dict=self.following_dict,
            embeddings=self.embeddings
        )
        
        logger.info("✅ Feature engineer initialized")
        
        # ════════════════════════════════════════════════════════════════
        # STEP 6: INITIALIZE RECALL CHANNELS
        # ════════════════════════════════════════════════════════════════
        
        logger.info("\n🔧 Step 6: Initializing recall channels...")
        
        channels_config = self.recall_config.get('channels', {})
        
        # Channel 1: Following
        self.following_recall = FollowingRecall(
            redis_client=self.redis,
            data=self.data,
            following_dict=self.following_dict,
            config=channels_config.get('following', {})
        )
        
        # Channel 2: CF
        self.cf_recall = CFRecall(
            redis_client=self.redis,
            data=self.data,
            cf_model=self.cf_model,
            config=channels_config.get('collaborative_filtering', {})
        )
        
        # Channel 3: Content
        self.content_recall = ContentRecall(
            redis_client=self.redis,
            embeddings=self.embeddings,
            faiss_index=self.faiss_index,
            faiss_post_ids=self.faiss_post_ids,
            data=self.data,
            config=channels_config.get('content_based', {})
        )
        
        # Channel 4: Trending
        self.trending_recall = TrendingRecall(
            redis_client=self.redis,
            data=self.data,
            config=channels_config.get('trending', {})
        )
        
        logger.info("✅ Recall channels initialized")
        
        # ════════════════════════════════════════════════════════════════
        # STEP 7: INITIALIZE RANKER & RERANKER
        # ════════════════════════════════════════════════════════════════
        
        logger.info("\n🔧 Step 7: Initializing ranker & reranker...")
        
        # Ranker
        self.ranker = MLRanker(
            model=self.ranking_model,
            scaler=self.ranking_scaler,
            feature_cols=self.ranking_feature_cols,
            feature_engineer=self.feature_engineer,
            config=self.ranking_config
        )
        
        # Reranker
        self.reranker = Reranker(config=self.reranking_config)
        
        logger.info("✅ Ranker & Reranker initialized")
        
        # ════════════════════════════════════════════════════════════════
        # STEP 8: INITIALIZE FRIEND RECOMMENDATION
        # ════════════════════════════════════════════════════════════════
        
        logger.info("\n🔧 Step 8: Initializing friend recommendation...")
        
        self.friend_recommender = FriendRecommendation(
            data=self.data,
            embeddings=self.embeddings,
            cf_model=self.cf_model,
            config=self.config.get('friend_recommendation', {})
        )
        
        logger.info("✅ Friend recommendation initialized")
        
        # ════════════════════════════════════════════════════════════════
        # METRICS TRACKING
        # ════════════════════════════════════════════════════════════════
        
        self.metrics = defaultdict(list)
        
        logger.info("\n✅ ONLINE PIPELINE READY!")
        logger.info("="*70 + "\n")
    
    # ════════════════════════════════════════════════════════════════════
    # MAIN FEED GENERATION
    # ════════════════════════════════════════════════════════════════════
    
    def generate_feed(
        self,
        user_id: int,
        limit: int = 50,
        exclude_seen: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Generate personalized feed for user
        
        Pipeline:
        1. Multi-channel recall (~1000 candidates)
        2. Feature extraction (47 features × N posts)
        3. ML ranking (LightGBM)
        4. Re-ranking & business rules
        5. Return top K
        
        Args:
            user_id: Target user ID
            limit: Number of posts to return
            exclude_seen: Post IDs to exclude
            
        Returns:
            List of post dicts with scores
        """
        start_time = time.time()
        
        try:
            # ════════════════════════════════════════════════════════════
            # STAGE 1: MULTI-CHANNEL RECALL
            # ════════════════════════════════════════════════════════════
            
            t1 = time.time()
            candidates = self._recall_candidates(user_id, target_count=1000)
            recall_latency = (time.time() - t1) * 1000
            self.metrics['recall_latency'].append(recall_latency)
            
            if not candidates:
                logger.warning(f"No candidates for user {user_id}")
                return []
            
            # Exclude seen posts
            if exclude_seen:
                candidates = [p for p in candidates if p not in exclude_seen]
            
            logger.debug(f"Recall: {len(candidates)} candidates in {recall_latency:.1f}ms")
            
            # ════════════════════════════════════════════════════════════
            # STAGE 2: ML RANKING
            # ════════════════════════════════════════════════════════════
            
            t2 = time.time()
            ranked_df = self.ranker.rank(user_id, candidates)
            ranking_latency = (time.time() - t2) * 1000
            self.metrics['ranking_latency'].append(ranking_latency)
            
            if ranked_df.empty:
                logger.warning(f"Ranking failed for user {user_id}")
                return []
            
            # Get top 100 for re-ranking
            ranked_df = ranked_df.head(100)
            
            logger.debug(f"Ranking: {len(ranked_df)} posts in {ranking_latency:.1f}ms")
            
            # ════════════════════════════════════════════════════════════
            # STAGE 3: RE-RANKING
            # ════════════════════════════════════════════════════════════
            
            t3 = time.time()
            final_feed = self.reranker.rerank(
                ranked_df=ranked_df,
                post_metadata=None,  # TODO: Add metadata
                limit=limit
            )
            reranking_latency = (time.time() - t3) * 1000
            self.metrics['reranking_latency'].append(reranking_latency)
            
            logger.debug(f"Reranking: {len(final_feed)} posts in {reranking_latency:.1f}ms")
            
            # ════════════════════════════════════════════════════════════
            # METRICS
            # ════════════════════════════════════════════════════════════
            
            total_latency = (time.time() - start_time) * 1000
            self.metrics['total_latency'].append(total_latency)
            
            logger.info(
                f"Feed generated for user {user_id}: "
                f"{len(final_feed)} posts | "
                f"Latency: {total_latency:.1f}ms "
                f"(R:{recall_latency:.0f} + M:{ranking_latency:.0f} + B:{reranking_latency:.0f})"
            )
            
            return final_feed
            
        except Exception as e:
            logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
            return []
    
    def _recall_candidates(self, user_id: int, target_count: int = 1000) -> List[int]:
        """
        Multi-channel recall
        
        Channels:
        1. Following (400) - Posts from followed users
        2. CF (300) - Posts liked by similar users
        3. Content (200) - Similar posts via embeddings
        4. Trending (100) - Popular posts
        """
        all_candidates = []
        
        # Channel 1: Following
        following_posts = self.following_recall.recall(user_id, k=400)
        all_candidates.extend(following_posts)
        
        # Channel 2: CF
        cf_posts = self.cf_recall.recall(user_id, k=300)
        all_candidates.extend(cf_posts)
        
        # Channel 3: Content
        content_posts = self.content_recall.recall(user_id, k=200)
        all_candidates.extend(content_posts)
        
        # Channel 4: Trending
        trending_posts = self.trending_recall.recall(k=100)
        all_candidates.extend(trending_posts)
        
        # Deduplicate (preserve order)
        unique_candidates = list(dict.fromkeys(all_candidates))
        
        return unique_candidates[:target_count]
    
    # ════════════════════════════════════════════════════════════════════
    # FRIEND RECOMMENDATION
    # ════════════════════════════════════════════════════════════════════
    
    def recommend_friends(
        self,
        user_id: int,
        k: int = 20
    ) -> List[Dict]:
        """
        Recommend potential friends
        
        Args:
            user_id: Target user ID
            k: Number of recommendations
            
        Returns:
            List of friend recommendations
        """
        try:
            recommendations = self.friend_recommender.recommend_friends(
                user_id=user_id,
                k=k
            )
            
            logger.info(f"Friend recommendations for user {user_id}: {len(recommendations)} users")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error recommending friends for user {user_id}: {e}", exc_info=True)
            return []
    
    # ════════════════════════════════════════════════════════════════════
    # REAL-TIME USER EMBEDDING UPDATE
    # ════════════════════════════════════════════════════════════════════
    
    def update_user_embedding_realtime(
        self,
        user_id: int,
        post_id: int,
        action: str
    ):
        """
        Update user embedding in real-time after interaction
        
        Strategy: Incremental weighted average
        new_embedding = (1 - α) × old_embedding + α × post_embedding
        
        Args:
            user_id: User ID
            post_id: Post ID interacted with
            action: Action type (like, comment, share, etc.)
        """
        # Check if real-time update enabled
        if not self.config.get('user_embedding', {}).get('real_time_update', {}).get('enabled', False):
            return
        
        # Check if action triggers update
        trigger_actions = self.config.get('user_embedding', {}).get('real_time_update', {}).get('trigger_actions', [])
        if action not in trigger_actions:
            return
        
        try:
            # Get post embedding
            if post_id not in self.embeddings['post']:
                logger.debug(f"No embedding for post {post_id}")
                return
            
            post_emb = self.embeddings['post'][post_id]
            
            # Get current user embedding
            if user_id in self.embeddings['user']:
                old_emb = self.embeddings['user'][user_id]
            else:
                # First interaction - use post embedding
                old_emb = post_emb
            
            # Learning rate (alpha)
            alpha = self.config.get('user_embedding', {}).get('real_time_update', {}).get('incremental', {}).get('learning_rate', 0.1)
            
            # Action weight
            action_weights = {
                'like': 1.0,
                'comment': 1.5,
                'share': 2.0,
                'save': 1.2
            }
            action_weight = action_weights.get(action, 1.0)
            
            # Weighted alpha
            weighted_alpha = alpha * action_weight
            
            # Update embedding (incremental weighted average)
            new_emb = (1 - weighted_alpha) * old_emb + weighted_alpha * post_emb
            
            # Normalize
            new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)
            
            # Update in memory
            self.embeddings['user'][user_id] = new_emb.astype(np.float32)
            
            # Update Redis cache (async)
            if self.redis is not None:
                try:
                    key = f"user:{user_id}:embedding"
                    value = new_emb.tobytes()
                    self.redis.setex(key, 604800, value)  # 7 days TTL
                except Exception as e:
                    logger.warning(f"Redis update failed: {e}")
            
            logger.debug(f"Updated embedding for user {user_id} after {action} on post {post_id}")
            
        except Exception as e:
            logger.error(f"Error updating user embedding: {e}")
    
    # ════════════════════════════════════════════════════════════════════
    # METRICS & MONITORING
    # ════════════════════════════════════════════════════════════════════
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        if not self.metrics['total_latency']:
            return {
                'total_requests': 0,
                'avg_latency_ms': 0,
                'p50_latency_ms': 0,
                'p95_latency_ms': 0,
                'p99_latency_ms': 0
            }
        
        latencies = self.metrics['total_latency']
        
        return {
            'total_requests': len(latencies),
            'avg_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'avg_recall_latency_ms': np.mean(self.metrics['recall_latency']),
            'avg_ranking_latency_ms': np.mean(self.metrics['ranking_latency']),
            'avg_reranking_latency_ms': np.mean(self.metrics['reranking_latency'])
        }
    
    def print_metrics(self):
        """Print performance metrics"""
        metrics = self.get_metrics()
        
        print("\n" + "="*70)
        print("PERFORMANCE METRICS")
        print("="*70)
        print(f"Total requests: {metrics['total_requests']}")
        print(f"\nLatency:")
        print(f"  Avg: {metrics['avg_latency_ms']:.1f}ms")
        print(f"  P50: {metrics['p50_latency_ms']:.1f}ms")
        print(f"  P95: {metrics['p95_latency_ms']:.1f}ms")
        print(f"  P99: {metrics['p99_latency_ms']:.1f}ms")
        print(f"\nStage breakdown:")
        print(f"  Recall: {metrics['avg_recall_latency_ms']:.1f}ms")
        print(f"  Ranking: {metrics['avg_ranking_latency_ms']:.1f}ms")
        print(f"  Reranking: {metrics['avg_reranking_latency_ms']:.1f}ms")
        print("="*70 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Test online pipeline"""
    
    # Initialize
    print("Initializing pipeline...")
    pipeline = OnlineInferencePipeline(
        config_path='configs/config_online.yaml',
        models_dir='models',
        data_dir='dataset',
        use_redis=False  # Disable Redis for testing
    )
    
    # Test feed generation
    print("\nTesting feed generation...")
    feed = pipeline.generate_feed(user_id=1, limit=50)
    
    print(f"\nGenerated feed: {len(feed)} posts")
    print(f"Sample: {feed[:5]}")
    
    # Test friend recommendation
    print("\nTesting friend recommendation...")
    friends = pipeline.recommend_friends(user_id=1, k=20)
    
    print(f"\nRecommended friends: {len(friends)}")
    print(f"Sample: {friends[:5]}")
    
    # Print metrics
    pipeline.print_metrics()



# """
# ONLINE INFERENCE PIPELINE - INTEGRATED WITH NEW OFFLINE ARTIFACTS
# ==================================================================
# Complete online serving pipeline using versioned artifacts

# Architecture:
# - Load models from ArtifactManager (latest version)
# - Multi-channel recall (Following, CF, Content, Trending)
# - Feature extraction (47 features)
# - ML Ranking (LightGBM)
# - Re-ranking & Business rules

# Target: < 200ms end-to-end latency
# """

# import os
# import sys
# import time
# import numpy as np
# import pandas as pd
# from typing import List, Dict, Optional, Tuple
# from datetime import datetime, timedelta
# from pathlib import Path
# import logging
# import warnings

# warnings.filterwarnings('ignore')

# # Add project root
# sys.path.append(str(Path(__file__).parent.parent.parent))

# # Try imports
# try:
#     from recommender.offline.artifact_manager import ArtifactManager
#     ARTIFACT_MANAGER_AVAILABLE = True
# except ImportError:
#     ARTIFACT_MANAGER_AVAILABLE = False
#     print("⚠️  ArtifactManager not available")

# try:
#     from recommender.common.feature_engineer import FeatureEngineer
#     FEATURE_ENGINEER_AVAILABLE = True
# except ImportError:
#     FEATURE_ENGINEER_AVAILABLE = False
#     print("⚠️  FeatureEngineer not available")

# try:
#     import redis
#     REDIS_AVAILABLE = True
# except ImportError:
#     REDIS_AVAILABLE = False
#     print("⚠️  redis not available. Install: pip install redis")

# try:
#     import lightgbm as lgb
#     LIGHTGBM_AVAILABLE = True
# except ImportError:
#     LIGHTGBM_AVAILABLE = False
#     print("⚠️  lightgbm not available")

# try:
#     import faiss
#     FAISS_AVAILABLE = True
# except ImportError:
#     FAISS_AVAILABLE = False
#     print("⚠️  faiss not available")


# from recommender.online.recall.following_recall import FollowingRecall
# from recommender.online.recall.base_recall import BaseRecall


# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# logger = logging.getLogger(__name__)


# class OnlineInferencePipeline:
#     """
#     Complete online inference pipeline
    
#     Features:
#     - Load from versioned artifacts (ArtifactManager)
#     - Multi-channel recall
#     - ML ranking with LightGBM
#     - Redis caching
#     - < 200ms latency
#     """
    
#     def __init__(
#         self,
#         models_dir: str = 'models',
#         redis_host: str = 'localhost',
#         redis_port: int = 6380,
#         redis_db: int = 0,
#         use_redis: bool = True
#     ):
#         """
#         Initialize online pipeline
        
#         Args:
#             models_dir: Directory with model artifacts
#             redis_host: Redis host
#             redis_port: Redis port
#             redis_db: Redis database number
#             use_redis: Enable Redis caching
#         """
#         logger.info("="*70)
#         logger.info("INITIALIZING ONLINE INFERENCE PIPELINE")
#         logger.info("="*70)
        
#         self.models_dir = Path(models_dir)
#         self.use_redis = use_redis and REDIS_AVAILABLE
        
#         # Performance metrics
#         self.metrics = {
#             'recall_latency': [],
#             'feature_latency': [],
#             'ranking_latency': [],
#             'reranking_latency': [],
#             'total_latency': []
#         }
        
#         # ════════════════════════════════════════════════════════════════
#         # STEP 1: CONNECT TO REDIS
#         # ════════════════════════════════════════════════════════════════
        
#         if self.use_redis:
#             try:
#                 self.redis_client = redis.Redis(
#                     host=redis_host,
#                     port=redis_port,
#                     db=redis_db,
#                     decode_responses=False  # For binary data
#                 )
#                 # Test connection
#                 self.redis_client.ping()
#                 logger.info(f"✅ Connected to Redis: {redis_host}:{redis_port}/{redis_db}")
#             except Exception as e:
#                 logger.warning(f"⚠️  Redis connection failed: {e}")
#                 logger.warning("⚠️  Falling back to no-cache mode")
#                 self.use_redis = False
#                 self.redis_client = None
#         else:
#             self.redis_client = None
#             logger.info("ℹ️  Redis disabled (using no-cache mode)")
        
#         # ════════════════════════════════════════════════════════════════
#         # STEP 2: LOAD ARTIFACTS FROM LATEST VERSION
#         # ════════════════════════════════════════════════════════════════
        
#         if not ARTIFACT_MANAGER_AVAILABLE:
#             raise ImportError("ArtifactManager required")
        
#         logger.info("\n📦 Loading artifacts...")
        
#         self.artifact_mgr = ArtifactManager(base_dir=self.models_dir)
        
#         # Get latest version
#         try:
#             self.current_version = self.artifact_mgr.get_latest_version()
#             logger.info(f"   Latest version: {self.current_version}")
#         except FileNotFoundError:
#             raise FileNotFoundError(
#                 f"No models found in {self.models_dir}. "
#                 "Run offline training first: python scripts/offline/main_offline_pipeline.py"
#             )
        
#         # Load all artifacts
#         artifacts = self.artifact_mgr.load_artifacts(self.current_version)
        
#         # Extract components
#         self.embeddings = artifacts['embeddings']
#         self.faiss_index = artifacts['faiss_index']
#         self.faiss_post_ids = artifacts['faiss_post_ids']
#         self.cf_model = artifacts['cf_model']
#         self.ranking_model = artifacts['ranking_model']
#         self.ranking_scaler = artifacts['ranking_scaler']
#         self.ranking_feature_cols = artifacts['ranking_feature_cols']
#         self.user_stats = artifacts['user_stats']
#         self.author_stats = artifacts['author_stats']
#         self.following_dict = artifacts['following_dict']
#         self.metadata = artifacts['metadata']
        
#         logger.info("✅ Artifacts loaded successfully!")
#         logger.info(f"   Post embeddings: {len(self.embeddings['post']):,}")
#         logger.info(f"   User embeddings: {len(self.embeddings['user']):,}")
#         logger.info(f"   CF users: {len(self.cf_model['user_ids']):,}")
#         logger.info(f"   Faiss index: {self.faiss_index.ntotal:,} vectors")
#         logger.info(f"   Ranking features: {len(self.ranking_feature_cols)}")
        
#         # ════════════════════════════════════════════════════════════════
#         # STEP 3: INITIALIZE FEATURE ENGINEER
#         # ════════════════════════════════════════════════════════════════
        
#         if not FEATURE_ENGINEER_AVAILABLE:
#             raise ImportError("FeatureEngineer required")
        
#         # Load data (dummy for online - only need structure)
#         # In production, this would connect to live database
#         logger.info("\n🔧 Initializing feature engineer...")
        
#         # Create dummy data structure (will be replaced by Redis/DB in production)
#         self.data = {
#             'user': pd.DataFrame(),
#             'post': pd.DataFrame(),
#             'postreaction': pd.DataFrame(),
#             'friendship': pd.DataFrame()
#         }
        
#         self.feature_engineer = FeatureEngineer(
#             data=self.data,
#             user_stats=self.user_stats,
#             author_stats=self.author_stats,
#             following_dict=self.following_dict,
#             embeddings=self.embeddings
#         )
        
#         logger.info("✅ Feature engineer initialized")
        
#         # ════════════════════════════════════════════════════════════════
#         # COMPLETE
#         # ════════════════════════════════════════════════════════════════
        
#         logger.info("\n✅ ONLINE PIPELINE READY!")
#         logger.info("="*70 + "\n")
    
#     # ════════════════════════════════════════════════════════════════════
#     # MAIN INFERENCE METHOD
#     # ════════════════════════════════════════════════════════════════════
    
#     def get_feed(
#         self,
#         user_id: int,
#         limit: int = 50,
#         exclude_seen: Optional[set] = None
#     ) -> List[Dict]:
#         """
#         Get personalized feed for user
        
#         Args:
#             user_id: Target user ID
#             limit: Number of posts to return
#             exclude_seen: Set of post IDs to exclude
        
#         Returns:
#             List of dicts with post_id, score, metadata
#         """
#         start_time = time.time()
        
#         try:
#             # ════════════════════════════════════════════════════════════
#             # STAGE 1: MULTI-CHANNEL RECALL (~1000 candidates)
#             # ════════════════════════════════════════════════════════════
            
#             t1 = time.time()
#             candidates = self._recall_candidates(user_id, target_count=1000)
#             recall_latency = (time.time() - t1) * 1000
#             self.metrics['recall_latency'].append(recall_latency)
            
#             if not candidates:
#                 logger.warning(f"No candidates found for user {user_id}")
#                 return []
            
#             # Filter out seen posts
#             if exclude_seen:
#                 candidates = [p for p in candidates if p not in exclude_seen]
            
#             logger.debug(f"Recall: {len(candidates)} candidates in {recall_latency:.1f}ms")
            
#             # ════════════════════════════════════════════════════════════
#             # STAGE 2: FEATURE EXTRACTION (47 features × N posts)
#             # ════════════════════════════════════════════════════════════
            
#             t2 = time.time()
#             features_df = self._extract_features_batch(user_id, candidates)
#             feature_latency = (time.time() - t2) * 1000
#             self.metrics['feature_latency'].append(feature_latency)
            
#             if features_df.empty:
#                 logger.warning(f"No features extracted for user {user_id}")
#                 return []
            
#             logger.debug(f"Features: {len(features_df)} posts in {feature_latency:.1f}ms")
            
#             # ════════════════════════════════════════════════════════════
#             # STAGE 3: ML RANKING (LightGBM)
#             # ════════════════════════════════════════════════════════════
            
#             t3 = time.time()
#             features_df = self._rank_candidates(features_df)
#             ranking_latency = (time.time() - t3) * 1000
#             self.metrics['ranking_latency'].append(ranking_latency)
            
#             # Get top 100 for re-ranking
#             features_df = features_df.nlargest(100, 'ml_score')
            
#             logger.debug(f"Ranking: {len(features_df)} posts in {ranking_latency:.1f}ms")
            
#             # ════════════════════════════════════════════════════════════
#             # STAGE 4: RE-RANKING & BUSINESS RULES
#             # ════════════════════════════════════════════════════════════
            
#             t4 = time.time()
#             final_feed = self._rerank_with_business_rules(features_df, limit)
#             reranking_latency = (time.time() - t4) * 1000
#             self.metrics['reranking_latency'].append(reranking_latency)
            
#             logger.debug(f"Reranking: {len(final_feed)} posts in {reranking_latency:.1f}ms")
            
#             # ════════════════════════════════════════════════════════════
#             # METRICS
#             # ════════════════════════════════════════════════════════════
            
#             total_latency = (time.time() - start_time) * 1000
#             self.metrics['total_latency'].append(total_latency)
            
#             logger.info(
#                 f"Feed generated for user {user_id}: "
#                 f"{len(final_feed)} posts | "
#                 f"Latency: {total_latency:.1f}ms "
#                 f"(R:{recall_latency:.0f} + F:{feature_latency:.0f} + "
#                 f"M:{ranking_latency:.0f} + B:{reranking_latency:.0f})"
#             )
            
#             return final_feed
            
#         except Exception as e:
#             logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
#             return []
    
#     # ════════════════════════════════════════════════════════════════════
#     # STAGE 1: MULTI-CHANNEL RECALL
#     # ════════════════════════════════════════════════════════════════════
    
#     def _recall_candidates(self, user_id: int, target_count: int = 1000) -> List[int]:
#         """
#         Multi-channel recall
        
#         Channels:
#         1. Following (400) - Posts from followed users
#         2. CF (300) - Posts liked by similar users
#         3. Content (200) - Similar posts via embeddings
#         4. Trending (100) - Popular posts
        
#         Returns:
#             List of unique post IDs
#         """
#         all_candidates = []
        
#         # Channel 1: Following
#         following_posts = self._recall_following(user_id, k=400)
#         all_candidates.extend(following_posts)
        
#         # Channel 2: Collaborative Filtering
#         cf_posts = self._recall_cf(user_id, k=300)
#         all_candidates.extend(cf_posts)
        
#         # Channel 3: Content-based (Embeddings)
#         content_posts = self._recall_content(user_id, k=200)
#         all_candidates.extend(content_posts)
        
#         # Channel 4: Trending
#         trending_posts = self._recall_trending(k=100)
#         all_candidates.extend(trending_posts)
        
#         # Deduplicate (preserve order)
#         unique_candidates = list(dict.fromkeys(all_candidates))
        
#         return unique_candidates[:target_count]
    
#     def _recall_following(self, user_id: int, k: int) -> List[int]:
#         """Recall posts from followed users"""
#         # Get followed users
#         followed_users = self.following_dict.get(user_id, [])
        
#         if not followed_users:
#             return []
        
#         # In production: Query Redis sorted set
#         # following:{user_id}:posts (sorted by timestamp)
        
#         # For now: Return empty (needs live data)
#         return []
    
#     def _recall_cf(self, user_id: int, k: int) -> List[int]:
#         """Recall via collaborative filtering"""
#         # Get similar users
#         user_similarities = self.cf_model['user_similarities']
#         similar_users = user_similarities.get(user_id, [])
        
#         if not similar_users:
#             return []
        
#         # Get posts liked by similar users
#         # In production: Query from Redis or database
        
#         # For now: Return top K most similar items
#         item_similarities = self.cf_model.get('item_similarities', {})
        
#         # Return first K items (placeholder)
#         all_items = list(item_similarities.keys())[:k] if item_similarities else []
        
#         return all_items
    
#     def _recall_content(self, user_id: int, k: int) -> List[int]:
#         """Recall via content similarity (embeddings)"""
#         # Get user embedding
#         user_emb = self.embeddings['user'].get(user_id)
        
#         if user_emb is None or not FAISS_AVAILABLE:
#             return []
        
#         # Normalize
#         user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
        
#         # Search Faiss
#         distances, indices = self.faiss_index.search(
#             user_emb.reshape(1, -1).astype(np.float32),
#             k * 2  # Over-fetch
#         )
        
#         # Convert indices to post IDs
#         post_ids = [int(self.faiss_post_ids[idx]) for idx in indices[0]]
        
#         return post_ids[:k]
    
#     def _recall_trending(self, k: int) -> List[int]:
#         """Recall trending posts"""
#         # In production: Query Redis sorted set
#         # trending:global:6h (sorted by engagement score)
        
#         # For now: Return empty
#         return []
    
#     # ════════════════════════════════════════════════════════════════════
#     # STAGE 2: FEATURE EXTRACTION
#     # ════════════════════════════════════════════════════════════════════
    
#     def _extract_features_batch(
#         self,
#         user_id: int,
#         post_ids: List[int]
#     ) -> pd.DataFrame:
#         """Extract 47 features for each (user, post) pair"""
#         features_list = []
#         timestamp = datetime.now()
        
#         for post_id in post_ids:
#             try:
#                 features = self.feature_engineer.extract_features(
#                     user_id,
#                     post_id,
#                     timestamp
#                 )
#                 features['post_id'] = post_id
#                 features_list.append(features)
#             except Exception as e:
#                 logger.debug(f"Feature extraction failed for post {post_id}: {e}")
#                 continue
        
#         if not features_list:
#             return pd.DataFrame()
        
#         features_df = pd.DataFrame(features_list)
        
#         # Ensure all features present
#         for col in self.ranking_feature_cols:
#             if col not in features_df.columns:
#                 features_df[col] = 0.0
        
#         return features_df
    
#     # ════════════════════════════════════════════════════════════════════
#     # STAGE 3: ML RANKING
#     # ════════════════════════════════════════════════════════════════════
    
#     def _rank_candidates(self, features_df: pd.DataFrame) -> pd.DataFrame:
#         """Rank using LightGBM model"""
#         # Select features
#         X = features_df[self.ranking_feature_cols].fillna(0)
        
#         # Scale
#         X_scaled = self.ranking_scaler.transform(X)
        
#         # Predict
#         scores = self.ranking_model.predict(X_scaled)
        
#         # Add to dataframe
#         features_df['ml_score'] = scores
        
#         return features_df
    
#     # ════════════════════════════════════════════════════════════════════
#     # STAGE 4: RE-RANKING
#     # ════════════════════════════════════════════════════════════════════
    
#     def _rerank_with_business_rules(
#         self,
#         ranked_df: pd.DataFrame,
#         limit: int
#     ) -> List[Dict]:
#         """Apply business rules and diversity"""
#         # Sort by ML score
#         ranked_df = ranked_df.sort_values('ml_score', ascending=False)
        
#         final_feed = []
#         last_author_id = None
#         same_author_count = 0
        
#         for _, row in ranked_df.iterrows():
#             post_id = int(row['post_id'])
#             score = float(row['ml_score'])
            
#             # Get author (from embeddings or stats)
#             # Placeholder: author_id = post_id % 1000
#             author_id = post_id % 1000
            
#             # Rule 1: Diversity (max 2 consecutive from same author)
#             if author_id == last_author_id:
#                 same_author_count += 1
#                 if same_author_count >= 2:
#                     continue
#             else:
#                 same_author_count = 0
#                 last_author_id = author_id
            
#             # Rule 2: Freshness boost (would need post creation time)
#             # Placeholder: No boost
            
#             final_feed.append({
#                 'post_id': post_id,
#                 'score': score,
#                 'author_id': author_id
#             })
            
#             if len(final_feed) >= limit:
#                 break
        
#         return final_feed
    
#     # ════════════════════════════════════════════════════════════════════
#     # UTILITIES
#     # ════════════════════════════════════════════════════════════════════
    
#     def get_metrics_summary(self) -> Dict:
#         """Get performance metrics summary"""
#         if not self.metrics['total_latency']:
#             return {}
        
#         summary = {}
#         for key, values in self.metrics.items():
#             if values:
#                 summary[key] = {
#                     'mean': np.mean(values),
#                     'p50': np.percentile(values, 50),
#                     'p95': np.percentile(values, 95),
#                     'p99': np.percentile(values, 99),
#                     'min': np.min(values),
#                     'max': np.max(values)
#                 }
        
#         return summary
    
#     def print_metrics(self):
#         """Print performance metrics"""
#         summary = self.get_metrics_summary()
        
#         print("\n" + "="*70)
#         print("ONLINE PIPELINE PERFORMANCE METRICS")
#         print("="*70)
        
#         for metric, stats in summary.items():
#             print(f"\n{metric}:")
#             print(f"  Mean:  {stats['mean']:.1f}ms")
#             print(f"  P50:   {stats['p50']:.1f}ms")
#             print(f"  P95:   {stats['p95']:.1f}ms")
#             print(f"  P99:   {stats['p99']:.1f}ms")
#             print(f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]ms")
        
#         print("="*70 + "\n")


# # ════════════════════════════════════════════════════════════════════════
# # MAIN TESTING
# # ════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     """
#     Test online inference pipeline
#     """
#     # Initialize pipeline
#     pipeline = OnlineInferencePipeline(
#         models_dir='models',
#         use_redis=False  # Disable Redis for testing
#     )
    
#     # Test with a few users
#     test_user_ids = [1, 2, 3, 4, 5]
    
#     print("\n" + "="*70)
#     print("TESTING INFERENCE")
#     print("="*70 + "\n")
    
#     for user_id in test_user_ids:
#         feed = pipeline.get_feed(user_id, limit=50)
#         print(f"User {user_id}: Generated {len(feed)} posts")
    
#     # Print metrics
#     pipeline.print_metrics()