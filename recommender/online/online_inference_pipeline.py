"""
ONLINE INFERENCE PIPELINE - INTEGRATED WITH NEW OFFLINE ARTIFACTS
==================================================================
Complete online serving pipeline using versioned artifacts

Architecture:
- Load models from ArtifactManager (latest version)
- Multi-channel recall (Following, CF, Content, Trending)
- Feature extraction (47 features)
- ML Ranking (LightGBM)
- Re-ranking & Business rules

Target: < 200ms end-to-end latency
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

# Try imports
try:
    from recommender.offline.artifact_manager import ArtifactManager
    ARTIFACT_MANAGER_AVAILABLE = True
except ImportError:
    ARTIFACT_MANAGER_AVAILABLE = False
    print("⚠️  ArtifactManager not available")

try:
    from recommender.common.feature_engineer import FeatureEngineer
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEER_AVAILABLE = False
    print("⚠️  FeatureEngineer not available")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️  redis not available. Install: pip install redis")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  lightgbm not available")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  faiss not available")


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class OnlineInferencePipeline:
    """
    Complete online inference pipeline
    
    Features:
    - Load from versioned artifacts (ArtifactManager)
    - Multi-channel recall
    - ML ranking with LightGBM
    - Redis caching
    - < 200ms latency
    """
    
    def __init__(
        self,
        models_dir: str = 'models',
        redis_host: str = 'localhost',
        redis_port: int = 6380,
        redis_db: int = 0,
        use_redis: bool = True
    ):
        """
        Initialize online pipeline
        
        Args:
            models_dir: Directory with model artifacts
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database number
            use_redis: Enable Redis caching
        """
        logger.info("="*70)
        logger.info("INITIALIZING ONLINE INFERENCE PIPELINE")
        logger.info("="*70)
        
        self.models_dir = Path(models_dir)
        self.use_redis = use_redis and REDIS_AVAILABLE
        
        # Performance metrics
        self.metrics = {
            'recall_latency': [],
            'feature_latency': [],
            'ranking_latency': [],
            'reranking_latency': [],
            'total_latency': []
        }
        
        # ════════════════════════════════════════════════════════════════
        # STEP 1: CONNECT TO REDIS
        # ════════════════════════════════════════════════════════════════
        
        if self.use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=False  # For binary data
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"✅ Connected to Redis: {redis_host}:{redis_port}/{redis_db}")
            except Exception as e:
                logger.warning(f"⚠️  Redis connection failed: {e}")
                logger.warning("⚠️  Falling back to no-cache mode")
                self.use_redis = False
                self.redis_client = None
        else:
            self.redis_client = None
            logger.info("ℹ️  Redis disabled (using no-cache mode)")
        
        # ════════════════════════════════════════════════════════════════
        # STEP 2: LOAD ARTIFACTS FROM LATEST VERSION
        # ════════════════════════════════════════════════════════════════
        
        if not ARTIFACT_MANAGER_AVAILABLE:
            raise ImportError("ArtifactManager required")
        
        logger.info("\n📦 Loading artifacts...")
        
        self.artifact_mgr = ArtifactManager(base_dir=self.models_dir)
        
        # Get latest version
        try:
            self.current_version = self.artifact_mgr.get_latest_version()
            logger.info(f"   Latest version: {self.current_version}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No models found in {self.models_dir}. "
                "Run offline training first: python scripts/offline/main_offline_pipeline.py"
            )
        
        # Load all artifacts
        artifacts = self.artifact_mgr.load_artifacts(self.current_version)
        
        # Extract components
        self.embeddings = artifacts['embeddings']
        self.faiss_index = artifacts['faiss_index']
        self.faiss_post_ids = artifacts['faiss_post_ids']
        self.cf_model = artifacts['cf_model']
        self.ranking_model = artifacts['ranking_model']
        self.ranking_scaler = artifacts['ranking_scaler']
        self.ranking_feature_cols = artifacts['ranking_feature_cols']
        self.user_stats = artifacts['user_stats']
        self.author_stats = artifacts['author_stats']
        self.following_dict = artifacts['following_dict']
        self.metadata = artifacts['metadata']
        
        logger.info("✅ Artifacts loaded successfully!")
        logger.info(f"   Post embeddings: {len(self.embeddings['post']):,}")
        logger.info(f"   User embeddings: {len(self.embeddings['user']):,}")
        logger.info(f"   CF users: {len(self.cf_model['user_ids']):,}")
        logger.info(f"   Faiss index: {self.faiss_index.ntotal:,} vectors")
        logger.info(f"   Ranking features: {len(self.ranking_feature_cols)}")
        
        # ════════════════════════════════════════════════════════════════
        # STEP 3: INITIALIZE FEATURE ENGINEER
        # ════════════════════════════════════════════════════════════════
        
        if not FEATURE_ENGINEER_AVAILABLE:
            raise ImportError("FeatureEngineer required")
        
        # Load data (dummy for online - only need structure)
        # In production, this would connect to live database
        logger.info("\n🔧 Initializing feature engineer...")
        
        # Create dummy data structure (will be replaced by Redis/DB in production)
        self.data = {
            'user': pd.DataFrame(),
            'post': pd.DataFrame(),
            'postreaction': pd.DataFrame(),
            'friendship': pd.DataFrame()
        }
        
        self.feature_engineer = FeatureEngineer(
            data=self.data,
            user_stats=self.user_stats,
            author_stats=self.author_stats,
            following_dict=self.following_dict,
            embeddings=self.embeddings
        )
        
        logger.info("✅ Feature engineer initialized")
        
        # ════════════════════════════════════════════════════════════════
        # COMPLETE
        # ════════════════════════════════════════════════════════════════
        
        logger.info("\n✅ ONLINE PIPELINE READY!")
        logger.info("="*70 + "\n")
    
    # ════════════════════════════════════════════════════════════════════
    # MAIN INFERENCE METHOD
    # ════════════════════════════════════════════════════════════════════
    
    def get_feed(
        self,
        user_id: int,
        limit: int = 50,
        exclude_seen: Optional[set] = None
    ) -> List[Dict]:
        """
        Get personalized feed for user
        
        Args:
            user_id: Target user ID
            limit: Number of posts to return
            exclude_seen: Set of post IDs to exclude
        
        Returns:
            List of dicts with post_id, score, metadata
        """
        start_time = time.time()
        
        try:
            # ════════════════════════════════════════════════════════════
            # STAGE 1: MULTI-CHANNEL RECALL (~1000 candidates)
            # ════════════════════════════════════════════════════════════
            
            t1 = time.time()
            candidates = self._recall_candidates(user_id, target_count=1000)
            recall_latency = (time.time() - t1) * 1000
            self.metrics['recall_latency'].append(recall_latency)
            
            if not candidates:
                logger.warning(f"No candidates found for user {user_id}")
                return []
            
            # Filter out seen posts
            if exclude_seen:
                candidates = [p for p in candidates if p not in exclude_seen]
            
            logger.debug(f"Recall: {len(candidates)} candidates in {recall_latency:.1f}ms")
            
            # ════════════════════════════════════════════════════════════
            # STAGE 2: FEATURE EXTRACTION (47 features × N posts)
            # ════════════════════════════════════════════════════════════
            
            t2 = time.time()
            features_df = self._extract_features_batch(user_id, candidates)
            feature_latency = (time.time() - t2) * 1000
            self.metrics['feature_latency'].append(feature_latency)
            
            if features_df.empty:
                logger.warning(f"No features extracted for user {user_id}")
                return []
            
            logger.debug(f"Features: {len(features_df)} posts in {feature_latency:.1f}ms")
            
            # ════════════════════════════════════════════════════════════
            # STAGE 3: ML RANKING (LightGBM)
            # ════════════════════════════════════════════════════════════
            
            t3 = time.time()
            features_df = self._rank_candidates(features_df)
            ranking_latency = (time.time() - t3) * 1000
            self.metrics['ranking_latency'].append(ranking_latency)
            
            # Get top 100 for re-ranking
            features_df = features_df.nlargest(100, 'ml_score')
            
            logger.debug(f"Ranking: {len(features_df)} posts in {ranking_latency:.1f}ms")
            
            # ════════════════════════════════════════════════════════════
            # STAGE 4: RE-RANKING & BUSINESS RULES
            # ════════════════════════════════════════════════════════════
            
            t4 = time.time()
            final_feed = self._rerank_with_business_rules(features_df, limit)
            reranking_latency = (time.time() - t4) * 1000
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
                f"(R:{recall_latency:.0f} + F:{feature_latency:.0f} + "
                f"M:{ranking_latency:.0f} + B:{reranking_latency:.0f})"
            )
            
            return final_feed
            
        except Exception as e:
            logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
            return []
    
    # ════════════════════════════════════════════════════════════════════
    # STAGE 1: MULTI-CHANNEL RECALL
    # ════════════════════════════════════════════════════════════════════
    
    def _recall_candidates(self, user_id: int, target_count: int = 1000) -> List[int]:
        """
        Multi-channel recall
        
        Channels:
        1. Following (400) - Posts from followed users
        2. CF (300) - Posts liked by similar users
        3. Content (200) - Similar posts via embeddings
        4. Trending (100) - Popular posts
        
        Returns:
            List of unique post IDs
        """
        all_candidates = []
        
        # Channel 1: Following
        following_posts = self._recall_following(user_id, k=400)
        all_candidates.extend(following_posts)
        
        # Channel 2: Collaborative Filtering
        cf_posts = self._recall_cf(user_id, k=300)
        all_candidates.extend(cf_posts)
        
        # Channel 3: Content-based (Embeddings)
        content_posts = self._recall_content(user_id, k=200)
        all_candidates.extend(content_posts)
        
        # Channel 4: Trending
        trending_posts = self._recall_trending(k=100)
        all_candidates.extend(trending_posts)
        
        # Deduplicate (preserve order)
        unique_candidates = list(dict.fromkeys(all_candidates))
        
        return unique_candidates[:target_count]
    
    def _recall_following(self, user_id: int, k: int) -> List[int]:
        """Recall posts from followed users"""
        # Get followed users
        followed_users = self.following_dict.get(user_id, [])
        
        if not followed_users:
            return []
        
        # In production: Query Redis sorted set
        # following:{user_id}:posts (sorted by timestamp)
        
        # For now: Return empty (needs live data)
        return []
    
    def _recall_cf(self, user_id: int, k: int) -> List[int]:
        """Recall via collaborative filtering"""
        # Get similar users
        user_similarities = self.cf_model['user_similarities']
        similar_users = user_similarities.get(user_id, [])
        
        if not similar_users:
            return []
        
        # Get posts liked by similar users
        # In production: Query from Redis or database
        
        # For now: Return top K most similar items
        item_similarities = self.cf_model.get('item_similarities', {})
        
        # Return first K items (placeholder)
        all_items = list(item_similarities.keys())[:k] if item_similarities else []
        
        return all_items
    
    def _recall_content(self, user_id: int, k: int) -> List[int]:
        """Recall via content similarity (embeddings)"""
        # Get user embedding
        user_emb = self.embeddings['user'].get(user_id)
        
        if user_emb is None or not FAISS_AVAILABLE:
            return []
        
        # Normalize
        user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
        
        # Search Faiss
        distances, indices = self.faiss_index.search(
            user_emb.reshape(1, -1).astype(np.float32),
            k * 2  # Over-fetch
        )
        
        # Convert indices to post IDs
        post_ids = [int(self.faiss_post_ids[idx]) for idx in indices[0]]
        
        return post_ids[:k]
    
    def _recall_trending(self, k: int) -> List[int]:
        """Recall trending posts"""
        # In production: Query Redis sorted set
        # trending:global:6h (sorted by engagement score)
        
        # For now: Return empty
        return []
    
    # ════════════════════════════════════════════════════════════════════
    # STAGE 2: FEATURE EXTRACTION
    # ════════════════════════════════════════════════════════════════════
    
    def _extract_features_batch(
        self,
        user_id: int,
        post_ids: List[int]
    ) -> pd.DataFrame:
        """Extract 47 features for each (user, post) pair"""
        features_list = []
        timestamp = datetime.now()
        
        for post_id in post_ids:
            try:
                features = self.feature_engineer.extract_features(
                    user_id,
                    post_id,
                    timestamp
                )
                features['post_id'] = post_id
                features_list.append(features)
            except Exception as e:
                logger.debug(f"Feature extraction failed for post {post_id}: {e}")
                continue
        
        if not features_list:
            return pd.DataFrame()
        
        features_df = pd.DataFrame(features_list)
        
        # Ensure all features present
        for col in self.ranking_feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0
        
        return features_df
    
    # ════════════════════════════════════════════════════════════════════
    # STAGE 3: ML RANKING
    # ════════════════════════════════════════════════════════════════════
    
    def _rank_candidates(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Rank using LightGBM model"""
        # Select features
        X = features_df[self.ranking_feature_cols].fillna(0)
        
        # Scale
        X_scaled = self.ranking_scaler.transform(X)
        
        # Predict
        scores = self.ranking_model.predict(X_scaled)
        
        # Add to dataframe
        features_df['ml_score'] = scores
        
        return features_df
    
    # ════════════════════════════════════════════════════════════════════
    # STAGE 4: RE-RANKING
    # ════════════════════════════════════════════════════════════════════
    
    def _rerank_with_business_rules(
        self,
        ranked_df: pd.DataFrame,
        limit: int
    ) -> List[Dict]:
        """Apply business rules and diversity"""
        # Sort by ML score
        ranked_df = ranked_df.sort_values('ml_score', ascending=False)
        
        final_feed = []
        last_author_id = None
        same_author_count = 0
        
        for _, row in ranked_df.iterrows():
            post_id = int(row['post_id'])
            score = float(row['ml_score'])
            
            # Get author (from embeddings or stats)
            # Placeholder: author_id = post_id % 1000
            author_id = post_id % 1000
            
            # Rule 1: Diversity (max 2 consecutive from same author)
            if author_id == last_author_id:
                same_author_count += 1
                if same_author_count >= 2:
                    continue
            else:
                same_author_count = 0
                last_author_id = author_id
            
            # Rule 2: Freshness boost (would need post creation time)
            # Placeholder: No boost
            
            final_feed.append({
                'post_id': post_id,
                'score': score,
                'author_id': author_id
            })
            
            if len(final_feed) >= limit:
                break
        
        return final_feed
    
    # ════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ════════════════════════════════════════════════════════════════════
    
    def get_metrics_summary(self) -> Dict:
        """Get performance metrics summary"""
        if not self.metrics['total_latency']:
            return {}
        
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary
    
    def print_metrics(self):
        """Print performance metrics"""
        summary = self.get_metrics_summary()
        
        print("\n" + "="*70)
        print("ONLINE PIPELINE PERFORMANCE METRICS")
        print("="*70)
        
        for metric, stats in summary.items():
            print(f"\n{metric}:")
            print(f"  Mean:  {stats['mean']:.1f}ms")
            print(f"  P50:   {stats['p50']:.1f}ms")
            print(f"  P95:   {stats['p95']:.1f}ms")
            print(f"  P99:   {stats['p99']:.1f}ms")
            print(f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]ms")
        
        print("="*70 + "\n")


# ════════════════════════════════════════════════════════════════════════
# MAIN TESTING
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Test online inference pipeline
    """
    # Initialize pipeline
    pipeline = OnlineInferencePipeline(
        models_dir='models',
        use_redis=False  # Disable Redis for testing
    )
    
    # Test with a few users
    test_user_ids = [1, 2, 3, 4, 5]
    
    print("\n" + "="*70)
    print("TESTING INFERENCE")
    print("="*70 + "\n")
    
    for user_id in test_user_ids:
        feed = pipeline.get_feed(user_id, limit=50)
        print(f"User {user_id}: Generated {len(feed)} posts")
    
    # Print metrics
    pipeline.print_metrics()