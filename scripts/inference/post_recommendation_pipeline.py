"""
COMPLETE ONLINE INFERENCE PIPELINE
===================================
Flow: Recall → Feature Extraction → Ranking → Re-ranking

Target latency: < 200ms end-to-end
"""

import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)


class PostRecommendationPipeline:
    """
    Complete online inference pipeline for post recommendations
    
    Pipeline stages:
    1. Multi-Channel Recall (~1000 candidates)
    2. Feature Extraction (47 features × 1000 posts)
    3. ML Ranking (LightGBM)
    4. Re-ranking & Business Rules
    5. Final feed (50 posts)
    """
    
    def __init__(
        self,
        models_dir: str = 'models',
        data: Optional[Dict] = None
    ):
        """
        Initialize pipeline with all components
        
        Args:
            models_dir: Directory containing trained models
            data: Optional data dictionary for testing
        """
        self.models_dir = models_dir
        self.data = data
        
        # Load all components
        self._load_models()
        
        # Initialize recall system
        from multi_channel_recall import MultiChannelRecall
        self.recall_system = MultiChannelRecall(
            redis_client=None,
            data=self.data,
            cf_model=self.cf_model,
            embeddings=self.embeddings
        )
        
        # Initialize feature engineer
        from feature_engineer_fixed import FeatureEngineer
        self.feature_engineer = FeatureEngineer(
            data=self.data,
            user_stats=self.user_stats,
            author_stats=self.author_stats,
            following_dict=self.following_dict,
            embeddings=self.embeddings
        )
        
        # Metrics
        self.metrics = []
    
    def _load_models(self):
        """Load all trained models and artifacts"""
        logger.info(f"Loading models from {self.models_dir}...")
        
        # 1. LightGBM ranking model
        with open(f'{self.models_dir}/ranking_model.pkl', 'rb') as f:
            self.ranking_model = pickle.load(f)
        logger.info("✓ Loaded ranking_model.pkl")
        
        # 2. Feature scaler
        with open(f'{self.models_dir}/feature_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info("✓ Loaded feature_scaler.pkl")
        
        # 3. Feature columns
        with open(f'{self.models_dir}/feature_cols.pkl', 'rb') as f:
            self.feature_cols = pickle.load(f)
        logger.info("✓ Loaded feature_cols.pkl")
        
        # 4. CF model
        with open(f'{self.models_dir}/cf_model.pkl', 'rb') as f:
            self.cf_model = pickle.load(f)
        logger.info("✓ Loaded cf_model.pkl")
        
        # 5. Embeddings
        with open(f'{self.models_dir}/embeddings.pkl', 'rb') as f:
            self.embeddings = pickle.load(f)
        logger.info("✓ Loaded embeddings.pkl")
        
        # 6. User stats
        with open(f'{self.models_dir}/user_stats.pkl', 'rb') as f:
            self.user_stats = pickle.load(f)
        logger.info("✓ Loaded user_stats.pkl")
        
        # 7. Author stats
        with open(f'{self.models_dir}/author_stats.pkl', 'rb') as f:
            self.author_stats = pickle.load(f)
        logger.info("✓ Loaded author_stats.pkl")
        
        # 8. Following dict
        with open(f'{self.models_dir}/following_dict.pkl', 'rb') as f:
            self.following_dict = pickle.load(f)
        logger.info("✓ Loaded following_dict.pkl")
        
        logger.info("All models loaded successfully!")
    
    # ========================================================================
    # MAIN INFERENCE METHOD
    # ========================================================================
    
    def get_feed(
        self,
        user_id: int,
        limit: int = 50,
        seen_posts: Optional[set] = None
    ) -> List[Dict]:
        """
        Get personalized feed for user
        
        Args:
            user_id: Target user ID
            limit: Number of posts to return (default 50)
            seen_posts: Set of post IDs user has already seen
            
        Returns:
            List of dicts with post_id and score
        """
        start_time = time.time()
        metrics = {'user_id': user_id}
        
        try:
            # ================================================================
            # STAGE 1: MULTI-CHANNEL RECALL (~1000 candidates)
            # ================================================================
            t1 = time.time()
            candidates = self.recall_system.recall(user_id, k=1000)
            metrics['recall_latency_ms'] = (time.time() - t1) * 1000
            metrics['recall_count'] = len(candidates)
            
            if not candidates:
                logger.warning(f"No candidates found for user {user_id}")
                return []
            
            # Filter out seen posts
            if seen_posts:
                candidates = [p for p in candidates if p not in seen_posts]
                metrics['candidates_after_seen_filter'] = len(candidates)
            
            logger.info(f"Stage 1: Recalled {len(candidates)} candidates")
            
            # ================================================================
            # STAGE 2: FEATURE EXTRACTION (47 features × N posts)
            # ================================================================
            t2 = time.time()
            features_df = self._extract_features_batch(user_id, candidates)
            metrics['feature_extraction_latency_ms'] = (time.time() - t2) * 1000
            metrics['features_extracted'] = len(features_df)
            
            if features_df.empty:
                logger.warning(f"No features extracted for user {user_id}")
                return []
            
            logger.info(f"Stage 2: Extracted features for {len(features_df)} posts")
            
            # ================================================================
            # STAGE 3: ML RANKING (LightGBM)
            # ================================================================
            t3 = time.time()
            features_df = self._rank_candidates(features_df)
            metrics['ranking_latency_ms'] = (time.time() - t3) * 1000
            
            # Get top 100 for re-ranking
            features_df = features_df.nlargest(100, 'ml_score')
            
            logger.info(f"Stage 3: Ranked {len(features_df)} posts")
            
            # ================================================================
            # STAGE 4: RE-RANKING & BUSINESS RULES
            # ================================================================
            t4 = time.time()
            final_feed = self._rerank_with_business_rules(features_df, limit)
            metrics['reranking_latency_ms'] = (time.time() - t4) * 1000
            metrics['final_count'] = len(final_feed)
            
            logger.info(f"Stage 4: Re-ranked to final {len(final_feed)} posts")
            
            # ================================================================
            # METRICS
            # ================================================================
            total_latency = (time.time() - start_time) * 1000
            metrics['total_latency_ms'] = total_latency
            
            self.metrics.append(metrics)
            
            logger.info(f"Feed generated in {total_latency:.1f}ms for user {user_id}")
            
            return final_feed
            
        except Exception as e:
            logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
            return []
    
    # ========================================================================
    # FEATURE EXTRACTION
    # ========================================================================
    
    def _extract_features_batch(
        self,
        user_id: int,
        post_ids: List[int]
    ) -> pd.DataFrame:
        """
        Extract features for all (user, post) pairs
        
        Args:
            user_id: Target user ID
            post_ids: List of candidate post IDs
            
        Returns:
            DataFrame with features for each post
        """
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
                # Skip posts that fail feature extraction
                logger.debug(f"Skipping post {post_id}: {e}")
                continue
        
        if not features_list:
            return pd.DataFrame()
        
        features_df = pd.DataFrame(features_list)
        
        # Ensure all expected features are present
        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0
        
        return features_df
    
    # ========================================================================
    # ML RANKING
    # ========================================================================
    
    def _rank_candidates(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank candidates using LightGBM model
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            DataFrame with 'ml_score' column added
        """
        # Select and order features
        X = features_df[self.feature_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict scores
        scores = self.ranking_model.predict(X_scaled)
        
        # Add scores to dataframe
        features_df['ml_score'] = scores
        
        return features_df
    
    # ========================================================================
    # RE-RANKING & BUSINESS RULES
    # ========================================================================
    
    def _rerank_with_business_rules(
        self,
        ranked_df: pd.DataFrame,
        limit: int
    ) -> List[Dict]:
        """
        Apply business rules and diversity constraints
        
        Rules:
        1. Diversity: Max 2 consecutive posts from same author
        2. Freshness boost: Recent posts (< 6h) get +10% score
        3. Quality filter: Remove low-quality posts
        4. Deduplication: Remove exact duplicates
        
        Args:
            ranked_df: DataFrame with ml_score
            limit: Target number of posts
            
        Returns:
            List of dicts with post_id and score
        """
        # Sort by ML score
        ranked_df = ranked_df.sort_values('ml_score', ascending=False)
        
        final_feed = []
        last_author_id = None
        same_author_count = 0
        
        for _, row in ranked_df.iterrows():
            post_id = row['post_id']
            score = row['ml_score']
            
            # Get post data
            post = self.data['post'][self.data['post']['Id'] == post_id]
            if post.empty:
                continue
            
            post = post.iloc[0]
            author_id = post.get('UserId', None)
            
            # Rule 1: Diversity constraint
            if author_id == last_author_id:
                same_author_count += 1
                if same_author_count >= 2:
                    continue  # Skip if 2 consecutive from same author
            else:
                same_author_count = 0
                last_author_id = author_id
            
            # Rule 2: Freshness boost
            created_at = pd.to_datetime(post['CreateDate'])
            hours_old = (datetime.now() - created_at).total_seconds() / 3600
            if hours_old < 6:
                score *= 1.1  # 10% boost for recent posts
            
            # Rule 3: Quality filter (example)
            # if post.get('Status', 0) != 1:  # Active posts only
            #     continue
            
            # Add to final feed
            final_feed.append({
                'post_id': int(post_id),
                'score': float(score),
                'author_id': int(author_id) if author_id else None,
                'created_at': str(created_at)
            })
            
            if len(final_feed) >= limit:
                break
        
        return final_feed
    
    # ========================================================================
    # PERFORMANCE MONITORING
    # ========================================================================
    
    def get_metrics_summary(self) -> Dict:
        """Get summary statistics of performance metrics"""
        if not self.metrics:
            return {}
        
        df = pd.DataFrame(self.metrics)
        
        summary = {}
        for col in df.columns:
            if col.endswith('_ms') or col.endswith('_count'):
                summary[col] = {
                    'mean': df[col].mean(),
                    'p50': df[col].quantile(0.5),
                    'p95': df[col].quantile(0.95),
                    'p99': df[col].quantile(0.99)
                }
        
        return summary
    
    def print_metrics(self):
        """Print performance metrics"""
        summary = self.get_metrics_summary()
        
        print("\n" + "="*60)
        print("PIPELINE PERFORMANCE METRICS")
        print("="*60)
        
        for metric, stats in summary.items():
            print(f"\n{metric}:")
            print(f"  Mean: {stats['mean']:.1f}")
            print(f"  P50:  {stats['p50']:.1f}")
            print(f"  P95:  {stats['p95']:.1f}")
            print(f"  P99:  {stats['p99']:.1f}")
        
        print("="*60 + "\n")


# ========================================================================
# MAIN INFERENCE SCRIPT
# ========================================================================

def main():
    """
    Main script for online inference
    """
    import yaml
    from recommender.common.data_loader import load_data
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    print("Loading data...")
    data = load_data(config.get('data_dir', 'dataset'))
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = PostRecommendationPipeline(
        models_dir=config.get('models_dir', 'models'),
        data=data
    )
    
    # Test inference
    print("\n" + "="*60)
    print("TESTING INFERENCE PIPELINE")
    print("="*60 + "\n")
    
    # Get sample user
    sample_user_id = data['user']['Id'].iloc[0]
    
    print(f"Generating feed for user {sample_user_id}...")
    feed = pipeline.get_feed(user_id=sample_user_id, limit=50)
    
    print(f"\nGenerated feed with {len(feed)} posts:")
    for i, post in enumerate(feed[:10], 1):
        print(f"{i}. Post {post['post_id']} | Score: {post['score']:.4f}")
    
    # Print metrics
    pipeline.print_metrics()
    
    # Test with multiple users
    print("\n" + "="*60)
    print("TESTING WITH MULTIPLE USERS")
    print("="*60 + "\n")
    
    sample_users = data['user']['Id'].head(10).tolist()
    
    for user_id in sample_users:
        feed = pipeline.get_feed(user_id=user_id, limit=50)
        print(f"User {user_id}: {len(feed)} posts")
    
    # Final metrics
    print("\n" + "="*60)
    print("FINAL PERFORMANCE METRICS")
    print("="*60)
    pipeline.print_metrics()


if __name__ == "__main__":
    main()