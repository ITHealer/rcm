"""
ML RANKER
=========
LightGBM-based ranking for candidate posts

Features:
- Load trained LightGBM model
- Batch feature extraction
- Fast inference (< 50ms for 1000 posts)
- Score normalization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
import time
import pickle

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ lightgbm not available")

logger = logging.getLogger(__name__)


class MLRanker:
    """
    ML-based ranker using LightGBM
    """
    
    def __init__(
        self,
        model,
        scaler,
        feature_cols: List[str],
        feature_engineer,
        config: Optional[Dict] = None
    ):
        """
        Initialize ranker
        
        Args:
            model: Trained LightGBM model
            scaler: Feature scaler (StandardScaler)
            feature_cols: List of feature column names
            feature_engineer: FeatureEngineer instance
            config: Configuration dict
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm required for MLRanker")
        
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.feature_engineer = feature_engineer
        self.config = config or {}
        
        # Metrics
        self.metrics = {
            'latency': [],
            'batch_size': []
        }
        
        logger.info(f"MLRanker initialized with {len(feature_cols)} features")
    
    def rank(
        self,
        user_id: int,
        post_ids: List[int]
    ) -> pd.DataFrame:
        """
        Rank posts using ML model
        
        Args:
            user_id: Target user ID
            post_ids: List of candidate post IDs
            
        Returns:
            DataFrame with columns: [post_id, ml_score]
            Sorted by ml_score (descending)
        """
        start_time = time.time()
        
        # Extract features for all candidates
        features_df = self._extract_features_batch(user_id, post_ids)
        
        if features_df.empty:
            logger.warning(f"No features extracted for user {user_id}")
            return pd.DataFrame(columns=['post_id', 'ml_score'])
        
        # Select and order features
        X = features_df[self.feature_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict scores
        scores = self.model.predict(X_scaled)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'post_id': features_df['post_id'].values,
            'ml_score': scores
        })
        
        # Sort by score (descending)
        result = result.sort_values('ml_score', ascending=False)
        
        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        self.metrics['latency'].append(latency_ms)
        self.metrics['batch_size'].append(len(post_ids))
        
        logger.debug(
            f"Ranked {len(post_ids)} posts in {latency_ms:.1f}ms"
        )
        
        return result
    
    def _extract_features_batch(
        self,
        user_id: int,
        post_ids: List[int]
    ) -> pd.DataFrame:
        """
        Extract features for batch of posts
        
        Args:
            user_id: Target user ID
            post_ids: List of post IDs
            
        Returns:
            DataFrame with features
        """
        features_list = []
        
        for post_id in post_ids:
            try:
                features = self.feature_engineer.extract_features(
                    user_id,
                    post_id
                )
                features['post_id'] = post_id
                features_list.append(features)
            except Exception as e:
                logger.debug(f"Feature extraction error for post {post_id}: {e}")
                continue
        
        if not features_list:
            return pd.DataFrame()
        
        features_df = pd.DataFrame(features_list)
        
        # Ensure all expected features are present
        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0
        
        return features_df
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        if not self.metrics['latency']:
            return {
                'calls': 0,
                'avg_latency_ms': 0,
                'avg_batch_size': 0
            }
        
        return {
            'calls': len(self.metrics['latency']),
            'avg_latency_ms': np.mean(self.metrics['latency']),
            'p50_latency_ms': np.median(self.metrics['latency']),
            'p95_latency_ms': np.percentile(self.metrics['latency'], 95),
            'avg_batch_size': np.mean(self.metrics['batch_size'])
        }
    
    def reset_metrics(self):
        """Reset metrics"""
        self.metrics = {
            'latency': [],
            'batch_size': []
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_ranker(
    models_dir: str,
    feature_engineer,
    config: Optional[Dict] = None
) -> MLRanker:
    """
    Load trained ranker from disk
    
    Args:
        models_dir: Directory containing model artifacts
        feature_engineer: FeatureEngineer instance
        config: Configuration dict
        
    Returns:
        MLRanker instance
    """
    from pathlib import Path
    
    models_path = Path(models_dir)
    
    # Load model
    model = lgb.Booster(model_file=str(models_path / 'ranking_model.txt'))
    
    # Load scaler
    with open(models_path / 'ranking_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load feature columns
    with open(models_path / 'ranking_feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    logger.info(f"Loaded ranker from {models_dir}")
    
    return MLRanker(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        feature_engineer=feature_engineer,
        config=config
    )