"""
ML Ranking Service
=================
Rank candidates using trained LightGBM model

Features:
- Load trained model, scaler, feature columns
- Extract features for user-post pairs
- Batch prediction for efficiency
- Return ranked list

Target latency: < 50ms for 1000 candidates
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class MLRanker:
    """
    Machine Learning based ranking using LightGBM
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
        Initialize ML ranker
        
        Args:
            model: Trained LightGBM model
            scaler: Feature scaler (StandardScaler)
            feature_cols: List of feature column names
            feature_engineer: FeatureEngineer instance
            config: Configuration
        """
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.feature_engineer = feature_engineer
        self.config = config or {}
        
        # Configuration
        self.batch_size = self.config.get('batch_size', 1000)
        self.top_k = self.config.get('top_k', 100)
        
        # Metrics
        self.metrics = {
            'total_rankings': 0,
            'total_candidates': 0,
            'total_time_ms': 0,
            'feature_extraction_ms': 0,
            'prediction_ms': 0
        }
        
        logger.info(f"ML Ranker initialized with {len(feature_cols)} features")
    
    def rank(self, user_id: int, candidate_ids: List[int]) -> pd.DataFrame:
        """
        Rank candidates using ML model
        
        Process:
        1. Extract features for all user-post pairs
        2. Scale features
        3. Predict scores
        4. Sort by score
        
        Args:
            user_id: User ID
            candidate_ids: List of candidate post IDs
            
        Returns:
            DataFrame with columns: post_id, score (sorted by score descending)
        """
        start_time = time.time()
        
        if not candidate_ids:
            return pd.DataFrame(columns=['post_id', 'score'])
        
        # Extract features
        t1 = time.time()
        features_df = self._extract_features(user_id, candidate_ids)
        self.metrics['feature_extraction_ms'] += (time.time() - t1) * 1000
        
        if features_df.empty:
            logger.warning(f"No features extracted for user {user_id}")
            return pd.DataFrame(columns=['post_id', 'score'])
        
        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0
        
        # Select and order features
        X = features_df[self.feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(0.0)
        
        # Scale features
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            logger.error(f"Scaling error: {e}")
            X_scaled = X.values
        
        # Predict scores
        t2 = time.time()
        try:
            scores = self.model.predict(X_scaled)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            scores = np.zeros(len(X_scaled))
        
        self.metrics['prediction_ms'] += (time.time() - t2) * 1000
        
        # Create result DataFrame
        result = pd.DataFrame({
            'post_id': candidate_ids[:len(scores)],
            'score': scores
        })
        
        # Sort by score
        result = result.sort_values('score', ascending=False).reset_index(drop=True)
        
        # Update metrics
        self.metrics['total_rankings'] += 1
        self.metrics['total_candidates'] += len(candidate_ids)
        self.metrics['total_time_ms'] += (time.time() - start_time) * 1000
        
        logger.debug(f"Ranked {len(candidate_ids)} candidates for user {user_id}, "
                    f"top score: {result['score'].iloc[0]:.4f if not result.empty else 0}")
        
        return result
    
    def _extract_features(self, user_id: int, post_ids: List[int]) -> pd.DataFrame:
        """
        Extract features for user-post pairs
        
        Args:
            user_id: User ID
            post_ids: List of post IDs
            
        Returns:
            DataFrame with features
        """
        try:
            # Create pairs
            pairs = [(user_id, post_id) for post_id in post_ids]
            
            # Extract features using FeatureEngineer
            features_df = self.feature_engineer.extract_batch(pairs)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return pd.DataFrame()
    
    def get_metrics(self) -> Dict:
        """Get ranking metrics"""
        avg_time = (
            self.metrics['total_time_ms'] / self.metrics['total_rankings']
            if self.metrics['total_rankings'] > 0 else 0
        )
        
        avg_feature_time = (
            self.metrics['feature_extraction_ms'] / self.metrics['total_rankings']
            if self.metrics['total_rankings'] > 0 else 0
        )
        
        avg_prediction_time = (
            self.metrics['prediction_ms'] / self.metrics['total_rankings']
            if self.metrics['total_rankings'] > 0 else 0
        )
        
        return {
            **self.metrics,
            'avg_time_ms': round(avg_time, 2),
            'avg_feature_extraction_ms': round(avg_feature_time, 2),
            'avg_prediction_ms': round(avg_prediction_time, 2)
        }
    
    def reset_metrics(self):
        """Reset metrics"""
        for key in self.metrics:
            self.metrics[key] = 0