"""
RANKING MODEL TRAINING
======================
Train LightGBM model for ranking stage

Features:
- Time decay weights support
- Early stopping
- Feature importance analysis
- MLflow tracking
- Model versioning
- Cross-validation

Target metrics:
- AUC > 0.75
- Precision@10 > 0.25
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

# Try imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  lightgbm not available. Install: pip install lightgbm")

try:
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available. Install: pip install scikit-learn")

try:
    import mlflow
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("ℹ️  MLflow not available (optional). Install: pip install mlflow")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("ℹ️  Plotting libraries not available (optional)")


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RankingModelTrainer:
    """
    Train LightGBM ranking model with MLflow tracking
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        experiment_name: str = "ranking_model",
        models_dir: str = "models",
        artifacts_dir: str = "artifacts"
    ):
        """
        Initialize trainer
        
        Args:
            config: Model hyperparameters
            experiment_name: MLflow experiment name
            models_dir: Directory for models
            artifacts_dir: Directory for artifacts
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm required")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")
        
        # Default hyperparameters
        self.params = config or {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }
        
        self.models_dir = Path(models_dir)
        self.artifacts_dir = Path(artifacts_dir)
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)
        
        # MLflow setup
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(experiment_name)
        
        # Model
        self.model = None
        self.feature_names = None
        self.training_history = {}
        
        logger.info(f"Initialized RankingModelTrainer")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Artifacts directory: {self.artifacts_dir}")
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        weights_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        weights_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 50
    ) -> lgb.Booster:
        """
        Train LightGBM model
        
        Args:
            X_train: (N, 47) feature matrix
            y_train: (N,) labels (0 or 1)
            weights_train: (N,) time decay weights (optional)
            X_val: Validation features
            y_val: Validation labels
            weights_val: Validation weights
            feature_names: Feature names
            num_boost_round: Max boosting rounds
            early_stopping_rounds: Early stopping rounds
            verbose_eval: Logging frequency
        
        Returns:
            Trained LightGBM model
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING LIGHTGBM MODEL")
        logger.info(f"{'='*70}")
        
        start_time = datetime.now()
        
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Log dataset info
        logger.info(f"\nDataset info:")
        logger.info(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        logger.info(f"  Train positive rate: {y_train.mean():.2%}")
        
        if X_val is not None:
            logger.info(f"  Val: {X_val.shape[0]:,} samples")
            logger.info(f"  Val positive rate: {y_val.mean():.2%}")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            weight=weights_train,
            feature_name=self.feature_names,
            free_raw_data=False
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                weight=weights_val,
                feature_name=self.feature_names,
                reference=train_data,
                free_raw_data=False
            )
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Callbacks
        callbacks = [
            lgb.log_evaluation(period=verbose_eval),
            lgb.record_evaluation(self.training_history)
        ]
        
        if early_stopping_rounds and X_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
        
        # Train
        logger.info(f"\n🔧 Training model...")
        logger.info(f"  Max boosting rounds: {num_boost_round}")
        logger.info(f"  Early stopping: {early_stopping_rounds}")
        
        # MLflow tracking
        if MLFLOW_AVAILABLE:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(self.params)
                mlflow.log_param("num_boost_round", num_boost_round)
                mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("n_train_samples", X_train.shape[0])
                if X_val is not None:
                    mlflow.log_param("n_val_samples", X_val.shape[0])
                
                # Train
                self.model = lgb.train(
                    self.params,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    callbacks=callbacks
                )
                
                # Log model
                mlflow.lightgbm.log_model(self.model, "model")
        else:
            # Train without MLflow
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )
        
        # Training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"\n✅ Training complete!")
        logger.info(f"  Best iteration: {self.model.best_iteration}")
        logger.info(f"  Training time: {training_time/60:.1f}m {training_time%60:.0f}s")
        
        return self.model
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    
    def evaluate(
        self,
        model: lgb.Booster,
        X: np.ndarray,
        y: np.ndarray,
        k_values: List[int] = [10, 20, 50]
    ) -> Dict[str, float]:
        """
        Evaluate model
        
        Args:
            model: Trained model
            X: Features
            y: Labels
            k_values: K values for Precision@K
        
        Returns:
            Metrics dictionary
        """
        # Predictions
        y_pred_proba = model.predict(X, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {}
        
        # Classification metrics
        metrics['auc'] = float(roc_auc_score(y, y_pred_proba))
        metrics['logloss'] = float(log_loss(y, y_pred_proba))
        metrics['accuracy'] = float((y == y_pred).mean())
        metrics['precision'] = float(precision_score(y, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y, y_pred, zero_division=0))
        metrics['f1'] = float(f1_score(y, y_pred, zero_division=0))
        
        # Ranking metrics (Precision@K, NDCG@K)
        for k in k_values:
            metrics[f'precision_at_{k}'] = self._precision_at_k(y, y_pred_proba, k)
            metrics[f'ndcg_at_{k}'] = self._ndcg_at_k(y, y_pred_proba, k)
        
        return metrics
    
    def _precision_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """Calculate Precision@K"""
        # Get top K predictions
        top_k_indices = np.argsort(y_pred)[::-1][:k]
        
        # Check how many are actually positive
        top_k_true = y_true[top_k_indices]
        
        precision = float(top_k_true.sum() / k)
        
        return precision
    
    def _ndcg_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """Calculate NDCG@K"""
        # Get top K predictions
        top_k_indices = np.argsort(y_pred)[::-1][:k]
        
        # DCG
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            relevance = y_true[idx]
            dcg += relevance / np.log2(i + 2)  # i+2 because i starts from 0
        
        # IDCG (Ideal DCG)
        ideal_indices = np.argsort(y_true)[::-1][:k]
        idcg = 0.0
        for i, idx in enumerate(ideal_indices):
            relevance = y_true[idx]
            idcg += relevance / np.log2(i + 2)
        
        # NDCG
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        
        return float(ndcg)
    
    # ========================================================================
    # FEATURE IMPORTANCE
    # ========================================================================
    
    def get_feature_importance(
        self,
        model: Optional[lgb.Booster] = None,
        importance_type: str = 'gain',
        top_k: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            model: Model (default: self.model)
            importance_type: 'gain' or 'split'
            top_k: Top K features
        
        Returns:
            DataFrame with feature importance
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model available")
        
        # Get importance
        importance = model.feature_importance(importance_type=importance_type)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # Sort and filter
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df = importance_df.head(top_k).reset_index(drop=True)
        importance_df['rank'] = importance_df.index + 1
        
        return importance_df[['rank', 'feature', 'importance']]
    
    def plot_feature_importance(
        self,
        model: Optional[lgb.Booster] = None,
        top_k: int = 20,
        save_path: Optional[str] = None
    ):
        """Plot feature importance"""
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting libraries not available")
            return
        
        importance_df = self.get_feature_importance(model, top_k=top_k)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance (Gain)')
        plt.ylabel('Feature')
        plt.title(f'Top {top_k} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        plt.close()
    
    # ========================================================================
    # SAVE/LOAD
    # ========================================================================
    
    def save_model(
        self,
        model: Optional[lgb.Booster] = None,
        version: Optional[str] = None,
        metrics: Optional[Dict] = None
    ):
        """
        Save model with versioning
        
        Args:
            model: Model to save (default: self.model)
            version: Version string (default: timestamp)
            metrics: Training metrics
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model to save")
        
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logger.info(f"\n💾 Saving model version: {version}")
        
        # Save model
        model_path = self.models_dir / f'model_v_{version}.txt'
        model.save_model(str(model_path))
        logger.info(f"  Model saved: {model_path}")
        
        # Save feature importance
        importance_df = self.get_feature_importance(model, top_k=47)
        importance_path = self.artifacts_dir / f'feature_importance_v_{version}.csv'
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"  Feature importance saved: {importance_path}")
        
        # Plot feature importance
        plot_path = self.artifacts_dir / f'feature_importance_v_{version}.png'
        self.plot_feature_importance(model, top_k=20, save_path=plot_path)
        
        # Save metrics
        if metrics:
            metrics_path = self.artifacts_dir / f'training_metrics_v_{version}.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"  Metrics saved: {metrics_path}")
        
        # Save metadata
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'params': self.params,
            'best_iteration': model.best_iteration,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'metrics': metrics
        }
        
        metadata_path = self.models_dir / f'metadata_v_{version}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  Metadata saved: {metadata_path}")
    
    def load_model(self, version: str) -> lgb.Booster:
        """Load model by version"""
        model_path = self.models_dir / f'model_v_{version}.txt'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = lgb.Booster(model_file=str(model_path))
        
        logger.info(f"Loaded model version: {version}")
        
        return model
    
    # ========================================================================
    # CROSS-VALIDATION
    # ========================================================================
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        n_folds: int = 5,
        num_boost_round: int = 1000
    ) -> Dict[str, List[float]]:
        """
        K-fold cross-validation
        
        Args:
            X: Features
            y: Labels
            weights: Sample weights
            n_folds: Number of folds
            num_boost_round: Boosting rounds
        
        Returns:
            CV metrics
        """
        logger.info(f"\n🔄 Cross-validation ({n_folds} folds)")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_metrics = {
            'auc': [],
            'logloss': [],
            'precision_at_10': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            logger.info(f"\nFold {fold}/{n_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            w_train = weights[train_idx] if weights is not None else None
            w_val = weights[val_idx] if weights is not None else None
            
            # Train
            model = self.train(
                X_train, y_train, w_train,
                X_val, y_val, w_val,
                num_boost_round=num_boost_round,
                early_stopping_rounds=50,
                verbose_eval=100
            )
            
            # Evaluate
            metrics = self.evaluate(model, X_val, y_val)
            
            cv_metrics['auc'].append(metrics['auc'])
            cv_metrics['logloss'].append(metrics['logloss'])
            cv_metrics['precision_at_10'].append(metrics['precision_at_10'])
            
            logger.info(f"  AUC: {metrics['auc']:.4f}")
            logger.info(f"  Precision@10: {metrics['precision_at_10']:.4f}")
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info(f"CROSS-VALIDATION SUMMARY")
        logger.info(f"{'='*70}")
        
        for metric_name, values in cv_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")
        
        return cv_metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("="*70)
    print("RANKING MODEL TRAINING")
    print("="*70)
    
    # Load data (example)
    logger.info("\n📦 Loading data...")
    
    # This should load your feature-engineered data
    # For demonstration, we'll show the expected structure
    
    try:
        with open('features_extracted.pkl', 'rb') as f:
            data = pickle.load(f)
        
        train_df = data['train_df']
        val_df = data['val_df']
        test_df = data['test_df']
        feature_cols = data['feature_cols']
        
        logger.info(f"Train: {len(train_df):,}")
        logger.info(f"Val: {len(val_df):,}")
        logger.info(f"Test: {len(test_df):,}")
        logger.info(f"Features: {len(feature_cols)}")
        
    except FileNotFoundError:
        logger.error("features_extracted.pkl not found!")
        logger.error("Please run feature engineering first")
        return
    
    # Prepare data
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['label'].values
    w_train = train_df['weight'].values if 'weight' in train_df.columns else None
    
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df['label'].values
    w_val = val_df['weight'].values if 'weight' in val_df.columns else None
    
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['label'].values
    
    # Initialize trainer
    trainer = RankingModelTrainer(
        experiment_name="ranking_model_lightgbm"
    )
    
    # Train
    model = trainer.train(
        X_train, y_train, w_train,
        X_val, y_val, w_val,
        feature_names=feature_cols,
        num_boost_round=1000,
        early_stopping_rounds=50
    )
    
    # Evaluate on all sets
    logger.info(f"\n📊 Evaluating model...")
    
    train_metrics = trainer.evaluate(model, X_train, y_train)
    val_metrics = trainer.evaluate(model, X_val, y_val)
    test_metrics = trainer.evaluate(model, X_test, y_test)
    
    logger.info(f"\nTrain metrics:")
    for k, v in train_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info(f"\nValidation metrics:")
    for k, v in val_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info(f"\nTest metrics:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # Feature importance
    logger.info(f"\n📊 Feature importance (top 20):")
    importance_df = trainer.get_feature_importance(model, top_k=20)
    print(importance_df.to_string(index=False))
    
    # Save model if metrics pass threshold
    if val_metrics['auc'] > 0.75 and val_metrics['precision_at_10'] > 0.25:
        logger.info(f"\n✅ Metrics pass threshold!")
        logger.info(f"  AUC: {val_metrics['auc']:.4f} > 0.75")
        logger.info(f"  Precision@10: {val_metrics['precision_at_10']:.4f} > 0.25")
        
        trainer.save_model(
            model,
            version=datetime.now().strftime('%Y%m%d'),
            metrics={
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            }
        )
    else:
        logger.warning(f"\n⚠️  Metrics below threshold")
        logger.warning(f"  AUC: {val_metrics['auc']:.4f} (target: > 0.75)")
        logger.warning(f"  Precision@10: {val_metrics['precision_at_10']:.4f} (target: > 0.25)")
    
    print(f"\n✅ Training complete!")


if __name__ == "__main__":
    main()