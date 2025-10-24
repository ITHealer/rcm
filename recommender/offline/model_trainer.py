"""
Ranking Model Training với LightGBM
Purpose: Train model dự đoán P(engagement | user, post)
UPDATED: Support time decay weights
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score
from typing import Dict, Tuple
import pickle
from recommender.common.feature_engineer import FeatureEngineer

class ModelTrainer:
    """
    Train ranking model với LightGBM
    UPDATED: Support time decay weights
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.model_params = config['model']['params']
        self.model = None
        self.scaler = None
        self.feature_cols = None
    
    def prepare_training_data(
        self,
        interactions_df: pd.DataFrame,
        data: Dict[str, pd.DataFrame],
        user_stats: Dict,
        author_stats: Dict,
        following_dict: Dict,
        embeddings: Dict
    ) -> pd.DataFrame:
        """
        Extract features cho training
        UPDATED: Preserve 'weight' column from time decay
        
        Args:
            interactions_df: DataFrame có thể có 'weight' column (from time decay)
            
        Returns:
            training_df: DataFrame với features + label + weight (if exists)
        """
        print("\n🔧 Preparing Training Data...")
        
        # Check if interactions_df has 'weight' column (from time decay)
        has_weights = 'weight' in interactions_df.columns
        
        if has_weights:
            print("   ✅ Time decay weights detected in input data")
        else:
            print("   ⚠️  No time decay weights in input data")
        
        feature_engineer = FeatureEngineer(
            data, user_stats, author_stats, following_dict, embeddings
        )
        
        # ============================================================
        # Normalize column names first (handle both formats)
        # ============================================================
        
        # Create a normalized copy
        df_normalized = interactions_df.copy()
        
        # Map columns to standard names
        column_mapping = {
            'UserId': 'user_id',
            'PostId': 'post_id',
            'ReactionTypeId': 'reaction_type_id',
            'CreateDate': 'created_at'
        }
        
        # Rename if old format
        df_normalized.rename(columns=column_mapping, inplace=True)
        
        print(f"   📋 Columns in data: {df_normalized.columns.tolist()}")
        
        # ============================================================
        # Separate positive and negative samples
        # ============================================================
        
        # Map actions to positive/negative
        # Positive: like, comment, share, save
        # Negative: view only
        
        positive_actions = ['like', 'comment', 'share', 'save']
        negative_actions = ['view']
        
        if 'action' in df_normalized.columns:
            # New format: has 'action' column
            positive = df_normalized[
                df_normalized['action'].isin(positive_actions)
            ].copy()
            
            negative = df_normalized[
                df_normalized['action'].isin(negative_actions)
            ].copy()
            
        else:
            # Old format: has 'reaction_type_id' column
            positive = df_normalized[
                df_normalized['reaction_type_id'].isin([1, 2, 3, 5])
            ].copy()
            
            negative = df_normalized[
                df_normalized['reaction_type_id'] == 4
            ].copy()
        
        print(f"   Raw positive samples: {len(positive):,}")
        print(f"   Raw negative samples: {len(negative):,}")
        
        # ============================================================
        # Handle data imbalance
        # ============================================================
        
        if len(positive) == 0 or len(negative) == 0:
            print("\n⚠️  WARNING: Insufficient data diversity!")
            print(f"   Positive samples: {len(positive)}")
            print(f"   Negative samples: {len(negative)}")
            
            # Create synthetic negative samples if needed
            if len(negative) == 0 and len(positive) > 0:
                print("\n🔧 Creating synthetic negative samples...")
                
                all_posts = set(data['post']['Id'].tolist())
                synthetic_negatives = []
                
                # Use normalized column names
                for user_id in positive['user_id'].unique()[:100]:
                    user_interacted_posts = set(
                        df_normalized[df_normalized['user_id'] == user_id]['post_id']
                    )
                    
                    available_posts = list(all_posts - user_interacted_posts)
                    
                    if len(available_posts) > 0:
                        n_samples = min(5, len(available_posts))
                        sampled_posts = np.random.choice(available_posts, n_samples, replace=False)
                        
                        for post_id in sampled_posts:
                            neg_sample = {
                                'user_id': user_id,
                                'post_id': post_id,
                                'action': 'view'
                            }
                            
                            # Preserve weight column if exists
                            if has_weights:
                                neg_sample['weight'] = 0.1  # Low weight for synthetic
                            
                            synthetic_negatives.append(neg_sample)
                
                if len(synthetic_negatives) > 0:
                    negative = pd.DataFrame(synthetic_negatives)
                    print(f"   ✅ Created {len(negative)} synthetic negative samples")
                else:
                    raise ValueError("Cannot create negative samples - insufficient data!")
            
            elif len(positive) == 0:
                raise ValueError("No positive samples found! Cannot train model.")
        
        # Balance: Up to 5x negative vs positive
        n_neg = min(len(positive) * 5, len(negative))
        negative_sampled = negative.sample(n=n_neg, random_state=42) if n_neg > 0 else negative
        
        print(f"   Using positive samples: {len(positive):,}")
        print(f"   Using negative samples: {len(negative_sampled):,}")
        print(f"   Ratio (neg:pos): {len(negative_sampled)/len(positive):.2f}:1")
        
        # ============================================================
        # Extract features
        # ============================================================
        
        training_data = []
        failed_count = 0
        
        # Use normalized column names (already lowercase)
        user_col = 'user_id'
        post_col = 'post_id'
        
        print("   Extracting features for positive samples...")
        for idx, row in positive.iterrows():
            try:
                features = feature_engineer.extract_features(row[user_col], row[post_col])
                features['label'] = 1
                
                # Preserve time decay weight if exists
                if has_weights and 'weight' in row:
                    features['weight'] = row['weight']
                
                training_data.append(features)
            except Exception as e:
                failed_count += 1
                continue
        
        print("   Extracting features for negative samples...")
        for idx, row in negative_sampled.iterrows():
            try:
                features = feature_engineer.extract_features(row[user_col], row[post_col])
                features['label'] = 0
                
                # Preserve time decay weight if exists
                if has_weights and 'weight' in row:
                    features['weight'] = row['weight']
                
                training_data.append(features)
            except Exception as e:
                failed_count += 1
                continue
        
        if failed_count > 0:
            print(f"   ⚠️  Failed to extract features for {failed_count} samples")
        
        training_df = pd.DataFrame(training_data)
        
        # ============================================================
        # Validate training data
        # ============================================================
        
        if len(training_df) == 0:
            raise ValueError("No training samples after feature extraction!")
        
        label_counts = training_df['label'].value_counts()
        print(f"\n   📊 Final label distribution:")
        print(f"      Positive (label=1): {label_counts.get(1, 0):,}")
        print(f"      Negative (label=0): {label_counts.get(0, 0):,}")
        
        if len(label_counts) < 2:
            raise ValueError(
                f"Training data contains only one class! "
                f"Labels: {label_counts.to_dict()}\n"
                f"Cannot train binary classifier with single class."
            )
        
        # Check if weights preserved
        if has_weights and 'weight' in training_df.columns:
            print(f"\n   📊 Time decay weights preserved:")
            print(f"      Mean: {training_df['weight'].mean():.4f}")
            print(f"      Median: {training_df['weight'].median():.4f}")
            print(f"      Range: [{training_df['weight'].min():.4f}, {training_df['weight'].max():.4f}]")
        
        # Shuffle
        training_df = training_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n✅ Training data prepared: {len(training_df):,} samples")
        
        return training_df
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> Tuple[lgb.Booster, StandardScaler, list]:
        """
        Train LightGBM model
        UPDATED: Use time decay weights if available
        
        Returns:
            (model, scaler, feature_cols)
        """
        print("\n🔧 Training Ranking Model...")
        
        # Check if training data has weights
        has_weights = 'weight' in train_df.columns and 'weight' in val_df.columns
        
        if has_weights:
            print("   ✅ Using time decay weights in training")
        
        # Separate features and labels (and weights if exists)
        exclude_cols = ['label']
        if has_weights:
            exclude_cols.append('weight')
        
        self.feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        X_train = train_df[self.feature_cols].fillna(0)
        y_train = train_df['label']
        
        X_val = val_df[self.feature_cols].fillna(0)
        y_val = val_df['label']
        
        # Extract weights if available
        if has_weights:
            train_weights = train_df['weight'].values
            val_weights = val_df['weight'].values
        else:
            train_weights = None
            val_weights = None
        
        print(f"   Train: {X_train.shape}")
        print(f"   Val: {X_val.shape}")
        print(f"   Features: {len(self.feature_cols)}")
        
        if has_weights:
            print(f"   Train weights: mean={train_weights.mean():.4f}, range=[{train_weights.min():.4f}, {train_weights.max():.4f}]")
        
        # Scale features
        print("   Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create LightGBM datasets WITH weights
        if has_weights:
            train_data = lgb.Dataset(
                X_train_scaled, 
                label=y_train,
                weight=train_weights  # ← Use time decay weights
            )
            val_data = lgb.Dataset(
                X_val_scaled, 
                label=y_val,
                weight=val_weights,  # ← Use time decay weights
                reference=train_data
            )
        else:
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
        
        # Train
        print("   Training model...")
        self.model = lgb.train(
            self.model_params,
            train_data,
            num_boost_round=self.model_params['num_boost_round'],
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(self.model_params['early_stopping_rounds']),
                lgb.log_evaluation(50)
            ]
        )
        
        print(f"✅ Training complete!")
        print(f"   Best iteration: {self.model.best_iteration}")
        print(f"   Best score: {self.model.best_score['val']['auc']:.4f}")
        
        return self.model, self.scaler, self.feature_cols
    
    def evaluate(
        self,
        test_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Returns:
            metrics: Dict of evaluation metrics
        """
        print("\n📊 Evaluating Model...")
        
        # Check if test data has both classes
        if len(test_df) == 0:
            print("⚠️  No test data available")
            return {
                'auc': 0.0,
                'logloss': 999.0,
                'precision': 0.0,
                'recall': 0.0,
                'precision@10': 0.0,
                'precision@20': 0.0,
                'precision@50': 0.0
            }
        
        label_counts = test_df['label'].value_counts()
        print(f"   Test label distribution: {label_counts.to_dict()}")
        
        if len(label_counts) < 2:
            print("⚠️  Test set contains only one class - cannot compute metrics")
            return {
                'auc': 0.0,
                'logloss': 999.0,
                'precision': 0.0,
                'recall': 0.0,
                'precision@10': 0.0,
                'precision@20': 0.0,
                'precision@50': 0.0
            }
        
        # Extract features
        X_test = test_df[self.feature_cols].fillna(0)
        y_test = test_df['label']
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict
        y_pred_proba = self.model.predict(X_test_scaled)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Compute metrics với error handling
        metrics = {}
        
        # AUC
        try:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
        except Exception as e:
            print(f"   ⚠️  Could not compute AUC: {e}")
            metrics['auc'] = 0.0
        
        # Log Loss
        try:
            y_pred_proba_clipped = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
            metrics['logloss'] = log_loss(y_test, y_pred_proba_clipped)
        except Exception as e:
            print(f"   ⚠️  Could not compute Log Loss: {e}")
            metrics['logloss'] = 999.0
        
        # Precision & Recall
        try:
            metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
        except Exception as e:
            print(f"   ⚠️  Could not compute Precision/Recall: {e}")
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
        
        # Precision@K
        for k in [10, 20, 50]:
            try:
                if len(y_test) >= k:
                    top_k_indices = np.argsort(y_pred_proba)[-k:]
                    precision_k = y_test.iloc[top_k_indices].sum() / k
                    metrics[f'precision@{k}'] = precision_k
                else:
                    metrics[f'precision@{k}'] = 0.0
            except Exception as e:
                print(f"   ⚠️  Could not compute Precision@{k}: {e}")
                metrics[f'precision@{k}'] = 0.0
        
        # Print results
        print("\n📈 Test Set Performance:")
        print(f"   AUC: {metrics['auc']:.4f}")
        print(f"   Log Loss: {metrics['logloss']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   Precision@10: {metrics['precision@10']:.4f}")
        print(f"   Precision@20: {metrics['precision@20']:.4f}")
        print(f"   Precision@50: {metrics['precision@50']:.4f}")
        
        return metrics
    
    def save_model(self, output_path: str):
        """Save model artifacts"""
        # Save LightGBM model
        self.model.save_model(output_path + '_model.txt')
        
        # Save scaler and feature cols
        with open(output_path + '_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(output_path + '_feature_cols.pkl', 'wb') as f:
            pickle.dump(self.feature_cols, f)
        
        print(f"✅ Model saved to: {output_path}")