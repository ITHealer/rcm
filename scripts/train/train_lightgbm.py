import os
import sys
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from recommender.common.feature_engineering import FeatureEngineer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

def train_lightgbm_model(data, user_stats, author_stats, following_dict, embeddings, train_interactions, val_interactions, test_interactions):
    print("Training LightGBM Model.")

    if not LIGHTGBM_AVAILABLE:
        print("LightGBM not available")
        return None, None, None

    # Create training dataset
    feature_engineer = FeatureEngineer(data, user_stats, author_stats, following_dict, embeddings)

    # Sử dụng train_interactions để tạo tập huấn luyện
    train_interactions_post = train_interactions  # All interactions are for posts
    # Positive samples (like, comment, share, save)
    positive_samples = train_interactions_post[train_interactions_post['ReactionTypeId'].isin([1, 2, 3, 5])]
    # Negative samples (view)
    negative_samples = train_interactions_post[train_interactions_post['ReactionTypeId'] == 4]
    # Sample negative samples (up to 5x positive)
    n_neg = min(len(positive_samples) * 5, len(negative_samples))
    negative_samples = negative_samples.sample(n=n_neg, random_state=42) if n_neg > 0 else negative_samples

    training_data = []
    # Extract features for positive samples
    for idx, row in positive_samples.iterrows():
        try:
            features = feature_engineer.extract_features(row['UserId'], row['PostId'])
            features['label'] = 1
            training_data.append(features)
        except Exception:
            continue
    # Extract features for negative samples
    for idx, row in negative_samples.iterrows():
        try:
            features = feature_engineer.extract_features(row['UserId'], row['PostId'])
            features['label'] = 0
            training_data.append(features)
        except Exception:
            continue

    training_df = pd.DataFrame(training_data)
    training_df = training_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Training dataset: {len(training_df)} samples")

    # Train LightGBM
    feature_cols = [col for col in training_df.columns if col not in ['label']]
    X = training_df[feature_cols].fillna(0)
    y = training_df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'verbose': -1
    }
    ranking_model = lgb.train(
        params, train_data, num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    print("Ranking model trained")

    # Evaluate on validation and test sets
    def evaluate_on_split(split_df, split_name):
        split_post = split_df  # All interactions are for posts
        pos = split_post[split_post['ReactionTypeId'].isin([1,2,3,5])]
        neg = split_post[split_post['ReactionTypeId'] == 4]
        n_neg = min(len(pos)*5, len(neg))
        neg = neg.sample(n=n_neg, random_state=42) if n_neg > 0 else neg
        eval_data = []
        for _, row in pos.iterrows():
            try:
                features = feature_engineer.extract_features(row['UserId'], row['PostId'])
                features['label'] = 1
                eval_data.append(features)
            except Exception:
                continue
        for _, row in neg.iterrows():
            try:
                features = feature_engineer.extract_features(row['UserId'], row['PostId'])
                features['label'] = 0
                eval_data.append(features)
            except Exception:
                continue
        if not eval_data:
            print(f"No data for {split_name} evaluation.")
            return
        eval_df = pd.DataFrame(eval_data)
        X_eval = eval_df[feature_cols].fillna(0)
        y_eval = eval_df['label']
        X_eval_scaled = scaler.transform(X_eval)
        y_pred_prob = ranking_model.predict(X_eval_scaled)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_eval, y_pred_prob)
        acc = accuracy_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred)
        print(f"{split_name} AUC: {auc:.4f} | ACC: {acc:.4f} | F1: {f1:.4f} | Samples: {len(y_eval)}")

    print("\nEvaluating on Validation Set")
    evaluate_on_split(val_interactions, "Validation")
    print("\nEvaluating on Test Set")
    evaluate_on_split(test_interactions, "Test")

    return ranking_model, scaler, feature_cols

def save_lightgbm_models(ranking_model, scaler, feature_cols, models_dir='models'):
    """Save LightGBM models and related artifacts to disk"""
    os.makedirs(models_dir, exist_ok=True)
    
    with open(os.path.join(models_dir, 'ranking_model.pkl'), 'wb') as f:
        pickle.dump(ranking_model, f)
    print("Saved ranking_model.pkl")
    
    with open(os.path.join(models_dir, 'feature_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved feature_scaler.pkl")
    
    with open(os.path.join(models_dir, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)
    print("Saved feature_cols.pkl")

if __name__ == "__main__":
    # For testing
    from train_data_loader import load_and_preprocess_data
    from train_embeddings import generate_embeddings

    data_dict = load_and_preprocess_data()
    embeddings = generate_embeddings(data_dict['data'])

    ranking_model, scaler, feature_cols = train_lightgbm_model(
        data_dict['data'], data_dict['user_stats'], data_dict['author_stats'],
        data_dict['following_dict'], embeddings, data_dict['train_interactions'],
        data_dict['val_interactions'], data_dict['test_interactions']
    )
    
    if ranking_model is not None:
        save_lightgbm_models(ranking_model, scaler, feature_cols)
    
    print("LightGBM training complete!")