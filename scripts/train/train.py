import os
import sys
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train_data_loader import load_and_preprocess_data
from train_collaborative import build_collaborative_filtering_model, save_collaborative_model
from train_embeddings import generate_embeddings, save_embeddings
from train_lightgbm import train_lightgbm_model, save_lightgbm_models

def main():
    print("="*70)
    print("RECOMMENDER SYSTEM TRAINING PIPELINE")
    print("="*70)

    # 1. Load and preprocess data
    print("\nSTEP 1: Data Loading & Preprocessing")
    data_dict = load_and_preprocess_data()
    data = data_dict['data']
    user_stats = data_dict['user_stats']
    author_stats = data_dict['author_stats']
    following_dict = data_dict['following_dict']
    train_interactions = data_dict['train_interactions']
    val_interactions = data_dict['val_interactions']
    test_interactions = data_dict['test_interactions']
    config = data_dict['config']

    MODELS_DIR = config.get('models_dir', 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 2. Build collaborative filtering model
    print("\nSTEP 2: Building Collaborative Filtering Model")
    cf_model = build_collaborative_filtering_model(data)
    save_collaborative_model(cf_model, MODELS_DIR)

    # 3. Generate embeddings
    print("\nSTEP 3: Generating Embeddings")
    embeddings = generate_embeddings(data)
    save_embeddings(embeddings, MODELS_DIR)

    # 4. Feature engineering and training
    print("\nSTEP 4: Feature Engineering & Model Training")
    ranking_model, scaler, feature_cols = train_lightgbm_model(
        data, user_stats, author_stats, following_dict, embeddings,
        train_interactions, val_interactions, test_interactions
    )
    
    if ranking_model is not None:
        save_lightgbm_models(ranking_model, scaler, feature_cols, MODELS_DIR)

    # 5. Save remaining artifacts
    print("\nSTEP 5: Saving Remaining Artifacts")
    save_items = [
        ('user_stats.pkl', user_stats),
        ('author_stats.pkl', author_stats),
        ('following_dict.pkl', following_dict),
        ('data_info.pkl', {
            'n_users': len(data['user']),
            'n_posts': len(data['post']),
            'n_interactions': len(data['postreaction']),
            'date_range': {
                'posts': [data['post']['CreateDate'].min(), data['post']['CreateDate'].max()],
                'interactions': [data['postreaction']['CreateDate'].min(), data['postreaction']['CreateDate'].max()]
            }
        })
    ]
    for fname, obj in save_items:
        with open(os.path.join(MODELS_DIR, fname), 'wb') as f:
            pickle.dump(obj, f)
        print(f"Saved {fname}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()