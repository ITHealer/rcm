import os
import sys
import yaml
from recommender.common.data_loader import load_data, filter_recent_data, compute_statistics, create_temporal_splits

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def get_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_and_preprocess_data():
    print("Loading and preprocessing data.")

    # Read config
    config = get_config()
    DATA_DIR = config.get('data_dir', 'dataset')
    LOOKBACK_DAYS = config.get('lookback_days', 90)

    # Load and preprocess data
    data = load_data(DATA_DIR)
    data = filter_recent_data(data, LOOKBACK_DAYS)

    user_stats, author_stats, following_dict = compute_statistics(data)
    train_interactions, val_interactions, test_interactions = create_temporal_splits(data)

    print(f"Data loaded: {len(data['user'])} users, {len(data['post'])} posts")

    return {
        'data': data,
        'user_stats': user_stats,
        'author_stats': author_stats,
        'following_dict': following_dict,
        'train_interactions': train_interactions,
        'val_interactions': val_interactions,
        'test_interactions': test_interactions,
        'config': config
    }

if __name__ == "__main__":
    result = load_and_preprocess_data()
    print("Data loading complete!")