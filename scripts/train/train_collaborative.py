import os
import sys
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from train_data_loader import load_and_preprocess_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def build_collaborative_filtering_model(data):
    print("Building Collaborative Filtering Model.")
    reaction_type_map = {
        1: 'like',
        2: 'love',
        3: 'laugh',
        4: 'wow',
        5: 'sad',
        6: 'angry',
        7: 'care'
    }
    score_map = {
        'like': 2,
        'love': 5,
        'laugh': 3,
        'wow': 3,
        'sad': -1,
        'angry': -5,
        'care': 4
    }

    df = data['postreaction']
    # All reactions in PostReaction table are for posts, no need to filter by EntityType
    df['reaction_name'] = df['ReactionTypeId'].map(reaction_type_map)
    df['implicit_score'] = df['reaction_name'].map(score_map).fillna(0)

    user_post_scores = df.groupby(['UserId', 'PostId'])['implicit_score'].sum().reset_index()

    user_ids = user_post_scores['UserId'].unique()
    post_ids = user_post_scores['PostId'].unique()

    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    post_id_to_idx = {pid: idx for idx, pid in enumerate(post_ids)}

    rows = [user_id_to_idx[uid] for uid in user_post_scores['UserId']]
    cols = [post_id_to_idx[pid] for pid in user_post_scores['PostId']]
    scores = user_post_scores['implicit_score'].values

    user_item_matrix = csr_matrix((scores, (rows, cols)), shape=(len(user_ids), len(post_ids)))
    item_similarity = cosine_similarity(user_item_matrix.T, dense_output=False)
    user_similarity = cosine_similarity(user_item_matrix, dense_output=False)

    cf_model = {
        'user_item_matrix': user_item_matrix,
        'item_similarity': item_similarity,
        'user_similarity_matrix': user_similarity,
        'post_ids': post_ids,
        'user_ids': user_ids
    }

    print("CF model built")
    return cf_model

def save_collaborative_model(cf_model, models_dir='models'):
    """Save collaborative filtering model to disk"""
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'cf_model.pkl'), 'wb') as f:
        pickle.dump(cf_model, f)
    print("Saved cf_model.pkl")

if __name__ == "__main__":
    data_dict = load_and_preprocess_data()
    cf_model = build_collaborative_filtering_model(data_dict['data'])
    save_collaborative_model(cf_model)
    print("Collaborative filtering model training complete!")