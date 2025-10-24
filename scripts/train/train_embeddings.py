import os
import sys
import numpy as np
import pickle
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

def generate_embeddings(data):
    """Generate post and user embeddings"""
    print("Generating Embeddings...")

    embeddings = {}

    if EMBEDDINGS_AVAILABLE:
        embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

        # Post embeddings
        texts = []
        for idx, row in data['post'].iterrows():
            text = str(row['Content']) if pd.notna(row['Content']) else ""
            # Nếu không có cột hashtags thì bỏ qua đoạn này
            if 'hashtags' in row and isinstance(row['hashtags'], list) and len(row['hashtags']) > 0:
                text += ' ' + ' '.join([f'#{tag}' for tag in row['hashtags']])
            texts.append(text)

        post_embeddings_array = embedding_model.encode(texts, batch_size=32, show_progress_bar=True, device='cpu')
        post_embeddings = {post_id: emb for post_id, emb in zip(data['post']['Id'], post_embeddings_array)}

        # User embeddings (weighted average of interacted posts)
        user_embeddings = {}
        for user_id in data['user']['Id']:
            user_interactions = data['postreaction'][
                (data['postreaction']['UserId'] == user_id)
            ]

            if len(user_interactions) == 0:
                user_embeddings[user_id] = np.zeros(post_embeddings_array.shape[1])
                continue

            weighted_sum = np.zeros(post_embeddings_array.shape[1])
            total_weight = 0

            # Map ReactionType sang tên hành động và điểm số
            reaction_type_map = {
                1: 'like', 2: 'love', 3: 'laugh', 4: 'wow',
                5: 'sad', 6: 'angry', 7: 'care'
            }
            score_map = {
                'like': 2, 'love': 5, 'laugh': 3, 'wow': 3,
                'sad': -1, 'angry': -5, 'care': 4
            }

            for _, row in user_interactions.iterrows():
                post_id = row['PostId']
                action = row['ReactionTypeId']

                if post_id in post_embeddings:
                    action_name = reaction_type_map.get(action, 'view')
                    weight = score_map.get(action_name, 0)
                    weighted_sum += post_embeddings[post_id] * weight
                    total_weight += weight

            if total_weight > 0:
                user_emb = weighted_sum / total_weight
                user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
            else:
                user_emb = np.zeros(post_embeddings_array.shape[1])

            user_embeddings[user_id] = user_emb

        embeddings = {'post': post_embeddings, 'user': user_embeddings}
        print("Embeddings generated")
    else:
        print("Embeddings not available")

    return embeddings

def save_embeddings(embeddings, models_dir='models'):
    """Save embeddings to disk"""
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)
    print("Saved embeddings.pkl")

if __name__ == "__main__":
    # For testing
    from train_data_loader import load_and_preprocess_data
    data_dict = load_and_preprocess_data()
    embeddings = generate_embeddings(data_dict['data'])
    save_embeddings(embeddings)
    print("Embeddings generation complete!")