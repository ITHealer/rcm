"""
Collaborative Filtering Precomputation
Purpose: Tính trước user-user và item-item similarities
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple, List
import pickle

class CFBuilder:
    """
    Build Collaborative Filtering similarity matrices
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.min_interactions = config['collaborative_filtering']['min_interactions']
        self.top_k = config['collaborative_filtering']['top_k_similar']
    
    def build_user_item_matrix(
        self, 
        interactions_df: pd.DataFrame
    ) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """
        Build sparse user-item interaction matrix
        
        Returns:
            (matrix, user_ids, post_ids)
        """
        print("\n🔧 Building User-Item Matrix...")
        
        # Filter users with minimum interactions
        user_counts = interactions_df.groupby('UserId').size()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        
        interactions_filtered = interactions_df[
            interactions_df['UserId'].isin(valid_users)
        ].copy()
        
        print(f"   Filtered to {len(valid_users):,} users with >={self.min_interactions} interactions")
        
        # Create mappings
        user_ids = interactions_filtered['UserId'].unique()
        post_ids = interactions_filtered['PostId'].unique()
        
        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        post_id_to_idx = {pid: idx for idx, pid in enumerate(post_ids)}
        
        # Assign implicit scores based on reaction type
        reaction_scores = {
            1: 1.0,   # Like
            2: 2.0,   # Comment
            3: 3.0,   # Share
            4: 0.1,   # View
            5: 1.5    # Save
        }
        
        interactions_filtered['score'] = interactions_filtered['ReactionTypeId'].map(
            lambda x: reaction_scores.get(x, 0.1)
        )
        
        # Build sparse matrix
        rows = interactions_filtered['UserId'].map(user_id_to_idx)
        cols = interactions_filtered['PostId'].map(post_id_to_idx)
        data = interactions_filtered['score']
        
        user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(post_ids))
        )
        
        print(f"✅ Matrix shape: {user_item_matrix.shape}")
        print(f"   Sparsity: {1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.4f}")
        
        return user_item_matrix, user_ids, post_ids
    
    def compute_user_similarities(
        self,
        user_item_matrix: csr_matrix,
        user_ids: np.ndarray
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Compute user-user similarities
        
        Returns:
            user_similarities: Dict[user_id -> [(similar_user_id, score), ...]]
        """
        print("\n🔧 Computing User-User Similarities...")
        
        # Compute cosine similarity (memory intensive!)
        # Process in chunks if needed
        print("   Computing cosine similarities...")
        user_sim_matrix = cosine_similarity(user_item_matrix, dense_output=False)
        
        print("   Extracting top-K similar users per user...")
        user_similarities = {}
        
        for idx, user_id in enumerate(user_ids):
            # Get similarity scores for this user
            sim_scores = user_sim_matrix[idx].toarray().flatten()
            
            # Get top K (excluding self)
            sim_scores[idx] = -1  # Exclude self
            top_k_indices = np.argsort(sim_scores)[-self.top_k:][::-1]
            
            similar_users = [
                (user_ids[i], float(sim_scores[i]))
                for i in top_k_indices
                if sim_scores[i] > 0
            ]
            
            user_similarities[user_id] = similar_users
        
        print(f"✅ Computed similarities for {len(user_similarities):,} users")
        
        return user_similarities
    
    def compute_item_similarities(
        self,
        user_item_matrix: csr_matrix,
        post_ids: np.ndarray
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Compute item-item similarities
        
        Returns:
            item_similarities: Dict[post_id -> [(similar_post_id, score), ...]]
        """
        print("\n🔧 Computing Item-Item Similarities...")
        
        # Transpose to get item-user matrix
        item_user_matrix = user_item_matrix.T
        
        print("   Computing cosine similarities...")
        item_sim_matrix = cosine_similarity(item_user_matrix, dense_output=False)
        
        print("   Extracting top-K similar items per item...")
        item_similarities = {}
        
        for idx, post_id in enumerate(post_ids):
            sim_scores = item_sim_matrix[idx].toarray().flatten()
            
            # Get top K (excluding self)
            sim_scores[idx] = -1
            top_k_indices = np.argsort(sim_scores)[-self.top_k:][::-1]
            
            similar_posts = [
                (post_ids[i], float(sim_scores[i]))
                for i in top_k_indices
                if sim_scores[i] > 0
            ]
            
            item_similarities[post_id] = similar_posts
        
        print(f"✅ Computed similarities for {len(item_similarities):,} items")
        
        return item_similarities
    
    def build_cf_model(
        self,
        interactions_df: pd.DataFrame
    ) -> Dict:
        """
        Complete CF pipeline
        
        Returns:
            cf_model: Dict containing all CF components
        """
        # Build matrix
        user_item_matrix, user_ids, post_ids = self.build_user_item_matrix(interactions_df)
        
        # Compute similarities
        user_similarities = self.compute_user_similarities(user_item_matrix, user_ids)
        item_similarities = self.compute_item_similarities(user_item_matrix, post_ids)
        
        cf_model = {
            'user_item_matrix': user_item_matrix,
            'user_ids': user_ids,
            'post_ids': post_ids,
            'user_similarities': user_similarities,
            'item_similarities': item_similarities,
            'metadata': {
                'n_users': len(user_ids),
                'n_posts': len(post_ids),
                'n_interactions': user_item_matrix.nnz,
                'top_k': self.top_k
            }
        }
        
        return cf_model
    
    def save_cf_model(self, cf_model: Dict, output_path: str):
        """Save CF model"""
        with open(output_path, 'wb') as f:
            pickle.dump(cf_model, f)
        
        print(f"✅ CF model saved to: {output_path}")