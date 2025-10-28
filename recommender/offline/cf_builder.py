# """
# Collaborative Filtering Precomputation
# Purpose: Tính trước user-user và item-item similarities
# """

# import numpy as np
# import pandas as pd
# from scipy.sparse import csr_matrix
# from sklearn.metrics.pairwise import cosine_similarity
# from typing import Dict, Tuple, List
# import pickle

# class CFBuilder:
#     """
#     Build Collaborative Filtering similarity matrices
#     """
    
#     def __init__(self, config: dict):
#         self.config = config
#         self.min_interactions = config['collaborative_filtering']['min_interactions']
#         self.top_k = config['collaborative_filtering']['top_k_similar']
    
#     def build_user_item_matrix(
#         self, 
#         interactions_df: pd.DataFrame
#     ) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
#         """
#         Build sparse user-item interaction matrix
        
#         Returns:
#             (matrix, user_ids, post_ids)
#         """
#         print("\n🔧 Building User-Item Matrix...")
        
#         # Filter users with minimum interactions
#         user_counts = interactions_df.groupby('UserId').size()
#         valid_users = user_counts[user_counts >= self.min_interactions].index
        
#         interactions_filtered = interactions_df[
#             interactions_df['UserId'].isin(valid_users)
#         ].copy()
        
#         print(f"   Filtered to {len(valid_users):,} users with >={self.min_interactions} interactions")
        
#         # Create mappings
#         user_ids = interactions_filtered['UserId'].unique()
#         post_ids = interactions_filtered['PostId'].unique()
        
#         user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
#         post_id_to_idx = {pid: idx for idx, pid in enumerate(post_ids)}
        
#         # Assign implicit scores based on reaction type
#         reaction_scores = {
#             1: 1.0,   # Like
#             2: 2.0,   # Comment
#             3: 3.0,   # Share
#             4: 0.1,   # View
#             5: 1.5    # Save
#         }
        
#         interactions_filtered['score'] = interactions_filtered['ReactionTypeId'].map(
#             lambda x: reaction_scores.get(x, 0.1)
#         )
        
#         # Build sparse matrix
#         rows = interactions_filtered['UserId'].map(user_id_to_idx)
#         cols = interactions_filtered['PostId'].map(post_id_to_idx)
#         data = interactions_filtered['score']
        
#         user_item_matrix = csr_matrix(
#             (data, (rows, cols)),
#             shape=(len(user_ids), len(post_ids))
#         )
        
#         print(f"✅ Matrix shape: {user_item_matrix.shape}")
#         print(f"   Sparsity: {1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.4f}")
        
#         return user_item_matrix, user_ids, post_ids
    
#     def compute_user_similarities(
#         self,
#         user_item_matrix: csr_matrix,
#         user_ids: np.ndarray
#     ) -> Dict[int, List[Tuple[int, float]]]:
#         """
#         Compute user-user similarities
        
#         Returns:
#             user_similarities: Dict[user_id -> [(similar_user_id, score), ...]]
#         """
#         print("\n🔧 Computing User-User Similarities...")
        
#         # Compute cosine similarity (memory intensive!)
#         # Process in chunks if needed
#         print("   Computing cosine similarities...")
#         user_sim_matrix = cosine_similarity(user_item_matrix, dense_output=False)
        
#         print("   Extracting top-K similar users per user...")
#         user_similarities = {}
        
#         for idx, user_id in enumerate(user_ids):
#             # Get similarity scores for this user
#             sim_scores = user_sim_matrix[idx].toarray().flatten()
            
#             # Get top K (excluding self)
#             sim_scores[idx] = -1  # Exclude self
#             top_k_indices = np.argsort(sim_scores)[-self.top_k:][::-1]
            
#             similar_users = [
#                 (user_ids[i], float(sim_scores[i]))
#                 for i in top_k_indices
#                 if sim_scores[i] > 0
#             ]
            
#             user_similarities[user_id] = similar_users
        
#         print(f"✅ Computed similarities for {len(user_similarities):,} users")
        
#         return user_similarities
    
#     def compute_item_similarities(
#         self,
#         user_item_matrix: csr_matrix,
#         post_ids: np.ndarray
#     ) -> Dict[int, List[Tuple[int, float]]]:
#         """
#         Compute item-item similarities
        
#         Returns:
#             item_similarities: Dict[post_id -> [(similar_post_id, score), ...]]
#         """
#         print("\n🔧 Computing Item-Item Similarities...")
        
#         # Transpose to get item-user matrix
#         item_user_matrix = user_item_matrix.T
        
#         print("   Computing cosine similarities...")
#         item_sim_matrix = cosine_similarity(item_user_matrix, dense_output=False)
        
#         print("   Extracting top-K similar items per item...")
#         item_similarities = {}
        
#         for idx, post_id in enumerate(post_ids):
#             sim_scores = item_sim_matrix[idx].toarray().flatten()
            
#             # Get top K (excluding self)
#             sim_scores[idx] = -1
#             top_k_indices = np.argsort(sim_scores)[-self.top_k:][::-1]
            
#             similar_posts = [
#                 (post_ids[i], float(sim_scores[i]))
#                 for i in top_k_indices
#                 if sim_scores[i] > 0
#             ]
            
#             item_similarities[post_id] = similar_posts
        
#         print(f"✅ Computed similarities for {len(item_similarities):,} items")
        
#         return item_similarities
    
#     def build_cf_model(
#         self,
#         interactions_df: pd.DataFrame
#     ) -> Dict:
#         """
#         Complete CF pipeline
        
#         Returns:
#             cf_model: Dict containing all CF components
#         """
#         # Build matrix
#         user_item_matrix, user_ids, post_ids = self.build_user_item_matrix(interactions_df)
        
#         # Compute similarities
#         user_similarities = self.compute_user_similarities(user_item_matrix, user_ids)
#         item_similarities = self.compute_item_similarities(user_item_matrix, post_ids)
        
#         cf_model = {
#             'user_item_matrix': user_item_matrix,
#             'user_ids': user_ids,
#             'post_ids': post_ids,
#             'user_similarities': user_similarities,
#             'item_similarities': item_similarities,
#             'metadata': {
#                 'n_users': len(user_ids),
#                 'n_posts': len(post_ids),
#                 'n_interactions': user_item_matrix.nnz,
#                 'top_k': self.top_k
#             }
#         }
        
#         return cf_model
    
#     def save_cf_model(self, cf_model: Dict, output_path: str):
#         """Save CF model"""
#         with open(output_path, 'wb') as f:
#             pickle.dump(cf_model, f)
        
#         print(f"✅ CF model saved to: {output_path}")


## version cập nhật 27/10 - claude
"""
COLLABORATIVE FILTERING BUILDER
================================
Build CF similarity matrices for user-user and item-item recommendations

Features:
- User-User similarities (top-K neighbors)
- Item-Item similarities (top-K similar posts)
- Sparse matrix optimization for memory efficiency
- Configurable similarity metrics and thresholds
- Save/load from pickle

Usage:
    builder = CFBuilder()
    cf_model = builder.build_cf_model(interactions_df)
    builder.save_cf_model(cf_model, 'cf_model.pkl')
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time

# Setup logging
logger = logging.getLogger(__name__)


class CFBuilder:
    """
    Build Collaborative Filtering Model
    
    Components:
    1. User-Item interaction matrix (sparse)
    2. User-User similarity matrix (top-K)
    3. Item-Item similarity matrix (top-K)
    """
    
    def __init__(
        self,
        min_interactions: int = 3,
        top_k_similar: int = 50,
        min_similarity: float = 0.0
    ):
        """
        Initialize CF Builder
        
        Args:
            min_interactions: Minimum interactions for user to be included
            top_k_similar: Number of similar users/items to keep per entity
            min_similarity: Minimum similarity score threshold
        """
        self.min_interactions = min_interactions
        self.top_k = top_k_similar
        self.min_similarity = min_similarity
        
        logger.info(f"CFBuilder initialized:")
        logger.info(f"   min_interactions: {min_interactions}")
        logger.info(f"   top_k_similar: {top_k_similar}")
        logger.info(f"   min_similarity: {min_similarity}")
    
    # ========================================================================
    # USER-ITEM MATRIX
    # ========================================================================
    
    def build_user_item_matrix(
        self,
        interactions_df: pd.DataFrame,
        user_id_col: str = 'user_id',
        post_id_col: str = 'post_id',
        reaction_type_col: str = 'reaction_type_id'
    ) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """
        Build sparse user-item interaction matrix
        
        Strategy:
        - Use implicit feedback (reaction types → scores)
        - Filter low-activity users (< min_interactions)
        - Build sparse CSR matrix for memory efficiency
        
        Args:
            interactions_df: DataFrame with user-post interactions
            user_id_col: Column name for user ID
            post_id_col: Column name for post ID  
            reaction_type_col: Column name for reaction type
        
        Returns:
            Tuple of (matrix, user_ids_array, post_ids_array)
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"BUILDING USER-ITEM MATRIX")
        logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        if interactions_df.empty:
            logger.warning("Empty interactions DataFrame!")
            return csr_matrix((0, 0)), np.array([]), np.array([])
        
        logger.info(f"Total interactions: {len(interactions_df):,}")
        
        # ────────────────────────────────────────────────────────────────────
        # Filter users with minimum interactions
        # ────────────────────────────────────────────────────────────────────
        
        user_counts = interactions_df.groupby(user_id_col).size()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        
        interactions_filtered = interactions_df[
            interactions_df[user_id_col].isin(valid_users)
        ].copy()
        
        logger.info(f"Filtered users:")
        logger.info(f"   Valid users: {len(valid_users):,} (>= {self.min_interactions} interactions)")
        logger.info(f"   Filtered interactions: {len(interactions_filtered):,}")
        
        if interactions_filtered.empty:
            logger.warning("No valid interactions after filtering!")
            return csr_matrix((0, 0)), np.array([]), np.array([])
        
        # ────────────────────────────────────────────────────────────────────
        # Create ID mappings
        # ────────────────────────────────────────────────────────────────────
        
        user_ids = np.sort(interactions_filtered[user_id_col].unique())
        post_ids = np.sort(interactions_filtered[post_id_col].unique())
        
        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        post_id_to_idx = {pid: idx for idx, pid in enumerate(post_ids)}
        
        logger.info(f"Matrix dimensions:")
        logger.info(f"   Users: {len(user_ids):,}")
        logger.info(f"   Posts: {len(post_ids):,}")
        
        # ────────────────────────────────────────────────────────────────────
        # Assign implicit scores based on reaction type
        # ────────────────────────────────────────────────────────────────────
        
        # Reaction type scoring (adjust based on your data)
        reaction_scores = {
            1: 1.0,   # Like
            2: 2.0,   # Comment (stronger signal)
            3: 3.0,   # Share (strongest signal)
            4: 0.1,   # View (weak signal)
            5: 1.5,   # Save
        }
        
        interactions_filtered['score'] = interactions_filtered[reaction_type_col].map(
            lambda x: reaction_scores.get(x, 0.5)  # Default score if reaction type unknown
        )
        
        logger.info(f"Reaction type distribution:")
        reaction_dist = interactions_filtered[reaction_type_col].value_counts().to_dict()
        for rt, count in sorted(reaction_dist.items()):
            logger.info(f"   Type {rt}: {count:,} ({count/len(interactions_filtered)*100:.1f}%)")
        
        # ────────────────────────────────────────────────────────────────────
        # Build sparse matrix
        # ────────────────────────────────────────────────────────────────────
        
        logger.info(f"Building sparse matrix...")
        
        row_indices = interactions_filtered[user_id_col].map(user_id_to_idx).values
        col_indices = interactions_filtered[post_id_col].map(post_id_to_idx).values
        scores = interactions_filtered['score'].values
        
        user_item_matrix = csr_matrix(
            (scores, (row_indices, col_indices)),
            shape=(len(user_ids), len(post_ids)),
            dtype=np.float32
        )
        
        elapsed_time = time.time() - start_time
        
        # Matrix stats
        density = user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])
        memory_mb = user_item_matrix.data.nbytes / (1024 * 1024)
        
        logger.info(f"✅ User-Item matrix built:")
        logger.info(f"   Shape: {user_item_matrix.shape}")
        logger.info(f"   Non-zero entries: {user_item_matrix.nnz:,}")
        logger.info(f"   Density: {density*100:.4f}%")
        logger.info(f"   Memory: {memory_mb:.1f} MB")
        logger.info(f"   Time: {elapsed_time:.1f}s")
        
        return user_item_matrix, user_ids, post_ids
    
    # ========================================================================
    # USER-USER SIMILARITIES
    # ========================================================================
    
    def compute_user_similarities(
        self,
        user_item_matrix: csr_matrix,
        user_ids: np.ndarray
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Compute user-user similarities (cosine)
        
        Strategy:
        - Compute pairwise cosine similarities between users
        - Keep only top-K similar users per user
        - Filter by minimum similarity threshold
        
        Args:
            user_item_matrix: Sparse user-item matrix
            user_ids: Array of user IDs
        
        Returns:
            Dict {user_id: [(similar_user_id, score), ...]}
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"COMPUTING USER-USER SIMILARITIES")
        logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        if user_item_matrix.shape[0] == 0:
            logger.warning("Empty user-item matrix!")
            return {}
        
        logger.info(f"Computing cosine similarities for {len(user_ids):,} users...")
        
        # Compute similarities (this returns dense matrix)
        user_sim_matrix = cosine_similarity(user_item_matrix, dense_output=False)
        
        logger.info(f"Extracting top-{self.top_k} similar users per user...")
        
        user_similarities = {}
        users_with_neighbors = 0
        
        for idx, user_id in enumerate(user_ids):
            # Get similarity scores for this user
            sim_scores = user_sim_matrix[idx].toarray().flatten()
            
            # Exclude self-similarity
            sim_scores[idx] = -1
            
            # Get top K indices
            top_k_indices = np.argsort(sim_scores)[-self.top_k:][::-1]
            
            # Build list of (similar_user_id, score) tuples
            similar_users = [
                (int(user_ids[i]), float(sim_scores[i]))
                for i in top_k_indices
                if sim_scores[i] >= self.min_similarity
            ]
            
            if similar_users:
                user_similarities[int(user_id)] = similar_users
                users_with_neighbors += 1
        
        elapsed_time = time.time() - start_time
        
        # Statistics
        avg_neighbors = np.mean([len(v) for v in user_similarities.values()]) if user_similarities else 0
        
        logger.info(f"✅ User similarities computed:")
        logger.info(f"   Users with neighbors: {users_with_neighbors:,} / {len(user_ids):,}")
        logger.info(f"   Average neighbors per user: {avg_neighbors:.1f}")
        logger.info(f"   Time: {elapsed_time:.1f}s ({len(user_ids)/elapsed_time:.0f} users/s)")
        
        return user_similarities
    
    # ========================================================================
    # ITEM-ITEM SIMILARITIES
    # ========================================================================
    
    def compute_item_similarities(
        self,
        user_item_matrix: csr_matrix,
        post_ids: np.ndarray
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Compute item-item similarities (cosine)
        
        Strategy:
        - Transpose matrix to get item-user matrix
        - Compute pairwise cosine similarities between items
        - Keep only top-K similar items per item
        
        Args:
            user_item_matrix: Sparse user-item matrix
            post_ids: Array of post IDs
        
        Returns:
            Dict {post_id: [(similar_post_id, score), ...]}
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"COMPUTING ITEM-ITEM SIMILARITIES")
        logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        if user_item_matrix.shape[1] == 0:
            logger.warning("Empty user-item matrix!")
            return {}
        
        logger.info(f"Computing cosine similarities for {len(post_ids):,} items...")
        
        # Transpose to get item-user matrix
        item_user_matrix = user_item_matrix.T
        
        # Compute similarities
        item_sim_matrix = cosine_similarity(item_user_matrix, dense_output=False)
        
        logger.info(f"Extracting top-{self.top_k} similar items per item...")
        
        item_similarities = {}
        items_with_neighbors = 0
        
        for idx, post_id in enumerate(post_ids):
            sim_scores = item_sim_matrix[idx].toarray().flatten()
            
            # Exclude self-similarity
            sim_scores[idx] = -1
            
            # Get top K indices
            top_k_indices = np.argsort(sim_scores)[-self.top_k:][::-1]
            
            # Build list
            similar_items = [
                (int(post_ids[i]), float(sim_scores[i]))
                for i in top_k_indices
                if sim_scores[i] >= self.min_similarity
            ]
            
            if similar_items:
                item_similarities[int(post_id)] = similar_items
                items_with_neighbors += 1
        
        elapsed_time = time.time() - start_time
        
        # Statistics
        avg_neighbors = np.mean([len(v) for v in item_similarities.values()]) if item_similarities else 0
        
        logger.info(f"✅ Item similarities computed:")
        logger.info(f"   Items with neighbors: {items_with_neighbors:,} / {len(post_ids):,}")
        logger.info(f"   Average neighbors per item: {avg_neighbors:.1f}")
        logger.info(f"   Time: {elapsed_time:.1f}s ({len(post_ids)/elapsed_time:.0f} items/s)")
        
        return item_similarities
    
    # ========================================================================
    # COMPLETE CF MODEL
    # ========================================================================
    
    def build_cf_model(
        self,
        interactions_df: pd.DataFrame,
        user_id_col: str = 'user_id',
        post_id_col: str = 'post_id',
        reaction_type_col: str = 'reaction_type_id'
    ) -> Dict:
        """
        Complete CF pipeline: Build user-item matrix + compute similarities
        
        Args:
            interactions_df: DataFrame with user-post interactions
            user_id_col: Column name for user ID
            post_id_col: Column name for post ID
            reaction_type_col: Column name for reaction type
        
        Returns:
            cf_model: Dict containing all CF components
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"BUILDING COMPLETE CF MODEL")
        logger.info(f"{'='*70}")
        
        overall_start = time.time()
        
        # Step 1: Build user-item matrix
        user_item_matrix, user_ids, post_ids = self.build_user_item_matrix(
            interactions_df,
            user_id_col=user_id_col,
            post_id_col=post_id_col,
            reaction_type_col=reaction_type_col
        )
        
        if user_item_matrix.shape[0] == 0:
            logger.error("Failed to build user-item matrix!")
            return {}
        
        # Step 2: Compute user similarities
        user_similarities = self.compute_user_similarities(user_item_matrix, user_ids)
        
        # Step 3: Compute item similarities
        item_similarities = self.compute_item_similarities(user_item_matrix, post_ids)
        
        # Build final model
        cf_model = {
            'user_item_matrix': user_item_matrix,
            'user_ids': user_ids,
            'post_ids': post_ids,
            'user_similarities': user_similarities,
            'item_similarities': item_similarities,
            'metadata': {
                'n_users': len(user_ids),
                'n_posts': len(post_ids),
                'n_interactions': int(user_item_matrix.nnz),
                'top_k': self.top_k,
                'min_interactions': self.min_interactions,
                'min_similarity': self.min_similarity,
                'users_with_neighbors': len(user_similarities),
                'items_with_neighbors': len(item_similarities)
            }
        }
        
        overall_time = time.time() - overall_start
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ CF MODEL COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"   Users: {len(user_ids):,}")
        logger.info(f"   Posts: {len(post_ids):,}")
        logger.info(f"   User similarities: {len(user_similarities):,}")
        logger.info(f"   Item similarities: {len(item_similarities):,}")
        logger.info(f"   Total time: {overall_time:.1f}s")
        
        return cf_model
    
    # ========================================================================
    # SAVE / LOAD
    # ========================================================================
    
    def save_cf_model(self, cf_model: Dict, path: str):
        """
        Save CF model to pickle file
        
        Args:
            cf_model: CF model dict
            path: Output path
        """
        logger.info(f"\n💾 Saving CF model to {path}...")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(cf_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size_mb = path.stat().st_size / (1024 * 1024)
        
        logger.info(f"✅ CF model saved:")
        logger.info(f"   File: {path}")
        logger.info(f"   Size: {file_size_mb:.1f} MB")
        logger.info(f"   Users: {cf_model['metadata']['n_users']:,}")
        logger.info(f"   Posts: {cf_model['metadata']['n_posts']:,}")
    
    @staticmethod
    def load_cf_model(path: str) -> Dict:
        """
        Load CF model from pickle file
        
        Args:
            path: Input path
        
        Returns:
            CF model dict
        """
        logger.info(f"Loading CF model from {path}...")
        
        path = Path(path)
        
        if not path.exists():
            logger.error(f"File not found: {path}")
            return {}
        
        with open(path, 'rb') as f:
            cf_model = pickle.load(f)
        
        logger.info(f"✅ CF model loaded:")
        logger.info(f"   Users: {cf_model['metadata']['n_users']:,}")
        logger.info(f"   Posts: {cf_model['metadata']['n_posts']:,}")
        
        return cf_model


# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

def main():
    """Test CF Builder"""
    
    logger.info(f"{'='*70}")
    logger.info(f"CF BUILDER TEST")
    logger.info(f"{'='*70}")
    
    # Initialize
    builder = CFBuilder(
        min_interactions=2,
        top_k_similar=10,
        min_similarity=0.0
    )
    
    # Create sample data
    np.random.seed(42)
    
    interactions_df = pd.DataFrame({
        'user_id': np.random.randint(1, 20, 100),
        'post_id': np.random.randint(1, 30, 100),
        'reaction_type_id': np.random.choice([1, 2, 3, 4, 5], 100)
    })
    
    logger.info(f"Sample data:")
    logger.info(f"   Interactions: {len(interactions_df)}")
    logger.info(f"   Unique users: {interactions_df['user_id'].nunique()}")
    logger.info(f"   Unique posts: {interactions_df['post_id'].nunique()}")
    
    # Build CF model
    cf_model = builder.build_cf_model(interactions_df)
    
    # Save
    builder.save_cf_model(cf_model, 'test_cf_model.pkl')
    
    # Load
    loaded_model = CFBuilder.load_cf_model('test_cf_model.pkl')
    
    logger.info(f"\n✅ TEST COMPLETE")
    logger.info(f"   User similarities: {len(loaded_model['user_similarities'])}")
    logger.info(f"   Item similarities: {len(loaded_model['item_similarities'])}")


if __name__ == "__main__":
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    main()