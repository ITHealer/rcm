# """
# Post & User Embeddings Generation
# Purpose: Tạo vector representations cho posts và users
# """

# import numpy as np
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from typing import Dict, List
# import pickle
# from tqdm import tqdm
# import torch

# class EmbeddingGenerator:
#     """
#     Generate embeddings for posts and users
#     """
    
#     def __init__(self, config: dict):
#         self.config = config
#         self.model_name = config['embeddings']['model_name']
#         self.embedding_dim = config['embeddings']['embedding_dim']
#         self.batch_size = config['embeddings']['batch_size']
        
#         print(f"📦 Loading embedding model: {self.model_name}")
#         self.model = SentenceTransformer(self.model_name)
        
#         # Use GPU if available
#         if torch.cuda.is_available():
#             self.model = self.model.cuda()
#             print("✅ Using GPU for embeddings")
#         else:
#             print("⚠️  Using CPU for embeddings")
    
#     def generate_post_embeddings(self, posts_df: pd.DataFrame) -> Dict[int, np.ndarray]:
#         """
#         Generate embeddings for all posts
        
#         Args:
#             posts_df: DataFrame với columns [Id, Content, Hashtags]
        
#         Returns:
#             post_embeddings: Dict[post_id -> embedding vector]
#         """
#         print("\n🔧 Generating Post Embeddings...")
#         print(f"   Total posts: {len(posts_df):,}")
        
#         post_embeddings = {}
        
#         # Prepare texts
#         texts = []
#         post_ids = []
        
#         for idx, row in posts_df.iterrows():
#             # Combine content + hashtags
#             text = str(row['Content'])
            
#             if pd.notna(row.get('Hashtags')) and row.get('Hashtags'):
#                 hashtags = row['Hashtags'] if isinstance(row['Hashtags'], list) else []
#                 if hashtags:
#                     text += " " + " ".join([f"#{tag}" for tag in hashtags])
            
#             texts.append(text)
#             post_ids.append(row['Id'])
        
#         # Batch encode
#         print(f"   Encoding {len(texts):,} posts in batches...")
        
#         embeddings = self.model.encode(
#             texts,
#             batch_size=self.batch_size,
#             show_progress_bar=True,
#             convert_to_numpy=True,
#             normalize_embeddings=True  # Normalize for cosine similarity
#         )
        
#         # Store in dict
#         for post_id, embedding in zip(post_ids, embeddings):
#             post_embeddings[post_id] = embedding
        
#         print(f"✅ Generated {len(post_embeddings):,} post embeddings")
#         print(f"   Embedding dimension: {embeddings.shape[1]}")
        
#         return post_embeddings
    
#     def generate_user_embeddings(
#         self, 
#         interactions_df: pd.DataFrame,
#         post_embeddings: Dict[int, np.ndarray]
#     ) -> Dict[int, np.ndarray]:
#         """
#         Generate user embeddings by aggregating their interaction history
        
#         User embedding = Weighted average of posts they interacted with
#         Weights: Like=1.0, Comment=2.0, Share=3.0
        
#         Args:
#             interactions_df: User interactions
#             post_embeddings: Post embeddings từ bước trước
        
#         Returns:
#             user_embeddings: Dict[user_id -> embedding vector]
#         """
#         print("\n🔧 Generating User Embeddings...")
        
#         # Reaction weights
#         reaction_weights = {
#             1: 1.0,   # Like
#             2: 2.0,   # Comment
#             3: 3.0,   # Share
#             4: 0.1,   # View
#             5: 1.5    # Save
#         }
        
#         user_embeddings = {}
        
#         # Group by user
#         for user_id, group in tqdm(interactions_df.groupby('UserId'), desc="Processing users"):
#             weighted_embeddings = []
#             weights = []
            
#             for _, row in group.iterrows():
#                 post_id = row['PostId']
#                 reaction_type = row['ReactionTypeId']
                
#                 # Get post embedding
#                 if post_id in post_embeddings:
#                     post_emb = post_embeddings[post_id]
#                     weight = reaction_weights.get(reaction_type, 0.1)
                    
#                     weighted_embeddings.append(post_emb * weight)
#                     weights.append(weight)
            
#             if weighted_embeddings:
#                 # Weighted average
#                 user_emb = np.average(weighted_embeddings, axis=0, weights=weights)
                
#                 # Normalize
#                 norm = np.linalg.norm(user_emb)
#                 if norm > 0:
#                     user_emb = user_emb / norm
                
#                 user_embeddings[user_id] = user_emb
        
#         print(f"✅ Generated {len(user_embeddings):,} user embeddings")
        
#         return user_embeddings
    
#     def save_embeddings(
#         self, 
#         post_embeddings: Dict[int, np.ndarray],
#         user_embeddings: Dict[int, np.ndarray],
#         output_path: str
#     ):
#         """Save embeddings to disk"""
#         embeddings = {
#             'post': post_embeddings,
#             'user': user_embeddings,
#             'metadata': {
#                 'model_name': self.model_name,
#                 'embedding_dim': self.embedding_dim,
#                 'n_posts': len(post_embeddings),
#                 'n_users': len(user_embeddings)
#             }
#         }
        
#         with open(output_path, 'wb') as f:
#             pickle.dump(embeddings, f)
        
#         print(f"✅ Embeddings saved to: {output_path}")


"""
EMBEDDING GENERATOR
===================
Generate post and user embeddings for content-based recommendation

Features:
- Post embeddings: Sentence-BERT encoding of post content
- User embeddings: Weighted aggregation of posts user interacted with
- Batch processing for efficiency
- GPU support (optional)
- Save/load from pickle

Usage:
    generator = EmbeddingGenerator()
    post_embs = generator.generate_post_embeddings(posts_df)
    user_embs = generator.generate_user_embeddings(interactions_df, post_embs)
    generator.save_embeddings({'posts': post_embs, 'users': user_embs}, 'embeddings.pkl')
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, List, Optional
from pathlib import Path
import time

# Setup logging
logger = logging.getLogger(__name__)

# Try import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.error("❌ sentence-transformers not available. Install: pip install sentence-transformers")


class EmbeddingGenerator:
    """
    Generate embeddings for posts and users
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: str = 'cpu',
        batch_size: int = 32
    ):
        """
        Initialize embedding generator
        
        Args:
            model_name: Sentence-BERT model name
                - 'all-MiniLM-L6-v2': 384-dim, fast, good quality (recommended)
                - 'all-mpnet-base-v2': 768-dim, slower, better quality
            device: 'cpu' or 'cuda'
            batch_size: Batch size for encoding
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")
        
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        # Load model
        logger.info(f"Loading Sentence-BERT model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"✅ Model loaded: {model_name}")
        logger.info(f"   Dimension: {self.dimension}")
        logger.info(f"   Device: {device}")
    
    # ========================================================================
    # POST EMBEDDINGS
    # ========================================================================
    
    def generate_post_embeddings(
        self,
        posts_df: pd.DataFrame,
        content_col: str = 'Content',
        post_id_col: str = 'PostId',
        show_progress: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        Generate embeddings for posts
        
        Args:
            posts_df: DataFrame with posts
            content_col: Column name for post content
            post_id_col: Column name for post ID
            show_progress: Show progress bar
        
        Returns:
            Dict {post_id: embedding_vector}
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"GENERATING POST EMBEDDINGS")
        logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        if posts_df.empty:
            logger.warning("Empty posts DataFrame!")
            return {}
        
        # Prepare texts
        logger.info(f"Preparing texts for {len(posts_df):,} posts...")
        
        texts = posts_df[content_col].fillna('').astype(str).tolist()
        post_ids = posts_df[post_id_col].astype(int).tolist()
        
        # Filter empty texts
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        texts = [texts[i] for i in valid_indices]
        post_ids = [post_ids[i] for i in valid_indices]
        
        logger.info(f"Valid posts: {len(texts):,} (filtered {len(posts_df) - len(texts):,} empty)")
        
        if not texts:
            logger.warning("No valid posts to embed!")
            return {}
        
        # Encode in batches
        logger.info(f"Encoding with batch_size={self.batch_size}...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        
        # Create dict
        post_embeddings = dict(zip(post_ids, embeddings))
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"✅ Post embeddings generated:")
        logger.info(f"   Count: {len(post_embeddings):,}")
        logger.info(f"   Dimension: {self.dimension}")
        logger.info(f"   Time: {elapsed_time:.1f}s ({len(post_embeddings)/elapsed_time:.0f} posts/s)")
        
        return post_embeddings
    
    # ========================================================================
    # USER EMBEDDINGS
    # ========================================================================
    
    def generate_user_embeddings(
        self,
        interactions_df: pd.DataFrame,
        post_embeddings: Dict[int, np.ndarray],
        user_id_col: str = 'user_id',
        post_id_col: str = 'post_id',
        weight_col: Optional[str] = 'time_decay_weight',
        min_interactions: int = 3
    ) -> Dict[int, np.ndarray]:
        """
        Generate user embeddings by aggregating posts they interacted with
        
        Strategy:
        - For each user, get posts they interacted with
        - Compute weighted average of post embeddings
        - Weights: time_decay_weight (if available) or equal weights
        
        Args:
            interactions_df: DataFrame with user-post interactions
            post_embeddings: Dict {post_id: embedding}
            user_id_col: Column name for user ID
            post_id_col: Column name for post ID
            weight_col: Column name for weights (time decay)
            min_interactions: Minimum interactions to generate embedding
        
        Returns:
            Dict {user_id: embedding_vector}
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"GENERATING USER EMBEDDINGS")
        logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        if interactions_df.empty:
            logger.warning("Empty interactions DataFrame!")
            return {}
        
        if not post_embeddings:
            logger.warning("Empty post_embeddings dict!")
            return {}
        
        logger.info(f"Processing {len(interactions_df):,} interactions...")
        logger.info(f"Post embeddings available: {len(post_embeddings):,}")
        
        user_embeddings = {}
        users_processed = 0
        users_skipped = 0
        
        # Group by user
        for user_id, group in interactions_df.groupby(user_id_col):
            user_id = int(user_id)
            
            # Get post IDs
            post_ids = group[post_id_col].astype(int).tolist()
            
            # Filter posts with embeddings
            valid_posts = [pid for pid in post_ids if pid in post_embeddings]
            
            if len(valid_posts) < min_interactions:
                users_skipped += 1
                continue
            
            # Get embeddings
            embs = np.array([post_embeddings[pid] for pid in valid_posts])
            
            # Get weights
            if weight_col and weight_col in group.columns:
                weights = group[group[post_id_col].astype(int).isin(valid_posts)][weight_col].values
                weights = weights[:len(embs)]  # Ensure same length
            else:
                weights = np.ones(len(embs))
            
            # Weighted average
            user_emb = np.average(embs, axis=0, weights=weights)
            
            # Normalize
            user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
            
            user_embeddings[user_id] = user_emb
            users_processed += 1
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"✅ User embeddings generated:")
        logger.info(f"   Count: {len(user_embeddings):,}")
        logger.info(f"   Dimension: {self.dimension}")
        logger.info(f"   Skipped: {users_skipped:,} (< {min_interactions} interactions)")
        logger.info(f"   Time: {elapsed_time:.1f}s ({len(user_embeddings)/elapsed_time:.0f} users/s)")
        
        return user_embeddings
    
    # ========================================================================
    # SAVE / LOAD
    # ========================================================================
    
    def save_embeddings(
        self,
        embeddings: Dict,
        path: str
    ):
        """
        Save embeddings to pickle file
        
        Args:
            embeddings: Dict {'posts': {...}, 'users': {...}}
            path: Output path
        """
        logger.info(f"\n💾 Saving embeddings to {path}...")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Get file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        
        logger.info(f"✅ Embeddings saved:")
        logger.info(f"   File: {path}")
        logger.info(f"   Size: {file_size_mb:.1f} MB")
        logger.info(f"   Posts: {len(embeddings.get('posts', {})):,}")
        logger.info(f"   Users: {len(embeddings.get('users', {})):,}")
    
    @staticmethod
    def load_embeddings(path: str) -> Dict:
        """
        Load embeddings from pickle file
        
        Args:
            path: Input path
        
        Returns:
            Dict {'posts': {...}, 'users': {...}}
        """
        logger.info(f"Loading embeddings from {path}...")
        
        path = Path(path)
        
        if not path.exists():
            logger.error(f"File not found: {path}")
            return {}
        
        with open(path, 'rb') as f:
            embeddings = pickle.load(f)
        
        logger.info(f"✅ Embeddings loaded:")
        logger.info(f"   Posts: {len(embeddings.get('posts', {})):,}")
        logger.info(f"   Users: {len(embeddings.get('users', {})):,}")
        
        return embeddings
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def compute_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            emb1: First embedding
            emb2: Second embedding
        
        Returns:
            Similarity score (0-1)
        """
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
    
    def get_info(self) -> Dict:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'device': self.device,
            'batch_size': self.batch_size
        }


# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

def main():
    """Test embedding generation"""
    
    logger.info(f"{'='*70}")
    logger.info(f"EMBEDDING GENERATOR TEST")
    logger.info(f"{'='*70}")
    
    # Initialize
    generator = EmbeddingGenerator(
        model_name='all-MiniLM-L6-v2',
        device='cpu',
        batch_size=32
    )
    
    # Create sample data
    posts_df = pd.DataFrame({
        'PostId': [1, 2, 3, 4, 5],
        'Content': [
            'I love machine learning!',
            'Python is great for data science',
            'Deep learning with PyTorch',
            'Natural language processing',
            'Computer vision with CNNs'
        ]
    })
    
    interactions_df = pd.DataFrame({
        'user_id': [101, 101, 102, 102, 103, 103, 103],
        'post_id': [1, 2, 2, 3, 1, 2, 4],
        'time_decay_weight': [1.0, 0.8, 1.0, 0.9, 0.7, 1.0, 0.85]
    })
    
    # Generate post embeddings
    post_embeddings = generator.generate_post_embeddings(posts_df)
    
    # Generate user embeddings
    user_embeddings = generator.generate_user_embeddings(
        interactions_df,
        post_embeddings
    )
    
    # Test similarity
    if len(post_embeddings) >= 2:
        emb1 = post_embeddings[1]
        emb2 = post_embeddings[2]
        sim = generator.compute_similarity(emb1, emb2)
        logger.info(f"\nSimilarity between post 1 and 2: {sim:.4f}")
    
    # Save
    embeddings = {
        'posts': post_embeddings,
        'users': user_embeddings
    }
    
    generator.save_embeddings(embeddings, 'test_embeddings.pkl')
    
    # Load
    loaded = EmbeddingGenerator.load_embeddings('test_embeddings.pkl')
    
    logger.info(f"\n✅ TEST COMPLETE")
    logger.info(f"   Post embeddings: {len(loaded['posts'])}")
    logger.info(f"   User embeddings: {len(loaded['users'])}")


if __name__ == "__main__":
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    main()