"""
Post & User Embeddings Generation
Purpose: Tạo vector representations cho posts và users
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import pickle
from tqdm import tqdm
import torch

class EmbeddingGenerator:
    """
    Generate embeddings for posts and users
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config['embeddings']['model_name']
        self.embedding_dim = config['embeddings']['embedding_dim']
        self.batch_size = config['embeddings']['batch_size']
        
        print(f"📦 Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Use GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("✅ Using GPU for embeddings")
        else:
            print("⚠️  Using CPU for embeddings")
    
    def generate_post_embeddings(self, posts_df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Generate embeddings for all posts
        
        Args:
            posts_df: DataFrame với columns [Id, Content, Hashtags]
        
        Returns:
            post_embeddings: Dict[post_id -> embedding vector]
        """
        print("\n🔧 Generating Post Embeddings...")
        print(f"   Total posts: {len(posts_df):,}")
        
        post_embeddings = {}
        
        # Prepare texts
        texts = []
        post_ids = []
        
        for idx, row in posts_df.iterrows():
            # Combine content + hashtags
            text = str(row['Content'])
            
            if pd.notna(row.get('Hashtags')) and row.get('Hashtags'):
                hashtags = row['Hashtags'] if isinstance(row['Hashtags'], list) else []
                if hashtags:
                    text += " " + " ".join([f"#{tag}" for tag in hashtags])
            
            texts.append(text)
            post_ids.append(row['Id'])
        
        # Batch encode
        print(f"   Encoding {len(texts):,} posts in batches...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        # Store in dict
        for post_id, embedding in zip(post_ids, embeddings):
            post_embeddings[post_id] = embedding
        
        print(f"✅ Generated {len(post_embeddings):,} post embeddings")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        
        return post_embeddings
    
    def generate_user_embeddings(
        self, 
        interactions_df: pd.DataFrame,
        post_embeddings: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """
        Generate user embeddings by aggregating their interaction history
        
        User embedding = Weighted average of posts they interacted with
        Weights: Like=1.0, Comment=2.0, Share=3.0
        
        Args:
            interactions_df: User interactions
            post_embeddings: Post embeddings từ bước trước
        
        Returns:
            user_embeddings: Dict[user_id -> embedding vector]
        """
        print("\n🔧 Generating User Embeddings...")
        
        # Reaction weights
        reaction_weights = {
            1: 1.0,   # Like
            2: 2.0,   # Comment
            3: 3.0,   # Share
            4: 0.1,   # View
            5: 1.5    # Save
        }
        
        user_embeddings = {}
        
        # Group by user
        for user_id, group in tqdm(interactions_df.groupby('UserId'), desc="Processing users"):
            weighted_embeddings = []
            weights = []
            
            for _, row in group.iterrows():
                post_id = row['PostId']
                reaction_type = row['ReactionTypeId']
                
                # Get post embedding
                if post_id in post_embeddings:
                    post_emb = post_embeddings[post_id]
                    weight = reaction_weights.get(reaction_type, 0.1)
                    
                    weighted_embeddings.append(post_emb * weight)
                    weights.append(weight)
            
            if weighted_embeddings:
                # Weighted average
                user_emb = np.average(weighted_embeddings, axis=0, weights=weights)
                
                # Normalize
                norm = np.linalg.norm(user_emb)
                if norm > 0:
                    user_emb = user_emb / norm
                
                user_embeddings[user_id] = user_emb
        
        print(f"✅ Generated {len(user_embeddings):,} user embeddings")
        
        return user_embeddings
    
    def save_embeddings(
        self, 
        post_embeddings: Dict[int, np.ndarray],
        user_embeddings: Dict[int, np.ndarray],
        output_path: str
    ):
        """Save embeddings to disk"""
        embeddings = {
            'post': post_embeddings,
            'user': user_embeddings,
            'metadata': {
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'n_posts': len(post_embeddings),
                'n_users': len(user_embeddings)
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"✅ Embeddings saved to: {output_path}")