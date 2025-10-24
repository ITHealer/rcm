"""
BATCH EMBEDDINGS GENERATOR
==========================
Generate embeddings for new posts and update user embeddings

Schedule: Run every 6 hours
Purpose: Keep embeddings fresh for new content

Flow:
1. Find new posts (created in last 6h)
2. Generate post embeddings
3. Update user embeddings (users active in last 24h)
4. Update Redis cache
5. Save checkpoint

Usage:
    python scripts/offline/batch_embeddings.py
    
Cron:
    0 */6 * * * cd /app && python scripts/offline/batch_embeddings.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pickle
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

# Imports
from recommender.common.data_loader import load_data
from recommender.offline.artifact_manager import ArtifactManager

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️  redis not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class BatchEmbeddingGenerator:
    """
    Generate embeddings in batches
    
    Features:
    - Incremental updates (only new content)
    - User embedding updates (active users)
    - Redis cache updates
    - Checkpoint saving
    """
    
    def __init__(
        self,
        config: Dict,
        models_dir: str = 'models',
        checkpoints_dir: str = 'checkpoints'
    ):
        """
        Initialize batch embedding generator
        
        Args:
            config: Configuration dict
            models_dir: Models directory
            checkpoints_dir: Checkpoints directory
        """
        self.config = config
        self.models_dir = Path(models_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.batch_size = config.get('embeddings', {}).get('batch_size', 512)
        self.model_name = config.get('embeddings', {}).get('model_name', 
                                     'sentence-transformers/all-MiniLM-L6-v2')
        
        # Load model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Artifact manager
        self.artifact_mgr = ArtifactManager(base_dir=str(self.models_dir))
        
        # Load existing embeddings
        self.existing_embeddings = self._load_existing_embeddings()
    
    def _load_existing_embeddings(self) -> Dict:
        """Load existing embeddings from latest version"""
        try:
            version = self.artifact_mgr.get_latest_version()
            logger.info(f"Loading existing embeddings from {version}")
            
            with open(self.models_dir / version / 'embeddings.pkl', 'rb') as f:
                embeddings = pickle.load(f)
            
            logger.info(f"Loaded {len(embeddings.get('post', {}))} post embeddings")
            logger.info(f"Loaded {len(embeddings.get('user', {}))} user embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.warning(f"Could not load existing embeddings: {e}")
            return {'post': {}, 'user': {}}
    
    def generate_post_embeddings(
        self,
        data: Dict,
        hours_lookback: int = 6
    ) -> Dict[int, np.ndarray]:
        """
        Generate embeddings for new posts
        
        Args:
            data: Data dictionary
            hours_lookback: Look back N hours for new posts
            
        Returns:
            Dict mapping post_id to embedding
        """
        logger.info(f"\n{'='*70}")
        logger.info("GENERATING POST EMBEDDINGS")
        logger.info(f"{'='*70}")
        
        posts_df = data['post']
        
        # Filter new posts
        cutoff_time = datetime.now() - timedelta(hours=hours_lookback)
        new_posts = posts_df[
            pd.to_datetime(posts_df['CreateDate']) >= cutoff_time
        ].copy()
        
        logger.info(f"Found {len(new_posts)} new posts in last {hours_lookback}h")
        
        if new_posts.empty:
            logger.info("No new posts to process")
            return {}
        
        # Generate embeddings
        new_embeddings = {}
        
        texts = []
        post_ids = []
        
        for _, post in new_posts.iterrows():
            post_id = int(post['Id'])
            
            # Skip if already embedded
            if post_id in self.existing_embeddings.get('post', {}):
                continue
            
            # Get content
            content = str(post.get('Content', ''))
            
            if len(content) < 10:  # Skip too short
                continue
            
            texts.append(content)
            post_ids.append(post_id)
        
        if not texts:
            logger.info("All new posts already have embeddings")
            return {}
        
        logger.info(f"Generating embeddings for {len(texts)} posts...")
        
        # Batch encoding
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Store
        for post_id, emb in zip(post_ids, embeddings):
            new_embeddings[post_id] = emb.astype(np.float32)
        
        logger.info(f"✅ Generated {len(new_embeddings)} new post embeddings")
        
        return new_embeddings
    
    def update_user_embeddings(
        self,
        data: Dict,
        hours_lookback: int = 24
    ) -> Dict[int, np.ndarray]:
        """
        Update user embeddings for active users
        
        Strategy: Weighted average of interacted posts
        
        Args:
            data: Data dictionary
            hours_lookback: Look back N hours for active users
            
        Returns:
            Dict mapping user_id to embedding
        """
        logger.info(f"\n{'='*70}")
        logger.info("UPDATING USER EMBEDDINGS")
        logger.info(f"{'='*70}")
        
        reactions_df = data['postreaction']
        
        # Get all post embeddings (existing + new)
        all_post_embeddings = self.existing_embeddings.get('post', {})
        
        if not all_post_embeddings:
            logger.warning("No post embeddings available")
            return {}
        
        # Filter active users
        cutoff_time = datetime.now() - timedelta(hours=hours_lookback)
        recent_reactions = reactions_df[
            pd.to_datetime(reactions_df['CreateDate']) >= cutoff_time
        ].copy()
        
        active_users = recent_reactions['UserId'].unique()
        logger.info(f"Found {len(active_users)} active users in last {hours_lookback}h")
        
        # Get interactions for last 14 days (for embedding)
        interaction_cutoff = datetime.now() - timedelta(days=14)
        user_interactions = reactions_df[
            pd.to_datetime(reactions_df['CreateDate']) >= interaction_cutoff
        ].copy()
        
        # Generate user embeddings
        updated_embeddings = {}
        
        for user_id in active_users:
            # Get user's interactions
            user_reactions = user_interactions[
                user_interactions['UserId'] == user_id
            ]
            
            if len(user_reactions) < 5:  # Skip users with too few interactions
                continue
            
            # Collect post embeddings
            post_embs = []
            weights = []
            
            for _, reaction in user_reactions.iterrows():
                post_id = int(reaction['PostId'])
                
                # Check if post has embedding
                if post_id not in all_post_embeddings:
                    continue
                
                post_emb = all_post_embeddings[post_id]
                
                # Weight by action type and recency
                action_weight = self._get_action_weight(int(reaction['ReactionTypeId']))
                
                # Time decay
                days_ago = (datetime.now() - pd.to_datetime(reaction['CreateDate'])).days
                time_weight = np.exp(-days_ago / 7.0)  # 7 day half-life
                
                weight = action_weight * time_weight
                
                post_embs.append(post_emb)
                weights.append(weight)
            
            if not post_embs:
                continue
            
            # Weighted average
            post_embs = np.array(post_embs)
            weights = np.array(weights)
            weights = weights / (weights.sum() + 1e-8)  # Normalize
            
            user_emb = np.average(post_embs, axis=0, weights=weights)
            
            # Normalize
            user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
            
            updated_embeddings[int(user_id)] = user_emb.astype(np.float32)
        
        logger.info(f"✅ Updated {len(updated_embeddings)} user embeddings")
        
        return updated_embeddings
    
    def _get_action_weight(self, reaction_type_id: int) -> float:
        """Get weight for action type"""
        weights = {
            1: 1.0,   # like
            2: 1.5,   # comment
            3: 2.0,   # share
            4: 0.5,   # view
            5: 1.2,   # save
            6: -3.0,  # hide
            7: -5.0   # report
        }
        return weights.get(reaction_type_id, 1.0)
    
    def save_checkpoint(self, embeddings: Dict, suffix: str = ""):
        """Save checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"embeddings_{timestamp}{suffix}.pkl"
        path = self.checkpoints_dir / filename
        
        with open(path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        logger.info(f"Saved checkpoint: {path}")
    
    def update_redis_cache(self, embeddings: Dict):
        """Update Redis cache with new embeddings"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, skipping cache update")
            return
        
        try:
            # Connect to Redis
            redis_client = redis.Redis(
                host='localhost',
                port=6380,
                db=0,
                decode_responses=False
            )
            
            logger.info("\n📦 Updating Redis cache...")
            
            # Update post embeddings
            if 'post' in embeddings:
                post_embs = embeddings['post']
                logger.info(f"Updating {len(post_embs)} post embeddings...")
                
                pipe = redis_client.pipeline()
                for post_id, emb in post_embs.items():
                    key = f"post:{post_id}:embedding"
                    value = emb.tobytes()
                    pipe.setex(key, 2592000, value)  # 30 days TTL
                pipe.execute()
            
            # Update user embeddings
            if 'user' in embeddings:
                user_embs = embeddings['user']
                logger.info(f"Updating {len(user_embs)} user embeddings...")
                
                pipe = redis_client.pipeline()
                for user_id, emb in user_embs.items():
                    key = f"user:{user_id}:embedding"
                    value = emb.tobytes()
                    pipe.setex(key, 604800, value)  # 7 days TTL
                pipe.execute()
            
            logger.info("✅ Redis cache updated")
            
        except Exception as e:
            logger.error(f"Redis update failed: {e}")


def main():
    """Main execution"""
    logger.info("="*70)
    logger.info("BATCH EMBEDDINGS GENERATOR")
    logger.info("="*70)
    logger.info(f"Started at: {datetime.now()}")
    
    # Load config
    with open('configs/config_offline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    logger.info("\nLoading data...")
    data = load_data('dataset')
    
    # Initialize generator
    generator = BatchEmbeddingGenerator(config)
    
    # Generate post embeddings (last 6h)
    new_post_embeddings = generator.generate_post_embeddings(data, hours_lookback=6)
    
    # Update user embeddings (active in last 24h)
    updated_user_embeddings = generator.update_user_embeddings(data, hours_lookback=24)
    
    # Merge with existing
    all_embeddings = generator.existing_embeddings.copy()
    
    # Update post embeddings
    if 'post' not in all_embeddings:
        all_embeddings['post'] = {}
    all_embeddings['post'].update(new_post_embeddings)
    
    # Update user embeddings
    if 'user' not in all_embeddings:
        all_embeddings['user'] = {}
    all_embeddings['user'].update(updated_user_embeddings)
    
    # Save checkpoint
    generator.save_checkpoint(all_embeddings, suffix="_incremental")
    
    # Update Redis cache
    updated_embeddings = {
        'post': new_post_embeddings,
        'user': updated_user_embeddings
    }
    generator.update_redis_cache(updated_embeddings)
    
    logger.info("\n" + "="*70)
    logger.info("BATCH EMBEDDINGS COMPLETE!")
    logger.info("="*70)
    logger.info(f"New post embeddings: {len(new_post_embeddings)}")
    logger.info(f"Updated user embeddings: {len(updated_user_embeddings)}")
    logger.info(f"Total post embeddings: {len(all_embeddings['post'])}")
    logger.info(f"Total user embeddings: {len(all_embeddings['user'])}")
    logger.info(f"Finished at: {datetime.now()}")


if __name__ == "__main__":
    main()