"""
COLLABORATIVE FILTERING SIMILARITY
===================================
Compute user-user similarity for CF recall channel

Algorithm: Cosine similarity on interaction vectors
Update: Daily
Storage: Redis cache (top 50 similar users per user)
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

# Try imports
try:
    from scipy.sparse import csr_matrix, save_npz, load_npz
    from sklearn.metrics.pairwise import cosine_similarity
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available. Install: pip install scipy scikit-learn")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️  redis not available. Install: pip install redis")

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("⚠️  psycopg2 not available. Install: pip install psycopg2-binary")


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CollaborativeFilteringSimilarity:
    """
    Compute user-user similarity using collaborative filtering
    
    Algorithm: Cosine similarity on user-item interaction matrix
    """
    
    def __init__(
        self,
        db_connection = None,
        redis_client = None,
        lookback_days: int = 30,
        top_k: int = 50,
        batch_size: int = 1000,
        models_dir: str = 'models'
    ):
        """
        Initialize CF similarity computer
        
        Args:
            db_connection: PostgreSQL connection
            redis_client: Redis client
            lookback_days: Days of interaction history
            top_k: Top K similar users to save
            batch_size: Batch size for similarity computation
            models_dir: Directory for models
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy and scikit-learn required")
        
        self.db_connection = db_connection
        self.redis_client = redis_client
        self.lookback_days = lookback_days
        self.top_k = top_k
        self.batch_size = batch_size
        self.models_dir = Path(models_dir)
        
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initializing CollaborativeFilteringSimilarity...")
        logger.info(f"Lookback days: {lookback_days}")
        logger.info(f"Top K: {top_k}")
        logger.info(f"Batch size: {batch_size}")
        
        # Action weights
        self.action_weights = {
            'view': 0.5,
            'like': 1.0,
            'love': 2.0,
            'comment': 1.5,
            'share': 2.0,
            'save': 1.8
        }
        
        # ReactionTypeId mapping
        self.reaction_type_map = {
            1: 'like',
            2: 'love',
            3: 'share',
            4: 'view',
            5: 'save',
        }
        
        # CSV mode
        self.use_csv = False
        self.csv_dir = Path('dataset')
        
        # Statistics
        self.stats = {
            'n_users': 0,
            'n_posts': 0,
            'n_interactions': 0,
            'sparsity': 0.0,
            'avg_interactions_per_user': 0.0,
            'users_with_similarities': 0,
            'avg_similarity_score': []
        }
    
    def enable_csv_mode(self, csv_dir: str = 'dataset'):
        """Enable CSV mode"""
        self.use_csv = True
        self.csv_dir = Path(csv_dir)
        logger.info(f"CSV mode enabled. Data directory: {csv_dir}")
    
    # ========================================================================
    # BUILD USER-ITEM MATRIX
    # ========================================================================
    
    def build_user_item_matrix(
        self,
        lookback_days: Optional[int] = None
    ) -> Tuple[csr_matrix, Dict[int, int], List[int]]:
        """
        Build sparse user-item interaction matrix
        
        Args:
            lookback_days: Days of history (default: self.lookback_days)
        
        Returns:
            matrix: Sparse (n_users, n_posts) matrix
            user_id_map: {user_id: index}
            user_ids: List of user IDs
        """
        if lookback_days is None:
            lookback_days = self.lookback_days
        
        logger.info(f"\n{'='*70}")
        logger.info(f"BUILDING USER-ITEM MATRIX")
        logger.info(f"{'='*70}")
        logger.info(f"Lookback days: {lookback_days}")
        
        start_time = datetime.now()
        
        # Load interactions
        if self.use_csv:
            interactions = self._load_interactions_csv(lookback_days)
        else:
            interactions = self._load_interactions_db(lookback_days)
        
        if len(interactions) == 0:
            logger.error("No interactions found!")
            return csr_matrix((0, 0)), {}, []
        
        logger.info(f"Loaded {len(interactions):,} interactions")
        
        # Apply action weights
        logger.info("\nApplying action weights...")
        
        if 'action' in interactions.columns:
            interactions['score'] = interactions['action'].map(self.action_weights).fillna(1.0)
        else:
            # Map ReactionTypeId to action
            interactions['action'] = interactions['ReactionTypeId'].map(self.reaction_type_map).fillna('view')
            interactions['score'] = interactions['action'].map(self.action_weights).fillna(1.0)
        
        # Aggregate by (user, post)
        logger.info("Aggregating scores...")
        
        if 'UserId' in interactions.columns:
            user_col = 'UserId'
            post_col = 'PostId'
        else:
            user_col = 'user_id'
            post_col = 'post_id'
        
        user_post_scores = interactions.groupby([user_col, post_col])['score'].sum().reset_index()
        
        logger.info(f"Aggregated to {len(user_post_scores):,} unique (user, post) pairs")
        
        # Create mappings
        logger.info("\nCreating ID mappings...")
        
        user_ids = sorted(user_post_scores[user_col].unique())
        post_ids = sorted(user_post_scores[post_col].unique())
        
        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        post_id_to_idx = {pid: idx for idx, pid in enumerate(post_ids)}
        
        logger.info(f"Users: {len(user_ids):,}")
        logger.info(f"Posts: {len(post_ids):,}")
        
        # Build sparse matrix
        logger.info("\nBuilding sparse matrix...")
        
        rows = [user_id_to_idx[uid] for uid in user_post_scores[user_col]]
        cols = [post_id_to_idx[pid] for pid in user_post_scores[post_col]]
        data = user_post_scores['score'].values
        
        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(post_ids)),
            dtype=np.float32
        )
        
        # Statistics
        self.stats['n_users'] = len(user_ids)
        self.stats['n_posts'] = len(post_ids)
        self.stats['n_interactions'] = len(user_post_scores)
        self.stats['sparsity'] = 100.0 * (1.0 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
        self.stats['avg_interactions_per_user'] = matrix.nnz / matrix.shape[0]
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ USER-ITEM MATRIX BUILT")
        logger.info(f"{'='*70}")
        logger.info(f"Shape: {matrix.shape}")
        logger.info(f"Non-zero: {matrix.nnz:,}")
        logger.info(f"Sparsity: {self.stats['sparsity']:.4f}%")
        logger.info(f"Avg interactions/user: {self.stats['avg_interactions_per_user']:.1f}")
        logger.info(f"Time: {total_time:.1f}s")
        
        # Save matrix
        matrix_path = self.models_dir / 'user_item_matrix.npz'
        save_npz(matrix_path, matrix)
        logger.info(f"Saved matrix to {matrix_path}")
        
        # Save mappings
        mappings = {
            'user_ids': user_ids,
            'user_id_to_idx': user_id_to_idx,
            'post_ids': post_ids,
            'post_id_to_idx': post_id_to_idx
        }
        
        mappings_path = self.models_dir / 'cf_mappings.pkl'
        with open(mappings_path, 'wb') as f:
            pickle.dump(mappings, f)
        logger.info(f"Saved mappings to {mappings_path}")
        
        return matrix, user_id_to_idx, user_ids
    
    def _load_interactions_csv(self, lookback_days: int) -> pd.DataFrame:
        """Load interactions from CSV"""
        interactions_path = self.csv_dir / 'PostReaction.csv'
        
        if not interactions_path.exists():
            logger.error(f"File not found: {interactions_path}")
            return pd.DataFrame()
        
        # Read CSV
        df = pd.read_csv(interactions_path)
        
        # Parse dates
        df['CreateDate'] = pd.to_datetime(df['CreateDate'])
        
        # Filter by time
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        df = df[df['CreateDate'] >= cutoff_date]
        
        return df
    
    def _load_interactions_db(self, lookback_days: int) -> pd.DataFrame:
        """Load interactions from database"""
        if not PSYCOPG2_AVAILABLE or self.db_connection is None:
            return pd.DataFrame()
        
        query = """
        SELECT 
            user_id,
            post_id,
            action_type as action,
            created_at
        FROM interactions
        WHERE created_at >= NOW() - INTERVAL '%s days'
        """
        
        cursor = self.db_connection.cursor()
        cursor.execute(query, (lookback_days,))
        
        rows = cursor.fetchall()
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=['user_id', 'post_id', 'action', 'created_at'])
        
        return df
    
    # ========================================================================
    # COMPUTE SIMILARITIES
    # ========================================================================
    
    def compute_similarities(
        self,
        matrix: csr_matrix,
        user_ids: List[int],
        top_k: Optional[int] = None
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Compute user-user cosine similarities
        
        Memory-efficient implementation using batches
        
        Args:
            matrix: User-item matrix (n_users, n_posts)
            user_ids: List of user IDs
            top_k: Top K similar users (default: self.top_k)
        
        Returns:
            {user_id: [(similar_user_id, score), ...]}
        """
        if top_k is None:
            top_k = self.top_k
        
        logger.info(f"\n{'='*70}")
        logger.info(f"COMPUTING USER-USER SIMILARITIES")
        logger.info(f"{'='*70}")
        logger.info(f"Top K: {top_k}")
        logger.info(f"Batch size: {self.batch_size}")
        
        start_time = datetime.now()
        
        n_users = matrix.shape[0]
        n_batches = (n_users + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Users: {n_users:,}")
        logger.info(f"Batches: {n_batches}")
        
        similarities_dict = {}
        
        # Process in batches to avoid memory issues
        for batch_idx in tqdm(range(n_batches), desc="Computing similarities"):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, n_users)
            
            # Get batch of users
            batch_matrix = matrix[batch_start:batch_end]
            
            # Compute similarity with ALL users
            # This returns (batch_size, n_users) matrix
            batch_similarities = cosine_similarity(batch_matrix, matrix, dense_output=False)
            
            # Convert to dense for top-k extraction
            batch_similarities_dense = batch_similarities.toarray()
            
            # Get top K for each user in batch
            for i in range(batch_similarities_dense.shape[0]):
                user_idx = batch_start + i
                user_id = user_ids[user_idx]
                
                user_sims = batch_similarities_dense[i]
                
                # Get top K (excluding self)
                # Sort in descending order
                top_k_indices = np.argsort(user_sims)[::-1]
                
                # Filter out self and get top K
                top_similar = []
                for idx in top_k_indices:
                    if idx == user_idx:
                        continue  # Skip self
                    
                    similar_user_id = user_ids[idx]
                    similarity_score = float(user_sims[idx])
                    
                    if similarity_score > 0:  # Only positive similarities
                        top_similar.append((similar_user_id, similarity_score))
                    
                    if len(top_similar) >= top_k:
                        break
                
                if len(top_similar) > 0:
                    similarities_dict[user_id] = top_similar
                    
                    # Track statistics
                    self.stats['avg_similarity_score'].extend([score for _, score in top_similar])
        
        # Final statistics
        total_time = (datetime.now() - start_time).total_seconds()
        
        self.stats['users_with_similarities'] = len(similarities_dict)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ SIMILARITIES COMPUTED")
        logger.info(f"{'='*70}")
        logger.info(f"Users with similarities: {len(similarities_dict):,}")
        
        if self.stats['avg_similarity_score']:
            avg_sim = np.mean(self.stats['avg_similarity_score'])
            max_sim = np.max(self.stats['avg_similarity_score'])
            min_sim = np.min(self.stats['avg_similarity_score'])
            logger.info(f"Average similarity: {avg_sim:.4f}")
            logger.info(f"Max similarity: {max_sim:.4f}")
            logger.info(f"Min similarity: {min_sim:.4f}")
        
        logger.info(f"Time: {total_time/60:.1f}m {total_time%60:.0f}s")
        logger.info(f"Throughput: {n_users / total_time:.0f} users/second")
        
        # Save similarities
        similarities_path = self.models_dir / 'user_similarities.pkl'
        with open(similarities_path, 'wb') as f:
            pickle.dump(similarities_dict, f)
        logger.info(f"Saved similarities to {similarities_path}")
        
        return similarities_dict
    
    # ========================================================================
    # SAVE TO REDIS
    # ========================================================================
    
    def save_to_redis(
        self,
        similarities: Dict[int, List[Tuple[int, float]]],
        ttl_hours: int = 24
    ):
        """
        Save similarities to Redis
        
        Args:
            similarities: {user_id: [(similar_user_id, score), ...]}
            ttl_hours: Time to live in hours
        """
        if not REDIS_AVAILABLE or self.redis_client is None:
            logger.warning("Redis not available. Skipping Redis save.")
            return
        
        logger.info(f"\n💾 Saving to Redis (TTL={ttl_hours} hours)...")
        
        ttl_seconds = ttl_hours * 3600
        
        saved_count = 0
        
        for user_id, similar_users in tqdm(similarities.items(), desc="Saving to Redis"):
            try:
                # Convert to JSON format
                similar_users_json = [
                    {'user_id': int(uid), 'similarity': float(score)}
                    for uid, score in similar_users
                ]
                
                # Save to Redis
                key = f"cf:similar:{user_id}"
                value = json.dumps(similar_users_json)
                
                self.redis_client.set(key, value, ex=ttl_seconds)
                
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Error saving to Redis for user {user_id}: {e}")
                continue
        
        logger.info(f"✅ Saved {saved_count:,} user similarities to Redis")
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    def validate_similarities(
        self,
        similarities: Dict[int, List[Tuple[int, float]]]
    ):
        """
        Validate computed similarities
        
        Checks:
        - Similarity scores in [0, 1]
        - No self-similarities
        - Sorted by score (descending)
        """
        logger.info(f"\n🔍 Validating similarities...")
        
        issues = []
        
        for user_id, similar_users in similarities.items():
            # Check for self-similarity
            for similar_user_id, score in similar_users:
                if similar_user_id == user_id:
                    issues.append(f"User {user_id} has self-similarity")
                
                # Check score range
                if not (0 <= score <= 1):
                    issues.append(f"User {user_id}: similarity {score:.4f} out of range [0, 1]")
            
            # Check sorted
            scores = [score for _, score in similar_users]
            if scores != sorted(scores, reverse=True):
                issues.append(f"User {user_id}: similarities not sorted descending")
        
        if issues:
            logger.warning(f"Found {len(issues)} validation issues:")
            for issue in issues[:10]:  # Show first 10
                logger.warning(f"  - {issue}")
        else:
            logger.info("✅ Validation passed!")
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def run_pipeline(self):
        """
        Run complete CF similarity pipeline
        
        Steps:
        1. Build user-item matrix
        2. Compute similarities
        3. Validate
        4. Save to Redis
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"COLLABORATIVE FILTERING SIMILARITY PIPELINE")
        logger.info(f"{'='*70}")
        
        start_time = datetime.now()
        
        # Step 1: Build matrix
        matrix, user_id_to_idx, user_ids = self.build_user_item_matrix()
        
        if matrix.shape[0] == 0:
            logger.error("Failed to build matrix. Exiting.")
            return
        
        # Step 2: Compute similarities
        similarities = self.compute_similarities(matrix, user_ids)
        
        # Step 3: Validate
        self.validate_similarities(similarities)
        
        # Step 4: Save to Redis
        self.save_to_redis(similarities)
        
        # Summary
        total_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ PIPELINE COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total time: {total_time/60:.1f}m {total_time%60:.0f}s")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    
    print("="*70)
    print("COLLABORATIVE FILTERING SIMILARITY")
    print("="*70)
    
    # Initialize
    cf_similarity = CollaborativeFilteringSimilarity(
        lookback_days=30,
        top_k=50,
        batch_size=1000,
        models_dir='models'
    )
    
    # Enable CSV mode
    cf_similarity.enable_csv_mode(csv_dir='dataset')
    
    # Run pipeline
    cf_similarity.run_pipeline()
    
    print(f"\n✅ CF similarity computation complete!")


if __name__ == "__main__":
    main()