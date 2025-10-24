"""
USER EMBEDDINGS COMPUTER
========================
Compute user embeddings from interaction history using weighted average

Strategy:
- Weighted average of post embeddings
- Time decay weights (exponential decay)
- L2 normalization
- Cold start handling

Update: Weekly (with model training)
Storage: Redis (TTL=7d) + PostgreSQL backup
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
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
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️  redis not available. Install: pip install redis")

try:
    import psycopg2
    from psycopg2.extras import execute_batch
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


class UserEmbeddingComputer:
    """
    Compute user embeddings from interaction history
    
    Strategy: Weighted average with time decay
    """
    
    def __init__(
        self,
        db_connection = None,
        redis_client = None,
        embedding_dim: int = 384,
        lookback_days: int = 14,
        batch_size: int = 1000,
        time_decay_half_life: float = 7.0
    ):
        """
        Initialize user embedding computer
        
        Args:
            db_connection: PostgreSQL connection
            redis_client: Redis client
            embedding_dim: Embedding dimension (default: 384 for all-MiniLM-L6-v2)
            lookback_days: Days of interaction history to use
            batch_size: Batch size for processing
            time_decay_half_life: Half-life for time decay (days)
        """
        self.db_connection = db_connection
        self.redis_client = redis_client
        self.embedding_dim = embedding_dim
        self.lookback_days = lookback_days
        self.batch_size = batch_size
        self.time_decay_half_life = time_decay_half_life
        
        logger.info(f"Initializing UserEmbeddingComputer...")
        logger.info(f"Embedding dim: {embedding_dim}")
        logger.info(f"Lookback days: {lookback_days}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Time decay half-life: {time_decay_half_life} days")
        
        # CSV mode (fallback)
        self.use_csv = False
        self.csv_dir = Path('dataset')
        self.models_dir = Path('models')
        
        # Action weights (for multi-action weighting)
        self.action_weights = {
            'like': 1.0,
            'love': 2.0,
            'comment': 3.0,
            'share': 5.0,
            'save': 4.0,
            'view': 0.1
        }
        
        # ReactionTypeId mapping (for CSV data)
        self.reaction_type_map = {
            1: 'like',
            2: 'love',
            3: 'share',
            4: 'view',
            5: 'save',
            # Add more as needed
        }
        
        # Statistics
        self.stats = {
            'total_users': 0,
            'users_processed': 0,
            'cold_start_users': 0,
            'avg_norm': [],
            'avg_interactions': []
        }
    
    def enable_csv_mode(self, csv_dir: str = 'dataset', models_dir: str = 'models'):
        """Enable CSV mode"""
        self.use_csv = True
        self.csv_dir = Path(csv_dir)
        self.models_dir = Path(models_dir)
        logger.info(f"CSV mode enabled. Data: {csv_dir}, Models: {models_dir}")
    
    # ========================================================================
    # TIME DECAY
    # ========================================================================
    
    def compute_time_decay(self, age_days: float) -> float:
        """
        Compute time decay weight using exponential decay
        
        Formula: weight = 0.5 ^ (age_days / half_life)
        
        Args:
            age_days: Age in days
        
        Returns:
            Decay weight (0-1)
        
        Examples:
            age=0 days: weight=1.0
            age=7 days: weight=0.5 (half-life)
            age=14 days: weight=0.25
            age=30 days: weight=0.06
        """
        return 0.5 ** (age_days / self.time_decay_half_life)
    
    # ========================================================================
    # COMPUTE SINGLE USER
    # ========================================================================
    
    def compute_single_user(
        self,
        user_id: int,
        lookback_days: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute embedding for single user
        
        Process:
        1. Get user interactions (last N days)
        2. Get post embeddings for interacted posts
        3. Apply time decay weights
        4. Apply action weights
        5. Weighted average
        6. L2 normalize
        
        Args:
            user_id: User ID
            lookback_days: Days of history (default: self.lookback_days)
        
        Returns:
            embedding: (embedding_dim,) normalized vector
        """
        if lookback_days is None:
            lookback_days = self.lookback_days
        
        # Get interactions
        if self.use_csv:
            interactions = self._get_user_interactions_csv(user_id, lookback_days)
            post_embeddings = self._load_post_embeddings_csv()
        else:
            interactions = self._get_user_interactions_db(user_id, lookback_days)
            post_embeddings = self._load_post_embeddings_db()
        
        # Handle cold start (no interactions)
        if len(interactions) == 0:
            logger.debug(f"User {user_id}: No interactions (cold start)")
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Compute weighted average
        weighted_sum = np.zeros(self.embedding_dim, dtype=np.float32)
        total_weight = 0.0
        
        now = datetime.now()
        
        for _, interaction in interactions.iterrows():
            # Get post embedding
            post_id = int(interaction.get('PostId', interaction.get('post_id')))
            
            if post_id not in post_embeddings:
                continue  # Skip if embedding missing
            
            post_embedding = post_embeddings[post_id]
            
            # Time decay weight
            created_at = interaction.get('CreateDate', interaction.get('created_at'))
            if isinstance(created_at, str):
                created_at = pd.to_datetime(created_at)
            
            age_days = (now - created_at).total_seconds() / 86400.0
            time_weight = self.compute_time_decay(age_days)
            
            # Action weight
            action = interaction.get('action', None)
            if action is None:
                # Map ReactionTypeId to action
                reaction_type = interaction.get('ReactionTypeId')
                action = self.reaction_type_map.get(reaction_type, 'view')
            
            action_weight = self.action_weights.get(action, 1.0)
            
            # Combined weight
            combined_weight = time_weight * action_weight
            
            # Accumulate
            weighted_sum += post_embedding * combined_weight
            total_weight += combined_weight
        
        # Compute average
        if total_weight > 0:
            user_embedding = weighted_sum / total_weight
            
            # L2 normalize
            norm = np.linalg.norm(user_embedding)
            if norm > 0:
                user_embedding = user_embedding / norm
            
            return user_embedding
        else:
            # No valid interactions
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _get_user_interactions_csv(
        self,
        user_id: int,
        lookback_days: int
    ) -> pd.DataFrame:
        """Get user interactions from CSV"""
        # Load interactions
        interactions_path = self.csv_dir / 'PostReaction.csv'
        
        if not interactions_path.exists():
            return pd.DataFrame()
        
        # Read only relevant columns
        df = pd.read_csv(
            interactions_path,
            usecols=['UserId', 'PostId', 'ReactionTypeId', 'CreateDate']
        )
        
        # Filter by user
        df = df[df['UserId'] == user_id]
        
        # Parse dates
        df['CreateDate'] = pd.to_datetime(df['CreateDate'])
        
        # Filter by time
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        df = df[df['CreateDate'] >= cutoff_date]
        
        return df
    
    def _get_user_interactions_db(
        self,
        user_id: int,
        lookback_days: int
    ) -> pd.DataFrame:
        """Get user interactions from database"""
        if not PSYCOPG2_AVAILABLE or self.db_connection is None:
            return pd.DataFrame()
        
        query = """
        SELECT 
            user_id,
            post_id,
            action_type as action,
            created_at
        FROM interactions
        WHERE 
            user_id = %s
            AND created_at >= NOW() - INTERVAL '%s days'
        ORDER BY created_at DESC
        """
        
        cursor = self.db_connection.cursor()
        cursor.execute(query, (user_id, lookback_days))
        
        rows = cursor.fetchall()
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=['user_id', 'post_id', 'action', 'created_at'])
        
        return df
    
    def _load_post_embeddings_csv(self) -> Dict[int, np.ndarray]:
        """Load post embeddings from pickle file"""
        embeddings_file = self.models_dir / 'post_embeddings.pkl'
        
        if not embeddings_file.exists():
            logger.warning(f"Post embeddings not found: {embeddings_file}")
            return {}
        
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        
        return embeddings
    
    def _load_post_embeddings_db(self) -> Dict[int, np.ndarray]:
        """Load post embeddings from database"""
        if not PSYCOPG2_AVAILABLE or self.db_connection is None:
            return {}
        
        query = "SELECT post_id, embedding FROM post_embeddings"
        
        cursor = self.db_connection.cursor()
        cursor.execute(query)
        
        rows = cursor.fetchall()
        
        embeddings = {}
        for post_id, embedding_bytes in rows:
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings[post_id] = embedding
        
        return embeddings
    
    # ========================================================================
    # COMPUTE ALL ACTIVE USERS
    # ========================================================================
    
    def compute_all_active_users(
        self,
        active_days: int = 30,
        min_interactions: int = 5
    ) -> Dict[int, np.ndarray]:
        """
        Compute embeddings for all active users
        
        Args:
            active_days: Consider users active in last N days
            min_interactions: Minimum interactions to compute embedding
        
        Returns:
            Dict[user_id -> embedding]
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"COMPUTING USER EMBEDDINGS")
        logger.info(f"{'='*70}")
        logger.info(f"Active days: {active_days}")
        logger.info(f"Lookback days: {self.lookback_days}")
        logger.info(f"Min interactions: {min_interactions}")
        
        start_time = datetime.now()
        
        # Get active users
        if self.use_csv:
            active_users = self._get_active_users_csv(active_days, min_interactions)
        else:
            active_users = self._get_active_users_db(active_days, min_interactions)
        
        logger.info(f"\nFound {len(active_users):,} active users")
        
        if len(active_users) == 0:
            logger.warning("No active users found!")
            return {}
        
        # Process in batches
        embeddings = {}
        n_batches = (len(active_users) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"\nProcessing in {n_batches} batches of {self.batch_size}...")
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, len(active_users))
            
            batch_users = active_users[batch_start:batch_end]
            
            batch_start_time = datetime.now()
            
            # Process batch
            for user_id in batch_users:
                try:
                    embedding = self.compute_single_user(user_id)
                    
                    # Check if cold start
                    if np.allclose(embedding, 0):
                        self.stats['cold_start_users'] += 1
                    else:
                        embeddings[user_id] = embedding
                        
                        # Track statistics
                        norm = np.linalg.norm(embedding)
                        self.stats['avg_norm'].append(norm)
                    
                    self.stats['users_processed'] += 1
                    
                except Exception as e:
                    logger.error(f"Error computing embedding for user {user_id}: {e}")
                    continue
            
            batch_time = (datetime.now() - batch_start_time).total_seconds()
            
            logger.info(
                f"Batch {batch_idx+1}/{n_batches}: "
                f"Processed {len(batch_users)} users ({batch_time:.1f}s)"
            )
        
        # Final statistics
        total_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ USER EMBEDDINGS COMPUTED")
        logger.info(f"{'='*70}")
        logger.info(f"Total users: {len(active_users):,}")
        logger.info(f"Users with embeddings: {len(embeddings):,}")
        logger.info(f"Cold start users: {self.stats['cold_start_users']:,} ({self.stats['cold_start_users']/len(active_users)*100:.1f}%)")
        
        if self.stats['avg_norm']:
            avg_norm = np.mean(self.stats['avg_norm'])
            logger.info(f"Average embedding norm: {avg_norm:.4f}")
        
        logger.info(f"Total time: {total_time/60:.1f}m {total_time%60:.0f}s")
        logger.info(f"Throughput: {len(active_users) / total_time:.0f} users/second")
        
        return embeddings
    
    def _get_active_users_csv(
        self,
        active_days: int,
        min_interactions: int
    ) -> List[int]:
        """Get active users from CSV"""
        # Load interactions
        interactions_path = self.csv_dir / 'PostReaction.csv'
        
        if not interactions_path.exists():
            return []
        
        df = pd.read_csv(interactions_path, usecols=['UserId', 'CreateDate'])
        
        # Parse dates
        df['CreateDate'] = pd.to_datetime(df['CreateDate'])
        
        # Filter by time
        cutoff_date = datetime.now() - timedelta(days=active_days)
        df = df[df['CreateDate'] >= cutoff_date]
        
        # Count interactions per user
        user_counts = df['UserId'].value_counts()
        
        # Filter by min interactions
        active_users = user_counts[user_counts >= min_interactions].index.tolist()
        
        return active_users
    
    def _get_active_users_db(
        self,
        active_days: int,
        min_interactions: int
    ) -> List[int]:
        """Get active users from database"""
        if not PSYCOPG2_AVAILABLE or self.db_connection is None:
            return []
        
        query = """
        SELECT user_id, COUNT(*) as interaction_count
        FROM interactions
        WHERE created_at >= NOW() - INTERVAL '%s days'
        GROUP BY user_id
        HAVING COUNT(*) >= %s
        ORDER BY interaction_count DESC
        """
        
        cursor = self.db_connection.cursor()
        cursor.execute(query, (active_days, min_interactions))
        
        rows = cursor.fetchall()
        
        active_users = [row[0] for row in rows]
        
        return active_users
    
    # ========================================================================
    # SAVE TO REDIS
    # ========================================================================
    
    def save_to_redis(
        self,
        embeddings: Dict[int, np.ndarray],
        ttl_days: int = 7
    ):
        """
        Save embeddings to Redis
        
        Args:
            embeddings: Dict[user_id -> embedding]
            ttl_days: Time to live in days
        """
        if not REDIS_AVAILABLE or self.redis_client is None:
            logger.warning("Redis not available. Skipping Redis save.")
            return
        
        logger.info(f"\n💾 Saving to Redis (TTL={ttl_days} days)...")
        
        ttl_seconds = ttl_days * 86400
        
        saved_count = 0
        
        for user_id, embedding in embeddings.items():
            try:
                # Convert to bytes
                embedding_bytes = embedding.astype(np.float32).tobytes()
                
                # Save to Redis
                key = f"user:{user_id}:embedding"
                self.redis_client.set(key, embedding_bytes, ex=ttl_seconds)
                
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Error saving to Redis for user {user_id}: {e}")
                continue
        
        logger.info(f"✅ Saved {saved_count:,} embeddings to Redis")
    
    # ========================================================================
    # SAVE TO DATABASE
    # ========================================================================
    
    def save_to_db(self, embeddings: Dict[int, np.ndarray]):
        """
        Save embeddings to PostgreSQL
        
        Args:
            embeddings: Dict[user_id -> embedding]
        """
        if self.use_csv:
            self._save_to_file(embeddings)
        else:
            self._save_to_postgres(embeddings)
    
    def _save_to_file(self, embeddings: Dict[int, np.ndarray]):
        """Save to pickle file"""
        output_file = self.models_dir / 'user_embeddings.pkl'
        
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        logger.info(f"✅ Saved {len(embeddings):,} embeddings to {output_file}")
    
    def _save_to_postgres(self, embeddings: Dict[int, np.ndarray]):
        """Save to PostgreSQL"""
        if not PSYCOPG2_AVAILABLE or self.db_connection is None:
            logger.warning("PostgreSQL not available. Skipping database save.")
            return
        
        logger.info(f"\n💾 Saving to PostgreSQL...")
        
        cursor = self.db_connection.cursor()
        
        # Prepare data
        data = []
        for user_id, embedding in embeddings.items():
            embedding_bytes = embedding.astype(np.float32).tobytes()
            data.append((user_id, embedding_bytes, datetime.now()))
        
        # Batch insert
        insert_query = """
        INSERT INTO user_embeddings (user_id, embedding, updated_at)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE
        SET embedding = EXCLUDED.embedding, updated_at = EXCLUDED.updated_at
        """
        
        execute_batch(cursor, insert_query, data, page_size=1000)
        self.db_connection.commit()
        
        logger.info(f"✅ Saved {len(embeddings):,} embeddings to database")
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    def validate_embeddings(self, embeddings: Dict[int, np.ndarray]):
        """
        Validate computed embeddings
        
        Checks:
        - Norms should be ~1.0 (normalized)
        - No NaN values
        - No inf values
        """
        logger.info(f"\n🔍 Validating embeddings...")
        
        norms = []
        nan_count = 0
        inf_count = 0
        
        for user_id, embedding in embeddings.items():
            norm = np.linalg.norm(embedding)
            norms.append(norm)
            
            if np.isnan(embedding).any():
                nan_count += 1
            
            if np.isinf(embedding).any():
                inf_count += 1
        
        # Statistics
        avg_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        logger.info(f"Validation results:")
        logger.info(f"  Average norm: {avg_norm:.4f} (should be ~1.0)")
        logger.info(f"  Std norm: {std_norm:.4f}")
        logger.info(f"  NaN embeddings: {nan_count}")
        logger.info(f"  Inf embeddings: {inf_count}")
        
        # Check for issues
        if abs(avg_norm - 1.0) > 0.1:
            logger.warning(f"⚠️  Average norm {avg_norm:.4f} deviates from 1.0")
        
        if nan_count > 0:
            logger.error(f"❌ Found {nan_count} embeddings with NaN values!")
        
        if inf_count > 0:
            logger.error(f"❌ Found {inf_count} embeddings with inf values!")
        
        if nan_count == 0 and inf_count == 0 and abs(avg_norm - 1.0) < 0.1:
            logger.info(f"✅ Validation passed!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    
    print("="*70)
    print("USER EMBEDDINGS COMPUTER")
    print("="*70)
    
    # Initialize
    computer = UserEmbeddingComputer(
        embedding_dim=384,
        lookback_days=14,
        batch_size=1000,
        time_decay_half_life=7.0
    )
    
    # Enable CSV mode
    computer.enable_csv_mode(csv_dir='dataset', models_dir='models')
    
    # Compute embeddings
    embeddings = computer.compute_all_active_users(
        active_days=30,
        min_interactions=5
    )
    
    # Validate
    if len(embeddings) > 0:
        computer.validate_embeddings(embeddings)
        
        # Save
        computer.save_to_db(embeddings)
    
    print(f"\n✅ Computed embeddings for {len(embeddings):,} users")


if __name__ == "__main__":
    main()