"""
Data Loading Pipeline with Time Decay
======================================
Supports both CSV (development) and PostgreSQL (production)

Features:
- Exponential time decay: weight = 0.5^(days/half_life)
- Action multipliers (view, like, comment, share, hide)
- Memory efficient with chunking
- Performance optimized for 1.4M+ rows
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader with time decay support
    Handles both CSV files and PostgreSQL database
    """
    
    # Action multipliers mapping
    ACTION_WEIGHTS = {
        'view': 0.5,
        'like': 1.0,
        'comment': 1.5,
        'share': 2.0,
        'save': 1.2,
        'hide': -3.0,
        'report': -5.0
    }
    
    # Map ReactionTypeId to action names (based on your schema)
    REACTION_TYPE_MAPPING = {
        1: 'like',      # Like
        2: 'comment',   # Comment
        3: 'share',     # Share
        4: 'view',      # View
        5: 'save',      # Save
        6: 'hide',      # Hide (if exists)
        7: 'report'     # Report (if exists)
    }
    
    def __init__(
        self, 
        db_connection=None, 
        config: Dict = None,
        data_dir: str = 'dataset'
    ):
        """
        Initialize DataLoader
        
        Args:
            db_connection: PostgreSQL connection (optional)
            config: Configuration dict
            data_dir: Directory containing CSV files (for development)
        """
        self.db_connection = db_connection
        self.config = config or {}
        self.data_dir = Path(data_dir)
        
        # Configuration
        self.lookback_days = self.config.get('lookback_days', 14)
        self.half_life_days = self.config.get('half_life_days', 7.0)
        self.min_weight = self.config.get('min_weight', 0.01)
        self.chunk_size = self.config.get('chunk_size', 100000)
        
        logger.info(f"DataLoader initialized with:")
        logger.info(f"  Lookback days: {self.lookback_days}")
        logger.info(f"  Half-life: {self.half_life_days} days")
        logger.info(f"  Min weight: {self.min_weight}")
        logger.info(f"  Chunk size: {self.chunk_size:,}")
    
    def load_training_data(
        self, 
        lookback_days: Optional[int] = None,
        use_csv: bool = True
    ) -> pd.DataFrame:
        """
        Load training data with lookback window
        
        Args:
            lookback_days: Number of days to look back (default: from config)
            use_csv: If True, load from CSV; else from PostgreSQL
        
        Returns:
            DataFrame with columns: user_id, post_id, action, created_at
        """
        lookback_days = lookback_days or self.lookback_days
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"LOADING TRAINING DATA")
        logger.info(f"{'='*70}")
        logger.info(f"Lookback window: {lookback_days} days")
        logger.info(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
        
        if use_csv or self.db_connection is None:
            df = self._load_from_csv()
        else:
            df = self._load_from_postgresql(cutoff_date)
        
        # Validate data
        self._validate_data(df)
        
        # Filter by date
        df = self._filter_by_date(df, cutoff_date)
        
        logger.info(f"✅ Loaded {len(df):,} interactions")
        logger.info(f"   Date range: {df['created_at'].min()} to {df['created_at'].max()}")
        logger.info(f"   Unique users: {df['user_id'].nunique():,}")
        logger.info(f"   Unique posts: {df['post_id'].nunique():,}")
        
        return df
    
    def _load_from_csv(self) -> pd.DataFrame:
        """Load from CSV files (development mode)"""
        logger.info("📂 Loading from CSV files...")
        
        csv_path = self.data_dir / 'PostReaction.csv'
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"PostReaction.csv not found in {self.data_dir}\n"
                f"Expected path: {csv_path}"
            )
        
        # Define column mappings and dtypes
        dtype_mapping = {
            'Id': 'int64',
            'UserId': 'int64',
            'PostId': 'int64',
            'ReactionTypeId': 'int8'
        }
        
        # Read CSV in chunks for memory efficiency
        chunks = []
        
        for chunk in pd.read_csv(
            csv_path, 
            dtype=dtype_mapping,
            parse_dates=['CreateDate'],
            chunksize=self.chunk_size
        ):
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        
        logger.info(f"   Loaded {len(df):,} rows from CSV")
        
        # Rename columns to standard format
        df = df.rename(columns={
            'UserId': 'user_id',
            'PostId': 'post_id',
            'ReactionTypeId': 'reaction_type_id',
            'CreateDate': 'created_at'
        })
        
        # Map ReactionTypeId to action names
        df['action'] = df['reaction_type_id'].map(self.REACTION_TYPE_MAPPING)
        
        # Handle unknown reaction types
        unknown_reactions = df[df['action'].isna()]['reaction_type_id'].unique()
        if len(unknown_reactions) > 0:
            logger.warning(f"⚠️  Unknown reaction types: {unknown_reactions}")
            logger.warning(f"   Mapping to 'view' by default")
            df['action'] = df['action'].fillna('view')
        
        return df[['user_id', 'post_id', 'action', 'created_at']]
    
    def _load_from_postgresql(self, cutoff_date: datetime) -> pd.DataFrame:
        """Load from PostgreSQL database (production mode)"""
        logger.info("🗄️  Loading from PostgreSQL...")
        
        if self.db_connection is None:
            raise ValueError("Database connection not provided")
        
        query = """
        SELECT 
            user_id,
            post_id,
            action,
            created_at
        FROM interactions
        WHERE created_at >= %s
        ORDER BY created_at DESC
        """
        
        try:
            df = pd.read_sql_query(
                query,
                self.db_connection,
                params=[cutoff_date]
            )
            
            logger.info(f"   Loaded {len(df):,} rows from PostgreSQL")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading from PostgreSQL: {e}")
            raise
    
    def _filter_by_date(
        self, 
        df: pd.DataFrame, 
        cutoff_date: datetime
    ) -> pd.DataFrame:
        """Filter interactions by date"""
        logger.info(f"\n🔍 Filtering by date...")
        
        original_count = len(df)
        
        # Ensure created_at is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Filter
        df = df[df['created_at'] >= cutoff_date].copy()
        
        filtered_count = original_count - len(df)
        
        if filtered_count > 0:
            logger.info(f"   Filtered out {filtered_count:,} old interactions")
        
        return df
    
    def _validate_data(self, df: pd.DataFrame):
        """Validate loaded data"""
        logger.info("\n✓ Validating data...")
        
        # Check required columns
        required_cols = ['user_id', 'post_id', 'action', 'created_at']
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for nulls
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            logger.warning("⚠️  Found null values:")
            for col, count in null_counts[null_counts > 0].items():
                logger.warning(f"   {col}: {count:,} nulls")
        
        # Check data types
        if not pd.api.types.is_integer_dtype(df['user_id']):
            logger.warning(f"   Converting user_id to int")
            df['user_id'] = df['user_id'].astype('int64')
        
        if not pd.api.types.is_integer_dtype(df['post_id']):
            logger.warning(f"   Converting post_id to int")
            df['post_id'] = df['post_id'].astype('int64')
        
        logger.info("   ✅ Data validation passed")
    
    def apply_time_decay(
        self, 
        df: pd.DataFrame,
        half_life_days: Optional[float] = None,
        reference_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Apply exponential time decay to interactions
        
        Formula: weight = action_multiplier × 0.5^(days_ago / half_life)
        
        Args:
            df: DataFrame with 'created_at' and 'action' columns
            half_life_days: Half-life for decay (default: from config)
            reference_date: Reference date for decay (default: now)
        
        Returns:
            DataFrame with 'weight' column added
        """
        half_life_days = half_life_days or self.half_life_days
        reference_date = reference_date or datetime.now()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"APPLYING TIME DECAY")
        logger.info(f"{'='*70}")
        logger.info(f"Half-life: {half_life_days} days")
        logger.info(f"Reference date: {reference_date.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Min weight threshold: {self.min_weight}")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Ensure created_at is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Calculate days ago
        df['days_ago'] = (reference_date - df['created_at']).dt.total_seconds() / 86400
        
        # Calculate time decay factor
        # Formula: 0.5^(days_ago / half_life)
        df['time_decay'] = np.power(0.5, df['days_ago'] / half_life_days)
        
        # Get action multipliers
        df['action_multiplier'] = df['action'].map(self.ACTION_WEIGHTS)
        
        # Handle unknown actions (default to 1.0)
        unknown_actions = df[df['action_multiplier'].isna()]['action'].unique()
        if len(unknown_actions) > 0:
            logger.warning(f"⚠️  Unknown actions: {unknown_actions}")
            logger.warning(f"   Using default multiplier: 1.0")
            df['action_multiplier'] = df['action_multiplier'].fillna(1.0)
        
        # Calculate final weight
        df['weight'] = df['action_multiplier'] * df['time_decay']
        
        # Apply minimum weight threshold
        df['weight'] = df['weight'].clip(lower=self.min_weight)
        
        # Log statistics
        self._log_weight_statistics(df)
        
        # Clean up temporary columns
        df = df.drop(columns=['days_ago', 'time_decay', 'action_multiplier'])
        
        logger.info(f"✅ Time decay applied successfully")
        
        return df
    
    def _log_weight_statistics(self, df: pd.DataFrame):
        """Log weight distribution statistics"""
        logger.info(f"\n📊 Weight Statistics:")
        logger.info(f"   Mean: {df['weight'].mean():.4f}")
        logger.info(f"   Median: {df['weight'].median():.4f}")
        logger.info(f"   Std: {df['weight'].std():.4f}")
        logger.info(f"   Min: {df['weight'].min():.4f}")
        logger.info(f"   Max: {df['weight'].max():.4f}")
        
        # Weight distribution by action
        logger.info(f"\n📊 Weight by Action:")
        weight_by_action = df.groupby('action')['weight'].agg(['mean', 'count'])
        for action, stats in weight_by_action.iterrows():
            logger.info(f"   {action:10s}: mean={stats['mean']:.4f}, count={stats['count']:,}")
        
        # Weight distribution over time
        logger.info(f"\n📊 Weight Over Time:")
        df['date'] = df['created_at'].dt.date
        weight_by_date = df.groupby('date')['weight'].mean().tail(7)
        for date, weight in weight_by_date.items():
            logger.info(f"   {date}: {weight:.4f}")
    
    def load_and_prepare_training_data(
        self,
        lookback_days: Optional[int] = None,
        half_life_days: Optional[float] = None,
        use_csv: bool = True
    ) -> pd.DataFrame:
        """
        Convenience method: Load data + apply time decay in one call
        
        Returns:
            DataFrame with 'weight' column
        """
        # Load data
        df = self.load_training_data(
            lookback_days=lookback_days,
            use_csv=use_csv
        )
        
        # Apply time decay
        df = self.apply_time_decay(
            df,
            half_life_days=half_life_days
        )
        
        return df
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_days: int = 3,
        val_days: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/val/test split
        
        Args:
            df: DataFrame with 'created_at' column
            test_days: Days for test set
            val_days: Days for validation set
        
        Returns:
            (train_df, val_df, test_df)
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"CREATING TRAIN/VAL/TEST SPLIT")
        logger.info(f"{'='*70}")
        
        # Sort by date
        df = df.sort_values('created_at').copy()
        
        # Calculate split dates
        max_date = df['created_at'].max()
        test_start = max_date - timedelta(days=test_days)
        val_start = test_start - timedelta(days=val_days)
        
        # Split
        train_df = df[df['created_at'] < val_start].copy()
        val_df = df[(df['created_at'] >= val_start) & (df['created_at'] < test_start)].copy()
        test_df = df[df['created_at'] >= test_start].copy()
        
        logger.info(f"Split dates:")
        logger.info(f"   Train: up to {val_start.strftime('%Y-%m-%d')}")
        logger.info(f"   Val: {val_start.strftime('%Y-%m-%d')} to {test_start.strftime('%Y-%m-%d')}")
        logger.info(f"   Test: from {test_start.strftime('%Y-%m-%d')}")
        
        logger.info(f"\nSplit sizes:")
        logger.info(f"   Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"   Val: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"   Test: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_training_data(
    lookback_days: int = 14,
    data_dir: str = 'dataset',
    db_connection=None
) -> pd.DataFrame:
    """
    Convenience function to load training data
    
    Args:
        lookback_days: Days to look back
        data_dir: Directory with CSV files
        db_connection: PostgreSQL connection (optional)
    
    Returns:
        DataFrame with interactions
    """
    config = {
        'lookback_days': lookback_days,
        'half_life_days': 7.0,
        'min_weight': 0.01
    }
    
    loader = DataLoader(
        db_connection=db_connection,
        config=config,
        data_dir=data_dir
    )
    
    use_csv = db_connection is None
    
    return loader.load_training_data(use_csv=use_csv)


def apply_time_decay(
    df: pd.DataFrame,
    half_life_days: float = 7.0,
    min_weight: float = 0.01
) -> pd.DataFrame:
    """
    Convenience function to apply time decay
    
    Args:
        df: DataFrame with 'created_at' and 'action' columns
        half_life_days: Half-life for decay
        min_weight: Minimum weight threshold
    
    Returns:
        DataFrame with 'weight' column
    """
    config = {
        'half_life_days': half_life_days,
        'min_weight': min_weight
    }
    
    loader = DataLoader(config=config)
    
    return loader.apply_time_decay(df)


# =============================================================================
# MAIN - FOR TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the DataLoader with your CSV files
    """
    
    print("="*70)
    print("TESTING DATA LOADER")
    print("="*70)
    
    # Configuration
    config = {
        'lookback_days': 14,
        'half_life_days': 7.0,
        'min_weight': 0.01,
        'chunk_size': 100000
    }
    
    # Initialize loader
    loader = DataLoader(
        db_connection=None,  # Use CSV mode
        config=config,
        data_dir='dataset'
    )
    
    # Test 1: Load data
    print("\n" + "="*70)
    print("TEST 1: LOAD TRAINING DATA")
    print("="*70)
    
    df = loader.load_training_data(use_csv=True)
    
    print(f"\n✅ Loaded {len(df):,} interactions")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Test 2: Apply time decay
    print("\n" + "="*70)
    print("TEST 2: APPLY TIME DECAY")
    print("="*70)
    
    df_weighted = loader.apply_time_decay(df)
    
    print(f"\n✅ Applied time decay")
    print(f"\nFirst 5 rows with weights:")
    print(df_weighted[['user_id', 'post_id', 'action', 'created_at', 'weight']].head())
    
    # Test 3: Load and prepare in one call
    print("\n" + "="*70)
    print("TEST 3: LOAD AND PREPARE (ONE CALL)")
    print("="*70)
    
    df_prepared = loader.load_and_prepare_training_data(use_csv=True)
    
    print(f"\n✅ Data prepared: {len(df_prepared):,} rows")
    
    # Test 4: Create train/val/test split
    print("\n" + "="*70)
    print("TEST 4: TRAIN/VAL/TEST SPLIT")
    print("="*70)
    
    train_df, val_df, test_df = loader.create_train_test_split(
        df_prepared,
        test_days=3,
        val_days=3
    )
    
    print("\n✅ Split created successfully")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)