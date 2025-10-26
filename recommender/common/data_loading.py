# """
# Data Loading Pipeline with Time Decay
# ======================================
# Supports both CSV (development) and PostgreSQL (production)

# Features:
# - Exponential time decay: weight = 0.5^(days/half_life)
# - Action multipliers (view, like, comment, share, hide)
# - Memory efficient with chunking
# - Performance optimized for 1.4M+ rows
# """

# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from typing import Optional, Dict, Tuple
# import logging
# from pathlib import Path
# import warnings

# warnings.filterwarnings('ignore')

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# class DataLoader:
#     """
#     Data loader with time decay support
#     Handles both CSV files and PostgreSQL database
#     """
    
#     # Action multipliers mapping
#     ACTION_WEIGHTS = {
#         'view': 0.5,
#         'like': 1.0,
#         'comment': 1.5,
#         'share': 2.0,
#         'save': 1.2,
#         'hide': -3.0,
#         'report': -5.0
#     }
    
#     # Map ReactionTypeId to action names (based on your schema)
#     REACTION_TYPE_MAPPING = {
#         1: 'like',      # Like
#         2: 'comment',   # Comment
#         3: 'share',     # Share
#         4: 'view',      # View
#         5: 'save',      # Save
#         6: 'hide',      # Hide (if exists)
#         7: 'report'     # Report (if exists)
#     }
    
#     def __init__(
#         self, 
#         db_connection=None, 
#         config: Dict = None,
#         data_dir: str = 'dataset'
#     ):
#         """
#         Initialize DataLoader
        
#         Args:
#             db_connection: PostgreSQL connection (optional)
#             config: Configuration dict
#             data_dir: Directory containing CSV files (for development)
#         """
#         self.db_connection = db_connection
#         self.config = config or {}
#         self.data_dir = Path(data_dir)
        
#         # Configuration
#         self.lookback_days = self.config.get('lookback_days', 14)
#         self.half_life_days = self.config.get('half_life_days', 7.0)
#         self.min_weight = self.config.get('min_weight', 0.01)
#         self.chunk_size = self.config.get('chunk_size', 100000)
        
#         logger.info(f"DataLoader initialized with:")
#         logger.info(f"  Lookback days: {self.lookback_days}")
#         logger.info(f"  Half-life: {self.half_life_days} days")
#         logger.info(f"  Min weight: {self.min_weight}")
#         logger.info(f"  Chunk size: {self.chunk_size:,}")
    
#     def load_training_data(
#         self, 
#         lookback_days: Optional[int] = None,
#         use_csv: bool = True
#     ) -> pd.DataFrame:
#         """
#         Load training data with lookback window
        
#         Args:
#             lookback_days: Number of days to look back (default: from config)
#             use_csv: If True, load from CSV; else from PostgreSQL
        
#         Returns:
#             DataFrame with columns: user_id, post_id, action, created_at
#         """
#         lookback_days = lookback_days or self.lookback_days
#         cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
#         logger.info(f"\n{'='*70}")
#         logger.info(f"LOADING TRAINING DATA")
#         logger.info(f"{'='*70}")
#         logger.info(f"Lookback window: {lookback_days} days")
#         logger.info(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
        
#         if use_csv or self.db_connection is None:
#             df = self._load_from_csv()
#         else:
#             df = self._load_from_postgresql(cutoff_date)
        
#         # Validate data
#         self._validate_data(df)
        
#         # Filter by date
#         df = self._filter_by_date(df, cutoff_date)
        
#         logger.info(f"✅ Loaded {len(df):,} interactions")
#         logger.info(f"   Date range: {df['created_at'].min()} to {df['created_at'].max()}")
#         logger.info(f"   Unique users: {df['user_id'].nunique():,}")
#         logger.info(f"   Unique posts: {df['post_id'].nunique():,}")
        
#         return df
    
#     def _load_from_csv(self) -> pd.DataFrame:
#         """Load from CSV files (development mode)"""
#         logger.info("📂 Loading from CSV files...")
        
#         csv_path = self.data_dir / 'PostReaction.csv'
        
#         if not csv_path.exists():
#             raise FileNotFoundError(
#                 f"PostReaction.csv not found in {self.data_dir}\n"
#                 f"Expected path: {csv_path}"
#             )
        
#         # Define column mappings and dtypes
#         dtype_mapping = {
#             'Id': 'int64',
#             'UserId': 'int64',
#             'PostId': 'int64',
#             'ReactionTypeId': 'int8'
#         }
        
#         # Read CSV in chunks for memory efficiency
#         chunks = []
        
#         for chunk in pd.read_csv(
#             csv_path, 
#             dtype=dtype_mapping,
#             parse_dates=['CreateDate'],
#             chunksize=self.chunk_size
#         ):
#             chunks.append(chunk)
        
#         df = pd.concat(chunks, ignore_index=True)
        
#         logger.info(f"   Loaded {len(df):,} rows from CSV")
        
#         # Rename columns to standard format
#         df = df.rename(columns={
#             'UserId': 'user_id',
#             'PostId': 'post_id',
#             'ReactionTypeId': 'reaction_type_id',
#             'CreateDate': 'created_at'
#         })
        
#         # Map ReactionTypeId to action names
#         df['action'] = df['reaction_type_id'].map(self.REACTION_TYPE_MAPPING)
        
#         # Handle unknown reaction types
#         unknown_reactions = df[df['action'].isna()]['reaction_type_id'].unique()
#         if len(unknown_reactions) > 0:
#             logger.warning(f"⚠️  Unknown reaction types: {unknown_reactions}")
#             logger.warning(f"   Mapping to 'view' by default")
#             df['action'] = df['action'].fillna('view')
        
#         return df[['user_id', 'post_id', 'action', 'created_at']]
    
#     def _load_from_postgresql(self, cutoff_date: datetime) -> pd.DataFrame:
#         """Load from PostgreSQL database (production mode)"""
#         logger.info("🗄️  Loading from PostgreSQL...")
        
#         if self.db_connection is None:
#             raise ValueError("Database connection not provided")
        
#         query = """
#         SELECT 
#             user_id,
#             post_id,
#             action,
#             created_at
#         FROM interactions
#         WHERE created_at >= %s
#         ORDER BY created_at DESC
#         """
        
#         try:
#             df = pd.read_sql_query(
#                 query,
#                 self.db_connection,
#                 params=[cutoff_date]
#             )
            
#             logger.info(f"   Loaded {len(df):,} rows from PostgreSQL")
            
#             return df
            
#         except Exception as e:
#             logger.error(f"❌ Error loading from PostgreSQL: {e}")
#             raise
    
#     def _filter_by_date(
#         self, 
#         df: pd.DataFrame, 
#         cutoff_date: datetime
#     ) -> pd.DataFrame:
#         """Filter interactions by date"""
#         logger.info(f"\n🔍 Filtering by date...")
        
#         original_count = len(df)
        
#         # Ensure created_at is datetime
#         if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
#             df['created_at'] = pd.to_datetime(df['created_at'])
        
#         # Filter
#         df = df[df['created_at'] >= cutoff_date].copy()
        
#         filtered_count = original_count - len(df)
        
#         if filtered_count > 0:
#             logger.info(f"   Filtered out {filtered_count:,} old interactions")
        
#         return df
    
#     def _validate_data(self, df: pd.DataFrame):
#         """Validate loaded data"""
#         logger.info("\n✓ Validating data...")
        
#         # Check required columns
#         required_cols = ['user_id', 'post_id', 'action', 'created_at']
#         missing_cols = set(required_cols) - set(df.columns)
        
#         if missing_cols:
#             raise ValueError(f"Missing required columns: {missing_cols}")
        
#         # Check for nulls
#         null_counts = df[required_cols].isnull().sum()
#         if null_counts.any():
#             logger.warning("⚠️  Found null values:")
#             for col, count in null_counts[null_counts > 0].items():
#                 logger.warning(f"   {col}: {count:,} nulls")
        
#         # Check data types
#         if not pd.api.types.is_integer_dtype(df['user_id']):
#             logger.warning(f"   Converting user_id to int")
#             df['user_id'] = df['user_id'].astype('int64')
        
#         if not pd.api.types.is_integer_dtype(df['post_id']):
#             logger.warning(f"   Converting post_id to int")
#             df['post_id'] = df['post_id'].astype('int64')
        
#         logger.info("   ✅ Data validation passed")
    
#     def apply_time_decay(
#         self, 
#         df: pd.DataFrame,
#         half_life_days: Optional[float] = None,
#         reference_date: Optional[datetime] = None
#     ) -> pd.DataFrame:
#         """
#         Apply exponential time decay to interactions
        
#         Formula: weight = action_multiplier × 0.5^(days_ago / half_life)
        
#         Args:
#             df: DataFrame with 'created_at' and 'action' columns
#             half_life_days: Half-life for decay (default: from config)
#             reference_date: Reference date for decay (default: now)
        
#         Returns:
#             DataFrame with 'weight' column added
#         """
#         half_life_days = half_life_days or self.half_life_days
#         reference_date = reference_date or datetime.now()
        
#         logger.info(f"\n{'='*70}")
#         logger.info(f"APPLYING TIME DECAY")
#         logger.info(f"{'='*70}")
#         logger.info(f"Half-life: {half_life_days} days")
#         logger.info(f"Reference date: {reference_date.strftime('%Y-%m-%d %H:%M:%S')}")
#         logger.info(f"Min weight threshold: {self.min_weight}")
        
#         # Create a copy to avoid modifying original
#         df = df.copy()
        
#         # Ensure created_at is datetime
#         if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
#             df['created_at'] = pd.to_datetime(df['created_at'])
        
#         # Calculate days ago
#         df['days_ago'] = (reference_date - df['created_at']).dt.total_seconds() / 86400
        
#         # Calculate time decay factor
#         # Formula: 0.5^(days_ago / half_life)
#         df['time_decay'] = np.power(0.5, df['days_ago'] / half_life_days)
        
#         # Get action multipliers
#         df['action_multiplier'] = df['action'].map(self.ACTION_WEIGHTS)
        
#         # Handle unknown actions (default to 1.0)
#         unknown_actions = df[df['action_multiplier'].isna()]['action'].unique()
#         if len(unknown_actions) > 0:
#             logger.warning(f"⚠️  Unknown actions: {unknown_actions}")
#             logger.warning(f"   Using default multiplier: 1.0")
#             df['action_multiplier'] = df['action_multiplier'].fillna(1.0)
        
#         # Calculate final weight
#         df['weight'] = df['action_multiplier'] * df['time_decay']
        
#         # Apply minimum weight threshold
#         df['weight'] = df['weight'].clip(lower=self.min_weight)
        
#         # Log statistics
#         self._log_weight_statistics(df)
        
#         # Clean up temporary columns
#         df = df.drop(columns=['days_ago', 'time_decay', 'action_multiplier'])
        
#         logger.info(f"✅ Time decay applied successfully")
        
#         return df
    
#     def _log_weight_statistics(self, df: pd.DataFrame):
#         """Log weight distribution statistics"""
#         logger.info(f"\n📊 Weight Statistics:")
#         logger.info(f"   Mean: {df['weight'].mean():.4f}")
#         logger.info(f"   Median: {df['weight'].median():.4f}")
#         logger.info(f"   Std: {df['weight'].std():.4f}")
#         logger.info(f"   Min: {df['weight'].min():.4f}")
#         logger.info(f"   Max: {df['weight'].max():.4f}")
        
#         # Weight distribution by action
#         logger.info(f"\n📊 Weight by Action:")
#         weight_by_action = df.groupby('action')['weight'].agg(['mean', 'count'])
#         for action, stats in weight_by_action.iterrows():
#             logger.info(f"   {action:10s}: mean={stats['mean']:.4f}, count={stats['count']:,}")
        
#         # Weight distribution over time
#         logger.info(f"\n📊 Weight Over Time:")
#         df['date'] = df['created_at'].dt.date
#         weight_by_date = df.groupby('date')['weight'].mean().tail(7)
#         for date, weight in weight_by_date.items():
#             logger.info(f"   {date}: {weight:.4f}")
    
#     def load_and_prepare_training_data(
#         self,
#         lookback_days: Optional[int] = None,
#         half_life_days: Optional[float] = None,
#         use_csv: bool = True
#     ) -> pd.DataFrame:
#         """
#         Convenience method: Load data + apply time decay in one call
        
#         Returns:
#             DataFrame with 'weight' column
#         """
#         # Load data
#         df = self.load_training_data(
#             lookback_days=lookback_days,
#             use_csv=use_csv
#         )
        
#         # Apply time decay
#         df = self.apply_time_decay(
#             df,
#             half_life_days=half_life_days
#         )
        
#         return df
    
#     def create_train_test_split(
#         self,
#         df: pd.DataFrame,
#         test_days: int = 3,
#         val_days: int = 3
#     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#         """
#         Create temporal train/val/test split
        
#         Args:
#             df: DataFrame with 'created_at' column
#             test_days: Days for test set
#             val_days: Days for validation set
        
#         Returns:
#             (train_df, val_df, test_df)
#         """
#         logger.info(f"\n{'='*70}")
#         logger.info(f"CREATING TRAIN/VAL/TEST SPLIT")
#         logger.info(f"{'='*70}")
        
#         # Sort by date
#         df = df.sort_values('created_at').copy()
        
#         # Calculate split dates
#         max_date = df['created_at'].max()
#         test_start = max_date - timedelta(days=test_days)
#         val_start = test_start - timedelta(days=val_days)
        
#         # Split
#         train_df = df[df['created_at'] < val_start].copy()
#         val_df = df[(df['created_at'] >= val_start) & (df['created_at'] < test_start)].copy()
#         test_df = df[df['created_at'] >= test_start].copy()
        
#         logger.info(f"Split dates:")
#         logger.info(f"   Train: up to {val_start.strftime('%Y-%m-%d')}")
#         logger.info(f"   Val: {val_start.strftime('%Y-%m-%d')} to {test_start.strftime('%Y-%m-%d')}")
#         logger.info(f"   Test: from {test_start.strftime('%Y-%m-%d')}")
        
#         logger.info(f"\nSplit sizes:")
#         logger.info(f"   Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
#         logger.info(f"   Val: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
#         logger.info(f"   Test: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
        
#         return train_df, val_df, test_df


# # =============================================================================
# # HELPER FUNCTIONS
# # =============================================================================

# def load_training_data(
#     lookback_days: int = 14,
#     data_dir: str = 'dataset',
#     db_connection=None
# ) -> pd.DataFrame:
#     """
#     Convenience function to load training data
    
#     Args:
#         lookback_days: Days to look back
#         data_dir: Directory with CSV files
#         db_connection: PostgreSQL connection (optional)
    
#     Returns:
#         DataFrame with interactions
#     """
#     config = {
#         'lookback_days': lookback_days,
#         'half_life_days': 7.0,
#         'min_weight': 0.01
#     }
    
#     loader = DataLoader(
#         db_connection=db_connection,
#         config=config,
#         data_dir=data_dir
#     )
    
#     use_csv = db_connection is None
    
#     return loader.load_training_data(use_csv=use_csv)


# def apply_time_decay(
#     df: pd.DataFrame,
#     half_life_days: float = 7.0,
#     min_weight: float = 0.01
# ) -> pd.DataFrame:
#     """
#     Convenience function to apply time decay
    
#     Args:
#         df: DataFrame with 'created_at' and 'action' columns
#         half_life_days: Half-life for decay
#         min_weight: Minimum weight threshold
    
#     Returns:
#         DataFrame with 'weight' column
#     """
#     config = {
#         'half_life_days': half_life_days,
#         'min_weight': min_weight
#     }
    
#     loader = DataLoader(config=config)
    
#     return loader.apply_time_decay(df)


# # =============================================================================
# # MAIN - FOR TESTING
# # =============================================================================

# if __name__ == "__main__":
#     """
#     Test the DataLoader with your CSV files
#     """
    
#     print("="*70)
#     print("TESTING DATA LOADER")
#     print("="*70)
    
#     # Configuration
#     config = {
#         'lookback_days': 14,
#         'half_life_days': 7.0,
#         'min_weight': 0.01,
#         'chunk_size': 100000
#     }
    
#     # Initialize loader
#     loader = DataLoader(
#         db_connection=None,  # Use CSV mode
#         config=config,
#         data_dir='dataset'
#     )
    
#     # Test 1: Load data
#     print("\n" + "="*70)
#     print("TEST 1: LOAD TRAINING DATA")
#     print("="*70)
    
#     df = loader.load_training_data(use_csv=True)
    
#     print(f"\n✅ Loaded {len(df):,} interactions")
#     print(f"\nFirst 5 rows:")
#     print(df.head())
    
#     # Test 2: Apply time decay
#     print("\n" + "="*70)
#     print("TEST 2: APPLY TIME DECAY")
#     print("="*70)
    
#     df_weighted = loader.apply_time_decay(df)
    
#     print(f"\n✅ Applied time decay")
#     print(f"\nFirst 5 rows with weights:")
#     print(df_weighted[['user_id', 'post_id', 'action', 'created_at', 'weight']].head())
    
#     # Test 3: Load and prepare in one call
#     print("\n" + "="*70)
#     print("TEST 3: LOAD AND PREPARE (ONE CALL)")
#     print("="*70)
    
#     df_prepared = loader.load_and_prepare_training_data(use_csv=True)
    
#     print(f"\n✅ Data prepared: {len(df_prepared):,} rows")
    
#     # Test 4: Create train/val/test split
#     print("\n" + "="*70)
#     print("TEST 4: TRAIN/VAL/TEST SPLIT")
#     print("="*70)
    
#     train_df, val_df, test_df = loader.create_train_test_split(
#         df_prepared,
#         test_days=3,
#         val_days=3
#     )
    
#     print("\n✅ Split created successfully")
    
#     print("\n" + "="*70)
#     print("ALL TESTS PASSED!")
#     print("="*70)


# recommender/common/data_loading.py
"""
Data Loading from MySQL & CSV (dev fallback)
===========================================

- Gộp interactions từ nhiều bảng MySQL:
  * PostView      -> action='view'      (ViewDate | CreateDate)
  * PostReaction  -> join ReactionType  -> action in {like, love, laugh, wow, sad, angry, care}
  * Comment       -> action='comment'
- Lọc theo window [since, until] (UTC), chunking LIMIT/OFFSET
- Trả kèm side tables: users, posts, friendships, post_hashtags, reaction_types, comments
- Chuẩn hoá schema interactions: user_id, post_id, action, created_at (UTC)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------- Utilities -------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _ensure_datetime_utc(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, utc=True, errors="coerce")
    else:
        if series.dt.tz is None:
            series = series.dt.tz_localize("UTC")
        else:
            series = series.dt.tz_convert("UTC")
    return series

def _clip_window(df: pd.DataFrame, col: str, since: datetime, until: datetime) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    d = df.copy()
    d[col] = _ensure_datetime_utc(d[col])
    m = (d[col] >= since) & (d[col] <= until)
    return d.loc[m].reset_index(drop=True)

def _read_sql_chunked(
    conn: Any, base_sql: str, base_params: Tuple, chunk_size: int, normalize_ts_cols: List[str]
) -> pd.DataFrame:
    """Chunking kiểu LIMIT/OFFSET (MySQL/SQLAlchemy compatible)."""
    limit = int(chunk_size)
    offset = 0
    chunks: List[pd.DataFrame] = []
    total = 0
    while True:
        sql = f"{base_sql} LIMIT %s OFFSET %s"
        params = (*base_params, limit, offset)
        df = pd.read_sql_query(sql, conn, params=params)
        n = len(df)
        if n == 0:
            break
        for c in normalize_ts_cols:
            if c in df.columns:
                df[c] = _ensure_datetime_utc(df[c])
        chunks.append(df)
        total += n
        offset += n
        logger.info(f"  -> loaded {n:,} rows (total {total:,})")
        if n < limit:
            break
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame()


# ----------------------------- DataLoader ------------------------------------

class DataLoader:
    def __init__(
        self,
        db_connection: Any = None,
        config: Optional[Dict] = None,
        data_dir: str = "dataset",
    ):
        self.conn = db_connection
        self.config = config or {}
        self.data_dir = Path(self.config.get("csv_dir", data_dir) or data_dir)
        self.chunk_size = int(self.config.get("chunk_size", 200_000))
        self.lookback_days = int(self.config.get("lookback_days", 14))

        # table names
        t = self.config.get("tables", {}) or {}
        self.tbl_view = t.get("post_view", "PostView")
        self.tbl_react = t.get("post_reaction", "PostReaction")
        self.tbl_reaction_type = t.get("reaction_type", "ReactionType")
        self.tbl_user = t.get("user", "User")
        self.tbl_post = t.get("post", "Post")
        self.tbl_hashtag = t.get("post_hashtag", "PostHashtag")
        self.tbl_friendship = t.get("friendship", "Friendship")
        self.tbl_comment = t.get("comment", "Comment")

        # ReactionType → action mapping (Code/Name). MẶC ĐỊNH KHỚP ẢNH DB: like/love/laugh/wow/sad/angry/care
        self.code_map = {
            **{"like": "like", "love": "love", "laugh": "laugh", "wow": "wow", "sad": "sad", "angry": "angry", "care": "care"},
            **(self.config.get("reaction_code_map") or {}),
        }
        self.name_map = {
            **{"Like": "like", "Love": "love", "Laugh": "laugh", "Wow": "wow", "Sad": "sad", "Angry": "angry", "Care": "care"},
            **(self.config.get("reaction_name_map") or {}),
        }

        # csv files
        self.csv_files = self.config.get("csv_files", {}) or {}

        logger.info("DataLoader initialized:")
        logger.info(f"  mode = {'DB' if self.conn is not None else 'CSV'}")
        logger.info(f"  lookback_days = {self.lookback_days} | chunk_size = {self.chunk_size:,}")

    # --------------------- public high-level APIs -----------------------------

    def load_training_bundle(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        use_csv: Optional[bool] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Trả về đầy đủ frames:
          - interactions (user_id, post_id, action, created_at)
          - views, reactions, comments, reaction_types, users, posts, friendships, post_hashtags
        """
        use_csv_mode = (use_csv if use_csv is not None else (self.conn is None))
        since, until = self._resolve_window(since, until)

        logger.info("\n" + "=" * 70)
        logger.info("LOADING TRAINING BUNDLE")
        logger.info("=" * 70)
        logger.info(f"Source: {'CSV' if use_csv_mode else 'DB'}; Window: {since.isoformat()} -> {until.isoformat()}")

        if use_csv_mode:
            frames = self._load_bundle_from_csv(since, until)
        else:
            frames = self._load_bundle_from_db(since, until)

        # Hợp nhất interactions
        inter = self._build_interactions(
            views=frames["views"], reactions=frames["reactions"],
            comments=frames["comments"], reaction_types=frames["reaction_types"]
        )
        inter = _clip_window(inter, "created_at", since, until)

        frames["interactions"] = inter
        self._log_interactions(inter)

        return frames

    # ------------------------ internal: window --------------------------------

    def _resolve_window(self, since: Optional[datetime], until: Optional[datetime]) -> Tuple[datetime, datetime]:
        now = _now_utc()
        u = (until or now).astimezone(timezone.utc)
        s = (since or (u - timedelta(days=self.lookback_days))).astimezone(timezone.utc)
        return s, u

    # ---------------------- internal: DB loaders ------------------------------

    def _load_bundle_from_db(self, since: datetime, until: datetime) -> Dict[str, pd.DataFrame]:
        return {
            "views": self._load_views_db(since, until),
            "reactions": self._load_reactions_db(since, until),
            "reaction_types": self._load_reaction_types_db(),
            "comments": self._load_comments_db(since, until),
            "users": self._load_users_db(),
            "posts": self._load_posts_db(),
            "friendships": self._load_friendships_db(),
            "post_hashtags": self._load_post_hashtags_db(),
        }

    def _load_views_db(self, since: datetime, until: datetime) -> pd.DataFrame:
        logger.info("DB: loading PostView ...")
        base = (
            f"SELECT Id, UserId, PostId, ViewDate, CreateDate, Status "
            f"FROM {self.tbl_view} "
            f"WHERE (ViewDate >= %s OR (ViewDate IS NULL AND CreateDate >= %s)) "
            f"AND (ViewDate <= %s OR (ViewDate IS NULL AND CreateDate <= %s)) "
            f"ORDER BY COALESCE(ViewDate, CreateDate) ASC"
        )
        df = _read_sql_chunked(self.conn, base, (since, since, until, until), self.chunk_size, ["ViewDate", "CreateDate"])
        if df.empty:
            return pd.DataFrame(columns=["user_id", "post_id", "action", "created_at"])
        df.rename(columns={"UserId": "user_id", "PostId": "post_id"}, inplace=True)
        df["created_at"] = df["ViewDate"].fillna(df["CreateDate"])
        df["action"] = "view"
        return df[["user_id", "post_id", "action", "created_at"]]

    def _load_reactions_db(self, since: datetime, until: datetime) -> pd.DataFrame:
        logger.info("DB: loading PostReaction ...")
        base = (
            f"SELECT Id, PostId, UserId, ReactionTypeId, CreateDate, Status "
            f"FROM {self.tbl_react} "
            f"WHERE CreateDate >= %s AND CreateDate <= %s "
            f"ORDER BY CreateDate ASC"
        )
        df = _read_sql_chunked(self.conn, base, (since, until), self.chunk_size, ["CreateDate"])
        if df.empty:
            return pd.DataFrame(columns=["user_id", "post_id", "reaction_type_id", "created_at"])
        df.rename(columns={"UserId": "user_id", "PostId": "post_id", "CreateDate": "created_at", "ReactionTypeId": "ReactionTypeId"}, inplace=True)
        return df[["user_id", "post_id", "ReactionTypeId", "created_at"]]

    def _load_reaction_types_db(self) -> pd.DataFrame:
        logger.info("DB: loading ReactionType ...")
        sql = f"SELECT Id AS ReactionTypeId, Code, Name FROM {self.tbl_reaction_type}"
        return pd.read_sql_query(sql, self.conn)

    def _load_comments_db(self, since: datetime, until: datetime) -> pd.DataFrame:
        logger.info("DB: loading Comment ...")
        base = (
            f"SELECT Id, PostId, UserId, CreateDate, Status "
            f"FROM {self.tbl_comment} "
            f"WHERE CreateDate >= %s AND CreateDate <= %s "
            f"ORDER BY CreateDate ASC"
        )
        df = _read_sql_chunked(self.conn, base, (since, until), self.chunk_size, ["CreateDate"])
        if df.empty:
            return pd.DataFrame(columns=["user_id", "post_id", "action", "created_at"])
        df.rename(columns={"UserId": "user_id", "PostId": "post_id", "CreateDate": "created_at"}, inplace=True)
        df["action"] = "comment"
        return df[["user_id", "post_id", "action", "created_at"]]

    def _load_users_db(self) -> pd.DataFrame:
        logger.info("DB: loading User ...")
        sql = f"SELECT Id, CreateDate FROM {self.tbl_user}"
        df = pd.read_sql_query(sql, self.conn)
        if "CreateDate" in df.columns:
            df["CreateDate"] = _ensure_datetime_utc(df["CreateDate"])
        return df

    def _load_posts_db(self) -> pd.DataFrame:
        logger.info("DB: loading Post ...")
        sql = f"SELECT Id, UserId, CreateDate, IsRepost, IsPin FROM {self.tbl_post}"
        df = pd.read_sql_query(sql, self.conn)
        if "CreateDate" in df.columns:
            df["CreateDate"] = _ensure_datetime_utc(df["CreateDate"])
        return df

    def _load_post_hashtags_db(self) -> pd.DataFrame:
        logger.info("DB: loading PostHashtag ...")
        sql = f"SELECT Id, PostId, HashtagId, CreateDate FROM {self.tbl_hashtag}"
        return pd.read_sql_query(sql, self.conn)

    def _load_friendships_db(self) -> pd.DataFrame:
        logger.info("DB: loading Friendship ...")
        sql = f"SELECT Id, UserId, FriendId, CreateDate FROM {self.tbl_friendship}"
        return pd.read_sql_query(sql, self.conn)

    # ---------------------- internal: CSV loaders -----------------------------

    def _load_bundle_from_csv(self, since: datetime, until: datetime) -> Dict[str, pd.DataFrame]:
        logger.info("CSV: loading all sources ...")

        def _csv(name: str, default: str) -> Path:
            return self.data_dir / self.config.get("csv_files", {}).get(name, default)

        def _read_csv(path: Path, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
            if not path.exists():
                logger.warning(f"CSV not found: {path}")
                return pd.DataFrame()
            return pd.read_csv(path, parse_dates=parse_dates)

        views = _read_csv(_csv("post_view", "PostView.csv"), parse_dates=["ViewDate", "CreateDate"])
        reactions = _read_csv(_csv("post_reaction", "PostReaction.csv"), parse_dates=["CreateDate"])
        reaction_types = _read_csv(_csv("reaction_type", "ReactionType.csv"))
        comments = _read_csv(_csv("comment", "Comment.csv"), parse_dates=["CreateDate"])
        users = _read_csv(_csv("user", "User.csv"), parse_dates=["CreateDate"])
        posts = _read_csv(_csv("post", "Post.csv"), parse_dates=["CreateDate"])
        post_hashtags = _read_csv(_csv("post_hashtag", "PostHashtag.csv"), parse_dates=["CreateDate"])
        friendships = _read_csv(_csv("friendship", "Friendship.csv"), parse_dates=["CreateDate"])

        # Normalize to UTC
        for df, col in [
            (views, "ViewDate"), (views, "CreateDate"),
            (reactions, "CreateDate"), (comments, "CreateDate"),
            (users, "CreateDate"), (posts, "CreateDate"),
        ]:
            if not df.empty and col in df.columns:
                df[col] = _ensure_datetime_utc(df[col])

        # Filter by window for raw sources
        if not views.empty:
            views = views[(views["ViewDate"].fillna(views["CreateDate"]) >= since) &
                          (views["ViewDate"].fillna(views["CreateDate"]) <= until)]
        if not reactions.empty:
            reactions = reactions[(reactions["CreateDate"] >= since) & (reactions["CreateDate"] <= until)]
        if not comments.empty:
            comments = comments[(comments["CreateDate"] >= since) & (comments["CreateDate"] <= until)]

        return {
            "views": views.reset_index(drop=True),
            "reactions": reactions.reset_index(drop=True),
            "reaction_types": reaction_types.reset_index(drop=True),
            "comments": comments.reset_index(drop=True),
            "users": users.reset_index(drop=True),
            "posts": posts.reset_index(drop=True),
            "friendships": friendships.reset_index(drop=True),
            "post_hashtags": post_hashtags.reset_index(drop=True),
        }

    # ------------------------ build interactions -----------------------------

    def _build_interactions(
        self,
        views: pd.DataFrame,
        reactions: pd.DataFrame,
        comments: pd.DataFrame,
        reaction_types: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build unified interactions dataframe.
        Hỗ trợ cả 2 trường hợp:
        - views/comments đã chuẩn hoá (user_id, post_id, action, created_at)
        - hoặc raw (UserId, PostId, ViewDate/CreateDate,…)
        """
        parts: List[pd.DataFrame] = []

        # ---------- VIEWS ----------
        if views is not None and not views.empty:
            # Case A: đã chuẩn hoá
            if {"user_id", "post_id", "created_at"}.issubset(views.columns):
                v = views.copy()
                if "action" not in v.columns:
                    v["action"] = "view"
                v["created_at"] = _ensure_datetime_utc(v["created_at"])
                parts.append(v[["user_id", "post_id", "action", "created_at"]])
            # Case B: raw
            elif {"UserId", "PostId"}.issubset(views.columns) and (
                "ViewDate" in views.columns or "CreateDate" in views.columns
            ):
                v = views.rename(columns={"UserId": "user_id", "PostId": "post_id"}).copy()
                if "ViewDate" in v.columns and "CreateDate" in v.columns:
                    base_ts = v["ViewDate"].fillna(v["CreateDate"])
                elif "ViewDate" in v.columns:
                    base_ts = v["ViewDate"]
                else:
                    base_ts = v["CreateDate"]
                v["created_at"] = _ensure_datetime_utc(base_ts)
                v["action"] = "view"
                parts.append(v[["user_id", "post_id", "action", "created_at"]])
            else:
                logger.warning("PostView columns not recognized. Skipping views in interactions build.")

        # ---------- REACTIONS ----------
        if reactions is not None and not reactions.empty:
            r = reactions.rename(
                columns={"UserId": "user_id", "PostId": "post_id", "CreateDate": "created_at"}
            ).copy()

            # Join ReactionType nếu có
            if reaction_types is not None and not reaction_types.empty:
                # reaction_types: ReactionTypeId, Code, Name (đã đúng theo query DB)
                rt = reaction_types.copy()
                r = r.merge(rt[["ReactionTypeId", "Code", "Name"]], on="ReactionTypeId", how="left")

            # Lấy code/name an toàn, KHÔNG dùng 'or' với Series
            if "Code" in r.columns:
                codes = r["Code"].astype(str).str.lower()
            else:
                codes = pd.Series([None] * len(r), index=r.index)

            if "Name" in r.columns:
                names = r["Name"].astype(str)
            else:
                names = pd.Series([None] * len(r), index=r.index)

            # Map Code/Name -> action với fallback 'like'
            def _map_action(code_val: Optional[str], name_val: Optional[str]) -> str:
                act = None
                if code_val is not None and pd.notna(code_val):
                    act = self.code_map.get(str(code_val).lower())
                if act is None and name_val is not None and pd.notna(name_val):
                    act = self.name_map.get(str(name_val))
                return act or "like"

            r["action"] = [ _map_action(c, n) for c, n in zip(codes, names) ]
            r["created_at"] = _ensure_datetime_utc(r["created_at"])
            parts.append(r[["user_id", "post_id", "action", "created_at"]])

        # ---------- COMMENTS ----------
        if comments is not None and not comments.empty:
            # Case A: đã chuẩn hoá
            if {"user_id", "post_id", "created_at"}.issubset(comments.columns):
                c = comments.copy()
                if "action" not in c.columns:
                    c["action"] = "comment"
                c["created_at"] = _ensure_datetime_utc(c["created_at"])
                parts.append(c[["user_id", "post_id", "action", "created_at"]])
            # Case B: raw
            elif {"UserId", "PostId", "CreateDate"}.issubset(comments.columns):
                c = comments.rename(
                    columns={"UserId": "user_id", "PostId": "post_id", "CreateDate": "created_at"}
                ).copy()
                c["action"] = "comment"
                c["created_at"] = _ensure_datetime_utc(c["created_at"])
                parts.append(c[["user_id", "post_id", "action", "created_at"]])
            else:
                logger.warning("Comment columns not recognized. Skipping comments in interactions build.")

        # ---------- CONCAT ----------
        if not parts:
            return pd.DataFrame(columns=["user_id", "post_id", "action", "created_at"])

        inter = pd.concat(parts, ignore_index=True)
        inter["created_at"] = _ensure_datetime_utc(inter["created_at"])
        inter = inter.dropna(subset=["user_id", "post_id", "created_at"])
        inter["user_id"] = inter["user_id"].astype("int64", errors="ignore")
        inter["post_id"] = inter["post_id"].astype("int64", errors="ignore")
        return inter



    # ---------------------------- logging ------------------------------------

    @staticmethod
    def _log_interactions(df: pd.DataFrame) -> None:
        if df.empty:
            logger.info("Loaded 0 interactions.")
            return
        logger.info(
            f"✅ Interactions: {len(df):,} | users={df['user_id'].nunique():,} | posts={df['post_id'].nunique():,}"
        )
        logger.info(f"   Range: {df['created_at'].min()} -> {df['created_at'].max()}")
