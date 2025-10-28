# recommender/online/app/utils/data_utils.py
"""
Data Utilities - Column Name Normalization
===========================================

Fix inconsistent column naming between MySQL and pandas DataFrames
"""

import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def normalize_column_names(df: pd.DataFrame, table_name: str = "") -> pd.DataFrame:
    """
    Normalize column names to handle case inconsistencies
    
    MySQL tables use PascalCase: UserId, FriendId, PostId
    But pandas might convert to lowercase: userid, friendid, postid
    
    This function standardizes all column names to PascalCase
    
    Args:
        df: Input DataFrame
        table_name: Name of table (for logging)
    
    Returns:
        DataFrame with normalized column names
    """
    if df.empty:
        return df
    
    # Mapping of common column variations
    column_mapping = {
        # User columns
        'userid': 'UserId',
        'user_id': 'UserId',
        'UserID': 'UserId',
        'USERID': 'UserId',
        
        # Friend columns
        'friendid': 'FriendId',
        'friend_id': 'FriendId',
        'FriendID': 'FriendId',
        'FRIENDID': 'FriendId',
        
        # Post columns
        'postid': 'PostId',
        'post_id': 'PostId',
        'PostID': 'PostId',
        'POSTID': 'PostId',
        
        # Other common columns
        'id': 'Id',
        'ID': 'Id',
        
        'status': 'Status',
        'STATUS': 'Status',
        
        'createdate': 'CreateDate',
        'create_date': 'CreateDate',
        'CreateDATE': 'CreateDate',
        
        'reactiontypeid': 'ReactionTypeId',
        'reaction_type_id': 'ReactionTypeId',
        'ReactionTypeID': 'ReactionTypeId',
    }
    
    # Create a copy to avoid modifying original
    df_normalized = df.copy()
    
    # Get current columns (case-insensitive lookup)
    current_columns = df_normalized.columns.tolist()
    rename_dict = {}
    
    for col in current_columns:
        # Try exact match first
        if col in column_mapping:
            rename_dict[col] = column_mapping[col]
        # Try lowercase match
        elif col.lower() in column_mapping:
            rename_dict[col] = column_mapping[col.lower()]
    
    if rename_dict:
        df_normalized.rename(columns=rename_dict, inplace=True)
        logger.debug(f"Normalized {len(rename_dict)} columns in {table_name}: {rename_dict}")
    
    return df_normalized


def validate_required_columns(df: pd.DataFrame, required_columns: list, table_name: str = "") -> bool:
    """
    Validate that DataFrame has all required columns
    
    Args:
        df: DataFrame to check
        required_columns: List of required column names
        table_name: Name of table (for logging)
    
    Returns:
        True if all columns exist, False otherwise
    """
    if df.empty:
        logger.warning(f"{table_name} DataFrame is empty")
        return False
    
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        logger.error(f"{table_name} missing required columns: {missing}")
        logger.error(f"Available columns: {df.columns.tolist()}")
        return False
    
    return True


def safe_groupby(df: pd.DataFrame, by_column: str, table_name: str = "") -> Dict[Any, pd.DataFrame]:
    """
    Safe groupby that handles missing columns gracefully
    
    Args:
        df: DataFrame to group
        by_column: Column to group by
        table_name: Name of table (for logging)
    
    Returns:
        Dictionary of groups or empty dict if error
    """
    if df is None or df.empty:
        logger.warning(f"Cannot groupby on empty {table_name} DataFrame")
        return {}
    
    if by_column not in df.columns:
        logger.error(f"Column '{by_column}' not found in {table_name}")
        logger.error(f"Available columns: {df.columns.tolist()}")
        return {}
    
    try:
        grouped = df.groupby(by_column)
        return {name: group for name, group in grouped}
    except Exception as e:
        logger.error(f"Error grouping {table_name} by {by_column}: {e}")
        return {}


def load_and_normalize_from_mysql(query: str, conn, table_name: str = "") -> pd.DataFrame:
    """
    Load data from MySQL and normalize column names
    
    Args:
        query: SQL query
        conn: Database connection
        table_name: Name of table (for logging)
    
    Returns:
        Normalized DataFrame
    """
    try:
        df = pd.read_sql(query, conn)
        df = normalize_column_names(df, table_name)
        logger.info(f"✅ Loaded {len(df):,} rows from {table_name}")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load {table_name}: {e}")
        return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Test case: DataFrame with inconsistent column names
    test_df = pd.DataFrame({
        'userid': [1, 2, 3],
        'friendid': [4, 5, 6],
        'status': [10, 10, 10]
    })
    
    print("Before normalization:")
    print(test_df.columns.tolist())
    
    normalized_df = normalize_column_names(test_df, "Friendship")
    
    print("\nAfter normalization:")
    print(normalized_df.columns.tolist())
    
    # Test groupby
    groups = safe_groupby(normalized_df, 'UserId', 'Friendship')
    print(f"\nGrouped by UserId: {len(groups)} groups")