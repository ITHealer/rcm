import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple
import ast
import yaml

def get_config(config_path: str = None) -> dict:
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_hashtags(hashtag_str):
    """
    Parse hashtags từ nhiều format

    Supports:
    - Python list: "['tag1', 'tag2']"
    - Hashtags với #: "#tag1 #tag2"
    - Plain text: "tag1 tag2"
    - Comma-separated: "tag1, tag2"
    """
    if pd.isna(hashtag_str) or hashtag_str == '' or hashtag_str == '[]':
        return []
    hashtag_str = str(hashtag_str).strip()
    # Format 1: Python list
    if hashtag_str.startswith('[') and hashtag_str.endswith(']'):
        try:
            result = ast.literal_eval(hashtag_str)
            if isinstance(result, list):
                return [str(tag).replace('#', '').strip() for tag in result]
        except:
            hashtag_str = hashtag_str.strip('[]').replace("'", "").replace('"', '')
    # Format 2 & 3: Space-separated
    if ' ' in hashtag_str:
        tags = hashtag_str.split()
        return [tag.replace('#', '').strip() for tag in tags if tag.strip()]
    # Format 4: Comma-separated
    if ',' in hashtag_str:
        tags = hashtag_str.split(',')
        return [tag.replace('#', '').strip() for tag in tags if tag.strip()]
    # Single tag
    return [hashtag_str.replace('#', '').strip()]

def load_data(data_dir: str = None) -> Dict[str, pd.DataFrame]:
    if data_dir is None:
        config = get_config()
        data_dir = config.get('data_dir', './dataset')
    data = {}

    # Định nghĩa cột ngày cho từng bảng
    datetime_columns = {
        'user': ['CreateDate', 'UpdateDate', 'DeleteDate'],
        'post': ['CreateDate', 'UpdateDate', 'DeleteDate'],
        'postreaction': ['CreateDate', 'UpdateDate', 'DeleteDate'],
        'postview': ['ViewDate', 'CreateDate', 'UpdateDate', 'DeleteDate'],
        'comment': ['CreateDate', 'UpdateDate', 'DeleteDate'],
        'sharelog': ['CreateDate', 'UpdateDate', 'DeleteDate'],
        'posthashtag': ['CreateDate', 'UpdateDate', 'DeleteDate'],
        'usersearchhistory': ['CreateDate', 'UpdateDate', 'DeleteDate'],
        'friendship': ['CreateDate', 'UpdateDate', 'DeleteDate'],
        'friendrequest': ['CreateDate', 'UpdateDate', 'DeleteDate'],
        'userhashtaginterest': ['CreateDate', 'UpdateDate', 'DeleteDate']
    }

    for fname in [
        'User.csv', 'Post.csv', 'PostReaction.csv', 'PostView.csv',
        'Comment.csv', 'ShareLog.csv', 'PostHashtag.csv', 'UserSearchHistory.csv',
        'Friendship.csv', 'FriendRequest.csv', 'UserHashtagInterest.csv'
    ]:
        fpath = os.path.join(data_dir, fname)
        key = fname.replace('.csv', '').lower()
        df = pd.read_csv(fpath)

        # Chuyển các cột ngày về kiểu datetime nếu có
        for col in datetime_columns.get(key, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Xử lý hashtags nếu có
        if 'hashtags' in df.columns:
            df['hashtags'] = df['hashtags'].apply(parse_hashtags)

        data[key] = df
    return data

def filter_recent_data(data: Dict[str, pd.DataFrame], lookback_days: int = None) -> Dict[str, pd.DataFrame]:
    if lookback_days is None:
        config = get_config()
        lookback_days = config.get('lookback_days', 90)
    print(f"\nFiltering Recent Data (Last {lookback_days} days).")
    # Lấy timezone từ dữ liệu nếu có
    tz = None
    if 'postreaction' in data and isinstance(data['postreaction']['CreateDate'].dtype, pd.DatetimeTZDtype):
        tz = data['postreaction']['CreateDate'].dt.tz
        now = pd.Timestamp(datetime.now()).tz_localize(tz)
    else:
        now = datetime.now()
    cutoff_date = now - timedelta(days=lookback_days)
    if 'postreaction' in data:
        recent_reactions = data['postreaction'][
            (data['postreaction']['CreateDate'] >= cutoff_date)
        ].copy()
        print(f"PostReactions: {len(data['postreaction']):,} → {len(recent_reactions):,}")
        relevant_post_ids = recent_reactions['PostId'].unique()
        if 'post' in data:
            data['post']['is_recent'] = data['post']['Id'].isin(relevant_post_ids)
        data['postreaction'] = recent_reactions
    print(f"Filtered to last {lookback_days} days")
    return data


# Hàm này tính toán các thống kê cần thiết cho việc feature engineering.
# Hàm này trả về  3 tham số
"""
- user_stats: Thống kê cho từng user (số lượng tương tác, tỷ lệ like, comment, share, thời gian xem trung bình,...)
- author_stats: Thống kê cho từng author (số bài viết, trung bình like, comment, share, engagement,...)
- following_dict: Dictionary cho biết user nào follow user nào (dạng {follower_id: set(followee_id)})
"""
def compute_statistics(data: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict, Dict]:
    """
    Pre-compute statistics cho feature engineering.
    Returns:
        (user_stats, author_stats, following_dict)
    """
    print("\nComputing Statistics.")
    # User statistics
    print("Computing user stats.")
    user_stats = {}
    # Precompute follower_count, following_count, post_count for all users
    follower_count_dict = data['friendship'].groupby('FriendId').size().to_dict() if 'friendship' in data else {}
    following_count_dict = data['friendship'].groupby('UserId').size().to_dict() if 'friendship' in data else {}
    post_count_dict = data['post'].groupby('UserId').size().to_dict() if 'post' in data else {}
    comment_count_dict = data['comment'].groupby('UserId').size().to_dict() if 'comment' in data else {}
    share_count_dict = data['sharelog'].groupby('UserId').size().to_dict() if 'sharelog' in data else {}
    view_count_dict = data['postview'].groupby('UserId').size().to_dict() if 'postview' in data else {}

    for user_id in data['user']['Id']:
        user_reactions = data['postreaction'][data['postreaction']['UserId'] == user_id] if 'postreaction' in data else pd.DataFrame()
        n_reactions = len(user_reactions)
        n_comments = comment_count_dict.get(user_id, 0)
        n_shares = share_count_dict.get(user_id, 0)
        n_views = view_count_dict.get(user_id, 0)
        total_actions = n_reactions + n_comments + n_shares + n_views
        
        if total_actions > 0:
            user_stats[user_id] = {
                'n_reactions': n_reactions,
                'like_rate': (user_reactions['ReactionTypeId'] == 1).sum() / total_actions if 'ReactionTypeId' in user_reactions.columns else 0,
                'comment_rate': n_comments / total_actions,
                'share_rate': n_shares / total_actions,
                'avg_dwell_time': n_views,  # Using view count as proxy for dwell time
                'follower_count': follower_count_dict.get(user_id, 0),
                'following_count': following_count_dict.get(user_id, 0),
                'post_count': post_count_dict.get(user_id, 0),
            }
        else:
            user_stats[user_id] = {
                'n_reactions': 0,
                'like_rate': 0,
                'comment_rate': 0,
                'share_rate': 0,
                'avg_dwell_time': 0,
                'follower_count': follower_count_dict.get(user_id, 0),
                'following_count': following_count_dict.get(user_id, 0),
                'post_count': post_count_dict.get(user_id, 0),
            }
    print(f"User stats for {len(user_stats)} users")
    # Author statistics
    print("Computing author stats.")
    author_stats = {}
    author_col = 'UserId'
    if 'post' in data and author_col in data['post'].columns:
        for author_id in data['post'][author_col].unique():
            author_posts = data['post'][data['post'][author_col] == author_id]
            # Compute engagement from PostReaction, Comment, ShareLog
            post_ids = author_posts['Id'].unique()
            total_likes = 0
            total_comments = 0
            total_shares = 0
            if 'postreaction' in data:
                reactions = data['postreaction'][data['postreaction']['PostId'].isin(post_ids)]
                total_likes = len(reactions[reactions['ReactionTypeId'] == 1])  # Assuming 1 is like
            if 'comment' in data:
                comments = data['comment'][data['comment']['PostId'].isin(post_ids)]
                total_comments = len(comments)
            if 'sharelog' in data:
                shares = data['sharelog'][data['sharelog']['PostId'].isin(post_ids)]
                total_shares = len(shares)
            total_engagement = total_likes + total_comments + total_shares
            author_stats[author_id] = {
                'n_posts': len(author_posts),
                'avg_likes': total_likes / len(author_posts) if len(author_posts) > 0 else 0,
                'avg_comments': total_comments / len(author_posts) if len(author_posts) > 0 else 0,
                'avg_shares': total_shares / len(author_posts) if len(author_posts) > 0 else 0,
                'avg_engagement': total_engagement / len(author_posts) if len(author_posts) > 0 else 0,
                'total_engagement': total_engagement,
            }
    print(f"Author stats for {len(author_stats)} authors")
    # Following dictionary (for fast lookup)
    print("Building following dictionary.")
    following_dict = {}
    if 'friendship' in data:
        for _, row in data['friendship'].iterrows():
            follower_id = row['UserId']
            followee_id = row['FriendId']
            if follower_id not in following_dict:
                following_dict[follower_id] = set()
            following_dict[follower_id].add(followee_id)
    print(f"Following dict for {len(following_dict)} users")
    return user_stats, author_stats, following_dict


# Chia dữ liệu thành train, val, test theo thời gian
"""
- test: là các tương tác xảy ra trong test_days. Mốc bắt đầu là ngày mới nhất trừ test_days.
- val: là các tương tác xảy ra trong khoảng từ (test_days + val_days). Mốc bắt đầu là ngày mới nhất trừ (test_days + val_days).
- train: là các tương tác xảy ra trước mốc bắt đầu của val.
"""

def create_temporal_splits(
    data: Dict[str, pd.DataFrame],
    test_days: int = 7,
    val_days: int = 7
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"\nCreating Temporal Splits...")
    print(f"Test: Last {test_days} days")
    print(f"Val: {test_days + val_days} to {test_days} days ago")
    print(f"Train: Everything before that")
    if 'postreaction' not in data:
        print("No postreaction data found, returning empty splits")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    reactions = data['postreaction'].copy()
    # Lấy timezone từ dữ liệu nếu có
    tz = None
    if isinstance(reactions['CreateDate'].dtype, pd.DatetimeTZDtype):
        tz = reactions['CreateDate'].dt.tz
        latest_date = reactions['CreateDate'].max()
        test_start = latest_date - pd.Timedelta(days=test_days)
        val_start = latest_date - pd.Timedelta(days=test_days + val_days)
    else:
        latest_date = reactions['CreateDate'].max()
        test_start = latest_date - timedelta(days=test_days)
        val_start = latest_date - timedelta(days=test_days + val_days)
    # Split
    test_reactions = reactions[
        reactions['CreateDate'] >= test_start
    ].copy()
    val_reactions = reactions[
        (reactions['CreateDate'] >= val_start) &
        (reactions['CreateDate'] < test_start)
    ].copy()
    train_reactions = reactions[
        reactions['CreateDate'] < val_start
    ].copy()
    print(f"\nSplit Summary:")
    print(f"Train: {len(train_reactions):,} reactions")
    print(f"Val:   {len(val_reactions):,} reactions")
    print(f"Test:  {len(test_reactions):,} reactions")
    print(f"Total: {len(reactions):,} reactions")

    return train_reactions, val_reactions, test_reactions


if __name__ == "__main__":
    config = get_config()
    data_dir = config.get('data_dir', './dataset')
    data = load_data(data_dir)
    data = filter_recent_data(data, lookback_days=config.get('lookback_days', 90))
    user_stats, author_stats, following_dict = compute_statistics(data)
    train_data, val_data, test_data = create_temporal_splits(data)
