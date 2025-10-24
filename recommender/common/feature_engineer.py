"""
FEATURE ENGINEERING FOR RANKING MODEL
======================================
Extract 47 features for recommendation ranking

Features:
- User Features (15)
- Post Features (18)
- Author Features (7)
- Interaction Features (7)

UPDATED: Support time decay weights + handle missing data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature extraction for ranking model
    Supports time decay weights
    Handles missing data gracefully
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        user_stats: Dict,
        author_stats: Dict,
        following_dict: Dict,
        embeddings: Optional[Dict] = None
    ):
        """
        Initialize feature engineer
        
        Args:
            data: Dict with 'user', 'post', 'postreaction', 'friendship'
            user_stats: User statistics
            author_stats: Author statistics
            following_dict: Following relationships
            embeddings: User/Post embeddings (optional)
        """
        self.data = data
        self.user_stats = user_stats
        self.author_stats = author_stats
        self.following_dict = following_dict
        self.embeddings = embeddings or {}
        
        # Caches for performance
        self.user_cache = {}
        self.post_cache = {}
        self.author_cache = {}
        
        # Reference timestamp for temporal features
        self.reference_time = datetime.now()
    
    def extract_features(
        self,
        user_id: int,
        post_id: int,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Extract all 47 features for (user, post) pair
        
        Args:
            user_id: User ID
            post_id: Post ID
            timestamp: Timestamp for temporal features
        
        Returns:
            Dict with 47 features
        """
        if timestamp is None:
            timestamp = self.reference_time
        
        features = {}
        
        try:
            # Get entities
            user = self._get_user_cached(user_id)
            post = self._get_post_cached(post_id)
            author_id = int(post.get('AuthorId', post.get('author_id', -1)))
            author = self._get_author_cached(author_id)
            
            # Extract all feature categories
            features.update(self._extract_user_features(user, user_id, timestamp))
            features.update(self._extract_post_features(post, timestamp))
            features.update(self._extract_author_features(author, author_id))
            features.update(self._extract_interaction_features(user_id, post_id, author_id, timestamp))
            
        except Exception as e:
            print(f"Warning: Feature extraction failed for user {user_id}, post {post_id}: {e}")
            # Return zero features
            features = self._get_zero_features()
        
        return features
    
    def _get_user_cached(self, user_id: int) -> pd.Series:
        """Get user with caching"""
        if user_id not in self.user_cache:
            user_df = self.data['user']
            
            # Handle both column name formats
            if 'Id' in user_df.columns:
                user_data = user_df[user_df['Id'] == user_id]
            else:
                user_data = user_df[user_df['user_id'] == user_id]
            
            if len(user_data) == 0:
                raise ValueError(f"User {user_id} not found")
            
            self.user_cache[user_id] = user_data.iloc[0]
        
        return self.user_cache[user_id]
    
    def _get_post_cached(self, post_id: int) -> pd.Series:
        """Get post with caching"""
        if post_id not in self.post_cache:
            post_df = self.data['post']
            
            # Handle both column name formats
            if 'Id' in post_df.columns:
                post_data = post_df[post_df['Id'] == post_id]
            else:
                post_data = post_df[post_df['post_id'] == post_id]
            
            if len(post_data) == 0:
                raise ValueError(f"Post {post_id} not found")
            
            self.post_cache[post_id] = post_data.iloc[0]
        
        return self.post_cache[post_id]
    
    def _get_author_cached(self, author_id: int) -> pd.Series:
        """Get author with caching"""
        if author_id not in self.author_cache:
            self.author_cache[author_id] = self._get_user_cached(author_id)
        
        return self.author_cache[author_id]
    
    # ========================================================================
    # USER FEATURES (15 features)
    # ========================================================================
    
    def _extract_user_features(
        self,
        user: pd.Series,
        user_id: int,
        timestamp: datetime
    ) -> Dict[str, float]:
        """Extract user features"""
        features = {}
        
        # Get user stats
        u_stats = self.user_stats.get(user_id, {})
        
        # 1. user_age_days - Account age
        create_date = user.get('CreateDate', user.get('created_at', timestamp))
        if pd.notna(create_date):
            if isinstance(create_date, str):
                create_date = pd.to_datetime(create_date)
            features['user_age_days'] = (timestamp - create_date).days
        else:
            features['user_age_days'] = 0.0
        
        # 2. user_total_posts - Total posts created
        features['user_total_posts'] = float(u_stats.get('n_posts', 0))
        
        # 3. user_total_followers - Number of followers
        # NOTE: Requires Friendship table with proper follower counts
        # COMMENTED: Data not available
        # features['user_total_followers'] = float(u_stats.get('n_followers', 0))
        features['user_total_followers'] = 0.0  # Placeholder
        
        # 4. user_total_following - Number following
        features['user_total_following'] = float(u_stats.get('n_following', 0))
        
        # 5. user_follower_following_ratio - Follower/Following ratio
        following = features['user_total_following']
        if following > 0:
            features['user_follower_following_ratio'] = features['user_total_followers'] / following
        else:
            features['user_follower_following_ratio'] = 0.0
        
        # 6. user_avg_likes_per_post - Average likes per post
        # COMMENTED: Requires aggregated post metrics
        # features['user_avg_likes_per_post'] = float(u_stats.get('avg_likes_per_post', 0))
        features['user_avg_likes_per_post'] = 0.0  # Placeholder
        
        # 7. user_engagement_rate_7d - Engagement rate last 7 days
        features['user_engagement_rate_7d'] = float(u_stats.get('engagement_rate_7d', 0))
        
        # 8. user_posts_per_day_7d - Posts per day (7 days)
        # COMMENTED: Requires temporal post aggregation
        # features['user_posts_per_day_7d'] = float(u_stats.get('posts_per_day_7d', 0))
        features['user_posts_per_day_7d'] = 0.0  # Placeholder
        
        # 9. user_likes_given_7d - Likes given (7 days)
        features['user_likes_given_7d'] = float(u_stats.get('likes_given_7d', 0))
        
        # 10. user_comments_given_7d - Comments given (7 days)
        features['user_comments_given_7d'] = float(u_stats.get('comments_given_7d', 0))
        
        # 11. user_session_count_7d - Session count (7 days)
        # COMMENTED: Requires session tracking
        # features['user_session_count_7d'] = float(u_stats.get('session_count_7d', 0))
        features['user_session_count_7d'] = 0.0  # Placeholder
        
        # 12. user_avg_session_duration - Average session duration
        # COMMENTED: Requires session tracking
        # features['user_avg_session_duration'] = float(u_stats.get('avg_session_duration', 0))
        features['user_avg_session_duration'] = 0.0  # Placeholder
        
        # 13. user_is_verified - Is verified
        # COMMENTED: Requires user verification field
        # features['user_is_verified'] = float(user.get('IsVerified', 0))
        features['user_is_verified'] = 0.0  # Placeholder
        
        # 14. user_has_profile_pic - Has profile picture
        # COMMENTED: Requires profile picture field
        # features['user_has_profile_pic'] = float(user.get('HasProfilePic', 0))
        features['user_has_profile_pic'] = 0.0  # Placeholder
        
        # 15. user_activity_level - Activity level (0-3)
        n_interactions = u_stats.get('n_interactions', 0)
        if n_interactions < 10:
            features['user_activity_level'] = 0.0  # Low
        elif n_interactions < 50:
            features['user_activity_level'] = 1.0  # Medium
        elif n_interactions < 200:
            features['user_activity_level'] = 2.0  # High
        else:
            features['user_activity_level'] = 3.0  # Very high
        
        return features
    
    # ========================================================================
    # POST FEATURES (18 features)
    # ========================================================================
    
    def _extract_post_features(
        self,
        post: pd.Series,
        timestamp: datetime
    ) -> Dict[str, float]:
        """Extract post features"""
        features = {}
        
        # Get post_id
        post_id = int(post.get('Id', post.get('post_id', -1)))
        
        # 16. post_age_hours - Post age in hours
        create_date = post.get('CreateDate', post.get('created_at', timestamp))
        if pd.notna(create_date):
            if isinstance(create_date, str):
                create_date = pd.to_datetime(create_date)
            post_age = (timestamp - create_date).total_seconds() / 3600.0
            features['post_age_hours'] = max(0, post_age)
        else:
            features['post_age_hours'] = 0.0
        
        # Get reactions for this post
        reactions_df = self.data['postreaction']
        
        # Handle column name variations
        if 'PostId' in reactions_df.columns:
            post_reactions = reactions_df[reactions_df['PostId'] == post_id]
        else:
            post_reactions = reactions_df[reactions_df['post_id'] == post_id]
        
        # 17. post_total_likes - Total likes
        if 'ReactionTypeId' in post_reactions.columns:
            features['post_total_likes'] = float(len(post_reactions[post_reactions['ReactionTypeId'] == 1]))
        elif 'action' in post_reactions.columns:
            features['post_total_likes'] = float(len(post_reactions[post_reactions['action'] == 'like']))
        else:
            features['post_total_likes'] = 0.0
        
        # 18. post_total_comments - Total comments
        if 'ReactionTypeId' in post_reactions.columns:
            features['post_total_comments'] = float(len(post_reactions[post_reactions['ReactionTypeId'] == 2]))
        elif 'action' in post_reactions.columns:
            features['post_total_comments'] = float(len(post_reactions[post_reactions['action'] == 'comment']))
        else:
            features['post_total_comments'] = 0.0
        
        # 19. post_total_shares - Total shares
        if 'ReactionTypeId' in post_reactions.columns:
            features['post_total_shares'] = float(len(post_reactions[post_reactions['ReactionTypeId'] == 3]))
        elif 'action' in post_reactions.columns:
            features['post_total_shares'] = float(len(post_reactions[post_reactions['action'] == 'share']))
        else:
            features['post_total_shares'] = 0.0
        
        # 20. post_total_views - Total views
        if 'ReactionTypeId' in post_reactions.columns:
            features['post_total_views'] = float(len(post_reactions[post_reactions['ReactionTypeId'] == 4]))
        elif 'action' in post_reactions.columns:
            features['post_total_views'] = float(len(post_reactions[post_reactions['action'] == 'view']))
        else:
            features['post_total_views'] = float(len(post_reactions))  # All reactions as views
        
        # 21-23. post_ctr_Xh - CTR at different time windows
        # COMMENTED: Requires temporal aggregation
        # features['post_ctr_1h'] = self._calculate_ctr(post_id, hours=1)
        # features['post_ctr_6h'] = self._calculate_ctr(post_id, hours=6)
        # features['post_ctr_24h'] = self._calculate_ctr(post_id, hours=24)
        features['post_ctr_1h'] = 0.0  # Placeholder
        features['post_ctr_6h'] = 0.0  # Placeholder
        features['post_ctr_24h'] = 0.0  # Placeholder
        
        # 24. post_engagement_velocity - Engagement velocity
        # COMMENTED: Requires temporal metrics
        # features['post_engagement_velocity'] = 0.0
        total_engagements = features['post_total_likes'] + features['post_total_comments'] + features['post_total_shares']
        if features['post_age_hours'] > 0:
            features['post_engagement_velocity'] = total_engagements / features['post_age_hours']
        else:
            features['post_engagement_velocity'] = 0.0
        
        # 25. post_has_image - Has image
        content = str(post.get('Content', ''))
        features['post_has_image'] = 1.0 if any(ext in content.lower() for ext in ['.jpg', '.png', '.jpeg', 'image']) else 0.0
        
        # 26. post_has_video - Has video
        features['post_has_video'] = 1.0 if any(ext in content.lower() for ext in ['.mp4', '.mov', 'video']) else 0.0
        
        # 27. post_has_link - Has link
        features['post_has_link'] = 1.0 if 'http' in content.lower() else 0.0
        
        # 28. post_text_length - Text length
        features['post_text_length'] = float(len(content))
        
        # 29. post_hashtag_count - Hashtag count
        features['post_hashtag_count'] = float(content.count('#'))
        
        # 30. post_mention_count - Mention count
        features['post_mention_count'] = float(content.count('@'))
        
        # 31. post_topic - Topic (categorical)
        # COMMENTED: Requires topic modeling
        # features['post_topic'] = 0.0
        features['post_topic'] = 0.0  # Placeholder
        
        # 32. post_language - Language (categorical)
        # COMMENTED: Requires language detection
        # features['post_language'] = 0.0
        features['post_language'] = 0.0  # Placeholder
        
        # 33. post_is_reply - Is reply
        # COMMENTED: Requires reply tracking
        # features['post_is_reply'] = 0.0
        features['post_is_reply'] = 0.0  # Placeholder
        
        return features
    
    # ========================================================================
    # AUTHOR FEATURES (7 features)
    # ========================================================================
    
    def _extract_author_features(
        self,
        author: pd.Series,
        author_id: int
    ) -> Dict[str, float]:
        """Extract author features"""
        features = {}
        
        # Get author stats
        a_stats = self.author_stats.get(author_id, {})
        
        # 34. author_total_followers
        # COMMENTED: Requires follower counts
        # features['author_total_followers'] = float(a_stats.get('n_followers', 0))
        features['author_total_followers'] = 0.0  # Placeholder
        
        # 35. author_total_posts
        features['author_total_posts'] = float(a_stats.get('n_posts', 0))
        
        # 36. author_avg_engagement_rate
        features['author_avg_engagement_rate'] = float(a_stats.get('avg_engagement_rate', 0))
        
        # 37. author_posts_per_day
        # COMMENTED: Requires temporal aggregation
        # features['author_posts_per_day'] = float(a_stats.get('posts_per_day', 0))
        features['author_posts_per_day'] = 0.0  # Placeholder
        
        # 38. author_is_verified
        # COMMENTED: Requires verification field
        # features['author_is_verified'] = float(author.get('IsVerified', 0))
        features['author_is_verified'] = 0.0  # Placeholder
        
        # 39. author_account_age_days
        create_date = author.get('CreateDate', author.get('created_at'))
        if pd.notna(create_date):
            if isinstance(create_date, str):
                create_date = pd.to_datetime(create_date)
            features['author_account_age_days'] = (self.reference_time - create_date).days
        else:
            features['author_account_age_days'] = 0.0
        
        # 40. author_follower_growth_7d
        # COMMENTED: Requires temporal follower tracking
        # features['author_follower_growth_7d'] = float(a_stats.get('follower_growth_7d', 0))
        features['author_follower_growth_7d'] = 0.0  # Placeholder
        
        return features
    
    # ========================================================================
    # INTERACTION FEATURES (7 features)
    # ========================================================================
    
    def _extract_interaction_features(
        self,
        user_id: int,
        post_id: int,
        author_id: int,
        timestamp: datetime
    ) -> Dict[str, float]:
        """Extract interaction features"""
        features = {}
        
        # 41. user_follows_author - Does user follow author?
        user_following = self.following_dict.get(user_id, set())
        features['user_follows_author'] = 1.0 if author_id in user_following else 0.0
        
        # 42. user_author_past_interactions - Past interactions count
        reactions_df = self.data['postreaction']
        
        # Get posts by this author
        posts_df = self.data['post']
        if 'AuthorId' in posts_df.columns:
            author_posts = set(posts_df[posts_df['AuthorId'] == author_id]['Id'])
        else:
            author_posts = set(posts_df[posts_df['author_id'] == author_id]['post_id'])
        
        # Count user interactions with author's posts
        if 'UserId' in reactions_df.columns and 'PostId' in reactions_df.columns:
            user_reactions = reactions_df[reactions_df['UserId'] == user_id]
            features['user_author_past_interactions'] = float(
                len(user_reactions[user_reactions['PostId'].isin(author_posts)])
            )
        elif 'user_id' in reactions_df.columns and 'post_id' in reactions_df.columns:
            user_reactions = reactions_df[reactions_df['user_id'] == user_id]
            features['user_author_past_interactions'] = float(
                len(user_reactions[user_reactions['post_id'].isin(author_posts)])
            )
        else:
            features['user_author_past_interactions'] = 0.0
        
        # 43. user_topic_affinity - Topic affinity
        # COMMENTED: Requires topic modeling
        # features['user_topic_affinity'] = 0.0
        features['user_topic_affinity'] = 0.0  # Placeholder
        
        # 44. user_author_similarity - CF similarity
        # Use embeddings if available
        if self.embeddings and 'user' in self.embeddings and 'post' in self.embeddings:
            user_emb = self.embeddings['user'].get(user_id)
            post_emb = self.embeddings['post'].get(post_id)
            
            if user_emb is not None and post_emb is not None:
                # Cosine similarity
                similarity = np.dot(user_emb, post_emb) / (
                    np.linalg.norm(user_emb) * np.linalg.norm(post_emb) + 1e-8
                )
                features['user_author_similarity'] = float(similarity)
            else:
                features['user_author_similarity'] = 0.0
        else:
            features['user_author_similarity'] = 0.0
        
        # 45. post_in_user_language - Language match
        # COMMENTED: Requires language detection
        # features['post_in_user_language'] = 1.0
        features['post_in_user_language'] = 1.0  # Assume match
        
        # 46. time_since_last_seen_author - Time since last interaction
        if 'CreateDate' in reactions_df.columns:
            user_author_reactions = reactions_df[
                (reactions_df['UserId'] == user_id) & 
                (reactions_df['PostId'].isin(author_posts))
            ]
            if len(user_author_reactions) > 0:
                last_interaction = pd.to_datetime(user_author_reactions['CreateDate'].max())
                hours_since = (timestamp - last_interaction).total_seconds() / 3600.0
                features['time_since_last_seen_author'] = max(0, hours_since)
            else:
                features['time_since_last_seen_author'] = 999.0  # Large value
        else:
            features['time_since_last_seen_author'] = 999.0
        
        # 47. contextual_freshness_boost - Freshness boost
        # Boost recent posts from followed authors
        if features['user_follows_author'] > 0:
            features['contextual_freshness_boost'] = 1.5
        else:
            features['contextual_freshness_boost'] = 1.0
        
        return features
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _get_zero_features(self) -> Dict[str, float]:
        """Return zero features (fallback)"""
        feature_names = [
            # User features (15)
            'user_age_days', 'user_total_posts', 'user_total_followers',
            'user_total_following', 'user_follower_following_ratio',
            'user_avg_likes_per_post', 'user_engagement_rate_7d',
            'user_posts_per_day_7d', 'user_likes_given_7d',
            'user_comments_given_7d', 'user_session_count_7d',
            'user_avg_session_duration', 'user_is_verified',
            'user_has_profile_pic', 'user_activity_level',
            
            # Post features (18)
            'post_age_hours', 'post_total_likes', 'post_total_comments',
            'post_total_shares', 'post_total_views', 'post_ctr_1h',
            'post_ctr_6h', 'post_ctr_24h', 'post_engagement_velocity',
            'post_has_image', 'post_has_video', 'post_has_link',
            'post_text_length', 'post_hashtag_count', 'post_mention_count',
            'post_topic', 'post_language', 'post_is_reply',
            
            # Author features (7)
            'author_total_followers', 'author_total_posts',
            'author_avg_engagement_rate', 'author_posts_per_day',
            'author_is_verified', 'author_account_age_days',
            'author_follower_growth_7d',
            
            # Interaction features (7)
            'user_follows_author', 'user_author_past_interactions',
            'user_topic_affinity', 'user_author_similarity',
            'post_in_user_language', 'time_since_last_seen_author',
            'contextual_freshness_boost'
        ]
        
        return {name: 0.0 for name in feature_names}
    
    def get_feature_names(self) -> List[str]:
        """Get list of all 47 feature names"""
        return list(self._get_zero_features().keys())


# ============================================================================
# BATCH FEATURE EXTRACTION
# ============================================================================

def extract_features_batch(
    interactions_df: pd.DataFrame,
    feature_engineer: FeatureEngineer,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Extract features for batch of interactions
    
    Args:
        interactions_df: DataFrame with user_id, post_id, (optional: created_at, weight)
        feature_engineer: FeatureEngineer instance
        show_progress: Show progress bar
    
    Returns:
        DataFrame with features + label
    """
    print(f"\n🔧 Extracting features for {len(interactions_df):,} samples...")
    
    features_list = []
    failed_count = 0
    
    # Determine column names
    if 'UserId' in interactions_df.columns:
        user_col = 'UserId'
        post_col = 'PostId'
        time_col = 'CreateDate'
    else:
        user_col = 'user_id'
        post_col = 'post_id'
        time_col = 'created_at'
    
    # Extract features
    total = len(interactions_df)
    
    for idx, row in interactions_df.iterrows():
        if show_progress and idx % 1000 == 0:
            print(f"   Progress: {idx:,}/{total:,} ({idx/total*100:.1f}%)", end='\r')
        
        try:
            timestamp = row.get(time_col, None)
            if timestamp and isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            features = feature_engineer.extract_features(
                row[user_col],
                row[post_col],
                timestamp
            )
            
            # Add metadata
            features['user_id'] = row[user_col]
            features['post_id'] = row[post_col]
            
            # Preserve weight if exists
            if 'weight' in row:
                features['weight'] = row['weight']
            
            features_list.append(features)
            
        except Exception as e:
            failed_count += 1
            continue
    
    if show_progress:
        print(f"   Progress: {total:,}/{total:,} (100.0%)")
    
    if failed_count > 0:
        print(f"   ⚠️  Failed to extract features for {failed_count:,} samples")
    
    df = pd.DataFrame(features_list)
    
    print(f"   ✅ Extracted features: {df.shape}")
    
    return df