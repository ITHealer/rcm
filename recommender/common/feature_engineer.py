# """
# FEATURE ENGINEERING FOR RANKING MODEL
# ======================================
# Extract 47 features for recommendation ranking

# Features:
# - User Features (15)
# - Post Features (18)
# - Author Features (7)
# - Interaction Features (7)

# UPDATED: Support time decay weights + handle missing data
# """

# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# from typing import Dict, List, Tuple, Optional
# import warnings

# warnings.filterwarnings('ignore')


# class FeatureEngineer:
#     """
#     Feature extraction for ranking model
#     Supports time decay weights
#     Handles missing data gracefully
#     """
    
#     def __init__(
#         self,
#         data: Dict[str, pd.DataFrame],
#         user_stats: Dict,
#         author_stats: Dict,
#         following_dict: Dict,
#         embeddings: Optional[Dict] = None
#     ):
#         """
#         Initialize feature engineer
        
#         Args:
#             data: Dict with 'user', 'post', 'postreaction', 'friendship'
#             user_stats: User statistics
#             author_stats: Author statistics
#             following_dict: Following relationships
#             embeddings: User/Post embeddings (optional)
#         """
#         self.data = data
#         self.user_stats = user_stats
#         self.author_stats = author_stats
#         self.following_dict = following_dict
#         self.embeddings = embeddings or {}
        
#         # Caches for performance
#         self.user_cache = {}
#         self.post_cache = {}
#         self.author_cache = {}
        
#         # Reference timestamp for temporal features
#         self.reference_time = datetime.now()
    
#     def extract_features(
#         self,
#         user_id: int,
#         post_id: int,
#         timestamp: Optional[datetime] = None
#     ) -> Dict[str, float]:
#         """
#         Extract all 47 features for (user, post) pair
        
#         Args:
#             user_id: User ID
#             post_id: Post ID
#             timestamp: Timestamp for temporal features
        
#         Returns:
#             Dict with 47 features
#         """
#         if timestamp is None:
#             timestamp = self.reference_time
        
#         features = {}
        
#         try:
#             # Get entities
#             user = self._get_user_cached(user_id)
#             post = self._get_post_cached(post_id)
#             author_id = int(post.get('AuthorId', post.get('author_id', -1)))
#             author = self._get_author_cached(author_id)
            
#             # Extract all feature categories
#             features.update(self._extract_user_features(user, user_id, timestamp))
#             features.update(self._extract_post_features(post, timestamp))
#             features.update(self._extract_author_features(author, author_id))
#             features.update(self._extract_interaction_features(user_id, post_id, author_id, timestamp))
            
#         except Exception as e:
#             print(f"Warning: Feature extraction failed for user {user_id}, post {post_id}: {e}")
#             # Return zero features
#             features = self._get_zero_features()
        
#         return features
    
#     def _get_user_cached(self, user_id: int) -> pd.Series:
#         """Get user with caching"""
#         if user_id not in self.user_cache:
#             user_df = self.data['user']
            
#             # Handle both column name formats
#             if 'Id' in user_df.columns:
#                 user_data = user_df[user_df['Id'] == user_id]
#             else:
#                 user_data = user_df[user_df['user_id'] == user_id]
            
#             if len(user_data) == 0:
#                 raise ValueError(f"User {user_id} not found")
            
#             self.user_cache[user_id] = user_data.iloc[0]
        
#         return self.user_cache[user_id]
    
#     def _get_post_cached(self, post_id: int) -> pd.Series:
#         """Get post with caching"""
#         if post_id not in self.post_cache:
#             post_df = self.data['post']
            
#             # Handle both column name formats
#             if 'Id' in post_df.columns:
#                 post_data = post_df[post_df['Id'] == post_id]
#             else:
#                 post_data = post_df[post_df['post_id'] == post_id]
            
#             if len(post_data) == 0:
#                 raise ValueError(f"Post {post_id} not found")
            
#             self.post_cache[post_id] = post_data.iloc[0]
        
#         return self.post_cache[post_id]
    
#     def _get_author_cached(self, author_id: int) -> pd.Series:
#         """Get author with caching"""
#         if author_id not in self.author_cache:
#             self.author_cache[author_id] = self._get_user_cached(author_id)
        
#         return self.author_cache[author_id]
    
#     # ========================================================================
#     # USER FEATURES (15 features)
#     # ========================================================================
    
#     def _extract_user_features(
#         self,
#         user: pd.Series,
#         user_id: int,
#         timestamp: datetime
#     ) -> Dict[str, float]:
#         """Extract user features"""
#         features = {}
        
#         # Get user stats
#         u_stats = self.user_stats.get(user_id, {})
        
#         # 1. user_age_days - Account age
#         create_date = user.get('CreateDate', user.get('created_at', timestamp))
#         if pd.notna(create_date):
#             if isinstance(create_date, str):
#                 create_date = pd.to_datetime(create_date)
#             features['user_age_days'] = (timestamp - create_date).days
#         else:
#             features['user_age_days'] = 0.0
        
#         # 2. user_total_posts - Total posts created
#         features['user_total_posts'] = float(u_stats.get('n_posts', 0))
        
#         # 3. user_total_followers - Number of followers
#         # NOTE: Requires Friendship table with proper follower counts
#         # COMMENTED: Data not available
#         # features['user_total_followers'] = float(u_stats.get('n_followers', 0))
#         features['user_total_followers'] = 0.0  # Placeholder
        
#         # 4. user_total_following - Number following
#         features['user_total_following'] = float(u_stats.get('n_following', 0))
        
#         # 5. user_follower_following_ratio - Follower/Following ratio
#         following = features['user_total_following']
#         if following > 0:
#             features['user_follower_following_ratio'] = features['user_total_followers'] / following
#         else:
#             features['user_follower_following_ratio'] = 0.0
        
#         # 6. user_avg_likes_per_post - Average likes per post
#         # COMMENTED: Requires aggregated post metrics
#         # features['user_avg_likes_per_post'] = float(u_stats.get('avg_likes_per_post', 0))
#         features['user_avg_likes_per_post'] = 0.0  # Placeholder
        
#         # 7. user_engagement_rate_7d - Engagement rate last 7 days
#         features['user_engagement_rate_7d'] = float(u_stats.get('engagement_rate_7d', 0))
        
#         # 8. user_posts_per_day_7d - Posts per day (7 days)
#         # COMMENTED: Requires temporal post aggregation
#         # features['user_posts_per_day_7d'] = float(u_stats.get('posts_per_day_7d', 0))
#         features['user_posts_per_day_7d'] = 0.0  # Placeholder
        
#         # 9. user_likes_given_7d - Likes given (7 days)
#         features['user_likes_given_7d'] = float(u_stats.get('likes_given_7d', 0))
        
#         # 10. user_comments_given_7d - Comments given (7 days)
#         features['user_comments_given_7d'] = float(u_stats.get('comments_given_7d', 0))
        
#         # 11. user_session_count_7d - Session count (7 days)
#         # COMMENTED: Requires session tracking
#         # features['user_session_count_7d'] = float(u_stats.get('session_count_7d', 0))
#         features['user_session_count_7d'] = 0.0  # Placeholder
        
#         # 12. user_avg_session_duration - Average session duration
#         # COMMENTED: Requires session tracking
#         # features['user_avg_session_duration'] = float(u_stats.get('avg_session_duration', 0))
#         features['user_avg_session_duration'] = 0.0  # Placeholder
        
#         # 13. user_is_verified - Is verified
#         # COMMENTED: Requires user verification field
#         # features['user_is_verified'] = float(user.get('IsVerified', 0))
#         features['user_is_verified'] = 0.0  # Placeholder
        
#         # 14. user_has_profile_pic - Has profile picture
#         # COMMENTED: Requires profile picture field
#         # features['user_has_profile_pic'] = float(user.get('HasProfilePic', 0))
#         features['user_has_profile_pic'] = 0.0  # Placeholder
        
#         # 15. user_activity_level - Activity level (0-3)
#         n_interactions = u_stats.get('n_interactions', 0)
#         if n_interactions < 10:
#             features['user_activity_level'] = 0.0  # Low
#         elif n_interactions < 50:
#             features['user_activity_level'] = 1.0  # Medium
#         elif n_interactions < 200:
#             features['user_activity_level'] = 2.0  # High
#         else:
#             features['user_activity_level'] = 3.0  # Very high
        
#         return features
    
#     # ========================================================================
#     # POST FEATURES (18 features)
#     # ========================================================================
    
#     def _extract_post_features(
#         self,
#         post: pd.Series,
#         timestamp: datetime
#     ) -> Dict[str, float]:
#         """Extract post features"""
#         features = {}
        
#         # Get post_id
#         post_id = int(post.get('Id', post.get('post_id', -1)))
        
#         # 16. post_age_hours - Post age in hours
#         create_date = post.get('CreateDate', post.get('created_at', timestamp))
#         if pd.notna(create_date):
#             if isinstance(create_date, str):
#                 create_date = pd.to_datetime(create_date)
#             post_age = (timestamp - create_date).total_seconds() / 3600.0
#             features['post_age_hours'] = max(0, post_age)
#         else:
#             features['post_age_hours'] = 0.0
        
#         # Get reactions for this post
#         reactions_df = self.data['postreaction']
        
#         # Handle column name variations
#         if 'PostId' in reactions_df.columns:
#             post_reactions = reactions_df[reactions_df['PostId'] == post_id]
#         else:
#             post_reactions = reactions_df[reactions_df['post_id'] == post_id]
        
#         # 17. post_total_likes - Total likes
#         if 'ReactionTypeId' in post_reactions.columns:
#             features['post_total_likes'] = float(len(post_reactions[post_reactions['ReactionTypeId'] == 1]))
#         elif 'action' in post_reactions.columns:
#             features['post_total_likes'] = float(len(post_reactions[post_reactions['action'] == 'like']))
#         else:
#             features['post_total_likes'] = 0.0
        
#         # 18. post_total_comments - Total comments
#         if 'ReactionTypeId' in post_reactions.columns:
#             features['post_total_comments'] = float(len(post_reactions[post_reactions['ReactionTypeId'] == 2]))
#         elif 'action' in post_reactions.columns:
#             features['post_total_comments'] = float(len(post_reactions[post_reactions['action'] == 'comment']))
#         else:
#             features['post_total_comments'] = 0.0
        
#         # 19. post_total_shares - Total shares
#         if 'ReactionTypeId' in post_reactions.columns:
#             features['post_total_shares'] = float(len(post_reactions[post_reactions['ReactionTypeId'] == 3]))
#         elif 'action' in post_reactions.columns:
#             features['post_total_shares'] = float(len(post_reactions[post_reactions['action'] == 'share']))
#         else:
#             features['post_total_shares'] = 0.0
        
#         # 20. post_total_views - Total views
#         if 'ReactionTypeId' in post_reactions.columns:
#             features['post_total_views'] = float(len(post_reactions[post_reactions['ReactionTypeId'] == 4]))
#         elif 'action' in post_reactions.columns:
#             features['post_total_views'] = float(len(post_reactions[post_reactions['action'] == 'view']))
#         else:
#             features['post_total_views'] = float(len(post_reactions))  # All reactions as views
        
#         # 21-23. post_ctr_Xh - CTR at different time windows
#         # COMMENTED: Requires temporal aggregation
#         # features['post_ctr_1h'] = self._calculate_ctr(post_id, hours=1)
#         # features['post_ctr_6h'] = self._calculate_ctr(post_id, hours=6)
#         # features['post_ctr_24h'] = self._calculate_ctr(post_id, hours=24)
#         features['post_ctr_1h'] = 0.0  # Placeholder
#         features['post_ctr_6h'] = 0.0  # Placeholder
#         features['post_ctr_24h'] = 0.0  # Placeholder
        
#         # 24. post_engagement_velocity - Engagement velocity
#         # COMMENTED: Requires temporal metrics
#         # features['post_engagement_velocity'] = 0.0
#         total_engagements = features['post_total_likes'] + features['post_total_comments'] + features['post_total_shares']
#         if features['post_age_hours'] > 0:
#             features['post_engagement_velocity'] = total_engagements / features['post_age_hours']
#         else:
#             features['post_engagement_velocity'] = 0.0
        
#         # 25. post_has_image - Has image
#         content = str(post.get('Content', ''))
#         features['post_has_image'] = 1.0 if any(ext in content.lower() for ext in ['.jpg', '.png', '.jpeg', 'image']) else 0.0
        
#         # 26. post_has_video - Has video
#         features['post_has_video'] = 1.0 if any(ext in content.lower() for ext in ['.mp4', '.mov', 'video']) else 0.0
        
#         # 27. post_has_link - Has link
#         features['post_has_link'] = 1.0 if 'http' in content.lower() else 0.0
        
#         # 28. post_text_length - Text length
#         features['post_text_length'] = float(len(content))
        
#         # 29. post_hashtag_count - Hashtag count
#         features['post_hashtag_count'] = float(content.count('#'))
        
#         # 30. post_mention_count - Mention count
#         features['post_mention_count'] = float(content.count('@'))
        
#         # 31. post_topic - Topic (categorical)
#         # COMMENTED: Requires topic modeling
#         # features['post_topic'] = 0.0
#         features['post_topic'] = 0.0  # Placeholder
        
#         # 32. post_language - Language (categorical)
#         # COMMENTED: Requires language detection
#         # features['post_language'] = 0.0
#         features['post_language'] = 0.0  # Placeholder
        
#         # 33. post_is_reply - Is reply
#         # COMMENTED: Requires reply tracking
#         # features['post_is_reply'] = 0.0
#         features['post_is_reply'] = 0.0  # Placeholder
        
#         return features
    
#     # ========================================================================
#     # AUTHOR FEATURES (7 features)
#     # ========================================================================
    
#     def _extract_author_features(
#         self,
#         author: pd.Series,
#         author_id: int
#     ) -> Dict[str, float]:
#         """Extract author features"""
#         features = {}
        
#         # Get author stats
#         a_stats = self.author_stats.get(author_id, {})
        
#         # 34. author_total_followers
#         # COMMENTED: Requires follower counts
#         # features['author_total_followers'] = float(a_stats.get('n_followers', 0))
#         features['author_total_followers'] = 0.0  # Placeholder
        
#         # 35. author_total_posts
#         features['author_total_posts'] = float(a_stats.get('n_posts', 0))
        
#         # 36. author_avg_engagement_rate
#         features['author_avg_engagement_rate'] = float(a_stats.get('avg_engagement_rate', 0))
        
#         # 37. author_posts_per_day
#         # COMMENTED: Requires temporal aggregation
#         # features['author_posts_per_day'] = float(a_stats.get('posts_per_day', 0))
#         features['author_posts_per_day'] = 0.0  # Placeholder
        
#         # 38. author_is_verified
#         # COMMENTED: Requires verification field
#         # features['author_is_verified'] = float(author.get('IsVerified', 0))
#         features['author_is_verified'] = 0.0  # Placeholder
        
#         # 39. author_account_age_days
#         create_date = author.get('CreateDate', author.get('created_at'))
#         if pd.notna(create_date):
#             if isinstance(create_date, str):
#                 create_date = pd.to_datetime(create_date)
#             features['author_account_age_days'] = (self.reference_time - create_date).days
#         else:
#             features['author_account_age_days'] = 0.0
        
#         # 40. author_follower_growth_7d
#         # COMMENTED: Requires temporal follower tracking
#         # features['author_follower_growth_7d'] = float(a_stats.get('follower_growth_7d', 0))
#         features['author_follower_growth_7d'] = 0.0  # Placeholder
        
#         return features
    
#     # ========================================================================
#     # INTERACTION FEATURES (7 features)
#     # ========================================================================
    
#     def _extract_interaction_features(
#         self,
#         user_id: int,
#         post_id: int,
#         author_id: int,
#         timestamp: datetime
#     ) -> Dict[str, float]:
#         """Extract interaction features"""
#         features = {}
        
#         # 41. user_follows_author - Does user follow author?
#         user_following = self.following_dict.get(user_id, set())
#         features['user_follows_author'] = 1.0 if author_id in user_following else 0.0
        
#         # 42. user_author_past_interactions - Past interactions count
#         reactions_df = self.data['postreaction']
        
#         # Get posts by this author
#         posts_df = self.data['post']
#         if 'AuthorId' in posts_df.columns:
#             author_posts = set(posts_df[posts_df['AuthorId'] == author_id]['Id'])
#         else:
#             author_posts = set(posts_df[posts_df['author_id'] == author_id]['post_id'])
        
#         # Count user interactions with author's posts
#         if 'UserId' in reactions_df.columns and 'PostId' in reactions_df.columns:
#             user_reactions = reactions_df[reactions_df['UserId'] == user_id]
#             features['user_author_past_interactions'] = float(
#                 len(user_reactions[user_reactions['PostId'].isin(author_posts)])
#             )
#         elif 'user_id' in reactions_df.columns and 'post_id' in reactions_df.columns:
#             user_reactions = reactions_df[reactions_df['user_id'] == user_id]
#             features['user_author_past_interactions'] = float(
#                 len(user_reactions[user_reactions['post_id'].isin(author_posts)])
#             )
#         else:
#             features['user_author_past_interactions'] = 0.0
        
#         # 43. user_topic_affinity - Topic affinity
#         # COMMENTED: Requires topic modeling
#         # features['user_topic_affinity'] = 0.0
#         features['user_topic_affinity'] = 0.0  # Placeholder
        
#         # 44. user_author_similarity - CF similarity
#         # Use embeddings if available
#         if self.embeddings and 'user' in self.embeddings and 'post' in self.embeddings:
#             user_emb = self.embeddings['user'].get(user_id)
#             post_emb = self.embeddings['post'].get(post_id)
            
#             if user_emb is not None and post_emb is not None:
#                 # Cosine similarity
#                 similarity = np.dot(user_emb, post_emb) / (
#                     np.linalg.norm(user_emb) * np.linalg.norm(post_emb) + 1e-8
#                 )
#                 features['user_author_similarity'] = float(similarity)
#             else:
#                 features['user_author_similarity'] = 0.0
#         else:
#             features['user_author_similarity'] = 0.0
        
#         # 45. post_in_user_language - Language match
#         # COMMENTED: Requires language detection
#         # features['post_in_user_language'] = 1.0
#         features['post_in_user_language'] = 1.0  # Assume match
        
#         # 46. time_since_last_seen_author - Time since last interaction
#         if 'CreateDate' in reactions_df.columns:
#             user_author_reactions = reactions_df[
#                 (reactions_df['UserId'] == user_id) & 
#                 (reactions_df['PostId'].isin(author_posts))
#             ]
#             if len(user_author_reactions) > 0:
#                 last_interaction = pd.to_datetime(user_author_reactions['CreateDate'].max())
#                 hours_since = (timestamp - last_interaction).total_seconds() / 3600.0
#                 features['time_since_last_seen_author'] = max(0, hours_since)
#             else:
#                 features['time_since_last_seen_author'] = 999.0  # Large value
#         else:
#             features['time_since_last_seen_author'] = 999.0
        
#         # 47. contextual_freshness_boost - Freshness boost
#         # Boost recent posts from followed authors
#         if features['user_follows_author'] > 0:
#             features['contextual_freshness_boost'] = 1.5
#         else:
#             features['contextual_freshness_boost'] = 1.0
        
#         return features
    
#     # ========================================================================
#     # HELPER METHODS
#     # ========================================================================
    
#     def _get_zero_features(self) -> Dict[str, float]:
#         """Return zero features (fallback)"""
#         feature_names = [
#             # User features (15)
#             'user_age_days', 'user_total_posts', 'user_total_followers',
#             'user_total_following', 'user_follower_following_ratio',
#             'user_avg_likes_per_post', 'user_engagement_rate_7d',
#             'user_posts_per_day_7d', 'user_likes_given_7d',
#             'user_comments_given_7d', 'user_session_count_7d',
#             'user_avg_session_duration', 'user_is_verified',
#             'user_has_profile_pic', 'user_activity_level',
            
#             # Post features (18)
#             'post_age_hours', 'post_total_likes', 'post_total_comments',
#             'post_total_shares', 'post_total_views', 'post_ctr_1h',
#             'post_ctr_6h', 'post_ctr_24h', 'post_engagement_velocity',
#             'post_has_image', 'post_has_video', 'post_has_link',
#             'post_text_length', 'post_hashtag_count', 'post_mention_count',
#             'post_topic', 'post_language', 'post_is_reply',
            
#             # Author features (7)
#             'author_total_followers', 'author_total_posts',
#             'author_avg_engagement_rate', 'author_posts_per_day',
#             'author_is_verified', 'author_account_age_days',
#             'author_follower_growth_7d',
            
#             # Interaction features (7)
#             'user_follows_author', 'user_author_past_interactions',
#             'user_topic_affinity', 'user_author_similarity',
#             'post_in_user_language', 'time_since_last_seen_author',
#             'contextual_freshness_boost'
#         ]
        
#         return {name: 0.0 for name in feature_names}
    
#     def get_feature_names(self) -> List[str]:
#         """Get list of all 47 feature names"""
#         return list(self._get_zero_features().keys())


# # ============================================================================
# # BATCH FEATURE EXTRACTION
# # ============================================================================

# def extract_features_batch(
#     interactions_df: pd.DataFrame,
#     feature_engineer: FeatureEngineer,
#     show_progress: bool = True
# ) -> pd.DataFrame:
#     """
#     Extract features for batch of interactions
    
#     Args:
#         interactions_df: DataFrame with user_id, post_id, (optional: created_at, weight)
#         feature_engineer: FeatureEngineer instance
#         show_progress: Show progress bar
    
#     Returns:
#         DataFrame with features + label
#     """
#     print(f"\n🔧 Extracting features for {len(interactions_df):,} samples...")
    
#     features_list = []
#     failed_count = 0
    
#     # Determine column names
#     if 'UserId' in interactions_df.columns:
#         user_col = 'UserId'
#         post_col = 'PostId'
#         time_col = 'CreateDate'
#     else:
#         user_col = 'user_id'
#         post_col = 'post_id'
#         time_col = 'created_at'
    
#     # Extract features
#     total = len(interactions_df)
    
#     for idx, row in interactions_df.iterrows():
#         if show_progress and idx % 1000 == 0:
#             print(f"   Progress: {idx:,}/{total:,} ({idx/total*100:.1f}%)", end='\r')
        
#         try:
#             timestamp = row.get(time_col, None)
#             if timestamp and isinstance(timestamp, str):
#                 timestamp = pd.to_datetime(timestamp)
            
#             features = feature_engineer.extract_features(
#                 row[user_col],
#                 row[post_col],
#                 timestamp
#             )
            
#             # Add metadata
#             features['user_id'] = row[user_col]
#             features['post_id'] = row[post_col]
            
#             # Preserve weight if exists
#             if 'weight' in row:
#                 features['weight'] = row['weight']
            
#             features_list.append(features)
            
#         except Exception as e:
#             failed_count += 1
#             continue
    
#     if show_progress:
#         print(f"   Progress: {total:,}/{total:,} (100.0%)")
    
#     if failed_count > 0:
#         print(f"   ⚠️  Failed to extract features for {failed_count:,} samples")
    
#     df = pd.DataFrame(features_list)
    
#     print(f"   ✅ Extracted features: {df.shape}")
    
#     return df



# import pandas as pd
# import numpy as np
# from datetime import datetime
# from typing import Dict, Optional
# import logging

# logger = logging.getLogger(__name__)


# class FeatureEngineer:
#     """
#     Extract features for ranking model with robust error handling
#     """
    
#     def __init__(
#         self,
#         data: Dict[str, pd.DataFrame],
#         user_stats: Dict,
#         author_stats: Dict,
#         following_dict: Dict,
#         embeddings: Optional[Dict] = None,
#         redis_client=None,     
#     ):
#         self.data = data
#         self.user_stats = user_stats
#         self.author_stats = author_stats
#         self.following_dict = following_dict
#         self.embeddings = embeddings if embeddings is not None else {}
#         self.redis = redis_client 
        
#         # Cache for faster lookup
#         self.user_df = data['user']
#         self.post_df = data['post']
        
#         # Create ID sets for fast validation
#         self.valid_user_ids = set(self.user_df['Id'].tolist())
#         self.valid_post_ids = set(self.post_df['Id'].tolist())
        
#     def extract_features(
#         self, 
#         user_id: int, 
#         post_id: int, 
#         timestamp: Optional[datetime] = None
#     ) -> Dict[str, float]:
#         """
#         Extract all features for (user, post) pair
        
#         Returns:
#             Dictionary of features
        
#         Raises:
#             ValueError: Only if both user and post are invalid (unrecoverable)
#         """
#         features = {}
        
#         # ================================================================
#         # VALIDATION: Check if user/post exist
#         # ================================================================
#         user_exists = user_id in self.valid_user_ids
#         post_exists = post_id in self.valid_post_ids
        
#         # If BOTH are invalid, this is unrecoverable
#         if not user_exists and not post_exists:
#             raise ValueError(f"Both User {user_id} and Post {post_id} not found")
        
#         # ================================================================
#         # USER FEATURES
#         # ================================================================
#         if user_exists:
#             user_row = self.user_df[self.user_df['Id'] == user_id]
#             user = user_row.iloc[0] if not user_row.empty else None
#             user_stat = self.user_stats.get(user_id, {})
#         else:
#             logger.warning(f"User {user_id} not found, using default features")
#             user = None
#             user_stat = {}
        
#         # Extract user features (with defaults)
#         features['user_n_reactions'] = user_stat.get('n_reactions', 0)
#         features['user_like_rate'] = user_stat.get('like_rate', 0.0)
#         features['user_comment_rate'] = user_stat.get('comment_rate', 0.0)
#         features['user_share_rate'] = user_stat.get('share_rate', 0.0)
#         features['user_avg_dwell_time'] = user_stat.get('avg_dwell_time', 0.0)
#         features['user_follower_count'] = user_stat.get('follower_count', 0)
#         features['user_following_count'] = user_stat.get('following_count', 0)
#         features['user_post_count'] = user_stat.get('post_count', 0)
        
#         # ================================================================
#         # POST FEATURES
#         # ================================================================
#         if post_exists:
#             post_row = self.post_df[self.post_df['Id'] == post_id]
#             post = post_row.iloc[0] if not post_row.empty else None
#         else:
#             logger.warning(f"Post {post_id} not found, using default features")
#             post = None
        
#         if post is not None:
#             # Get author
#             author_id = post.get('UserId', None) if isinstance(post, pd.Series) else post.get('UserId')
            
#             # Author features
#             if author_id and author_id in self.valid_user_ids:
#                 author_stat = self.author_stats.get(author_id, {})
#                 features['author_n_posts'] = author_stat.get('n_posts', 0)
#                 features['author_avg_likes'] = author_stat.get('avg_likes', 0.0)
#                 features['author_avg_comments'] = author_stat.get('avg_comments', 0.0)
#                 features['author_avg_shares'] = author_stat.get('avg_shares', 0.0)
#                 features['author_avg_engagement'] = author_stat.get('avg_engagement', 0.0)
#                 features['author_total_engagement'] = author_stat.get('total_engagement', 0.0)
#             else:
#                 # Author not found - use defaults
#                 features['author_n_posts'] = 0
#                 features['author_avg_likes'] = 0.0
#                 features['author_avg_comments'] = 0.0
#                 features['author_avg_shares'] = 0.0
#                 features['author_avg_engagement'] = 0.0
#                 features['author_total_engagement'] = 0.0
            
#             # Following relationship
#             features['is_following_author'] = 0
#             if author_id and user_id in self.following_dict:
#                 if author_id in self.following_dict.get(user_id, []):
#                     features['is_following_author'] = 1
            
#             # Post metadata features
#             features['post_status'] = int(post.get('Status', 0))
#             features['post_privacy'] = int(post.get('Privacy', 0))
#             features['post_is_repost'] = int(post.get('IsRepost', 0))
#             features['post_is_pin'] = int(post.get('IsPin', 0))
#             features['post_original_post_id'] = int(post.get('OriginalPostId', 0))
            
#             # Time since post creation
#             if timestamp is not None and 'CreateDate' in post:
#                 try:
#                     post_time = pd.to_datetime(post['CreateDate'])
#                     if pd.notna(timestamp) and pd.notna(post_time):
#                         features['hours_since_post'] = (timestamp - post_time).total_seconds() / 3600
#                     else:
#                         features['hours_since_post'] = 0.0
#                 except Exception as e:
#                     logger.warning(f"Error computing time delta: {e}")
#                     features['hours_since_post'] = 0.0
#             else:
#                 features['hours_since_post'] = 0.0
#         else:
#             # Post not found - use all defaults
#             features['author_n_posts'] = 0
#             features['author_avg_likes'] = 0.0
#             features['author_avg_comments'] = 0.0
#             features['author_avg_shares'] = 0.0
#             features['author_avg_engagement'] = 0.0
#             features['author_total_engagement'] = 0.0
#             features['is_following_author'] = 0
#             features['post_status'] = 0
#             features['post_privacy'] = 0
#             features['post_is_repost'] = 0
#             features['post_is_pin'] = 0
#             features['post_original_post_id'] = 0
#             features['hours_since_post'] = 0.0
        
#         # ================================================================
#         # EMBEDDINGS (if available)
#         # ================================================================
#         if self.embeddings and 'post' in self.embeddings:
#             if post_id in self.embeddings['post']:
#                 post_emb = self.embeddings['post'][post_id]
#                 for i, val in enumerate(post_emb):
#                     features[f'post_emb_{i}'] = float(val)
        
#         return features
    
#     def extract_features_batch(
#         self,
#         user_post_pairs: list,
#         timestamp: Optional[datetime] = None
#     ) -> pd.DataFrame:
#         """
#         Extract features for multiple (user, post) pairs
        
#         Args:
#             user_post_pairs: List of (user_id, post_id) tuples
#             timestamp: Optional timestamp for temporal features
            
#         Returns:
#             DataFrame with features for each pair
#         """
#         features_list = []
        
#         for user_id, post_id in user_post_pairs:
#             try:
#                 features = self.extract_features(user_id, post_id, timestamp)
#                 features['user_id'] = user_id
#                 features['post_id'] = post_id
#                 features_list.append(features)
#             except ValueError as e:
#                 # Skip unrecoverable errors
#                 logger.error(f"Skipping pair ({user_id}, {post_id}): {e}")
#                 continue
#             except Exception as e:
#                 # Log unexpected errors
#                 logger.error(f"Unexpected error for ({user_id}, {post_id}): {e}")
#                 continue
        
#         if not features_list:
#             # Return empty DataFrame with expected columns
#             return pd.DataFrame()
        
#         return pd.DataFrame(features_list)


# # ================================================================
# # DATA VALIDATION UTILITIES
# # ================================================================

# def validate_training_data(
#     interactions_df: pd.DataFrame,
#     data: Dict[str, pd.DataFrame]
# ) -> pd.DataFrame:
#     """
#     Validate and clean training data
    
#     Filters out:
#     - Invalid user IDs (< 0, not in users table)
#     - Invalid post IDs (< 0, not in posts table)
#     - Duplicate interactions
    
#     Returns:
#         Cleaned DataFrame
#     """
#     logger.info(f"Original interactions: {len(interactions_df)}")
    
#     # Get valid IDs
#     valid_user_ids = set(data['user']['Id'].tolist())
#     valid_post_ids = set(data['post']['Id'].tolist())
    
#     # Filter invalid user IDs
#     before = len(interactions_df)
#     interactions_df = interactions_df[
#         (interactions_df['UserId'] > 0) &
#         (interactions_df['UserId'].isin(valid_user_ids))
#     ]
#     after = len(interactions_df)
#     logger.info(f"Removed {before - after} interactions with invalid user IDs")
    
#     # Filter invalid post IDs
#     before = len(interactions_df)
#     interactions_df = interactions_df[
#         (interactions_df['PostId'] > 0) &
#         (interactions_df['PostId'].isin(valid_post_ids))
#     ]
#     after = len(interactions_df)
#     logger.info(f"Removed {before - after} interactions with invalid post IDs")
    
#     # Remove duplicates
#     before = len(interactions_df)
#     interactions_df = interactions_df.drop_duplicates(subset=['UserId', 'PostId', 'ReactionTypeId'])
#     after = len(interactions_df)
#     logger.info(f"Removed {before - after} duplicate interactions")
    
#     logger.info(f"Final valid interactions: {len(interactions_df)}")
    
#     return interactions_df


# # ================================================================
# # EXAMPLE USAGE
# # ================================================================

# if __name__ == "__main__":
#     """
#     Example: How to use the fixed FeatureEngineer
#     """
#     import yaml
#     from recommender.common.data_loader import load_data
    
#     # Setup logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
    
#     # Load config
#     with open('configs/config.yaml', 'r') as f:
#         config = yaml.safe_load(f)
    
#     # Load data
#     data = load_data(config.get('data_dir', 'dataset'))
    
#     # Example: Validate training data
#     interactions = data['postreaction']
#     cleaned_interactions = validate_training_data(interactions, data)
    
#     print(f"\n{'='*60}")
#     print(f"Data Validation Results:")
#     print(f"{'='*60}")
#     print(f"Original: {len(interactions)} interactions")
#     print(f"Cleaned:  {len(cleaned_interactions)} interactions")
#     print(f"Removed:  {len(interactions) - len(cleaned_interactions)} invalid")
#     print(f"{'='*60}\n")


# # recommender/common/feature_engineer.py
# """
# Feature Engineering (row-by-row + vector)
# =========================================

# - Dùng đầy đủ các bảng MySQL:
#   users(Id, CreateDate), posts(Id, UserId, CreateDate, IsRepost, IsPin),
#   friendships(Id, UserId, FriendId), post_hashtags(Id, PostId, HashtagId)
# - interactions chuẩn: user_id, post_id, action, created_at

# Label (ENGAGEMENT):
# - Positive = {like, love, laugh, wow, sad, angry, care, comment, share, save}
# - Negative = {view}
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from datetime import datetime, timezone
# from typing import Dict, Optional, Tuple, Any, List

# import numpy as np
# import pandas as pd

# # >>> include all emotion reactions <<<
# POSITIVE_ACTIONS = {"like", "love", "laugh", "wow", "sad", "angry", "care", "comment", "share", "save"}
# NEGATIVE_ACTIONS = {"view"}


# def _now_utc() -> datetime:
#     return datetime.now(timezone.utc)

# def _ensure_datetime_utc(series: pd.Series) -> pd.Series:
#     if not pd.api.types.is_datetime64_any_dtype(series):
#         series = pd.to_datetime(series, utc=True, errors="coerce")
#     else:
#         if series.dt.tz is None:
#             series = series.dt.tz_localize("UTC")
#         else:
#             series = series.dt.tz_convert("UTC")
#     return series

# def _cosine(a: np.ndarray, b: np.ndarray) -> float:
#     if a is None or b is None:
#         return 0.0
#     na, nb = np.linalg.norm(a), np.linalg.norm(b)
#     if na == 0.0 or nb == 0.0:
#         return 0.0
#     return float(np.dot(a, b) / (na * nb))


# # --------------------------- Row-by-row FE -----------------------------------

# @dataclass
# class FeatureEngineer:
#     data: Dict[str, pd.DataFrame] | None
#     user_stats: Dict | None
#     author_stats: Dict | None
#     following: Dict[int, set] | None
#     embeddings: Dict[str, Dict[int, np.ndarray]] | None

#     def __post_init__(self):
#         self.users = (self.data or {}).get("users")
#         self.posts = (self.data or {}).get("posts")
#         self.friendships = (self.data or {}).get("friendships")
#         self.post_hashtags = (self.data or {}).get("post_hashtags")

#         # post -> author
#         self.post_author: Dict[int, int] = {}
#         if isinstance(self.posts, pd.DataFrame) and {"Id", "UserId"}.issubset(self.posts.columns):
#             self.post_author = dict(zip(self.posts["Id"].astype(int), self.posts["UserId"].astype(int)))

#         # flags & created
#         self.post_is_repost = dict(zip(self.posts["Id"], self.posts.get("IsRepost", 0))) if isinstance(self.posts, pd.DataFrame) and "Id" in self.posts.columns else {}
#         self.post_is_pin = dict(zip(self.posts["Id"], self.posts.get("IsPin", 0))) if isinstance(self.posts, pd.DataFrame) and "Id" in self.posts.columns else {}
#         self.post_created = {}
#         if isinstance(self.posts, pd.DataFrame) and "CreateDate" in self.posts.columns:
#             tmp = self.posts[["Id", "CreateDate"]].copy()
#             tmp["CreateDate"] = _ensure_datetime_utc(tmp["CreateDate"])
#             self.post_created = dict(zip(tmp["Id"].astype(int), tmp["CreateDate"]))

#         # hashtags
#         self.post_hashtag_count: Dict[int, int] = {}
#         if isinstance(self.post_hashtags, pd.DataFrame) and "PostId" in self.post_hashtags.columns:
#             cnt = self.post_hashtags.groupby("PostId").size().rename("cnt")
#             self.post_hashtag_count = cnt.to_dict()

#         # user created
#         self.user_created: Dict[int, pd.Timestamp] = {}
#         if isinstance(self.users, pd.DataFrame) and {"Id", "CreateDate"}.issubset(self.users.columns):
#             tmp = self.users[["Id", "CreateDate"]].copy()
#             tmp["CreateDate"] = _ensure_datetime_utc(tmp["CreateDate"])
#             self.user_created = dict(zip(tmp["Id"].astype(int), tmp["CreateDate"]))

#         # author follower count
#         self.author_follower_count: Dict[int, int] = {}
#         if isinstance(self.friendships, pd.DataFrame) and {"UserId", "FriendId"}.issubset(self.friendships.columns):
#             cnt = self.friendships.groupby("FriendId").size()
#             self.author_follower_count = cnt.to_dict()

#         # embeddings map
#         self.post_vecs = (self.embeddings or {}).get("post")
#         self.user_vecs = (self.embeddings or {}).get("user")

#     def extract_features(self, user_id: int, post_id: int) -> Dict[str, float]:
#         feats: Dict[str, float] = {}

#         # user stats (precomputed dict)
#         ust = (self.user_stats or {}).get(int(user_id), {})
#         feats["user_total_interactions"] = float(ust.get("total_interactions", 0.0))
#         feats["user_positive_rate"] = float(ust.get("positive_rate", 0.0))

#         # account age
#         acc_age = 0.0
#         if int(user_id) in self.user_created:
#             acc_age = max((_now_utc() - self.user_created[int(user_id)]).total_seconds() / 86400.0, 0.0)
#         feats["user_account_age_days"] = acc_age

#         feats["post_total_interactions"] = 0.0
#         feats["post_positive_rate"] = 0.0

#         feats["post_is_repost"] = float(self.post_is_repost.get(int(post_id), 0))
#         feats["post_is_pin"] = float(self.post_is_pin.get(int(post_id), 0))
#         feats["post_hashtag_count"] = float(self.post_hashtag_count.get(int(post_id), 0))

#         age_h = 0.0
#         if int(post_id) in self.post_created:
#             age_h = max((_now_utc() - self.post_created[int(post_id)]).total_seconds() / 3600.0, 0.0)
#         feats["post_age_hours"] = age_h

#         author_id = self.post_author.get(int(post_id), -1)
#         feats["author_id_known"] = 1.0 if author_id != -1 else 0.0

#         ast = (self.author_stats or {}).get(int(author_id), {}) if author_id != -1 else {}
#         feats["author_total_interactions"] = float(ast.get("total_interactions", 0.0))
#         feats["author_positive_rate"] = float(ast.get("positive_rate", 0.0))
#         feats["author_follower_count"] = float(self.author_follower_count.get(int(author_id), 0))

#         foll = (self.following or {}).get(int(user_id), set())
#         feats["is_following_author"] = 1.0 if author_id in foll else 0.0

#         # cosine(sim) nếu có embeddings
#         sim = 0.0
#         if self.post_vecs is not None:
#             pv = self.post_vecs.get(int(post_id))
#             uv = self.user_vecs.get(int(user_id)) if self.user_vecs else None
#             if pv is not None and uv is not None:
#                 na, nb = np.linalg.norm(pv), np.linalg.norm(uv)
#                 sim = float(np.dot(pv, uv) / (na * nb)) if (na > 0 and nb > 0) else 0.0
#         feats["user_post_cosine"] = sim

#         return feats


# # ---------------------- Vector pipeline (batch FE) ---------------------------

# def build_training_matrices(
#     interactions: pd.DataFrame,
#     users_df: Optional[pd.DataFrame] = None,
#     posts_df: Optional[pd.DataFrame] = None,
#     friendships_df: Optional[pd.DataFrame] = None,
#     post_hashtags_df: Optional[pd.DataFrame] = None,
#     embeddings: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
#     now_ts: Optional[datetime] = None,
# ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
#     if interactions is None or interactions.empty:
#         raise ValueError("interactions is empty")

#     df = interactions.copy()
#     if "created_at" in df.columns:
#         df["created_at"] = _ensure_datetime_utc(df["created_at"])

#     # Label theo POSITIVE_ACTIONS
#     if "label" not in df.columns:
#         df["label"] = df["action"].isin(POSITIVE_ACTIONS).astype(int)

#     # --- user stats ---
#     u_cnt = df.groupby("user_id").size().rename("user_total_interactions")
#     u_pos = df.groupby("user_id")["label"].mean().rename("user_positive_rate")
#     u = pd.concat([u_cnt, u_pos], axis=1).reset_index()
#     df = df.merge(u, on="user_id", how="left")

#     # user account age
#     df["user_account_age_days"] = 0.0
#     if isinstance(users_df, pd.DataFrame) and {"Id", "CreateDate"}.issubset(users_df.columns):
#         uu = users_df[["Id", "CreateDate"]].rename(columns={"Id": "user_id", "CreateDate": "user_created_at"}).copy()
#         uu["user_created_at"] = _ensure_datetime_utc(uu["user_created_at"])
#         base = now_ts or _now_utc()
#         df = df.merge(uu, on="user_id", how="left")
#         age_days = (base - df["user_created_at"]).dt.total_seconds() / 86400.0
#         df["user_account_age_days"] = age_days.fillna(0.0).clip(lower=0.0)
#         df.drop(columns=["user_created_at"], inplace=True)

#     # --- post stats ---
#     p_cnt = df.groupby("post_id").size().rename("post_total_interactions")
#     p_pos = df.groupby("post_id")["label"].mean().rename("post_positive_rate")
#     p = pd.concat([p_cnt, p_pos], axis=1).reset_index()
#     df = df.merge(p, on="post_id", how="left")

#     # posts meta + age + author_id
#     df["post_age_hours"] = 0.0
#     df["post_is_repost"] = 0.0
#     df["post_is_pin"] = 0.0
#     df["author_id_known"] = 0.0
#     if isinstance(posts_df, pd.DataFrame) and {"Id", "UserId", "CreateDate"}.issubset(posts_df.columns):
#         pp = posts_df[["Id", "UserId", "CreateDate", "IsRepost", "IsPin"]].rename(
#             columns={"Id": "post_id", "UserId": "author_id", "CreateDate": "post_created_at"}
#         ).copy()
#         pp["post_created_at"] = _ensure_datetime_utc(pp["post_created_at"])
#         base = now_ts or _now_utc()
#         df = df.merge(pp, on="post_id", how="left")
#         df["author_id_known"] = df["author_id"].notnull().astype(float)
#         age_h = (base - df["post_created_at"]).dt.total_seconds() / 3600.0
#         df["post_age_hours"] = age_h.fillna(0.0).clip(lower=0.0)
#         df["post_is_repost"] = df["IsRepost"].fillna(0).astype(float)
#         df["post_is_pin"] = df["IsPin"].fillna(0).astype(float)
#         df.drop(columns=["post_created_at", "IsRepost", "IsPin"], inplace=True)

#     # hashtag count
#     if isinstance(post_hashtags_df, pd.DataFrame) and "PostId" in post_hashtags_df.columns:
#         # đếm số hashtag theo PostId → ph_cnt
#         hcnt = (
#             post_hashtags_df.groupby("PostId")
#             .size()
#             .rename("ph_cnt")
#             .reset_index()
#             .rename(columns={"PostId": "post_id"})
#         )
#         df = df.merge(hcnt, on="post_id", how="left")
#         df["post_hashtag_count"] = df["ph_cnt"].fillna(0.0)
#         df.drop(columns=["ph_cnt"], inplace=True)
#     else:
#         df["post_hashtag_count"] = 0.0


#     # # author stats + is_following + follower_count
#     # df["author_total_interactions"] = 0.0
#     # df["author_positive_rate"] = 0.0
#     # df["author_follower_count"] = 0.0
#     # df["is_following_author"] = 0.0

#     # if "author_id" in df.columns:
#     #     a_stats = (
#     #         df[["author_id", "label"]].dropna().groupby("author_id")["label"]
#     #         .agg(["count", "mean"]).rename(columns={"count": "author_total_interactions", "mean": "author_positive_rate"})
#     #         .reset_index()
#     #     )
#     #     df = df.merge(a_stats, on="author_id", how="left")

#     #     if isinstance(friendships_df, pd.DataFrame) and {"UserId", "FriendId"}.issubset(friendships_df.columns):
#     #         fcnt = friendships_df.groupby("FriendId").size().rename("author_follower_count").reset_index().rename(columns={"FriendId": "author_id"})
#     #         df = df.merge(fcnt, on="author_id", how="left")
#     #         tmp = friendships_df.rename(columns={"UserId": "user_id", "FriendId": "author_id"})[["user_id", "author_id"]].copy()
#     #         tmp["is_following_author"] = 1.0
#     #         df = df.merge(tmp, on=["user_id", "author_id"], how="left")
#     #         df["is_following_author"] = df["is_following_author"].fillna(0.0)
#     # author stats + is_following + follower_count
#     # luôn tạo các cột mặc định để tránh KeyError
#     df["author_total_interactions"] = 0.0
#     df["author_positive_rate"] = 0.0
#     df["author_follower_count"] = 0.0
#     df["is_following_author"] = 0.0

#     if "author_id" in df.columns:
#         # Tính stats theo author_id nếu đã có
#         a_stats = (
#             df[["author_id", "label"]]
#             .dropna()
#             .groupby("author_id")["label"]
#             .agg(["count", "mean"])
#             .rename(columns={"count": "author_total_interactions", "mean": "author_positive_rate"})
#             .reset_index()
#         )
#         df = df.merge(a_stats, on="author_id", how="left", suffixes=("", "_a"))
#         # nếu merge tạo cột _a thì chọn đúng cột và dọn cột thừa
#         for col in ["author_total_interactions", "author_positive_rate"]:
#             if f"{col}_a" in df.columns:
#                 df[col] = df[f"{col}_a"].fillna(df[col])
#                 df.drop(columns=[f"{col}_a"], inplace=True)

#         # follower_count + following flag nếu có bảng Friendship
#         if isinstance(friendships_df, pd.DataFrame) and {"UserId", "FriendId"}.issubset(friendships_df.columns):
#             # follower count theo FriendId = author
#             fcnt = (
#                 friendships_df.groupby("FriendId")
#                 .size()
#                 .rename("author_follower_count")
#                 .reset_index()
#                 .rename(columns={"FriendId": "author_id"})
#             )
#             df = df.merge(fcnt, on="author_id", how="left", suffixes=("", "_fc"))
#             if "author_follower_count_fc" in df.columns:
#                 df["author_follower_count"] = df["author_follower_count_fc"].fillna(df["author_follower_count"])
#                 df.drop(columns=["author_follower_count_fc"], inplace=True)

#             # is_following_author: user_id -> author_id có trong Friendship (UserId, FriendId)
#             tmp = friendships_df.rename(columns={"UserId": "user_id", "FriendId": "author_id"})[["user_id", "author_id"]].copy()
#             tmp["is_following_author"] = 1.0
#             df = df.merge(tmp, on=["user_id", "author_id"], how="left", suffixes=("", "_if"))
#             # nếu sinh cột _if thì chọn đúng cột
#             if "is_following_author_if" in df.columns:
#                 df["is_following_author"] = df["is_following_author_if"].fillna(df["is_following_author"])
#                 df.drop(columns=["is_following_author_if"], inplace=True)

#     # đảm bảo không NaN
#     df["author_total_interactions"] = df["author_total_interactions"].fillna(0.0)
#     df["author_positive_rate"] = df["author_positive_rate"].fillna(0.0)
#     df["author_follower_count"] = df["author_follower_count"].fillna(0.0)
#     df["is_following_author"] = df["is_following_author"].fillna(0.0)


#     # embeddings similarity (optional)
#     df["user_post_cosine"] = 0.0
#     if embeddings and isinstance(embeddings.get("post"), dict):
#         post_vecs = embeddings["post"]
#         user_vecs = embeddings.get("user")
#         def _to_vec(ids: pd.Series, vec_dict: Dict[int, np.ndarray]):
#             return [vec_dict.get(int(x)) if pd.notnull(x) else None for x in ids]
#         pvec = _to_vec(df["post_id"], post_vecs)
#         uvec = _to_vec(df["user_id"], user_vecs) if user_vecs else [None] * len(df)
#         sims = []
#         for pv, uv in zip(pvec, uvec):
#             if pv is None or uv is None:
#                 sims.append(0.0); continue
#             na, nb = np.linalg.norm(pv), np.linalg.norm(uv)
#             sims.append(float(np.dot(pv, uv) / (na * nb)) if (na > 0 and nb > 0) else 0.0)
#         df["user_post_cosine"] = sims

#     # final feature set
#     feature_cols = [
#         "user_total_interactions", "user_positive_rate", "user_account_age_days",
#         "post_total_interactions", "post_positive_rate", "post_age_hours",
#         "post_is_repost", "post_is_pin", "post_hashtag_count",
#         "author_id_known", "author_total_interactions", "author_positive_rate",
#         "author_follower_count", "is_following_author",
#         "user_post_cosine",
#     ]
#     for c in feature_cols:
#         if c not in df.columns:
#             df[c] = 0.0

#     X = df[feature_cols].fillna(0.0)
#     y = df["label"].astype(int)

#     # PATCH: trả cả 'action' để orchestrator tính multiplier cho time-decay
#     interactions_df = df[["user_id", "post_id", "action", "created_at"]].copy()

#     meta = {
#         "n_rows": int(len(df)),
#         "n_features": int(len(feature_cols)),
#         "features": feature_cols,
#         "positive_rate": float(y.mean()) if len(y) else 0.0,
#     }
#     return X, y, interactions_df, meta


# recommender/common/feature_engineer.py
"""
Feature Engineering (row-by-row + vector)
=========================================

- Bảng: users(Id, CreateDate), posts(Id, UserId, CreateDate, IsRepost, IsPin),
        friendships(UserId, FriendId), post_hashtags(PostId, HashtagId)
- interactions chuẩn: user_id, post_id, action, created_at
- Label: Positive = {like, love, laugh, wow, sad, angry, care, comment, share, save}
         Negative = {view}
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd

POSITIVE_ACTIONS = {"like", "love", "laugh", "wow", "sad", "angry", "care", "comment", "share", "save"}
NEGATIVE_ACTIONS = {"view"}

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

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

@dataclass
class FeatureEngineer:
    data: Dict[str, pd.DataFrame] | None
    user_stats: Dict | None
    author_stats: Dict | None
    following: Dict[int, set] | None
    embeddings: Dict[str, Dict[int, np.ndarray]] | None

    def __post_init__(self):
        self.users = (self.data or {}).get("users")
        self.posts = (self.data or {}).get("posts")
        self.friendships = (self.data or {}).get("friendships")
        self.post_hashtags = (self.data or {}).get("post_hashtags")

        self.post_author: Dict[int, int] = {}
        if isinstance(self.posts, pd.DataFrame) and {"Id", "UserId"}.issubset(self.posts.columns):
            self.post_author = dict(zip(self.posts["Id"].astype(int), self.posts["UserId"].astype(int)))

        self.post_is_repost = dict(zip(self.posts["Id"], self.posts.get("IsRepost", 0))) if isinstance(self.posts, pd.DataFrame) and "Id" in self.posts.columns else {}
        self.post_is_pin = dict(zip(self.posts["Id"], self.posts.get("IsPin", 0))) if isinstance(self.posts, pd.DataFrame) and "Id" in self.posts.columns else {}
        self.post_created = {}
        if isinstance(self.posts, pd.DataFrame) and "CreateDate" in self.posts.columns:
            tmp = self.posts[["Id", "CreateDate"]].copy()
            tmp["CreateDate"] = _ensure_datetime_utc(tmp["CreateDate"])
            self.post_created = dict(zip(tmp["Id"].astype(int), tmp["CreateDate"]))

        self.post_hashtag_count: Dict[int, int] = {}
        if isinstance(self.post_hashtags, pd.DataFrame) and "PostId" in self.post_hashtags.columns:
            cnt = self.post_hashtags.groupby("PostId").size().rename("cnt")
            self.post_hashtag_count = cnt.to_dict()

        self.user_created: Dict[int, pd.Timestamp] = {}
        if isinstance(self.users, pd.DataFrame) and {"Id", "CreateDate"}.issubset(self.users.columns):
            tmp = self.users[["Id", "CreateDate"]].copy()
            tmp["CreateDate"] = _ensure_datetime_utc(tmp["CreateDate"])
            self.user_created = dict(zip(tmp["Id"].astype(int), tmp["CreateDate"]))

        self.author_follower_count: Dict[int, int] = {}
        if isinstance(self.friendships, pd.DataFrame) and {"UserId", "FriendId"}.issubset(self.friendships.columns):
            cnt = self.friendships.groupby("FriendId").size()
            self.author_follower_count = cnt.to_dict()

        self.post_vecs = (self.embeddings or {}).get("post")
        self.user_vecs = (self.embeddings or {}).get("user")

    def extract_features(self, user_id: int, post_id: int) -> Dict[str, float]:
        feats: Dict[str, float] = {}

        # user stats
        ust = (self.user_stats or {}).get(int(user_id), {})
        feats["user_total_interactions"] = float(ust.get("total_interactions", 0.0))
        feats["user_positive_rate"] = float(ust.get("positive_rate", 0.0))

        acc_age = 0.0
        if int(user_id) in self.user_created:
            acc_age = max((_now_utc() - self.user_created[int(user_id)]).total_seconds() / 86400.0, 0.0)
        feats["user_account_age_days"] = acc_age

        feats["post_total_interactions"] = 0.0
        feats["post_positive_rate"] = 0.0

        feats["post_is_repost"] = float(self.post_is_repost.get(int(post_id), 0))
        feats["post_is_pin"] = float(self.post_is_pin.get(int(post_id), 0))
        feats["post_hashtag_count"] = float(self.post_hashtag_count.get(int(post_id), 0))

        age_h = 0.0
        if int(post_id) in self.post_created:
            age_h = max((_now_utc() - self.post_created[int(post_id)]).total_seconds() / 3600.0, 0.0)
        feats["post_age_hours"] = age_h

        author_id = self.post_author.get(int(post_id), -1)
        feats["author_id_known"] = 1.0 if author_id != -1 else 0.0

        ast = (self.author_stats or {}).get(int(author_id), {}) if author_id != -1 else {}
        feats["author_total_interactions"] = float(ast.get("total_interactions", 0.0))
        feats["author_positive_rate"] = float(ast.get("positive_rate", 0.0))
        feats["author_follower_count"] = float(self.author_follower_count.get(int(author_id), 0))

        foll = (self.following or {}).get(int(user_id), set())
        feats["is_following_author"] = 1.0 if author_id in foll else 0.0

        sim = 0.0
        if self.post_vecs is not None:
            pv = self.post_vecs.get(int(post_id))
            uv = self.user_vecs.get(int(user_id)) if self.user_vecs else None
            if pv is not None and uv is not None:
                na, nb = np.linalg.norm(pv), np.linalg.norm(uv)
                sim = float(np.dot(pv, uv) / (na * nb)) if (na > 0 and nb > 0) else 0.0
        feats["user_post_cosine"] = sim

        return feats

# ---------------------- Vector pipeline (batch FE) ---------------------------

def build_training_matrices(
    interactions: pd.DataFrame,
    users_df: Optional[pd.DataFrame] = None,
    posts_df: Optional[pd.DataFrame] = None,
    friendships_df: Optional[pd.DataFrame] = None,
    post_hashtags_df: Optional[pd.DataFrame] = None,
    embeddings: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    now_ts: Optional[datetime] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    if interactions is None or interactions.empty:
        raise ValueError("interactions is empty")

    df = interactions.copy()
    if "created_at" in df.columns:
        df["created_at"] = _ensure_datetime_utc(df["created_at"])

    if "label" not in df.columns:
        df["label"] = df["action"].isin(POSITIVE_ACTIONS).astype(int)

    # user stats
    u_cnt = df.groupby("user_id").size().rename("user_total_interactions")
    u_pos = df.groupby("user_id")["label"].mean().rename("user_positive_rate")
    u = pd.concat([u_cnt, u_pos], axis=1).reset_index()
    df = df.merge(u, on="user_id", how="left")

    df["user_account_age_days"] = 0.0
    if isinstance(users_df, pd.DataFrame) and {"Id", "CreateDate"}.issubset(users_df.columns):
        uu = users_df[["Id", "CreateDate"]].rename(columns={"Id": "user_id", "CreateDate": "user_created_at"}).copy()
        uu["user_created_at"] = _ensure_datetime_utc(uu["user_created_at"])
        base = now_ts or _now_utc()
        df = df.merge(uu, on="user_id", how="left")
        age_days = (base - df["user_created_at"]).dt.total_seconds() / 86400.0
        df["user_account_age_days"] = age_days.fillna(0.0).clip(lower=0.0)
        df.drop(columns=["user_created_at"], inplace=True)

    # post stats
    p_cnt = df.groupby("post_id").size().rename("post_total_interactions")
    p_pos = df.groupby("post_id")["label"].mean().rename("post_positive_rate")
    p = pd.concat([p_cnt, p_pos], axis=1).reset_index()
    df = df.merge(p, on="post_id", how="left")

    # posts meta
    df["post_age_hours"] = 0.0
    df["post_is_repost"] = 0.0
    df["post_is_pin"] = 0.0
    df["author_id_known"] = 0.0
    if isinstance(posts_df, pd.DataFrame) and {"Id", "UserId", "CreateDate"}.issubset(posts_df.columns):
        pp = posts_df[["Id", "UserId", "CreateDate", "IsRepost", "IsPin"]].rename(
            columns={"Id": "post_id", "UserId": "author_id", "CreateDate": "post_created_at"}
        ).copy()
        pp["post_created_at"] = _ensure_datetime_utc(pp["post_created_at"])
        base = now_ts or _now_utc()
        df = df.merge(pp, on="post_id", how="left")
        df["author_id_known"] = df["author_id"].notnull().astype(float)
        age_h = (base - df["post_created_at"]).dt.total_seconds() / 3600.0
        df["post_age_hours"] = age_h.fillna(0.0).clip(lower=0.0)
        df["post_is_repost"] = df["IsRepost"].fillna(0).astype(float)
        df["post_is_pin"] = df["IsPin"].fillna(0).astype(float)
        df.drop(columns=["post_created_at", "IsRepost", "IsPin"], inplace=True)

    # hashtag count
    if isinstance(post_hashtags_df, pd.DataFrame) and "PostId" in post_hashtags_df.columns:
        hcnt = (
            post_hashtags_df.groupby("PostId")
            .size()
            .rename("ph_cnt")
            .reset_index()
            .rename(columns={"PostId": "post_id"})
        )
        df = df.merge(hcnt, on="post_id", how="left")
        df["post_hashtag_count"] = df["ph_cnt"].fillna(0.0)
        df.drop(columns=["ph_cnt"], inplace=True)
    else:
        df["post_hashtag_count"] = 0.0

    # author stats + follower_count + is_following
    df["author_total_interactions"] = 0.0
    df["author_positive_rate"] = 0.0
    df["author_follower_count"] = 0.0
    df["is_following_author"] = 0.0

    if "author_id" in df.columns:
        a_stats = (
            df[["author_id", "label"]]
            .dropna()
            .groupby("author_id")["label"]
            .agg(["count", "mean"])
            .rename(columns={"count": "author_total_interactions", "mean": "author_positive_rate"})
            .reset_index()
        )
        df = df.merge(a_stats, on="author_id", how="left", suffixes=("", "_a"))
        for col in ["author_total_interactions", "author_positive_rate"]:
            if f"{col}_a" in df.columns:
                df[col] = df[f"{col}_a"].fillna(df[col])
                df.drop(columns=[f"{col}_a"], inplace=True)

        if isinstance(friendships_df, pd.DataFrame) and {"UserId", "FriendId"}.issubset(friendships_df.columns):
            fcnt = (
                friendships_df.groupby("FriendId")
                .size()
                .rename("author_follower_count")
                .reset_index()
                .rename(columns={"FriendId": "author_id"})
            )
            df = df.merge(fcnt, on="author_id", how="left", suffixes=("", "_fc"))
            if "author_follower_count_fc" in df.columns:
                df["author_follower_count"] = df["author_follower_count_fc"].fillna(df["author_follower_count"])
                df.drop(columns=["author_follower_count_fc"], inplace=True)

            tmp = friendships_df.rename(columns={"UserId": "user_id", "FriendId": "author_id"})[["user_id", "author_id"]].copy()
            tmp["is_following_author"] = 1.0
            df = df.merge(tmp, on=["user_id", "author_id"], how="left", suffixes=("", "_if"))
            if "is_following_author_if" in df.columns:
                df["is_following_author"] = df["is_following_author_if"].fillna(df["is_following_author"])
                df.drop(columns=["is_following_author_if"], inplace=True)

    df["author_total_interactions"] = df["author_total_interactions"].fillna(0.0)
    df["author_positive_rate"] = df["author_positive_rate"].fillna(0.0)
    df["author_follower_count"] = df["author_follower_count"].fillna(0.0)
    df["is_following_author"] = df["is_following_author"].fillna(0.0)

    # embeddings similarity (optional)
    df["user_post_cosine"] = 0.0
    if embeddings and isinstance(embeddings.get("post"), dict):
        post_vecs = embeddings["post"]
        user_vecs = embeddings.get("user")
        def _to_vec(ids: pd.Series, vec_dict: Dict[int, np.ndarray]):
            return [vec_dict.get(int(x)) if pd.notnull(x) else None for x in ids]
        pvec = _to_vec(df["post_id"], post_vecs)
        uvec = _to_vec(df["user_id"], user_vecs) if user_vecs else [None] * len(df)
        sims = []
        for pv, uv in zip(pvec, uvec):
            if pv is None or uv is None:
                sims.append(0.0); continue
            na, nb = np.linalg.norm(pv), np.linalg.norm(uv)
            sims.append(float(np.dot(pv, uv) / (na * nb)) if (na > 0 and nb > 0) else 0.0)
        df["user_post_cosine"] = sims

    feature_cols = [
        "user_total_interactions", "user_positive_rate", "user_account_age_days",
        "post_total_interactions", "post_positive_rate", "post_age_hours",
        "post_is_repost", "post_is_pin", "post_hashtag_count",
        "author_id_known", "author_total_interactions", "author_positive_rate",
        "author_follower_count", "is_following_author",
        "user_post_cosine",
    ]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feature_cols].fillna(0.0)
    y = df["label"].astype(int)

    interactions_df = df[["user_id", "post_id", "action", "created_at"]].copy()

    meta = {
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "features": feature_cols,
        "positive_rate": float(y.mean()) if len(y) else 0.0,
    }
    return X, y, interactions_df, meta
