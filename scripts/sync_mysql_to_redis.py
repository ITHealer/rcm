"""
Data Sync Script for Team App

This script syncs data from MySQL database to Redis cache.
Team App should run this periodically (e.g., every minute) to keep Redis updated.

Usage:
    python sync_mysql_to_redis.py --full    # Full sync (all data)
    python sync_mysql_to_redis.py           # Incremental sync (only changed data)
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Set, Any
import time

from recommender.online.app.services.cache_service import cache_service
from recommender.online.app.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MySQLToRedisSync:
    """
    Sync data from MySQL to Redis
    
    Architecture:
    - Team App owns MySQL database
    - This script reads from MySQL and writes to Redis
    - AI team only reads from Redis
    """
    
    def __init__(self):
        """Initialize sync service"""
        self.last_sync_time = None
        
    def connect(self):
        """Connect to MySQL and Redis"""
        # Connect to Redis
        logger.info("Connecting to Redis...")
        cache_service.connect()
        
        if not cache_service.is_connected():
            raise Exception("Failed to connect to Redis")
        
        logger.info("✅ Connected to Redis")
        
        # TODO: Connect to MySQL
        # import mysql.connector
        # self.mysql_conn = mysql.connector.connect(
        #     host=settings.MYSQL_HOST,
        #     port=settings.MYSQL_PORT,
        #     database=settings.MYSQL_DATABASE,
        #     user=settings.MYSQL_USER,
        #     password=settings.MYSQL_PASSWORD
        # )
        # logger.info("✅ Connected to MySQL")
    
    def close(self):
        """Close connections"""
        cache_service.close()
        # TODO: Close MySQL connection
        # self.mysql_conn.close()
    
    # ==================== USER PROFILES ====================
    
    def sync_user_profiles(self, user_ids: List[int] = None, full_sync: bool = False):
        """
        Sync user profiles from MySQL to Redis
        
        Args:
            user_ids: List of specific user IDs to sync (None = all users)
            full_sync: If True, sync all users regardless of last_sync_time
        """
        logger.info("=" * 60)
        logger.info("Syncing user profiles...")
        logger.info("=" * 60)
        
        try:
            # TODO: Query MySQL for user profiles
            # Example query:
            """
            SELECT 
                id, username, email, age, gender, 
                interests, activity_level, created_at, updated_at
            FROM users
            WHERE 
                (updated_at > %s OR %s = TRUE)
                AND (id IN %s OR %s IS NULL)
            ORDER BY id
            LIMIT %s
            """
            
            # For now, using mock data
            logger.info("TODO: Implement actual MySQL query")
            logger.info("Using mock data for demonstration...")
            
            # Mock data example
            mock_users = [
                {
                    "id": 1,
                    "username": "user1",
                    "age": 25,
                    "gender": "M",
                    "interests": ["technology", "sports"],
                    "activity_level": 0.8,
                    "created_at": "2024-01-01T00:00:00"
                },
                {
                    "id": 2,
                    "username": "user2",
                    "age": 30,
                    "gender": "F",
                    "interests": ["fashion", "travel"],
                    "activity_level": 0.6,
                    "created_at": "2024-01-02T00:00:00"
                }
            ]
            
            # Sync to Redis
            synced_count = 0
            for user in mock_users:
                user_id = user["id"]
                
                # Cache user profile
                success = cache_service.set_user_profile(
                    user_id=user_id,
                    profile=user,
                    ttl=settings.USER_PROFILE_TTL
                )
                
                if success:
                    synced_count += 1
                    if synced_count % 100 == 0:
                        logger.info(f"Synced {synced_count} user profiles...")
            
            logger.info(f"✅ Synced {synced_count} user profiles to Redis")
            
        except Exception as e:
            logger.error(f"❌ Error syncing user profiles: {e}", exc_info=True)
            raise
    
    # ==================== POST FEATURES ====================
    
    def sync_post_features(self, post_ids: List[int] = None, full_sync: bool = False):
        """
        Sync post features from MySQL to Redis
        
        Args:
            post_ids: List of specific post IDs to sync (None = all posts)
            full_sync: If True, sync all posts regardless of last_sync_time
        """
        logger.info("=" * 60)
        logger.info("Syncing post features...")
        logger.info("=" * 60)
        
        try:
            # TODO: Query MySQL for post features
            # Example query:
            """
            SELECT 
                p.id, p.author_id, p.content, p.category,
                p.likes_count, p.comments_count, p.shares_count,
                p.created_at, p.updated_at,
                u.followers_count as author_followers
            FROM posts p
            JOIN users u ON p.author_id = u.id
            WHERE 
                (p.updated_at > %s OR %s = TRUE)
                AND (p.id IN %s OR %s IS NULL)
            ORDER BY p.id
            LIMIT %s
            """
            
            logger.info("TODO: Implement actual MySQL query")
            logger.info("Using mock data for demonstration...")
            
            # Mock data
            mock_posts = [
                {
                    "id": 101,
                    "author_id": 1,
                    "category": "technology",
                    "likes_count": 50,
                    "comments_count": 10,
                    "shares_count": 5,
                    "author_followers": 1000,
                    "post_age_hours": 2.5,
                    "created_at": "2025-01-27T10:00:00"
                },
                {
                    "id": 102,
                    "author_id": 2,
                    "category": "fashion",
                    "likes_count": 100,
                    "comments_count": 20,
                    "shares_count": 10,
                    "author_followers": 5000,
                    "post_age_hours": 1.0,
                    "created_at": "2025-01-27T12:00:00"
                }
            ]
            
            # Batch sync to Redis
            posts_dict = {post["id"]: post for post in mock_posts}
            success = cache_service.set_posts_features_batch(
                posts_features=posts_dict,
                ttl=settings.POST_FEATURES_TTL
            )
            
            if success:
                logger.info(f"✅ Synced {len(mock_posts)} post features to Redis")
            else:
                logger.error("❌ Failed to sync post features")
            
        except Exception as e:
            logger.error(f"❌ Error syncing post features: {e}", exc_info=True)
            raise
    
    # ==================== FOLLOWING RELATIONSHIPS ====================
    
    def sync_following_relationships(self, user_ids: List[int] = None):
        """
        Sync following relationships from MySQL to Redis
        
        Args:
            user_ids: List of specific user IDs to sync (None = all users)
        """
        logger.info("=" * 60)
        logger.info("Syncing following relationships...")
        logger.info("=" * 60)
        
        try:
            # TODO: Query MySQL for following relationships
            # Example query:
            """
            SELECT 
                follower_id as user_id,
                GROUP_CONCAT(following_id) as following_ids
            FROM user_follows
            WHERE 
                follower_id IN %s OR %s IS NULL
            GROUP BY follower_id
            """
            
            logger.info("TODO: Implement actual MySQL query")
            logger.info("Using mock data for demonstration...")
            
            # Mock data
            mock_following = {
                1: {2, 3, 4, 5},  # User 1 follows users 2, 3, 4, 5
                2: {1, 3, 6},     # User 2 follows users 1, 3, 6
            }
            
            synced_count = 0
            for user_id, following_set in mock_following.items():
                success = cache_service.set_following_list(
                    user_id=user_id,
                    author_ids=following_set,
                    ttl=settings.FOLLOWING_CACHE_TTL
                )
                
                if success:
                    synced_count += 1
            
            logger.info(f"✅ Synced following relationships for {synced_count} users")
            
        except Exception as e:
            logger.error(f"❌ Error syncing following relationships: {e}", exc_info=True)
            raise
    
    # ==================== FOLLOWING POSTS ====================
    
    def sync_following_posts(self, user_ids: List[int] = None):
        """
        Sync recent posts from followed authors for each user
        
        Args:
            user_ids: List of specific user IDs to sync (None = all users)
        """
        logger.info("=" * 60)
        logger.info("Syncing following posts...")
        logger.info("=" * 60)
        
        try:
            # TODO: Query MySQL for recent posts from followed authors
            # Example query:
            """
            SELECT 
                uf.follower_id as user_id,
                p.id as post_id,
                UNIX_TIMESTAMP(p.created_at) as timestamp
            FROM user_follows uf
            JOIN posts p ON uf.following_id = p.author_id
            WHERE 
                p.created_at > DATE_SUB(NOW(), INTERVAL 48 HOUR)
                AND (uf.follower_id IN %s OR %s IS NULL)
            ORDER BY p.created_at DESC
            """
            
            logger.info("TODO: Implement actual MySQL query")
            logger.info("Using mock data for demonstration...")
            
            # Mock data: user_id -> [(post_id, timestamp), ...]
            mock_following_posts = {
                1: [
                    (101, time.time() - 3600),      # 1 hour ago
                    (102, time.time() - 7200),      # 2 hours ago
                    (103, time.time() - 10800),     # 3 hours ago
                ],
                2: [
                    (104, time.time() - 1800),      # 30 min ago
                    (105, time.time() - 5400),      # 1.5 hours ago
                ]
            }
            
            synced_count = 0
            for user_id, posts in mock_following_posts.items():
                for post_id, timestamp in posts:
                    success = cache_service.add_following_post(
                        user_id=user_id,
                        post_id=post_id,
                        timestamp=timestamp,
                        ttl=172800  # 48 hours
                    )
                    
                    if success:
                        synced_count += 1
            
            logger.info(f"✅ Synced {synced_count} following posts")
            
        except Exception as e:
            logger.error(f"❌ Error syncing following posts: {e}", exc_info=True)
            raise
    
    # ==================== TRENDING POSTS ====================
    
    def sync_trending_posts(self):
        """
        Calculate and sync trending posts from MySQL to Redis
        
        Trending score = (likes + 2*comments + 3*shares) / age_hours^0.8
        """
        logger.info("=" * 60)
        logger.info("Calculating and syncing trending posts...")
        logger.info("=" * 60)
        
        try:
            # TODO: Query MySQL and calculate trending scores
            # Example query:
            """
            SELECT 
                p.id,
                p.category,
                p.likes_count,
                p.comments_count,
                p.shares_count,
                TIMESTAMPDIFF(HOUR, p.created_at, NOW()) as age_hours
            FROM posts p
            WHERE 
                p.created_at > DATE_SUB(NOW(), INTERVAL 24 HOUR)
            ORDER BY 
                (p.likes_count + 2*p.comments_count + 3*p.shares_count) / 
                POW(TIMESTAMPDIFF(HOUR, p.created_at, NOW()) + 1, 0.8) DESC
            LIMIT 500
            """
            
            logger.info("TODO: Implement actual MySQL query")
            logger.info("Using mock data for demonstration...")
            
            # Mock trending posts with scores
            mock_trending = [
                (101, 95.5),   # post_id, trending_score
                (102, 87.3),
                (103, 75.2),
                (104, 68.9),
                (105, 55.4),
            ]
            
            # Sync global trending
            success = cache_service.set_trending_posts(
                posts_with_scores=mock_trending,
                category=None,
                ttl=settings.TRENDING_CACHE_TTL
            )
            
            if success:
                logger.info(f"✅ Synced {len(mock_trending)} global trending posts")
            
            # TODO: Also sync category-specific trending
            # cache_service.set_trending_posts(
            #     posts_with_scores=category_trending,
            #     category="technology",
            #     ttl=settings.TRENDING_CACHE_TTL
            # )
            
        except Exception as e:
            logger.error(f"❌ Error syncing trending posts: {e}", exc_info=True)
            raise
    
    # ==================== CF SIMILARITIES ====================
    
    def sync_cf_similarities(self):
        """
        Sync CF user similarities from offline computed results
        
        These are precomputed by the AI team's offline pipeline.
        """
        logger.info("=" * 60)
        logger.info("Syncing CF similarities...")
        logger.info("=" * 60)
        
        try:
            # TODO: Load from offline computed file
            # Example: Read from CSV or numpy file
            """
            import pandas as pd
            similarities_df = pd.read_csv('offline/cf_similarities.csv')
            
            for user_id in similarities_df['user_id'].unique():
                user_sims = similarities_df[similarities_df['user_id'] == user_id]
                similar_users = list(zip(
                    user_sims['similar_user_id'].values,
                    user_sims['similarity_score'].values
                ))
                cache_service.set_similar_users(user_id, similar_users)
            """
            
            logger.info("TODO: Implement loading from offline computed file")
            logger.info("Using mock data for demonstration...")
            
            # Mock CF similarities
            mock_cf_similarities = {
                1: [(2, 0.85), (3, 0.72), (4, 0.68)],
                2: [(1, 0.85), (3, 0.79), (5, 0.65)],
            }
            
            synced_count = 0
            for user_id, similar_users in mock_cf_similarities.items():
                success = cache_service.set_similar_users(
                    user_id=user_id,
                    similar_users=similar_users,
                    ttl=settings.CF_SIMILARITIES_TTL
                )
                
                if success:
                    synced_count += 1
            
            logger.info(f"✅ Synced CF similarities for {synced_count} users")
            
        except Exception as e:
            logger.error(f"❌ Error syncing CF similarities: {e}", exc_info=True)
            raise
    
    # ==================== USER INTERACTIONS ====================
    
    def sync_user_likes(self, user_ids: List[int] = None):
        """
        Sync user likes from MySQL to Redis
        
        Args:
            user_ids: List of specific user IDs to sync (None = all users)
        """
        logger.info("=" * 60)
        logger.info("Syncing user likes...")
        logger.info("=" * 60)
        
        try:
            # TODO: Query MySQL for user likes
            # Example query:
            """
            SELECT 
                user_id,
                GROUP_CONCAT(post_id) as liked_post_ids
            FROM user_likes
            WHERE 
                user_id IN %s OR %s IS NULL
            GROUP BY user_id
            """
            
            logger.info("TODO: Implement actual MySQL query")
            logger.info("Using mock data for demonstration...")
            
            # Mock user likes
            mock_likes = {
                1: {101, 102, 103, 104},
                2: {102, 105, 106},
            }
            
            synced_count = 0
            for user_id, liked_posts in mock_likes.items():
                success = cache_service.set_user_liked_posts(
                    user_id=user_id,
                    post_ids=liked_posts,
                    ttl=settings.USER_LIKES_TTL
                )
                
                if success:
                    synced_count += 1
            
            logger.info(f"✅ Synced likes for {synced_count} users")
            
        except Exception as e:
            logger.error(f"❌ Error syncing user likes: {e}", exc_info=True)
            raise
    
    # ==================== MAIN SYNC FUNCTION ====================
    
    def full_sync(self):
        """
        Perform full sync of all data
        """
        logger.info("=" * 80)
        logger.info("🔄 STARTING FULL SYNC - MySQL → Redis")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Sync all data types
            self.sync_user_profiles(full_sync=True)
            self.sync_post_features(full_sync=True)
            self.sync_following_relationships()
            self.sync_following_posts()
            self.sync_trending_posts()
            self.sync_cf_similarities()
            self.sync_user_likes()
            
            duration = time.time() - start_time
            
            logger.info("=" * 80)
            logger.info(f"✅ FULL SYNC COMPLETED in {duration:.2f}s")
            logger.info("=" * 80)
            
            # Log cache stats
            stats = cache_service.get_cache_stats()
            logger.info(f"📊 Cache Stats:")
            logger.info(f"   - Memory Used: {stats.get('used_memory_human', 'unknown')}")
            logger.info(f"   - Hit Rate: {stats.get('hit_rate', 0):.2f}%")
            logger.info(f"   - Total Commands: {stats.get('total_commands_processed', 0)}")
            
        except Exception as e:
            logger.error(f"❌ FULL SYNC FAILED: {e}", exc_info=True)
            raise
    
    def incremental_sync(self):
        """
        Perform incremental sync (only changed data since last sync)
        """
        logger.info("=" * 80)
        logger.info("🔄 STARTING INCREMENTAL SYNC - MySQL → Redis")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Sync only changed data
            self.sync_user_profiles(full_sync=False)
            self.sync_post_features(full_sync=False)
            self.sync_trending_posts()  # Always recalculate
            
            # Update last sync time
            self.last_sync_time = datetime.now()
            
            duration = time.time() - start_time
            
            logger.info("=" * 80)
            logger.info(f"✅ INCREMENTAL SYNC COMPLETED in {duration:.2f}s")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ INCREMENTAL SYNC FAILED: {e}", exc_info=True)
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Sync data from MySQL to Redis for recommendation system"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Perform full sync (default: incremental)"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously (sync every N seconds)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Sync interval in seconds (default: 60)"
    )
    
    args = parser.parse_args()
    
    # Initialize sync service
    sync_service = MySQLToRedisSync()
    
    try:
        # Connect to databases
        sync_service.connect()
        
        if args.continuous:
            # Continuous sync mode
            logger.info(f"Running in continuous mode (interval: {args.interval}s)")
            
            while True:
                try:
                    if args.full:
                        sync_service.full_sync()
                    else:
                        sync_service.incremental_sync()
                    
                    logger.info(f"💤 Sleeping for {args.interval} seconds...")
                    time.sleep(args.interval)
                    
                except KeyboardInterrupt:
                    logger.info("⚠️ Received interrupt, stopping...")
                    break
                except Exception as e:
                    logger.error(f"❌ Sync error: {e}", exc_info=True)
                    logger.info(f"💤 Waiting {args.interval}s before retry...")
                    time.sleep(args.interval)
        else:
            # Single sync
            if args.full:
                sync_service.full_sync()
            else:
                sync_service.incremental_sync()
    
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        return 1
    
    finally:
        sync_service.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())