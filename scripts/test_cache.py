"""
Test Cache Service

Simple script to test Redis cache operations.
Use this to verify cache is working correctly.

Usage:
    python test_cache.py
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
from recommender.online.app.services.cache_service import cache_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_user_profile():
    """Test user profile operations"""
    logger.info("=" * 60)
    logger.info("Testing User Profile Operations")
    logger.info("=" * 60)
    
    # Test data
    user_id = 12345
    profile = {
        "id": user_id,
        "username": "test_user",
        "age": 25,
        "gender": "M",
        "interests": ["technology", "sports", "travel"],
        "activity_level": 0.85,
        "created_at": "2024-01-01T00:00:00"
    }
    
    # Set profile
    logger.info(f"Setting user profile for user {user_id}...")
    success = cache_service.set_user_profile(user_id, profile)
    assert success, "Failed to set user profile"
    logger.info("✅ Set user profile successfully")
    
    # Get profile
    logger.info(f"Getting user profile for user {user_id}...")
    retrieved_profile = cache_service.get_user_profile(user_id)
    assert retrieved_profile is not None, "Failed to get user profile"
    assert retrieved_profile["username"] == "test_user"
    logger.info("✅ Retrieved user profile successfully")
    logger.info(f"Profile: {retrieved_profile}")
    
    # Delete profile
    logger.info(f"Deleting user profile for user {user_id}...")
    success = cache_service.delete_user_profile(user_id)
    assert success, "Failed to delete user profile"
    logger.info("✅ Deleted user profile successfully")
    
    # Verify deletion
    retrieved_profile = cache_service.get_user_profile(user_id)
    assert retrieved_profile is None, "Profile still exists after deletion"
    logger.info("✅ Verified deletion")


def test_post_features():
    """Test post features operations"""
    logger.info("=" * 60)
    logger.info("Testing Post Features Operations")
    logger.info("=" * 60)
    
    # Test data
    post_id = 67890
    features = {
        "id": post_id,
        "author_id": 123,
        "category": "technology",
        "likes_count": 100,
        "comments_count": 20,
        "shares_count": 10,
        "author_followers": 5000,
        "post_age_hours": 2.5
    }
    
    # Set features
    logger.info(f"Setting post features for post {post_id}...")
    success = cache_service.set_post_features(post_id, features)
    assert success, "Failed to set post features"
    logger.info("✅ Set post features successfully")
    
    # Get features
    logger.info(f"Getting post features for post {post_id}...")
    retrieved_features = cache_service.get_post_features(post_id)
    assert retrieved_features is not None, "Failed to get post features"
    assert retrieved_features["likes_count"] == 100
    logger.info("✅ Retrieved post features successfully")
    logger.info(f"Features: {retrieved_features}")


def test_batch_operations():
    """Test batch operations"""
    logger.info("=" * 60)
    logger.info("Testing Batch Operations")
    logger.info("=" * 60)
    
    # Create multiple posts
    posts_features = {}
    for i in range(1, 11):
        post_id = 1000 + i
        posts_features[post_id] = {
            "id": post_id,
            "author_id": i,
            "likes_count": i * 10,
            "created_at": f"2025-01-{i:02d}T00:00:00"
        }
    
    # Batch set
    logger.info(f"Batch setting {len(posts_features)} posts...")
    start_time = time.time()
    success = cache_service.set_posts_features_batch(posts_features)
    duration_ms = (time.time() - start_time) * 1000
    assert success, "Failed to batch set posts"
    logger.info(f"✅ Batch set completed in {duration_ms:.2f}ms")
    
    # Batch get
    post_ids = list(posts_features.keys())
    logger.info(f"Batch getting {len(post_ids)} posts...")
    start_time = time.time()
    retrieved_posts = cache_service.get_posts_features_batch(post_ids)
    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"✅ Batch get completed in {duration_ms:.2f}ms")
    logger.info(f"Retrieved {len(retrieved_posts)}/{len(post_ids)} posts")
    logger.info(f"Hit rate: {len(retrieved_posts)/len(post_ids)*100:.1f}%")


def test_following_operations():
    """Test following operations"""
    logger.info("=" * 60)
    logger.info("Testing Following Operations")
    logger.info("=" * 60)
    
    user_id = 999
    following_ids = {1, 2, 3, 4, 5}
    
    # Set following list
    logger.info(f"Setting following list for user {user_id}...")
    success = cache_service.set_following_list(user_id, following_ids)
    assert success, "Failed to set following list"
    logger.info("✅ Set following list successfully")
    
    # Get following list
    logger.info(f"Getting following list for user {user_id}...")
    retrieved_following = cache_service.get_following_list(user_id)
    assert retrieved_following == following_ids
    logger.info("✅ Retrieved following list successfully")
    logger.info(f"Following: {retrieved_following}")
    
    # Add following
    logger.info("Adding new following...")
    success = cache_service.add_following(user_id, 6)
    assert success, "Failed to add following"
    
    retrieved_following = cache_service.get_following_list(user_id)
    assert 6 in retrieved_following
    logger.info("✅ Added following successfully")
    
    # Remove following
    logger.info("Removing following...")
    success = cache_service.remove_following(user_id, 6)
    assert success, "Failed to remove following"
    
    retrieved_following = cache_service.get_following_list(user_id)
    assert 6 not in retrieved_following
    logger.info("✅ Removed following successfully")


def test_following_posts():
    """Test following posts operations"""
    logger.info("=" * 60)
    logger.info("Testing Following Posts Operations")
    logger.info("=" * 60)
    
    user_id = 888
    
    # Add posts to following feed
    current_time = time.time()
    posts = [
        (2001, current_time - 3600),    # 1 hour ago
        (2002, current_time - 7200),    # 2 hours ago
        (2003, current_time - 10800),   # 3 hours ago
    ]
    
    logger.info(f"Adding {len(posts)} posts to following feed...")
    for post_id, timestamp in posts:
        success = cache_service.add_following_post(user_id, post_id, timestamp)
        assert success, f"Failed to add post {post_id}"
    
    logger.info("✅ Added posts to following feed")
    
    # Get following posts
    logger.info("Getting following posts...")
    retrieved_posts = cache_service.get_following_posts(user_id, limit=10)
    logger.info(f"✅ Retrieved {len(retrieved_posts)} following posts")
    
    # Verify order (most recent first)
    assert retrieved_posts[0][0] == 2001  # Most recent
    assert retrieved_posts[-1][0] == 2003  # Oldest
    logger.info("✅ Posts are in correct order (most recent first)")


def test_trending_posts():
    """Test trending posts operations"""
    logger.info("=" * 60)
    logger.info("Testing Trending Posts Operations")
    logger.info("=" * 60)
    
    # Set trending posts
    trending_posts = [
        (3001, 95.5),
        (3002, 87.3),
        (3003, 75.2),
        (3004, 68.9),
        (3005, 55.4),
    ]
    
    logger.info(f"Setting {len(trending_posts)} trending posts...")
    success = cache_service.set_trending_posts(trending_posts)
    assert success, "Failed to set trending posts"
    logger.info("✅ Set trending posts successfully")
    
    # Get trending posts
    logger.info("Getting trending posts...")
    retrieved_trending = cache_service.get_trending_posts(limit=10)
    logger.info(f"✅ Retrieved {len(retrieved_trending)} trending posts")
    
    # Verify order (highest score first)
    assert retrieved_trending[0][0] == 3001  # Highest score
    assert retrieved_trending[0][1] == 95.5
    logger.info("✅ Trending posts in correct order")


def test_seen_posts():
    """Test seen posts operations"""
    logger.info("=" * 60)
    logger.info("Testing Seen Posts Operations")
    logger.info("=" * 60)
    
    user_id = 777
    seen_post_ids = [4001, 4002, 4003, 4004, 4005]
    
    # Mark posts as seen
    logger.info(f"Marking {len(seen_post_ids)} posts as seen...")
    success = cache_service.mark_posts_as_seen(user_id, seen_post_ids)
    assert success, "Failed to mark posts as seen"
    logger.info("✅ Marked posts as seen")
    
    # Get seen posts
    logger.info("Getting seen posts...")
    retrieved_seen = cache_service.get_seen_posts(user_id)
    assert len(retrieved_seen) == len(seen_post_ids)
    assert all(pid in retrieved_seen for pid in seen_post_ids)
    logger.info(f"✅ Retrieved {len(retrieved_seen)} seen posts")


def test_cf_similarities():
    """Test CF similarities operations"""
    logger.info("=" * 60)
    logger.info("Testing CF Similarities Operations")
    logger.info("=" * 60)
    
    user_id = 666
    similar_users = [
        (101, 0.85),
        (102, 0.72),
        (103, 0.68),
        (104, 0.55),
    ]
    
    # Set similar users
    logger.info(f"Setting {len(similar_users)} similar users...")
    success = cache_service.set_similar_users(user_id, similar_users)
    assert success, "Failed to set similar users"
    logger.info("✅ Set similar users successfully")
    
    # Get similar users
    logger.info("Getting similar users...")
    retrieved_similar = cache_service.get_similar_users(user_id, limit=10)
    logger.info(f"✅ Retrieved {len(retrieved_similar)} similar users")
    
    # Verify order (highest similarity first)
    assert retrieved_similar[0][0] == 101  # Highest similarity
    assert retrieved_similar[0][1] == 0.85
    logger.info("✅ Similar users in correct order")


def test_cache_stats():
    """Test cache statistics"""
    logger.info("=" * 60)
    logger.info("Testing Cache Statistics")
    logger.info("=" * 60)
    
    stats = cache_service.get_cache_stats()
    
    logger.info("Cache Statistics:")
    logger.info(f"  Memory Used: {stats.get('used_memory_human', 'unknown')}")
    logger.info(f"  Memory Peak: {stats.get('used_memory_peak_human', 'unknown')}")
    logger.info(f"  Hit Rate: {stats.get('hit_rate', 0):.2f}%")
    logger.info(f"  Total Commands: {stats.get('total_commands_processed', 0)}")
    logger.info(f"  Ops/sec: {stats.get('instantaneous_ops_per_sec', 0)}")
    logger.info(f"  Connected Clients: {stats.get('connected_clients', 0)}")
    logger.info(f"  Keyspace Hits: {stats.get('keyspace_hits', 0)}")
    logger.info(f"  Keyspace Misses: {stats.get('keyspace_misses', 0)}")
    logger.info(f"  Evicted Keys: {stats.get('evicted_keys', 0)}")
    logger.info(f"  Expired Keys: {stats.get('expired_keys', 0)}")


def main():
    """Main test function"""
    logger.info("=" * 80)
    logger.info("🧪 Starting Cache Service Tests")
    logger.info("=" * 80)
    
    try:
        # Connect to Redis
        logger.info("Connecting to Redis...")
        cache_service.connect()
        
        if not cache_service.is_connected():
            logger.error("❌ Failed to connect to Redis")
            return 1
        
        logger.info("✅ Connected to Redis")
        logger.info("")
        
        # Run all tests
        test_user_profile()
        logger.info("")
        
        test_post_features()
        logger.info("")
        
        test_batch_operations()
        logger.info("")
        
        test_following_operations()
        logger.info("")
        
        test_following_posts()
        logger.info("")
        
        test_trending_posts()
        logger.info("")
        
        test_seen_posts()
        logger.info("")
        
        test_cf_similarities()
        logger.info("")
        
        test_cache_stats()
        logger.info("")
        
        logger.info("=" * 80)
        logger.info("✅ All Tests Passed!")
        logger.info("=" * 80)
        
        return 0
        
    except AssertionError as e:
        logger.error(f"❌ Test failed: {e}")
        return 1
    
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1
    
    finally:
        # Close connection
        cache_service.close()


if __name__ == "__main__":
    sys.exit(main())