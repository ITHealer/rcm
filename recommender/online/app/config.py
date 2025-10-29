"""
Application Configuration

Environment variables and settings for the recommendation service.
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    
    Team App will sync data from MySQL to Redis.
    AI team reads from Redis only.
    """
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignore extra fields not defined in class
    )
    
    # ==================== BASIC SETTINGS ====================
    
    PROJECT_NAME: str = "Social Network Recommendation API"
    ENVIRONMENT: str = "development"  # development, staging, production
    DEBUG: bool = True
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]  # In production, specify exact origins
    
    # ==================== REDIS CONFIGURATION ====================
    
    # Redis Connection
    REDIS_HOST: str = "recommendation_redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_SOCKET_TIMEOUT: float = 0.01  # 10ms timeout for fast operations
    
    # Redis Pool
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_RETRY_ON_TIMEOUT: bool = True
    
    # Cache TTL (Time To Live in seconds)
    USER_PROFILE_TTL: int = 3600  # 1 hour
    POST_FEATURES_TTL: int = 86400  # 24 hours
    FOLLOWING_CACHE_TTL: int = 1800  # 30 minutes
    TRENDING_CACHE_TTL: int = 300  # 5 minutes
    SEEN_POSTS_TTL: int = 604800  # 7 days
    CF_SIMILARITIES_TTL: int = 86400  # 24 hours
    USER_LIKES_TTL: int = 86400  # 24 hours
    
    # ==================== TIMEOUTS ====================
    
    # Request timeout (overall)
    REQUEST_TIMEOUT: float = 0.2  # 200ms
    
    # Component timeouts
    RECALL_TIMEOUT: float = 0.05  # 50ms
    RANKING_TIMEOUT: float = 0.05  # 50ms
    RERANKING_TIMEOUT: float = 0.02  # 20ms
    
    # ==================== MODEL PATHS ====================
    
    # Offline model paths (read-only)
    OFFLINE_PATH: str = "configs"
    # OFFLINE_PATH2: str = os.getenv("OFFLINE_PATH", "../offline")
    
    MODEL_PATH: str = ""  # Will be set dynamically from latest version
    MODEL_VERSION: str = "latest"
    
    # Embeddings
    POST_EMBEDDINGS_PATH: str = ""
    USER_EMBEDDINGS_PATH: str = ""
    POST_ID_TO_IDX_PATH: str = ""
    
    # Faiss Index
    FAISS_INDEX_PATH: str = ""
    
    # ==================== BUSINESS LOGIC ====================
    
    # Recall sizes per channel
    FOLLOWING_RECALL_SIZE: int = 400
    CF_RECALL_SIZE: int = 300
    CONTENT_RECALL_SIZE: int = 200
    TRENDING_RECALL_SIZE: int = 100
    
    # Total candidates after recall
    MAX_RECALL_SIZE: int = 1000
    
    # After ML ranking
    MAX_RANKING_SIZE: int = 100
    
    # Final feed size
    FINAL_FEED_SIZE: int = 50
    
    # Following feed time window
    FOLLOWING_TIME_WINDOW_HOURS: int = 48  # Only get posts from last 48 hours
    
    # Content-based recall time window
    CONTENT_TIME_WINDOW_DAYS: int = 7  # Only get posts from last 7 days
    
    # Trending calculation window
    TRENDING_TIME_WINDOW_HOURS: int = 24  # Calculate trending from last 24 hours
    
    # ==================== FEATURE FLAGS ====================
    # Control which features are enabled/disabled at runtime
    
    # Recall features
    FEATURE_CONTENT_BASED_RECALL: bool = True
    FEATURE_CF_RECALL: bool = True
    FEATURE_TRENDING_RECALL: bool = True
    FEATURE_FOLLOWING_RECALL: bool = True
    
    # Ranking features
    FEATURE_FAISS: bool = True
    FEATURE_RERANKING: bool = True
    
    # Post-processing features
    FEATURE_FRESHNESS_BOOST: bool = True
    FEATURE_DIVERSITY: bool = True
    FEATURE_AD_INSERTION: bool = False
    
    # Seen posts tracking
    FEATURE_MARK_SEEN: bool = True
    MARK_SEEN_AFTER_GENERATION: bool = False  # Mark seen after feed generation
    
    # API features
    RELOAD: bool = True  # Allow hot reload in development
    
    # ==================== DEBUG FLAGS ====================
    
    DEBUG_SQL: bool = False  # Log SQL queries
    DEBUG_REDIS: bool = False  # Log Redis operations
    DEBUG_TIMING: bool = True  # Log timing information
    
    # ==================== TESTING ====================
    
    USE_TEST_DATA: bool = False  # Use test data instead of real data
    TEST_DATA_DIR: str = "tests/data"
    
    # ==================== RE-RANKING RULES ====================
    
    # Diversity
    MAX_CONSECUTIVE_SAME_AUTHOR: int = 2
    CATEGORY_DIVERSITY_WINDOW: int = 5
    
    # Freshness boost
    FRESHNESS_BOOST_THRESHOLD_HOURS: float = 3.0
    FRESHNESS_BOOST_FACTOR: float = 1.2
    
    # Ad insertion positions
    AD_POSITIONS: List[int] = [5, 15, 30]
    
    # ==================== FEATURE EXTRACTION ====================
    
    # Number of features for ML model
    NUM_FEATURES: int = 47
    
    # Feature categories
    NUM_USER_FEATURES: int = 10
    NUM_POST_FEATURES: int = 15
    NUM_INTERACTION_FEATURES: int = 12
    NUM_CONTEXT_FEATURES: int = 5
    NUM_HISTORICAL_FEATURES: int = 5
    
    # ==================== LOGGING ====================
    
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "logs/recommendation.log"
    
    # ==================== MONITORING ====================
    
    # Prometheus metrics
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # ==================== RATE LIMITING ====================
    
    # Requests per user per minute
    RATE_LIMIT_PER_USER: int = 60
    
    # Global rate limit (requests per second)
    GLOBAL_RATE_LIMIT: int = 1000
    
    # ==================== DATABASE (for Team App sync) ====================
    
    # MySQL connection for Team App (NOT used by AI team directly)
    # These are here for reference only
    # MYSQL_HOST: str = "localhost"
    # MYSQL_PORT: int = 3306
    # MYSQL_DATABASE: str = "social_network"
    # MYSQL_USER: str = "app_user"
    # MYSQL_PASSWORD: str = ""
    MYSQL_HOST: str = "14.225.220.56"            # thay "ip" bằng địa chỉ IP thực tế
    MYSQL_PORT: int = 15479
    MYSQL_DATABASE: str = "wayjet_system"
    MYSQL_USER: str = "way_root"
    MYSQL_PASSWORD: str = "YmhNWpppahN92AtJotFDoHnCoW38keDp"
    
    # Sync intervals (for Team App scripts)
    SYNC_INTERVAL_SECONDS: int = 60  # Sync every minute
    BATCH_SYNC_SIZE: int = 1000  # Sync 1000 records at a time
    
    # ==================== DEVELOPMENT ====================
    
    # Allow flushing cache in development
    ALLOW_CACHE_FLUSH: bool = True  # Set to False in production


# Create settings instance
settings = Settings()


# Validate settings on import
def validate_settings():
    """
    Validate critical settings
    """
    # Check Redis connection info
    if not settings.REDIS_HOST:
        raise ValueError("REDIS_HOST must be set")
    
    if settings.REDIS_PORT <= 0 or settings.REDIS_PORT > 65535:
        raise ValueError("REDIS_PORT must be between 1 and 65535")
    
    # Check timeouts
    if settings.REQUEST_TIMEOUT <= 0:
        raise ValueError("REQUEST_TIMEOUT must be positive")
    
    component_timeout = (
        settings.RECALL_TIMEOUT + 
        settings.RANKING_TIMEOUT + 
        settings.RERANKING_TIMEOUT
    )
    
    if component_timeout > settings.REQUEST_TIMEOUT:
        raise ValueError(
            f"Sum of component timeouts ({component_timeout}s) "
            f"exceeds REQUEST_TIMEOUT ({settings.REQUEST_TIMEOUT}s)"
        )
    
    # Check recall sizes
    if settings.MAX_RECALL_SIZE <= 0:
        raise ValueError("MAX_RECALL_SIZE must be positive")
    
    if settings.FINAL_FEED_SIZE > settings.MAX_RANKING_SIZE:
        raise ValueError("FINAL_FEED_SIZE cannot exceed MAX_RANKING_SIZE")
    
    # Warn about debug mode in production
    if settings.ENVIRONMENT == "production" and settings.DEBUG:
        import warnings
        warnings.warn("DEBUG is True in production environment!")


# Run validation
validate_settings()