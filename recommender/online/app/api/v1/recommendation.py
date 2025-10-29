# """
# Recommendation API Endpoints
# ============================
# Feed generation endpoints for users
# """

# from fastapi import APIRouter, Query, HTTPException, status
# from typing import List, Optional
# from pydantic import BaseModel, Field
# import logging

# logger = logging.getLogger(__name__)

# router = APIRouter()

# # Will be injected by main.py
# recommendation_service = None


# def set_recommendation_service(service):
#     """Set recommendation service instance"""
#     global recommendation_service
#     recommendation_service = service


# # ==================== REQUEST/RESPONSE MODELS ====================

# class FeedRequest(BaseModel):
#     """Feed request model"""
#     user_id: int = Field(..., description="User ID", gt=0)
#     limit: int = Field(50, description="Number of posts to return", ge=1, le=100)
#     exclude_seen: Optional[List[int]] = Field(None, description="Post IDs to exclude")


# class PostItem(BaseModel):
#     """Post item in feed"""
#     post_id: int
#     score: float
#     rank: int
#     author_id: Optional[int] = None
#     category: Optional[str] = None
#     created_at: Optional[str] = None
#     is_ad: bool = False


# class FeedMetadata(BaseModel):
#     """Feed generation metadata"""
#     total_time_ms: float
#     recall_time_ms: Optional[float] = None
#     ranking_time_ms: Optional[float] = None
#     reranking_time_ms: Optional[float] = None
#     recall_count: Optional[int] = None
#     ranked_count: Optional[int] = None
#     final_count: int
#     timestamp: str


# class FeedResponse(BaseModel):
#     """Feed response model"""
#     user_id: int
#     feed: List[dict]
#     metadata: FeedMetadata


# # ==================== ENDPOINTS ====================

# @router.get("/feed/{user_id}", response_model=FeedResponse)
# async def get_feed(
#     user_id: int,
#     limit: int = Query(50, ge=1, le=100, description="Number of posts"),
#     exclude_seen: Optional[str] = Query(None, description="Comma-separated post IDs to exclude")
# ):
#     """
#     Get personalized feed for user
    
#     **Process:**
#     1. Multi-channel recall (~1000 candidates)
#     2. ML ranking (LightGBM)
#     3. Re-ranking (business rules)
#     4. Return top N posts
    
#     **Performance:**
#     - Target latency: < 200ms
#     - Cached results: < 50ms
    
#     **Args:**
#     - user_id: User ID (required)
#     - limit: Number of posts to return (default: 50, max: 100)
#     - exclude_seen: Comma-separated post IDs to exclude (optional)
    
#     **Returns:**
#     - Feed with posts and metadata
    
#     **Example:**
#     ```
#     GET /api/v1/recommendation/feed/123?limit=20
#     ```
#     """
#     if not recommendation_service:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Recommendation service not initialized"
#         )
    
#     # Parse exclude_seen
#     exclude_list = None
#     if exclude_seen:
#         try:
#             exclude_list = [int(pid.strip()) for pid in exclude_seen.split(',') if pid.strip()]
#         except ValueError:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Invalid exclude_seen format. Use comma-separated integers."
#             )
    
#     # Generate feed
#     try:
#         result = recommendation_service.get_feed(
#             user_id=user_id,
#             limit=limit,
#             exclude_seen=exclude_list
#         )
        
#         return result
        
#     except Exception as e:
#         logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to generate feed: {str(e)}"
#         )


# @router.post("/feed", response_model=FeedResponse)
# async def get_feed_post(request: FeedRequest):
#     """
#     Get personalized feed for user (POST version)
    
#     Same as GET /feed/{user_id} but accepts JSON body.
#     Useful when exclude_seen list is large.
    
#     **Example:**
#     ```json
#     {
#         "user_id": 123,
#         "limit": 20,
#         "exclude_seen": [1, 2, 3, 4, 5]
#     }
#     ```
#     """
#     if not recommendation_service:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Recommendation service not initialized"
#         )
    
#     try:
#         result = recommendation_service.generate_feed(
#             user_id=request.user_id,
#             limit=request.limit,
#             exclude_seen=request.exclude_seen
#         )
        
#         return result
        
#     except Exception as e:
#         logger.error(f"Error generating feed for user {request.user_id}: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to generate feed: {str(e)}"
#         )


# @router.post("/feed/{user_id}/mark_seen")
# async def mark_posts_seen(
#     user_id: int,
#     post_ids: List[int]
# ):
#     """
#     Mark posts as seen by user
    
#     **Args:**
#     - user_id: User ID
#     - post_ids: List of post IDs to mark as seen
    
#     **Example:**
#     ```json
#     POST /api/v1/recommendation/feed/123/mark_seen
#     [1, 2, 3, 4, 5]
#     ```
#     """
#     from recommender.online.app.services.cache_service import cache_service
    
#     if not cache_service.is_connected():
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Cache service not available"
#         )
    
#     try:
#         success = cache_service.mark_posts_as_seen(user_id, post_ids)
        
#         return {
#             "success": success,
#             "user_id": user_id,
#             "marked_count": len(post_ids)
#         }
        
#     except Exception as e:
#         logger.error(f"Error marking posts seen for user {user_id}: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to mark posts seen: {str(e)}"
#         )


# @router.get("/metrics")
# async def get_metrics():
#     """
#     Get recommendation service metrics
    
#     **Returns:**
#     - Service-level metrics
#     - Per-channel metrics
#     - Performance statistics
    
#     **Example:**
#     ```
#     GET /api/v1/recommendation/metrics
#     ```
#     """
#     if not recommendation_service:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Recommendation service not initialized"
#         )
    
#     try:
#         metrics = recommendation_service.get_metrics()
#         return metrics
        
#     except Exception as e:
#         logger.error(f"Error getting metrics: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to get metrics: {str(e)}"
#         )


# @router.post("/metrics/reset")
# async def reset_metrics():
#     """
#     Reset all metrics to zero
    
#     **Returns:**
#     - Success confirmation
    
#     **Example:**
#     ```
#     POST /api/v1/recommendation/metrics/reset
#     ```
#     """
#     if not recommendation_service:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Recommendation service not initialized"
#         )
    
#     try:
#         # Reset service metrics
#         recommendation_service.metrics = {
#             'total_requests': 0,
#             'total_time_ms': 0,
#             'recall_time_ms': 0,
#             'ranking_time_ms': 0,
#             'reranking_time_ms': 0
#         }
        
#         # Reset channel metrics
#         recommendation_service.following_recall.reset_metrics()
#         recommendation_service.cf_recall.reset_metrics()
#         recommendation_service.content_recall.reset_metrics()
#         recommendation_service.trending_recall.reset_metrics()
#         recommendation_service.ranker.reset_metrics()
#         recommendation_service.reranker.reset_metrics()
        
#         return {
#             "success": True,
#             "message": "All metrics reset successfully"
#         }
        
#     except Exception as e:
#         logger.error(f"Error resetting metrics: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to reset metrics: {str(e)}"
#         )

"""
Recommendation API Endpoints
============================
Feed generation endpoints for users

FILE: recommender/online/app/routers/recommendation.py
"""

from fastapi import APIRouter, Query, HTTPException, status
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# ==================== GLOBAL SERVICE INSTANCE ====================
# Will be injected by main.py via set_recommendation_service()
recommendation_service = None


def set_recommendation_service(service):
    """
    Set recommendation service instance
    Called from main.py during startup
    
    Args:
        service: RecommendationService instance
    """
    global recommendation_service
    recommendation_service = service
    logger.info("✅ Recommendation service injected into router")


# ==================== REQUEST/RESPONSE MODELS ====================

class FeedRequest(BaseModel):
    """Feed request model"""
    user_id: int = Field(..., description="User ID", gt=0)
    limit: int = Field(50, description="Number of posts to return", ge=1, le=100)
    exclude_seen: Optional[List[int]] = Field(None, description="Post IDs to exclude")


class PostItem(BaseModel):
    """Post item in feed"""
    post_id: int
    score: float
    rank: int
    author_id: Optional[int] = None
    category: Optional[str] = None
    created_at: Optional[str] = None
    is_ad: bool = False


class FeedMetadata(BaseModel):
    """Feed generation metadata"""
    total_time_ms: float
    recall_time_ms: Optional[float] = None
    ranking_time_ms: Optional[float] = None
    reranking_time_ms: Optional[float] = None
    recall_count: Optional[int] = None
    ranked_count: Optional[int] = None
    final_count: int
    timestamp: str


class FeedResponse(BaseModel):
    """
    Feed response model
    ✅ CRITICAL: Must be dictionary-based, not list
    """
    user_id: int
    feed: List[Dict[str, Any]]  # List of post dictionaries
    metadata: FeedMetadata
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "feed": [
                    {
                        "post_id": 456,
                        "score": 0.95,
                        "rank": 1,
                        "author_id": 789,
                        "created_at": "2025-10-29T10:00:00Z"
                    }
                ],
                "metadata": {
                    "total_time_ms": 150.5,
                    "recall_time_ms": 50.2,
                    "ranking_time_ms": 80.3,
                    "reranking_time_ms": 20.0,
                    "recall_count": 1000,
                    "ranked_count": 100,
                    "final_count": 50,
                    "timestamp": "2025-10-29T10:00:00.123456Z"
                }
            }
        }


# ==================== HELPER FUNCTION ====================

def _check_service_available():
    """
    Check if recommendation service is available
    Raises HTTPException if not
    """
    if not recommendation_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service not initialized. Please wait for service startup."
        )


# ==================== ENDPOINTS ====================

@router.get("/feed/{user_id}", response_model=FeedResponse)
async def get_feed(
    user_id: int,
    limit: int = Query(50, ge=1, le=100, description="Number of posts to return"),
    exclude_seen: Optional[str] = Query(None, description="Comma-separated post IDs to exclude")
):
    """
    Get personalized feed for user
    
    **Process:**
    1. Multi-channel recall (~1000 candidates)
       - Following feed
       - Collaborative filtering
       - Content-based (embeddings)
       - Trending posts
    2. ML ranking (LightGBM model)
    3. Re-ranking (business rules, diversity)
    4. Return top N posts
    
    **Performance:**
    - Target latency: < 200ms p95
    - Cached results: < 50ms
    
    **Args:**
    - user_id: User ID (required, must be > 0)
    - limit: Number of posts to return (default: 50, max: 100)
    - exclude_seen: Comma-separated post IDs to exclude (optional)
    
    **Returns:**
    - FeedResponse with posts and metadata
    
    **Example:**
    ```bash
    # Basic request
    GET /api/v1/recommendation/feed/123?limit=20
    
    # With exclude_seen
    GET /api/v1/recommendation/feed/123?limit=20&exclude_seen=1,2,3,4,5
    ```
    """
    # Check service availability
    _check_service_available()
    
    # Parse exclude_seen parameter
    exclude_list = None
    if exclude_seen:
        try:
            exclude_list = [int(pid.strip()) for pid in exclude_seen.split(',') if pid.strip()]
            logger.info(f"📝 Excluding {len(exclude_list)} posts for user {user_id}")
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid exclude_seen format. Use comma-separated integers. Error: {str(e)}"
            )
    
    # Generate feed
    try:
        logger.info(f"📰 Getting feed for user {user_id}, limit={limit}")
        
        # ✅ Call service directly (không dùng Depends)
        result = recommendation_service.get_feed(
            user_id=user_id,
            limit=limit,
            exclude_seen=exclude_list
        )
        
        # ✅ Validate result is dictionary
        if not isinstance(result, dict):
            logger.error(f"Service returned invalid type: {type(result)}, expected dict")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error: Invalid response format from service"
            )
        
        # ✅ Ensure required fields exist
        if 'feed' not in result:
            logger.warning(f"Service result missing 'feed' field, adding empty list")
            result['feed'] = []
        
        if 'metadata' not in result:
            logger.warning(f"Service result missing 'metadata' field, adding defaults")
            result['metadata'] = {
                'total_time_ms': 0.0,
                'final_count': len(result.get('feed', [])),
                'timestamp': datetime.utcnow().isoformat()
            }
        
        if 'user_id' not in result:
            result['user_id'] = user_id
        
        logger.info(f"✅ Feed generated: {len(result.get('feed', []))} posts for user {user_id}")
        
        return result
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"❌ Error generating feed for user {user_id}: {e}", exc_info=True)
        
        # ✅ Return proper dictionary even on error
        return FeedResponse(
            user_id=user_id,
            feed=[],
            metadata=FeedMetadata(
                total_time_ms=0.0,
                final_count=0,
                timestamp=datetime.utcnow().isoformat()
            )
        )


@router.post("/feed", response_model=FeedResponse)
async def get_feed_post(request: FeedRequest):
    """
    Get personalized feed for user (POST version)
    
    Same as GET /feed/{user_id} but accepts JSON body.
    Useful when exclude_seen list is very large (>100 items).
    
    **Example:**
    ```bash
    POST /api/v1/recommendation/feed
    Content-Type: application/json
    
    {
        "user_id": 123,
        "limit": 20,
        "exclude_seen": [1, 2, 3, 4, 5, ...]
    }
    ```
    """
    # Check service availability
    _check_service_available()
    
    try:
        logger.info(f"📰 Getting feed (POST) for user {request.user_id}, limit={request.limit}")
        
        # Call service
        result = recommendation_service.get_feed(
            user_id=request.user_id,
            limit=request.limit,
            exclude_seen=request.exclude_seen
        )
        
        # Validate result
        if not isinstance(result, dict):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error: Invalid response format"
            )
        
        # Ensure required fields
        if 'feed' not in result:
            result['feed'] = []
        if 'metadata' not in result:
            result['metadata'] = {
                'total_time_ms': 0.0,
                'final_count': len(result.get('feed', [])),
                'timestamp': datetime.utcnow().isoformat()
            }
        if 'user_id' not in result:
            result['user_id'] = request.user_id
        
        return result
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"❌ Error generating feed for user {request.user_id}: {e}", exc_info=True)
        
        return FeedResponse(
            user_id=request.user_id,
            feed=[],
            metadata=FeedMetadata(
                total_time_ms=0.0,
                final_count=0,
                timestamp=datetime.utcnow().isoformat()
            )
        )


@router.post("/feed/{user_id}/mark_seen")
async def mark_posts_seen(
    user_id: int,
    post_ids: List[int]
):
    """
    Mark posts as seen by user
    
    Used to track which posts user has already viewed.
    These posts can be excluded from future feed requests.
    
    **Args:**
    - user_id: User ID
    - post_ids: List of post IDs to mark as seen
    
    **Example:**
    ```bash
    POST /api/v1/recommendation/feed/123/mark_seen
    Content-Type: application/json
    
    [1, 2, 3, 4, 5]
    ```
    """
    try:
        from recommender.online.app.services.cache_service import cache_service
        
        if not cache_service.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache service not available"
            )
        
        success = cache_service.mark_posts_as_seen(user_id, post_ids)
        
        logger.info(f"✅ Marked {len(post_ids)} posts as seen for user {user_id}")
        
        return {
            "success": success,
            "user_id": user_id,
            "marked_count": len(post_ids),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ImportError:
        # Cache service not available
        logger.warning("Cache service not available, skipping mark_seen")
        return {
            "success": False,
            "user_id": user_id,
            "marked_count": 0,
            "message": "Cache service not available"
        }
        
    except Exception as e:
        logger.error(f"❌ Error marking posts seen for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to mark posts seen: {str(e)}"
        )


@router.get("/metrics")
async def get_metrics():
    """
    Get recommendation service metrics
    
    **Returns:**
    - Service-level metrics (request count, latency)
    - Per-channel metrics (following, CF, content, trending)
    - Model performance statistics
    
    **Example:**
    ```bash
    GET /api/v1/recommendation/metrics
    ```
    """
    _check_service_available()
    
    try:
        metrics = recommendation_service.get_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Error getting metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.post("/metrics/reset")
async def reset_metrics():
    """
    Reset all metrics to zero
    
    Useful for testing or starting fresh metric collection.
    
    **Returns:**
    - Success confirmation
    
    **Example:**
    ```bash
    POST /api/v1/recommendation/metrics/reset
    ```
    """
    _check_service_available()
    
    try:
        # Reset service-level metrics
        if hasattr(recommendation_service, 'metrics'):
            recommendation_service.metrics = {
                'total_requests': 0,
                'total_time_ms': 0,
                'recall_time_ms': 0,
                'ranking_time_ms': 0,
                'reranking_time_ms': 0
            }
        
        # Reset component metrics
        components = [
            'following_recall',
            'cf_recall',
            'content_recall',
            'trending_recall',
            'ranker',
            'reranker'
        ]
        
        for comp in components:
            if hasattr(recommendation_service, comp):
                component = getattr(recommendation_service, comp)
                if hasattr(component, 'reset_metrics'):
                    component.reset_metrics()
        
        logger.info("✅ All metrics reset successfully")
        
        return {
            "success": True,
            "message": "All metrics reset successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error resetting metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset metrics: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    **Returns:**
    - Service health status
    - Component availability
    
    **Example:**
    ```bash
    GET /api/v1/recommendation/health
    ```
    """
    status_ok = recommendation_service is not None
    
    return {
        "status": "healthy" if status_ok else "unhealthy",
        "service": "recommendation",
        "available": status_ok,
        "timestamp": datetime.utcnow().isoformat()
    }