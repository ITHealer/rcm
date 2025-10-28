"""
Recommendation API Endpoints
============================
Feed generation endpoints for users
"""

from fastapi import APIRouter, Query, HTTPException, status
from typing import List, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Will be injected by main.py
recommendation_service = None


def set_recommendation_service(service):
    """Set recommendation service instance"""
    global recommendation_service
    recommendation_service = service


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
    """Feed response model"""
    user_id: int
    feed: List[dict]
    metadata: FeedMetadata


# ==================== ENDPOINTS ====================

@router.get("/feed/{user_id}", response_model=FeedResponse)
async def get_feed(
    user_id: int,
    limit: int = Query(50, ge=1, le=100, description="Number of posts"),
    exclude_seen: Optional[str] = Query(None, description="Comma-separated post IDs to exclude")
):
    """
    Get personalized feed for user
    
    **Process:**
    1. Multi-channel recall (~1000 candidates)
    2. ML ranking (LightGBM)
    3. Re-ranking (business rules)
    4. Return top N posts
    
    **Performance:**
    - Target latency: < 200ms
    - Cached results: < 50ms
    
    **Args:**
    - user_id: User ID (required)
    - limit: Number of posts to return (default: 50, max: 100)
    - exclude_seen: Comma-separated post IDs to exclude (optional)
    
    **Returns:**
    - Feed with posts and metadata
    
    **Example:**
    ```
    GET /api/v1/recommendation/feed/123?limit=20
    ```
    """
    if not recommendation_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service not initialized"
        )
    
    # Parse exclude_seen
    exclude_list = None
    if exclude_seen:
        try:
            exclude_list = [int(pid.strip()) for pid in exclude_seen.split(',') if pid.strip()]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid exclude_seen format. Use comma-separated integers."
            )
    
    # Generate feed
    try:
        result = recommendation_service.generate_feed(
            user_id=user_id,
            limit=limit,
            exclude_seen=exclude_list
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate feed: {str(e)}"
        )


@router.post("/feed", response_model=FeedResponse)
async def get_feed_post(request: FeedRequest):
    """
    Get personalized feed for user (POST version)
    
    Same as GET /feed/{user_id} but accepts JSON body.
    Useful when exclude_seen list is large.
    
    **Example:**
    ```json
    {
        "user_id": 123,
        "limit": 20,
        "exclude_seen": [1, 2, 3, 4, 5]
    }
    ```
    """
    if not recommendation_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service not initialized"
        )
    
    try:
        result = recommendation_service.generate_feed(
            user_id=request.user_id,
            limit=request.limit,
            exclude_seen=request.exclude_seen
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating feed for user {request.user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate feed: {str(e)}"
        )


@router.post("/feed/{user_id}/mark_seen")
async def mark_posts_seen(
    user_id: int,
    post_ids: List[int]
):
    """
    Mark posts as seen by user
    
    **Args:**
    - user_id: User ID
    - post_ids: List of post IDs to mark as seen
    
    **Example:**
    ```json
    POST /api/v1/recommendation/feed/123/mark_seen
    [1, 2, 3, 4, 5]
    ```
    """
    from recommender.online.app.services.cache_service import cache_service
    
    if not cache_service.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache service not available"
        )
    
    try:
        success = cache_service.mark_posts_as_seen(user_id, post_ids)
        
        return {
            "success": success,
            "user_id": user_id,
            "marked_count": len(post_ids)
        }
        
    except Exception as e:
        logger.error(f"Error marking posts seen for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to mark posts seen: {str(e)}"
        )


@router.get("/metrics")
async def get_metrics():
    """
    Get recommendation service metrics
    
    **Returns:**
    - Service-level metrics
    - Per-channel metrics
    - Performance statistics
    
    **Example:**
    ```
    GET /api/v1/recommendation/metrics
    ```
    """
    if not recommendation_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service not initialized"
        )
    
    try:
        metrics = recommendation_service.get_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.post("/metrics/reset")
async def reset_metrics():
    """
    Reset all metrics to zero
    
    **Returns:**
    - Success confirmation
    
    **Example:**
    ```
    POST /api/v1/recommendation/metrics/reset
    ```
    """
    if not recommendation_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service not initialized"
        )
    
    try:
        # Reset service metrics
        recommendation_service.metrics = {
            'total_requests': 0,
            'total_time_ms': 0,
            'recall_time_ms': 0,
            'ranking_time_ms': 0,
            'reranking_time_ms': 0
        }
        
        # Reset channel metrics
        recommendation_service.following_recall.reset_metrics()
        recommendation_service.cf_recall.reset_metrics()
        recommendation_service.content_recall.reset_metrics()
        recommendation_service.trending_recall.reset_metrics()
        recommendation_service.ranker.reset_metrics()
        recommendation_service.reranker.reset_metrics()
        
        return {
            "success": True,
            "message": "All metrics reset successfully"
        }
        
    except Exception as e:
        logger.error(f"Error resetting metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset metrics: {str(e)}"
        )