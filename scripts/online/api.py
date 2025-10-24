"""
FASTAPI APPLICATION - RECOMMENDATION API
========================================
RESTful API for recommendation system

Endpoints:
- GET /feed - Get personalized feed
- GET /friends - Get friend recommendations
- POST /interaction - Log user interaction
- GET /trending - Get trending posts
- GET /health - Health check
- GET /metrics - System metrics
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import logging

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import pipeline
from recommender.online.online_inference_pipeline import OnlineInferencePipeline

# Setup logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class FeedRequest(BaseModel):
    """Feed request model"""
    user_id: int
    limit: int = 50
    exclude_seen: Optional[List[int]] = None


class FriendRecommendationRequest(BaseModel):
    """Friend recommendation request"""
    user_id: int
    limit: int = 20


class InteractionRequest(BaseModel):
    """User interaction logging"""
    user_id: int
    post_id: int
    action: str  # like, comment, share, save, view, hide, report


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    components: dict


class MetricsResponse(BaseModel):
    """Metrics response"""
    total_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Recommendation API",
    description="Social network recommendation system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize pipeline on startup
    
    FIXED: Load config from config file, not from individual arguments
    """
    global pipeline
    
    try:
        logger.info("="*70)
        logger.info("🚀 STARTING RECOMMENDATION API")
        logger.info("="*70)
        
        # Get configuration paths from environment or use defaults
        config_path = os.getenv('CONFIG_PATH', 'configs/config_online.yaml')
        models_dir = os.getenv('MODELS_DIR', 'models')
        data_dir = os.getenv('DATA_DIR', 'dataset')
        
        # Check if Redis is available (from environment)
        redis_host = os.getenv('REDIS_HOST')
        use_redis = redis_host is not None
        
        logger.info(f"Config path: {config_path}")
        logger.info(f"Models dir: {models_dir}")
        logger.info(f"Data dir: {data_dir}")
        logger.info(f"Redis: {'enabled' if use_redis else 'disabled'}")
        
        # Initialize pipeline with CORRECT arguments
        # The pipeline reads Redis config from config_online.yaml
        pipeline = OnlineInferencePipeline(
            config_path=config_path,
            models_dir=models_dir,
            data_dir=data_dir,
            use_redis=use_redis
        )
        
        logger.info("✅ Pipeline initialized successfully!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}", exc_info=True)
        # Don't exit - let health check report the issue
        pipeline = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down API server")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Recommendation API",
        "version": "1.0.0",
        "status": "running" if pipeline is not None else "error",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check() -> HealthResponse:
    """
    Health check endpoint
    
    Returns system health status
    """
    components = {
        "pipeline": pipeline is not None,
        "redis": False,
        "models": False
    }
    
    if pipeline is not None:
        components["redis"] = pipeline.redis is not None
        components["models"] = pipeline.ranking_model is not None
    
    status = "healthy" if all(components.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        components=components
    )


@app.get("/metrics")
async def get_metrics() -> MetricsResponse:
    """
    Get system metrics
    
    Returns performance metrics
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    metrics = pipeline.get_metrics()
    
    return MetricsResponse(
        total_requests=metrics.get('total_requests', 0),
        avg_latency_ms=metrics.get('avg_latency_ms', 0),
        p50_latency_ms=metrics.get('p50_latency_ms', 0),
        p95_latency_ms=metrics.get('p95_latency_ms', 0),
        p99_latency_ms=metrics.get('p99_latency_ms', 0),
        error_rate=0.0  # TODO: Track errors
    )


@app.get("/version")
async def get_version():
    """Get current model version"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "version": pipeline.current_version,
        "metadata": pipeline.metadata
    }


@app.get("/feed")
async def get_feed(
    user_id: int = Query(..., description="User ID"),
    limit: int = Query(50, ge=1, le=100, description="Number of posts"),
    exclude_seen: Optional[str] = Query(None, description="Comma-separated post IDs to exclude")
):
    """
    Get personalized feed for user
    
    Args:
        user_id: Target user ID
        limit: Number of posts to return (1-100)
        exclude_seen: Comma-separated post IDs to exclude (e.g., "1,2,3")
    
    Returns:
        List of recommended posts with scores
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Parse exclude_seen
        exclude_list = None
        if exclude_seen:
            try:
                exclude_list = [int(x.strip()) for x in exclude_seen.split(',')]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid exclude_seen format")
        
        # Generate feed
        feed = pipeline.generate_feed(
            user_id=user_id,
            limit=limit,
            exclude_seen=exclude_list
        )
        
        return {
            "user_id": user_id,
            "posts": feed,
            "count": len(feed),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/friends")
async def get_friend_recommendations(
    user_id: int = Query(..., description="User ID"),
    limit: int = Query(20, ge=1, le=50, description="Number of recommendations")
):
    """
    Get friend recommendations for user
    
    Args:
        user_id: Target user ID
        limit: Number of recommendations (1-50)
    
    Returns:
        List of recommended friends with scores
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        recommendations = pipeline.recommend_friends(
            user_id=user_id,
            k=limit
        )
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error recommending friends for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interaction")
async def log_interaction(request: InteractionRequest):
    """
    Log user interaction and update user embedding in real-time
    
    Args:
        request: Interaction details (user_id, post_id, action)
    
    Returns:
        Success status
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Validate action
        valid_actions = ['like', 'comment', 'share', 'save', 'view', 'hide', 'report']
        if request.action not in valid_actions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action. Must be one of: {', '.join(valid_actions)}"
            )
        
        # Update user embedding (real-time)
        pipeline.update_user_embedding_realtime(
            user_id=request.user_id,
            post_id=request.post_id,
            action=request.action
        )
        
        # TODO: Log to database for tracking
        
        return {
            "success": True,
            "message": f"Interaction logged: user {request.user_id} {request.action}d post {request.post_id}",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging interaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trending")
async def get_trending(
    limit: int = Query(100, ge=1, le=200, description="Number of trending posts")
):
    """
    Get trending posts (not personalized)
    
    Args:
        limit: Number of posts (1-200)
    
    Returns:
        List of trending posts
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        trending_posts = pipeline.trending_recall.recall(k=limit)
        
        return {
            "posts": trending_posts,
            "count": len(trending_posts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting trending posts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DEBUG ENDPOINTS (Remove in production)
# ============================================================================

@app.get("/debug/components")
async def debug_components():
    """Debug: Check loaded components"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "embeddings": {
            "post_count": len(pipeline.embeddings.get('post', {})),
            "user_count": len(pipeline.embeddings.get('user', {}))
        },
        "faiss_index": pipeline.faiss_index is not None,
        "faiss_posts": len(pipeline.faiss_post_ids),
        "cf_model": pipeline.cf_model is not None,
        "ranking_model": pipeline.ranking_model is not None,
        "redis": pipeline.redis is not None,
        "recall_channels": {
            "following": pipeline.following_recall is not None,
            "cf": pipeline.cf_recall is not None,
            "content": pipeline.content_recall is not None,
            "trending": pipeline.trending_recall is not None
        }
    }


@app.get("/debug/user/{user_id}")
async def debug_user(user_id: int):
    """Debug: Check user data"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "user_id": user_id,
        "has_embedding": user_id in pipeline.embeddings.get('user', {}),
        "has_stats": user_id in pipeline.user_stats,
        "following_count": len(pipeline.following_dict.get(user_id, []))
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Get config from environment
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8010))
    
    print(f"\n🚀 Starting API server on {host}:{port}\n")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )