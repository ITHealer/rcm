"""
FASTAPI SERVER FOR ONLINE INFERENCE
====================================
REST API endpoints for recommendation service

Endpoints:
- GET /feed - Get personalized feed
- GET /health - Health check
- GET /metrics - Performance metrics
- GET /version - Model version info

Usage: uvicorn api:app --reload
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))
# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️  fastapi not available. Install: pip install fastapi uvicorn")

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import pipeline
from recommender.online.online_inference_pipeline import OnlineInferencePipeline


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ════════════════════════════════════════════════════════════════════════

class FeedRequest(BaseModel):
    """Request model for feed generation"""
    user_id: int = Field(..., description="User ID", example=123)
    limit: int = Field(50, description="Number of posts", ge=1, le=100)
    exclude_seen: Optional[List[int]] = Field(
        None,
        description="Post IDs to exclude"
    )


class FeedResponse(BaseModel):
    """Response model for feed"""
    user_id: int
    posts: List[Dict]
    count: int
    latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_version: str
    redis_connected: bool
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Performance metrics response"""
    total_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


# ════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ════════════════════════════════════════════════════════════════════════

if not FASTAPI_AVAILABLE:
    raise ImportError("FastAPI required. Install: pip install fastapi uvicorn")

app = FastAPI(
    title="Recommendation API",
    description="Online recommendation service with multi-channel recall and ML ranking",
    version="1.0.0"
)

# Global pipeline instance
pipeline = None
start_time = datetime.now()


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline
    
    logger.info("="*70)
    logger.info("STARTING RECOMMENDATION API")
    logger.info("="*70)
    
    try:
        # Initialize pipeline
        pipeline = OnlineInferencePipeline(
            models_dir=os.getenv(r'D:\hongthai\projects\wayjet_recommendation\recommendation_wayjet\models\v20251024_120504', 'models'),
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', 6380)),
            redis_db=int(os.getenv('REDIS_DB', 0)),
            use_redis=os.getenv('USE_REDIS', 'true').lower() == 'true'
        )
        
        logger.info("✅ API READY!")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("\n" + "="*70)
    logger.info("SHUTTING DOWN API")
    logger.info("="*70)
    
    # Print final metrics
    if pipeline:
        pipeline.print_metrics()
    
    logger.info("✅ API SHUTDOWN COMPLETE")
    logger.info("="*70 + "\n")


# ════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "feed": "/feed?user_id=123&limit=50",
            "health": "/health",
            "metrics": "/metrics",
            "version": "/version"
        }
    }


@app.get("/feed", response_model=FeedResponse)
async def get_feed(
    user_id: int = Query(..., description="User ID", example=123),
    limit: int = Query(50, description="Number of posts", ge=1, le=100),
    exclude_seen: Optional[str] = Query(
        None,
        description="Comma-separated post IDs to exclude",
        example="1,2,3"
    )
):
    """
    Get personalized feed for user
    
    Args:
        user_id: Target user ID
        limit: Number of posts (1-100)
        exclude_seen: Comma-separated post IDs to exclude
    
    Returns:
        Personalized feed with posts ranked by relevance
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Pipeline not initialized."
        )
    
    try:
        # Parse exclude_seen
        exclude_set = None
        if exclude_seen:
            exclude_set = {int(pid) for pid in exclude_seen.split(',')}
        
        # Generate feed
        import time
        start = time.time()
        
        feed = pipeline.get_feed(
            user_id=user_id,
            limit=limit,
            exclude_seen=exclude_set
        )
        
        latency = (time.time() - start) * 1000
        
        # Response
        return FeedResponse(
            user_id=user_id,
            posts=feed,
            count=len(feed),
            latency_ms=latency,
            model_version=pipeline.current_version
        )
        
    except Exception as e:
        logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/feed", response_model=FeedResponse)
async def get_feed_post(request: FeedRequest):
    """
    Get personalized feed (POST version)
    
    Body:
        {
            "user_id": 123,
            "limit": 50,
            "exclude_seen": [1, 2, 3]
        }
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Pipeline not initialized."
        )
    
    try:
        # Generate feed
        import time
        start = time.time()
        
        exclude_set = set(request.exclude_seen) if request.exclude_seen else None
        
        feed = pipeline.get_feed(
            user_id=request.user_id,
            limit=request.limit,
            exclude_seen=exclude_set
        )
        
        latency = (time.time() - start) * 1000
        
        return FeedResponse(
            user_id=request.user_id,
            posts=feed,
            count=len(feed),
            latency_ms=latency,
            model_version=pipeline.current_version
        )
        
    except Exception as e:
        logger.error(f"Error generating feed for user {request.user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Service health status
    """
    if pipeline is None:
        return HealthResponse(
            status="starting",
            model_version="unknown",
            redis_connected=False,
            uptime_seconds=(datetime.now() - start_time).total_seconds()
        )
    
    # Check Redis
    redis_connected = False
    if pipeline.redis_client:
        try:
            pipeline.redis_client.ping()
            redis_connected = True
        except:
            pass
    
    return HealthResponse(
        status="healthy",
        model_version=pipeline.current_version,
        redis_connected=redis_connected,
        uptime_seconds=(datetime.now() - start_time).total_seconds()
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get performance metrics
    
    Returns:
        Latency statistics
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready."
        )
    
    metrics_summary = pipeline.get_metrics_summary()
    
    if not metrics_summary or 'total_latency' not in metrics_summary:
        return MetricsResponse(
            total_requests=0,
            avg_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0
        )
    
    total_stats = metrics_summary['total_latency']
    
    return MetricsResponse(
        total_requests=len(pipeline.metrics['total_latency']),
        avg_latency_ms=total_stats['mean'],
        p50_latency_ms=total_stats['p50'],
        p95_latency_ms=total_stats['p95'],
        p99_latency_ms=total_stats['p99']
    )


@app.get("/version")
async def get_version():
    """
    Get model version info
    
    Returns:
        Model version and metadata
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready."
        )
    
    return {
        "model_version": pipeline.current_version,
        "metadata": pipeline.metadata,
        "artifacts": {
            "post_embeddings": len(pipeline.embeddings['post']),
            "user_embeddings": len(pipeline.embeddings['user']),
            "cf_users": len(pipeline.cf_model['user_ids']),
            "faiss_vectors": pipeline.faiss_index.ntotal,
            "ranking_features": len(pipeline.ranking_feature_cols)
        }
    }


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Run API server
    
    Usage:
        python api.py
        
    Or:
        uvicorn api:app --reload --host 0.0.0.0 --port 8000
    """
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8010,
        reload=True,
        log_level="info"
    )