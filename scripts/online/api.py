# """
# FASTAPI APPLICATION - RECOMMENDATION API
# ========================================
# RESTful API for recommendation system

# Endpoints:
# - GET /feed - Get personalized feed
# - GET /friends - Get friend recommendations
# - POST /interaction - Log user interaction
# - GET /trending - Get trending posts
# - GET /health - Health check
# - GET /metrics - System metrics
# """

# import os
# import sys
# from pathlib import Path
# from typing import List, Optional
# from datetime import datetime
# import logging

# # Add project root
# sys.path.append(str(Path(__file__).parent.parent.parent))

# # FastAPI imports
# from fastapi import FastAPI, HTTPException, Query
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# # Import pipeline
# from recommender.online.online_inference_pipeline import OnlineInferencePipeline

# # Setup logging
# logging.basicConfig(
#     level=os.getenv('LOG_LEVEL', 'INFO'),
#     format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # ============================================================================
# # PYDANTIC MODELS
# # ============================================================================

# class FeedRequest(BaseModel):
#     """Feed request model"""
#     user_id: int
#     limit: int = 50
#     exclude_seen: Optional[List[int]] = None


# class FriendRecommendationRequest(BaseModel):
#     """Friend recommendation request"""
#     user_id: int
#     limit: int = 20


# class InteractionRequest(BaseModel):
#     """User interaction logging"""
#     user_id: int
#     post_id: int
#     action: str  # like, comment, share, save, view, hide, report


# class HealthResponse(BaseModel):
#     """Health check response"""
#     status: str
#     timestamp: str
#     version: str
#     components: dict


# class MetricsResponse(BaseModel):
#     """Metrics response"""
#     total_requests: int
#     avg_latency_ms: float
#     p50_latency_ms: float
#     p95_latency_ms: float
#     p99_latency_ms: float
#     error_rate: float


# # ============================================================================
# # FASTAPI APP
# # ============================================================================

# app = FastAPI(
#     title="Recommendation API",
#     description="Social network recommendation system",
#     version="1.0.0"
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure properly in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global pipeline instance
# pipeline = None

# # ============================================================================
# # STARTUP & SHUTDOWN
# # ============================================================================

# @app.on_event("startup")
# async def startup_event():
#     """
#     Initialize pipeline on startup
    
#     FIXED: Load config from config file, not from individual arguments
#     """
#     global pipeline
    
#     try:
#         logger.info("="*70)
#         logger.info("🚀 STARTING RECOMMENDATION API")
#         logger.info("="*70)
        
#         # Get configuration paths from environment or use defaults
#         config_path = os.getenv('CONFIG_PATH', 'configs/config_online.yaml')
#         models_dir = os.getenv('MODELS_DIR', 'models')
#         data_dir = os.getenv('DATA_DIR', 'dataset')
        
#         # Check if Redis is available (from environment)
#         redis_host = os.getenv('REDIS_HOST')
#         use_redis = redis_host is not None
        
#         logger.info(f"Config path: {config_path}")
#         logger.info(f"Models dir: {models_dir}")
#         logger.info(f"Data dir: {data_dir}")
#         logger.info(f"Redis: {'enabled' if use_redis else 'disabled'}")
        
#         # Initialize pipeline with CORRECT arguments
#         # The pipeline reads Redis config from config_online.yaml
#         pipeline = OnlineInferencePipeline(
#             config_path=config_path,
#             models_dir=models_dir,
#             data_dir=data_dir,
#             use_redis=use_redis
#         )
        
#         logger.info("✅ Pipeline initialized successfully!")
#         logger.info("="*70)
        
#     except Exception as e:
#         logger.error(f"❌ Failed to initialize pipeline: {e}", exc_info=True)
#         # Don't exit - let health check report the issue
#         pipeline = None


# @app.on_event("shutdown")
# async def shutdown_event():
#     """Cleanup on shutdown"""
#     logger.info("👋 Shutting down API server")


# # ============================================================================
# # ENDPOINTS
# # ============================================================================

# @app.get("/")
# async def root():
#     """Root endpoint"""
#     return {
#         "message": "Recommendation API",
#         "version": "1.0.0",
#         "status": "running" if pipeline is not None else "error",
#         "docs": "/docs"
#     }


# @app.get("/health")
# async def health_check() -> HealthResponse:
#     """
#     Health check endpoint
    
#     Returns system health status
#     """
#     components = {
#         "pipeline": pipeline is not None,
#         "redis": False,
#         "models": False
#     }
    
#     if pipeline is not None:
#         components["redis"] = pipeline.redis is not None
#         components["models"] = pipeline.ranking_model is not None
    
#     status = "healthy" if all(components.values()) else "degraded"
    
#     return HealthResponse(
#         status=status,
#         timestamp=datetime.now().isoformat(),
#         version="1.0.0",
#         components=components
#     )


# @app.get("/metrics")
# async def get_metrics() -> MetricsResponse:
#     """
#     Get system metrics
    
#     Returns performance metrics
#     """
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
#     metrics = pipeline.get_metrics()
    
#     return MetricsResponse(
#         total_requests=metrics.get('total_requests', 0),
#         avg_latency_ms=metrics.get('avg_latency_ms', 0),
#         p50_latency_ms=metrics.get('p50_latency_ms', 0),
#         p95_latency_ms=metrics.get('p95_latency_ms', 0),
#         p99_latency_ms=metrics.get('p99_latency_ms', 0),
#         error_rate=0.0  # TODO: Track errors
#     )


# @app.get("/version")
# async def get_version():
#     """Get current model version"""
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
#     return {
#         "version": pipeline.current_version,
#         "metadata": pipeline.metadata
#     }


# @app.get("/feed")
# async def get_feed(
#     user_id: int = Query(..., description="User ID"),
#     limit: int = Query(50, ge=1, le=100, description="Number of posts"),
#     exclude_seen: Optional[str] = Query(None, description="Comma-separated post IDs to exclude")
# ):
#     """
#     Get personalized feed for user
    
#     Args:
#         user_id: Target user ID
#         limit: Number of posts to return (1-100)
#         exclude_seen: Comma-separated post IDs to exclude (e.g., "1,2,3")
    
#     Returns:
#         List of recommended posts with scores
#     """
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
#     try:
#         # Parse exclude_seen
#         exclude_list = None
#         if exclude_seen:
#             try:
#                 exclude_list = [int(x.strip()) for x in exclude_seen.split(',')]
#             except ValueError:
#                 raise HTTPException(status_code=400, detail="Invalid exclude_seen format")
        
#         # Generate feed
#         feed = pipeline.generate_feed(
#             user_id=user_id,
#             limit=limit,
#             exclude_seen=exclude_list
#         )
        
#         return {
#             "user_id": user_id,
#             "posts": feed,
#             "count": len(feed),
#             "timestamp": datetime.now().isoformat()
#         }
        
#     except Exception as e:
#         logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/friends")
# async def get_friend_recommendations(
#     user_id: int = Query(..., description="User ID"),
#     limit: int = Query(20, ge=1, le=50, description="Number of recommendations")
# ):
#     """
#     Get friend recommendations for user
    
#     Args:
#         user_id: Target user ID
#         limit: Number of recommendations (1-50)
    
#     Returns:
#         List of recommended friends with scores
#     """
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
#     try:
#         recommendations = pipeline.recommend_friends(
#             user_id=user_id,
#             k=limit
#         )
        
#         return {
#             "user_id": user_id,
#             "recommendations": recommendations,
#             "count": len(recommendations),
#             "timestamp": datetime.now().isoformat()
#         }
        
#     except Exception as e:
#         logger.error(f"Error recommending friends for user {user_id}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/interaction")
# async def log_interaction(request: InteractionRequest):
#     """
#     Log user interaction and update user embedding in real-time
    
#     Args:
#         request: Interaction details (user_id, post_id, action)
    
#     Returns:
#         Success status
#     """
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
#     try:
#         # Validate action
#         valid_actions = ['like', 'comment', 'share', 'save', 'view', 'hide', 'report']
#         if request.action not in valid_actions:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Invalid action. Must be one of: {', '.join(valid_actions)}"
#             )
        
#         # Update user embedding (real-time)
#         pipeline.update_user_embedding_realtime(
#             user_id=request.user_id,
#             post_id=request.post_id,
#             action=request.action
#         )
        
#         # TODO: Log to database for tracking
        
#         return {
#             "success": True,
#             "message": f"Interaction logged: user {request.user_id} {request.action}d post {request.post_id}",
#             "timestamp": datetime.now().isoformat()
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error logging interaction: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/trending")
# async def get_trending(
#     limit: int = Query(100, ge=1, le=200, description="Number of trending posts")
# ):
#     """
#     Get trending posts (not personalized)
    
#     Args:
#         limit: Number of posts (1-200)
    
#     Returns:
#         List of trending posts
#     """
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
#     try:
#         trending_posts = pipeline.trending_recall.recall(k=limit)
        
#         return {
#             "posts": trending_posts,
#             "count": len(trending_posts),
#             "timestamp": datetime.now().isoformat()
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting trending posts: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))


# # ============================================================================
# # DEBUG ENDPOINTS (Remove in production)
# # ============================================================================

# @app.get("/debug/components")
# async def debug_components():
#     """Debug: Check loaded components"""
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
#     return {
#         "embeddings": {
#             "post_count": len(pipeline.embeddings.get('post', {})),
#             "user_count": len(pipeline.embeddings.get('user', {}))
#         },
#         "faiss_index": pipeline.faiss_index is not None,
#         "faiss_posts": len(pipeline.faiss_post_ids),
#         "cf_model": pipeline.cf_model is not None,
#         "ranking_model": pipeline.ranking_model is not None,
#         "redis": pipeline.redis is not None,
#         "recall_channels": {
#             "following": pipeline.following_recall is not None,
#             "cf": pipeline.cf_recall is not None,
#             "content": pipeline.content_recall is not None,
#             "trending": pipeline.trending_recall is not None
#         }
#     }


# @app.get("/debug/user/{user_id}")
# async def debug_user(user_id: int):
#     """Debug: Check user data"""
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
#     return {
#         "user_id": user_id,
#         "has_embedding": user_id in pipeline.embeddings.get('user', {}),
#         "has_stats": user_id in pipeline.user_stats,
#         "following_count": len(pipeline.following_dict.get(user_id, []))
#     }


# # ============================================================================
# # ERROR HANDLERS
# # ============================================================================

# @app.exception_handler(Exception)
# async def global_exception_handler(request, exc):
#     """Global exception handler"""
#     logger.error(f"Unhandled exception: {exc}", exc_info=True)
#     return JSONResponse(
#         status_code=500,
#         content={
#             "error": "Internal server error",
#             "detail": str(exc),
#             "timestamp": datetime.now().isoformat()
#         }
#     )


# # ============================================================================
# # MAIN (for testing)
# # ============================================================================

# if __name__ == "__main__":
#     import uvicorn
    
#     # Get config from environment
#     host = os.getenv('API_HOST', '0.0.0.0')
#     port = int(os.getenv('API_PORT', 8010))
    
#     print(f"\n🚀 Starting API server on {host}:{port}\n")
    
#     uvicorn.run(
#         "api:app",
#         host=host,
#         port=port,
#         reload=True,
#         log_level="info"
#     )

# """
# FASTAPI APPLICATION - RECOMMENDATION API
# ========================================
# RESTful API for recommendation system

# Endpoints:
# - GET /feed - Get personalized feed
# - GET /friends - Get friend recommendations
# - POST /interaction - Log user interaction
# - GET /trending - Get trending posts
# - GET /health - Health check
# - GET /metrics - System metrics
# """

# import os
# import sys
# from pathlib import Path
# from typing import List, Optional
# from datetime import datetime
# import time
# import logging

# # Add project root
# sys.path.append(str(Path(__file__).parent.parent.parent))

# # FastAPI imports
# from fastapi import FastAPI, HTTPException, Query
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# # Import pipeline
# from recommender.online.online_inference_pipeline import OnlineInferencePipeline
# # NEW: telemetry async logger (non-blocking)
# from recommender.online.telemetry import AsyncDBLogger
# from recommender.online.realtime_ingestor import BackendIngestor

# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=os.getenv('LOG_LEVEL', 'INFO'),
#     format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
# )

# # ============================================================================
# # PYDANTIC MODELS
# # ============================================================================

# class FeedRequest(BaseModel):
#     user_id: int
#     limit: int = 50
#     exclude_seen: Optional[List[int]] = None

# class FriendRecommendationRequest(BaseModel):
#     user_id: int
#     limit: int = 20

# class InteractionRequest(BaseModel):
#     user_id: int
#     post_id: int
#     action: str  # like, comment, share, save, view, hide, report
#     session_id: Optional[str] = None
#     meta: Optional[dict] = None

# class HealthResponse(BaseModel):
#     status: str
#     timestamp: str
#     version: str
#     components: dict

# class MetricsResponse(BaseModel):
#     total_requests: int
#     avg_latency_ms: float
#     p50_latency_ms: float
#     p95_latency_ms: float
#     p99_latency_ms: float
#     error_rate: float

# # ============================================================================
# # FASTAPI APP
# # ============================================================================

# app = FastAPI(
#     title="Recommendation API",
#     description="Social network recommendation system",
#     version="1.0.0"
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure properly in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global pipeline & telemetry
# pipeline = None
# app.state.telemetry = None

# # ============================================================================
# # STARTUP & SHUTDOWN
# # ============================================================================

# @app.on_event("startup")
# async def startup_event():
#     """
#     Initialize pipeline on startup
#     """
#     global pipeline
#     try:
#         logger.info("="*70)
#         logger.info("🚀 STARTING RECOMMENDATION API")
#         logger.info("="*70)

#         config_path = os.getenv('CONFIG_PATH', 'configs/config_online.yaml')
#         models_dir = os.getenv('MODELS_DIR', 'models')
#         data_dir   = os.getenv('DATA_DIR', 'dataset')

#         redis_host = os.getenv('REDIS_HOST')
#         use_redis = redis_host is not None

#         logger.info(f"Config path: {config_path}")
#         logger.info(f"Models dir: {models_dir}")
#         logger.info(f"Data dir: {data_dir}")
#         logger.info(f"Redis: {'enabled' if use_redis else 'disabled'}")

#         pipeline = OnlineInferencePipeline(
#             config_path=config_path,
#             models_dir=models_dir,
#             data_dir=data_dir,
#             use_redis=use_redis
#         )
#         logger.info("✅ Pipeline initialized successfully!")
#         try:
#             ingest_cfg = pipeline.config.get("realtime_ingest", {}) or {}
#             app.state.ingestor = BackendIngestor(
#                 cfg=ingest_cfg,
#                 mysql_engine=pipeline.db_engine,       # backend MySQL (đã init ở pipeline)
#                 redis_client=pipeline.redis,
#                 pipeline=pipeline                       # để update embedding optional
#             )
#             app.state.ingestor.start()
#         except Exception as e:
#             logger.warning("Failed to start realtime ingestor: %s", e)

#         # NEW: init AsyncDBLogger (không block request)
#         try:
#             tel_cfg = pipeline.config.get("telemetry", {}) if pipeline and hasattr(pipeline, "config") else {}
#             app.state.telemetry = AsyncDBLogger(tel_cfg, redis_client=pipeline.redis)
#             logger.info("✅ Telemetry logger initialized")
#         except Exception as e:
#             logger.warning(f"Telemetry init failed: {e}")
#             app.state.telemetry = None

#         logger.info("="*70)
#     except Exception as e:
#         logger.error(f"❌ Failed to initialize pipeline: {e}", exc_info=True)
#         pipeline = None

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Cleanup on shutdown"""
#     logger.info("👋 Shutting down API server")
#     if getattr(app.state, "telemetry", None):
#         try:
#             app.state.telemetry.shutdown()
#         except Exception:
#             pass

# # ============================================================================
# # ENDPOINTS
# # ============================================================================

# @app.get("/")
# async def root():
#     return {
#         "message": "Recommendation API",
#         "version": "1.0.0",
#         "status": "running" if pipeline is not None else "error",
#         "docs": "/docs"
#     }

# @app.get("/health")
# async def health_check() -> HealthResponse:
#     components = {
#         "pipeline": pipeline is not None,
#         "redis": False,
#         "models": False
#     }
#     if pipeline is not None:
#         components["redis"] = pipeline.redis is not None
#         components["models"] = pipeline.ranking_model is not None
#     status = "healthy" if all(components.values()) else "degraded"
#     return HealthResponse(
#         status=status,
#         timestamp=datetime.now().isoformat(),
#         version="1.0.0",
#         components=components
#     )

# @app.get("/metrics")
# async def get_metrics() -> MetricsResponse:
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     metrics = pipeline.get_metrics()
#     return MetricsResponse(
#         total_requests=metrics.get('total_requests', 0),
#         avg_latency_ms=metrics.get('avg_latency_ms', 0),
#         p50_latency_ms=metrics.get('p50_latency_ms', 0),
#         p95_latency_ms=metrics.get('p95_latency_ms', 0),
#         p99_latency_ms=metrics.get('p99_latency_ms', 0),
#         error_rate=0.0  # TODO: Track errors
#     )

# @app.get("/version")
# async def get_version():
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     return {"version": pipeline.current_version, "metadata": pipeline.metadata}

# @app.get("/feed")
# async def get_feed(
#     user_id: int = Query(..., description="User ID"),
#     limit: int = Query(50, ge=1, le=100, description="Number of posts"),
#     exclude_seen: Optional[str] = Query(None, description="Comma-separated post IDs to exclude")
# ):
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     try:
#         # Parse exclude_seen
#         exclude_list = None
#         if exclude_seen:
#             try:
#                 exclude_list = [int(x.strip()) for x in exclude_seen.split(',')]
#             except ValueError:
#                 raise HTTPException(status_code=400, detail="Invalid exclude_seen format")

#         t0 = time.time()
#         feed = pipeline.generate_feed(user_id=user_id, limit=limit, exclude_seen=exclude_list)
#         t1 = time.time()

#         # NEW: non-blocking telemetry (ghi DB ở background)
#         tel = getattr(app.state, "telemetry", None)
#         if tel:
#             # các field stage_ms lấy từ metrics trong pipeline
#             stage_ms = {
#                 "recall": int(pipeline.metrics['recall_latency'][-1]) if pipeline.metrics['recall_latency'] else 0,
#                 "ranking": int(pipeline.metrics['ranking_latency'][-1]) if pipeline.metrics['ranking_latency'] else 0,
#                 "rerank": int(pipeline.metrics['reranking_latency'][-1]) if pipeline.metrics['reranking_latency'] else 0,
#             }
#             recall_diag = getattr(pipeline, "_last_recall_diag", None)
#             tel.log_feed_request({
#                 "user_id": user_id,
#                 "timestamp": t1,
#                 "recall": {
#                     "following": getattr(recall_diag, "following", 0) if recall_diag else 0,
#                     "cf": getattr(recall_diag, "cf", 0) if recall_diag else 0,
#                     "content": getattr(recall_diag, "content", 0) if recall_diag else 0,
#                     "trending": getattr(recall_diag, "trending", 0) if recall_diag else 0,
#                     "total": getattr(recall_diag, "total", 0) if recall_diag else 0,
#                 },
#                 "final_count": len(feed or []),
#                 "reasons": getattr(pipeline, "_last_no_feed_reasons", []),
#                 "latency_ms": int((t1 - t0) * 1000),
#                 "stage_ms": stage_ms
#             })

#         return {
#             "user_id": user_id,
#             "posts": feed,
#             "count": len(feed),
#             "timestamp": datetime.now().isoformat()
#         }
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/friends")
# async def get_friend_recommendations(
#     user_id: int = Query(..., description="User ID"),
#     limit: int = Query(20, ge=1, le=50, description="Number of recommendations")
# ):
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     try:
#         recommendations = pipeline.recommend_friends(user_id=user_id, k=limit)
#         return {
#             "user_id": user_id,
#             "recommendations": recommendations,
#             "count": len(recommendations),
#             "timestamp": datetime.now().isoformat()
#         }
#     except Exception as e:
#         logger.error(f"Error recommending friends for user {user_id}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/interaction")
# async def log_interaction(request: InteractionRequest):
#     """
#     Log user interaction and update user embedding in real-time
#     + mirror realtime signals to Redis (ranking realtime features)
#     + update co-visit (CF realtime nhẹ)
#     """
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     try:
#         valid_actions = ['like', 'comment', 'share', 'save', 'view', 'hide', 'report']
#         if request.action not in valid_actions:
#             raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {', '.join(valid_actions)}")

#         # 1) Update user embedding realtime (Content-based phản ứng ngay)
#         pipeline.update_user_embedding_realtime(
#             user_id=request.user_id,
#             post_id=request.post_id,
#             action=request.action
#         )

#         # 2) Mirror realtime counters to Redis (ranking cảm nhận tương tác mới)
#         r = pipeline.redis
#         if r is not None:
#             try:
#                 # user 24h counters
#                 r.hincrby(f"user:{request.user_id}:engagement_24h", request.action, 1)
#                 r.expire(f"user:{request.user_id}:engagement_24h", 24*3600)

#                 # post 1h counters
#                 r.hincrby(f"post:{request.post_id}:engagement_1h", request.action, 1)
#                 r.expire(f"post:{request.post_id}:engagement_1h", 3600)

#                 # author 1h counters
#                 author_id = pipeline.get_author_id(request.post_id)
#                 if author_id:
#                     r.hincrby(f"author:{author_id}:engagement_1h", request.action, 1)
#                     r.expire(f"author:{author_id}:engagement_1h", 3600)

#                 # recent items (phục vụ CF realtime nhẹ)
#                 r.lpush(f"user:{request.user_id}:recent_items", request.post_id)
#                 r.ltrim(f"user:{request.user_id}:recent_items", 0, 99)
#                 r.expire(f"user:{request.user_id}:recent_items", 7*24*3600)

#                 # co-visit (CF realtime nhẹ): tăng điểm đồng xem/đồng tương tác
#                 recent = r.lrange(f"user:{request.user_id}:recent_items", 0, 50) or []
#                 for rid in recent:
#                     try:
#                         rid = int(rid)
#                         if rid == request.post_id:
#                             continue
#                         r.zincrby(f"covisit:{rid}", 1.0, request.post_id)
#                         r.zincrby(f"covisit:{request.post_id}", 1.0, rid)
#                         r.expire(f"covisit:{rid}", 7*24*3600)
#                         r.expire(f"covisit:{request.post_id}", 7*24*3600)
#                     except Exception:
#                         pass
#             except Exception as e:
#                 logger.debug(f"Realtime mirror to Redis failed: {e}")

#         # 3) Telemetry (non-blocking DB logging)
#         tel = getattr(app.state, "telemetry", None)
#         if tel:
#             try:
#                 tel.log_interaction({
#                     "user_id": request.user_id,
#                     "post_id": request.post_id,
#                     "author_id": author_id if 'author_id' in locals() else None,
#                     "action": request.action,
#                     "timestamp": time.time(),
#                     "session_id": request.session_id,
#                     "meta": request.meta or {}
#                 })
#             except Exception as e:
#                 logger.debug(f"telemetry log_interaction failed: {e}")

#         return {
#             "success": True,
#             "message": f"Interaction logged: user {request.user_id} {request.action}d post {request.post_id}",
#             "timestamp": datetime.now().isoformat()
#         }
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error logging interaction: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/trending")
# async def get_trending(
#     limit: int = Query(100, ge=1, le=200, description="Number of trending posts")
# ):
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     try:
#         trending_posts = pipeline.trending_recall.recall(k=limit)
#         return {
#             "posts": trending_posts,
#             "count": len(trending_posts),
#             "timestamp": datetime.now().isoformat()
#         }
#     except Exception as e:
#         logger.error(f"Error getting trending posts: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

# # ============================================================================
# # DEBUG ENDPOINTS (Remove in production)
# # ============================================================================

# @app.get("/debug/components")
# async def debug_components():
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     return {
#         "embeddings": {
#             "post_count": len(pipeline.embeddings.get('post', {})),
#             "user_count": len(pipeline.embeddings.get('user', {}))
#         },
#         "faiss_index": pipeline.faiss_index is not None,
#         "faiss_posts": len(pipeline.faiss_post_ids),
#         "cf_model": pipeline.cf_model is not None,
#         "ranking_model": pipeline.ranking_model is not None,
#         "redis": pipeline.redis is not None,
#         "recall_channels": {
#             "following": pipeline.following_recall is not None,
#             "cf": pipeline.cf_recall is not None,
#             "content": pipeline.content_recall is not None,
#             "trending": pipeline.trending_recall is not None
#         }
#     }

# @app.get("/debug/user/{user_id}")
# async def debug_user(user_id: int):
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     return {
#         "user_id": user_id,
#         "has_embedding": user_id in pipeline.embeddings.get('user', {}),
#         "has_stats": user_id in pipeline.user_stats,
#         "following_count": len(pipeline.following_dict.get(user_id, []))
#     }

# # ============================================================================
# # ERROR HANDLERS
# # ============================================================================

# @app.exception_handler(Exception)
# async def global_exception_handler(request, exc):
#     logger.error(f"Unhandled exception: {exc}", exc_info=True)
#     return JSONResponse(
#         status_code=500,
#         content={
#             "error": "Internal server error",
#             "detail": str(exc),
#             "timestamp": datetime.now().isoformat()
#         }
#     )

# # ============================================================================
# # MAIN (for testing)
# # ============================================================================

# if __name__ == "__main__":
#     import uvicorn
#     host = os.getenv('API_HOST', '0.0.0.0')
#     port = int(os.getenv('API_PORT', 8010))
#     print(f"\n🚀 Starting API server on {host}:{port}\n")
#     uvicorn.run("api:app", host=host, port=port, reload=True, log_level="info")





# """
# FASTAPI APPLICATION - RECOMMENDATION API
# ========================================
# """

# from __future__ import annotations
# import os
# import sys
# import time
# from pathlib import Path
# from typing import List, Optional
# from datetime import datetime
# import logging

# # Add project root
# sys.path.append(str(Path(__file__).parent.parent.parent))

# # FastAPI imports
# from fastapi import FastAPI, HTTPException, Query
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel

# from dotenv import load_dotenv
# load_dotenv()

# from recommender.online.online_inference_pipeline import OnlineInferencePipeline
# from recommender.online.telemetry import AsyncDBLogger
# from recommender.online.realtime_ingestor import BackendIngestor

# logging.basicConfig(
#     level=os.getenv('LOG_LEVEL', 'INFO'),
#     format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # --------------------- Pydantic models ----------------------

# class InteractionRequest(BaseModel):
#     user_id: int
#     post_id: int
#     action: str                     # like, comment, share, save, view, hide, report
#     session_id: Optional[str] = None
#     meta: Optional[dict] = None

# class HealthResponse(BaseModel):
#     status: str
#     timestamp: str
#     version: str
#     components: dict

# class MetricsResponse(BaseModel):
#     total_requests: int
#     avg_latency_ms: float
#     p50_latency_ms: float
#     p95_latency_ms: float
#     p99_latency_ms: float
#     error_rate: float

# # --------------------- FastAPI app --------------------------

# app = FastAPI(
#     title="Recommendation API",
#     description="Social network recommendation system",
#     version="1.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # configure in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# pipeline: Optional[OnlineInferencePipeline] = None
# app.state.telemetry: Optional[AsyncDBLogger] = None

# # --------------------- Startup / Shutdown -------------------

# @app.on_event("startup")
# async def startup_event():
#     global pipeline
#     try:
#         logger.info("="*70)
#         logger.info("🚀 STARTING RECOMMENDATION API")
#         logger.info("="*70)

#         config_path = os.getenv('CONFIG_PATH', 'configs/config_online.yaml')
#         models_dir  = os.getenv('MODELS_DIR', 'models')
#         data_dir    = os.getenv('DATA_DIR', 'dataset')

#         use_redis = os.getenv('REDIS_HOST') is not None

#         pipeline = OnlineInferencePipeline(
#             config_path=config_path,
#             models_dir=models_dir,
#             data_dir=data_dir,
#             use_redis=use_redis
#         )
#         logger.info("✅ Pipeline initialized")

#         # Telemetry async logger (Postgres DSN trong config.telemetry.mysql.url)
#         try:
#             tel_cfg = pipeline.config.get("telemetry", {}) if pipeline else {}
#             app.state.telemetry = AsyncDBLogger(tel_cfg, redis_client=pipeline.redis)
#             logger.info("✅ Telemetry logger initialized")
#         except Exception as e:
#             logger.warning(f"Telemetry init failed: {e}")
#             app.state.telemetry = None

#         # Quick ping
#         try:
#             be_ok = pipeline.ping_backend_db()
#             logger.info("Backend DB ping: %s", "OK" if be_ok else "FAIL")
#         except Exception as e:
#             logger.warning(f"Backend DB ping error: {e}")
#         try:
#             # tel_ok = app.state.telemetry.ping() if app.state.telemetry else False
#             tel = getattr(app.state, "telemetry", None)
#             tel_ok = False
#             if tel and hasattr(tel, "ping"):
#                 try:
#                     tel_ok = tel.ping()
#                 except Exception:
#                     tel_ok = False
#             logger.info("AI Logs DB ping: %s", "OK" if tel_ok else "FAIL")

#             # logger.info("AI Logs DB ping: %s", "OK" if tel_ok else "FAIL")
#         except Exception as e:
#             logger.warning(f"AI Logs DB ping error: {e}")

#         logger.info("="*70)
#     except Exception as e:
#         logger.error(f"Failed to init API: {e}", exc_info=True)
#         pipeline = None

# @app.on_event("shutdown")
# async def shutdown_event():
#     logger.info("👋 Shutting down API server")
#     if getattr(app.state, "telemetry", None):
#         try:
#             app.state.telemetry.shutdown()
#         except Exception:
#             pass

# # --------------------- Endpoints ----------------------------

# @app.get("/")
# async def root():
#     return {"message": "Recommendation API", "version": "1.0.0",
#             "status": "running" if pipeline else "error", "docs": "/docs"}

# @app.get("/health")
# async def health_check() -> HealthResponse:
#     components = {
#         "pipeline": pipeline is not None,
#         "redis": bool(pipeline and pipeline.redis is not None),
#         "models": bool(pipeline and pipeline.ranking_model is not None)
#     }
#     status = "healthy" if all(components.values()) else "degraded"
#     return HealthResponse(status=status, timestamp=datetime.now().isoformat(),
#                           version="1.0.0", components=components)

# @app.get("/metrics")
# async def get_metrics() -> MetricsResponse:
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     m = pipeline.get_metrics()
#     return MetricsResponse(
#         total_requests=m.get('total_requests', 0),
#         avg_latency_ms=m.get('avg_latency_ms', 0.0),
#         p50_latency_ms=m.get('p50_latency_ms', 0.0),
#         p95_latency_ms=m.get('p95_latency_ms', 0.0),
#         p99_latency_ms=m.get('p99_latency_ms', 0.0),
#         error_rate=0.0
#     )

# @app.get("/version")
# async def get_version():
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     return {"version": pipeline.current_version, "metadata": pipeline.metadata}

# @app.get("/feed")
# async def get_feed(
#     user_id: int = Query(..., description="User ID"),
#     limit: int = Query(50, ge=1, le=100, description="Number of posts"),
#     exclude_seen: Optional[str] = Query(None, description="Comma-separated post IDs to exclude")
# ):
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     try:
#         exclude_list = None
#         if exclude_seen:
#             try:
#                 exclude_list = [int(x.strip()) for x in exclude_seen.split(',')]
#             except ValueError:
#                 raise HTTPException(status_code=400, detail="Invalid exclude_seen format")

#         t0 = time.time()
#         feed = pipeline.generate_feed(user_id=user_id, limit=limit, exclude_seen=exclude_list)
#         t1 = time.time()

#         tel = getattr(app.state, "telemetry", None)
#         if tel:
#             stage_ms = {
#                 "recall": int(pipeline.metrics['recall_latency'][-1]) if pipeline.metrics['recall_latency'] else 0,
#                 "ranking": int(pipeline.metrics['ranking_latency'][-1]) if pipeline.metrics['ranking_latency'] else 0,
#                 "rerank": int(pipeline.metrics['reranking_latency'][-1]) if pipeline.metrics['reranking_latency'] else 0,
#             }
#             rd = getattr(pipeline, "_last_recall_diag", None)
#             tel.log_feed_request({
#                 "user_id": user_id,
#                 "timestamp": t1,
#                 "recall": {
#                     "following": getattr(rd, "following", 0) if rd else 0,
#                     "cf": getattr(rd, "cf", 0) if rd else 0,
#                     "content": getattr(rd, "content", 0) if rd else 0,
#                     "trending": getattr(rd, "trending", 0) if rd else 0,
#                     "total": getattr(rd, "total", 0) if rd else 0,
#                 },
#                 "final_count": len(feed or []),
#                 "reasons": getattr(pipeline, "_last_no_feed_reasons", []),
#                 "latency_ms": int((t1 - t0) * 1000),
#                 "stage_ms": stage_ms
#             })

#         return {
#             "user_id": user_id,
#             "posts": feed,
#             "count": len(feed),
#             "timestamp": datetime.now().isoformat()
#         }
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/interaction")
# async def log_interaction(request: InteractionRequest):
#     """
#     - Update user embedding realtime
#     - Mirror realtime counters (Redis) for ranking
#     - Update co-visitation (CF realtime nhẹ)
#     - Log to Postgres (AI logs) asynchronously
#     """
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     try:
#         valid_actions = ['like', 'comment', 'share', 'save', 'view', 'hide', 'report']
#         if request.action not in valid_actions:
#             raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {', '.join(valid_actions)}")

#         # 1) update embedding realtime
#         pipeline.update_user_embedding_realtime(request.user_id, request.post_id, request.action)

#         # 2) mirror realtime to Redis
#         r = pipeline.redis
#         author_id = None
#         if r is not None:
#             try:
#                 r.hincrby(f"user:{request.user_id}:engagement_24h", request.action, 1)
#                 r.expire(f"user:{request.user_id}:engagement_24h", 24*3600)

#                 r.hincrby(f"post:{request.post_id}:engagement_1h", request.action, 1)
#                 r.expire(f"post:{request.post_id}:engagement_1h", 3600)

#                 author_id = pipeline.get_author_id(request.post_id)
#                 if author_id:
#                     r.hincrby(f"author:{author_id}:engagement_1h", request.action, 1)
#                     r.expire(f"author:{author_id}:engagement_1h", 3600)

#                 r.lpush(f"user:{request.user_id}:recent_items", request.post_id)
#                 r.ltrim(f"user:{request.user_id}:recent_items", 0, 99)
#                 r.expire(f"user:{request.user_id}:recent_items", 7*24*3600)

#                 recent = r.lrange(f"user:{request.user_id}:recent_items", 0, 50) or []
#                 for rid in recent:
#                     try:
#                         rid = int(rid)
#                         if rid == request.post_id: 
#                             continue
#                         r.zincrby(f"covisit:{rid}", 1.0, request.post_id)
#                         r.zincrby(f"covisit:{request.post_id}", 1.0, rid)
#                         r.expire(f"covisit:{rid}", 7*24*3600)
#                         r.expire(f"covisit:{request.post_id}", 7*24*3600)
#                     except Exception:
#                         pass
#             except Exception as e:
#                 logger.debug(f"Realtime Redis mirror failed: {e}")

#         # 3) telemetry (Postgres)
#         tel = getattr(app.state, "telemetry", None)
#         if tel:
#             try:
#                 tel.log_interaction({
#                     "user_id": request.user_id,
#                     "post_id": request.post_id,
#                     "author_id": author_id,
#                     "action": request.action,
#                     "timestamp": time.time(),
#                     "session_id": request.session_id,
#                     "meta": request.meta or {}
#                 })
#             except Exception as e:
#                 logger.debug(f"telemetry log_interaction failed: {e}")

#         return {
#             "success": True,
#             "message": f"Interaction logged: user {request.user_id} {request.action}d post {request.post_id}",
#             "timestamp": datetime.now().isoformat()
#         }
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error logging interaction: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/trending")
# async def get_trending(
#     limit: int = Query(100, ge=1, le=200, description="Number of trending posts")
# ):
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     try:
#         posts = pipeline.trending_recall.recall(k=limit)
#         return {"posts": posts, "count": len(posts), "timestamp": datetime.now().isoformat()}
#     except Exception as e:
#         logger.error(f"Error getting trending posts: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/debug/components")
# async def debug_components():
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     return {
#         "embeddings": {
#             "post_count": len(pipeline.embeddings.get('post', {})),
#             "user_count": len(pipeline.embeddings.get('user', {}))
#         },
#         "faiss_index": pipeline.faiss_index is not None,
#         "faiss_posts": len(pipeline.faiss_post_ids),
#         "cf_model": pipeline.cf_model is not None,
#         "ranking_model": pipeline.ranking_model is not None,
#         "redis": pipeline.redis is not None,
#         "recall_channels": {
#             "following": pipeline.following_recall is not None,
#             "cf": pipeline.cf_recall is not None,
#             "content": pipeline.content_recall is not None,
#             "trending": pipeline.trending_recall is not None
#         }
#     }

# @app.get("/debug/user/{user_id}")
# async def debug_user(user_id: int):
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     return {
#         "user_id": user_id,
#         "has_embedding": user_id in pipeline.embeddings.get('user', {}),
#         "has_stats": user_id in pipeline.user_stats,
#         "following_count": len(pipeline.following_dict.get(user_id, []))
#     }

# @app.get("/debug/db")
# async def debug_db():
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not initialized")
#     backend = pipeline.backend_db_stats()
#     ai_logs_ok = False
#     if getattr(app.state, "telemetry", None):
#         try:
#             ai_logs_ok = app.state.telemetry.ping()
#         except Exception:
#             ai_logs_ok = False
#     return {"backend_db": backend, "ai_logs_db": {"connected": bool(ai_logs_ok)}}

# # --------------------- Errors & main ------------------------

# @app.exception_handler(Exception)
# async def global_exception_handler(request, exc):
#     logger.error(f"Unhandled exception: {exc}", exc_info=True)
#     return JSONResponse(status_code=500, content={
#         "error": "Internal server error",
#         "detail": str(exc),
#         "timestamp": datetime.now().isoformat()
#     })

# if __name__ == "__main__":
#     import uvicorn
#     host = os.getenv('API_HOST', '0.0.0.0')
#     port = int(os.getenv('API_PORT', 8010))
#     uvicorn.run("api:app", host=host, port=port, reload=True, log_level="info")


"""
FASTAPI APPLICATION - RECOMMENDATION API
========================================
RESTful API for recommendation system

Endpoints:
- GET /feed         : Get personalized feed
- GET /friends      : Get friend recommendations
- POST /interaction : Log user interaction (+realtime mirrors to Redis)
- POST /webhook/post_created : BE notifies new post (fanout to followers)
- GET /trending     : Get trending posts
- GET /health       : Health check
- GET /metrics      : System metrics
- GET /debug/*      : Debug endpoints (dev only)
"""

import os
import sys
import time
import math
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import pipeline
from recommender.online.online_inference_pipeline import OnlineInferencePipeline
# Async telemetry logger (non-blocking)
from recommender.online.telemetry import AsyncDBLogger
# Realtime backend ingestor (poll MySQL/Redis etc.) - optional
try:
    from recommender.online.realtime_ingestor import BackendIngestor
except Exception:  # dev fallback
    BackendIngestor = None

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class FeedRequest(BaseModel):
    user_id: int
    limit: int = 50
    exclude_seen: Optional[List[int]] = None

class FriendRecommendationRequest(BaseModel):
    user_id: int
    limit: int = 20

class InteractionRequest(BaseModel):
    user_id: int
    post_id: int
    action: str  # like, comment, share, save, view, hide, report
    session_id: Optional[str] = None
    meta: Optional[dict] = None

class PostCreatedWebhook(BaseModel):
    post_id: int
    author_id: int
    created_at: Optional[str] = None  # ISO8601 or epoch str
    followers: Optional[List[int]] = None  # optional: BE can send ready list
    meta: Optional[dict] = None           # optional: attach more info

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    components: dict

class MetricsResponse(BaseModel):
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

# Global singletons
pipeline: Optional[OnlineInferencePipeline] = None
app.state.telemetry: Optional[AsyncDBLogger] = None
app.state.ingestor: Optional[Any] = None  # BackendIngestor

# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _epoch_from_created_at(created_at: Optional[str]) -> int:
    """
    Accepts:
      - None -> now
      - "1697650000" or "1697650000.123" -> epoch str
      - "2025-10-26T03:40:00Z"/ISO -> parsed to epoch UTC
    Returns epoch seconds (int).
    """
    if not created_at:
        return int(time.time())
    try:
        # integer epoch (seconds / ms)
        if created_at.isdigit():
            val = int(created_at)
            return int(val / 1000) if val > 10_000_000_000 else val
        # float epoch
        f = float(created_at)
        return int(f)
    except Exception:
        pass
    try:
        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        return int(dt.timestamp())
    except Exception:
        return int(time.time())

def _resolve_followers(author_id: int) -> List[int]:
    """
    Resolve followers for author:
      1) Redis (author:{author_id}:followers:set or :followers list)
      2) pipeline.following_dict (dev/offline)
      3) Backend DB Friendship/Follow (optional; only if engine provided)
    """
    followers: List[int] = []
    # 1) Redis
    try:
        r = getattr(pipeline, "redis", None)
        if r:
            key_set = f"author:{author_id}:followers:set"
            key_list = f"author:{author_id}:followers"
            if r.exists(key_set):
                raw = r.smembers(key_set) or []
                followers = [int(x) for x in raw]
            elif r.exists(key_list):
                raw = r.lrange(key_list, 0, -1) or []
                followers = [int(x) for x in raw]
            if followers:
                return followers
    except Exception:
        pass

    # 2) following_dict (structure: user_id -> [author_ids])
    try:
        fd: Dict[int, List[int]] = getattr(pipeline, "following_dict", {}) or {}
        if fd:
            followers = [uid for uid, authors in fd.items() if author_id in authors]
            if followers:
                return followers
    except Exception:
        pass

    # 3) Backend DB (optional, avoid heavy usage)
    try:
        engine = getattr(pipeline, "db_engine", None)
        if engine:
            with engine.connect() as conn:
                # Example schema fallback: Friendship(UserId, FriendId, Status=10)
                rs = conn.execute(
                    "SELECT UserId FROM Friendship WHERE FriendId=%s AND Status=10",
                    (author_id,)
                )
                rows = rs.fetchall()
                followers = [int(r[0]) for r in rows]
                return followers
    except Exception:
        pass

    return followers

def _get_author_id_from_pipeline(post_id: int) -> Optional[int]:
    """
    Try to resolve author_id for a post using:
      1) Redis meta
      2) pipeline dataframes (dev)
    """
    # 1) Redis
    try:
        r = getattr(pipeline, "redis", None)
        if r:
            aid = r.hget(f"post:{post_id}:meta", "author_id")
            if aid is not None:
                return int(aid)
    except Exception:
        pass

    # 2) Dev dataframe
    try:
        df = getattr(pipeline, "data", {}).get("post")
        if df is not None:
            row = df[df["Id"] == post_id]
            if not row.empty:
                return int(row.iloc[0]["UserId"])
    except Exception:
        pass
    return None

def _fanout_following(author_id: int, post_id: int, ts_epoch: int, followers: List[int]) -> int:
    """
    Fanout the new post to followers' sorted sets in Redis.
    Returns number of followers updated.
    """
    r = getattr(pipeline, "redis", None)
    if not r or not followers:
        return 0
    updated = 0
    pipe = r.pipeline(transaction=False)
    for fid in followers:
        pipe.zadd(f"following:{fid}:posts", {post_id: ts_epoch})
        pipe.expire(f"following:{fid}:posts", 48 * 3600)
        updated += 1
    try:
        pipe.execute()
    except Exception as e:
        logger.warning(f"Fanout pipeline execute failed: {e}")
    return updated

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize pipeline on startup
    """
    global pipeline
    try:
        logger.info("="*70)
        logger.info("🚀 STARTING RECOMMENDATION API")
        logger.info("="*70)

        config_path = os.getenv('CONFIG_PATH', 'configs/config_online.yaml')
        models_dir = os.getenv('MODELS_DIR', 'models')
        data_dir   = os.getenv('DATA_DIR', 'dataset')

        redis_host = os.getenv('REDIS_HOST')
        use_redis = redis_host is not None

        logger.info(f"Config path: {config_path}")
        logger.info(f"Models dir: {models_dir}")
        logger.info(f"Data dir: {data_dir}")
        logger.info(f"Redis: {'enabled' if use_redis else 'disabled'}")

        pipeline = OnlineInferencePipeline(
            config_path=config_path,
            models_dir=models_dir,
            data_dir=data_dir,
            use_redis=use_redis
        )
        logger.info("✅ Pipeline initialized successfully!")

        # Start realtime ingestor (optional)
        try:
            if BackendIngestor is not None:
                ingest_cfg = pipeline.config.get("realtime_ingest", {}) or {}
                app.state.ingestor = BackendIngestor(
                    cfg=ingest_cfg,
                    mysql_engine=pipeline.db_engine,   # backend MySQL (in pipeline if enabled)
                    redis_client=pipeline.redis,
                    pipeline=pipeline
                )
                app.state.ingestor.start()
        except Exception as e:
            logger.warning("Failed to start realtime ingestor: %s", e)

        # Async telemetry (non-blocking)
        try:
            tel_cfg = pipeline.config.get("telemetry", {}) if pipeline and hasattr(pipeline, "config") else {}
            app.state.telemetry = AsyncDBLogger(tel_cfg, redis_client=pipeline.redis)
            logger.info("✅ Telemetry logger initialized")
        except Exception as e:
            logger.warning(f"Telemetry init failed: {e}")
            app.state.telemetry = None

        logger.info("="*70)
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}", exc_info=True)
        pipeline = None

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down API server")
    if getattr(app.state, "ingestor", None):
        try:
            app.state.ingestor.stop()
        except Exception:
            pass
    if getattr(app.state, "telemetry", None):
        try:
            app.state.telemetry.shutdown()
        except Exception:
            pass

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Recommendation API",
        "version": "1.0.0",
        "status": "running" if pipeline is not None else "error",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check() -> HealthResponse:
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
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    metrics = pipeline.get_metrics()
    return MetricsResponse(
        total_requests=metrics.get('total_requests', 0),
        avg_latency_ms=metrics.get('avg_latency_ms', 0),
        p50_latency_ms=metrics.get('p50_latency_ms', 0),
        p95_latency_ms=metrics.get('p95_latency_ms', 0),
        p99_latency_ms=metrics.get('p99_latency_ms', 0),
        error_rate=0.0  # TODO: track errors
    )

@app.get("/version")
async def get_version():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return {"version": pipeline.current_version, "metadata": pipeline.metadata}

@app.get("/feed")
async def get_feed(
    user_id: int = Query(..., description="User ID"),
    limit: int = Query(50, ge=1, le=100, description="Number of posts"),
    exclude_seen: Optional[str] = Query(None, description="Comma-separated post IDs to exclude")
):
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

        t0 = time.time()
        feed = pipeline.generate_feed(user_id=user_id, limit=limit, exclude_seen=exclude_list)
        t1 = time.time()

        # Non-blocking telemetry
        tel = getattr(app.state, "telemetry", None)
        if tel:
            stage_ms = {
                "recall": int(pipeline.metrics['recall_latency'][-1]) if pipeline.metrics['recall_latency'] else 0,
                "ranking": int(pipeline.metrics['ranking_latency'][-1]) if pipeline.metrics['ranking_latency'] else 0,
                "rerank": int(pipeline.metrics['reranking_latency'][-1]) if pipeline.metrics['reranking_latency'] else 0,
            }
            recall_diag = getattr(pipeline, "_last_recall_diag", None)
            tel.log_feed_request({
                "user_id": user_id,
                "timestamp": t1,
                "recall": {
                    "following": getattr(recall_diag, "following", 0) if recall_diag else 0,
                    "cf": getattr(recall_diag, "cf", 0) if recall_diag else 0,
                    "content": getattr(recall_diag, "content", 0) if recall_diag else 0,
                    "trending": getattr(recall_diag, "trending", 0) if recall_diag else 0,
                    "total": getattr(recall_diag, "total", 0) if recall_diag else 0,
                },
                "final_count": len(feed or []),
                "reasons": getattr(pipeline, "_last_no_feed_reasons", []),
                "latency_ms": int((t1 - t0) * 1000),
                "stage_ms": stage_ms
            })

        return {
            "user_id": user_id,
            "posts": feed,
            "count": len(feed),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/friends")
async def get_friend_recommendations(
    user_id: int = Query(..., description="User ID"),
    limit: int = Query(20, ge=1, le=50, description="Number of recommendations")
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        recommendations = pipeline.recommend_friends(user_id=user_id, k=limit)
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
    + mirror realtime signals to Redis (ranking realtime features)
    + update lightweight co-visit (CF realtime nhẹ)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        valid_actions = ['like', 'comment', 'share', 'save', 'view', 'hide', 'report']
        if request.action not in valid_actions:
            raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {', '.join(valid_actions)}")

        # 1) Update user embedding realtime (content-based reacts quickly)
        pipeline.update_user_embedding_realtime(
            user_id=request.user_id,
            post_id=request.post_id,
            action=request.action
        )

        # 2) Mirror realtime to Redis
        r = getattr(pipeline, "redis", None)
        author_id = _get_author_id_from_pipeline(request.post_id)
        if r is not None:
            try:
                # user 24h counters
                r.hincrby(f"user:{request.user_id}:eng24h", request.action, 1)
                r.expire(f"user:{request.user_id}:eng24h", 24*3600)

                # post 1h counters
                r.hincrby(f"post:{request.post_id}:eng1h", request.action, 1)
                r.expire(f"post:{request.post_id}:eng1h", 3600)

                # author 1h counters
                if author_id:
                    r.hincrby(f"author:{author_id}:eng1h", request.action, 1)
                    r.expire(f"author:{author_id}:eng1h", 3600)

                # co-visit (CF realtime nhẹ)
                r.lpush(f"user:{request.user_id}:recent_items", request.post_id)
                r.ltrim(f"user:{request.user_id}:recent_items", 0, 99)
                r.expire(f"user:{request.user_id}:recent_items", 7*24*3600)
                recent = r.lrange(f"user:{request.user_id}:recent_items", 0, 50) or []
                for rid in recent:
                    try:
                        rid = int(rid)
                        if rid == request.post_id:
                            continue
                        r.zincrby(f"covisit:{rid}", 1.0, request.post_id)
                        r.zincrby(f"covisit:{request.post_id}", 1.0, rid)
                        r.expire(f"covisit:{rid}", 7*24*3600)
                        r.expire(f"covisit:{request.post_id}", 7*24*3600)
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Realtime mirror to Redis failed: {e}")

        # 3) Telemetry (non-blocking DB logging)
        tel = getattr(app.state, "telemetry", None)
        if tel:
            try:
                tel.log_interaction({
                    "user_id": request.user_id,
                    "post_id": request.post_id,
                    "author_id": author_id,
                    "action": request.action,
                    "timestamp": time.time(),
                    "session_id": request.session_id,
                    "meta": request.meta or {}
                })
            except Exception as e:
                logger.debug(f"telemetry log_interaction failed: {e}")

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

@app.post("/webhook/post_created")
async def webhook_post_created(
    payload: PostCreatedWebhook,
    x_webhook_token: Optional[str] = Header(default=None, convert_underscores=False)
):
    """
    Webhook từ BE khi tác giả tạo bài mới.
    Công việc:
      1) Ghi meta & active_posts vào Redis (cho trending/job nền & content-based)
      2) Fanout sorted-set vào cache following của followers
      3) Ghi telemetry (async)
    Bảo vệ: header X-Webhook-Token == WEBHOOK_TOKEN (env)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Auth header
    expected = os.getenv("WEBHOOK_TOKEN")
    if expected:
        if not x_webhook_token or x_webhook_token != expected:
            raise HTTPException(status_code=401, detail="Invalid webhook token")

    try:
        post_id = int(payload.post_id)
        author_id = int(payload.author_id)
        ts_epoch = _epoch_from_created_at(payload.created_at)

        # Redis meta + active posts
        r = getattr(pipeline, "redis", None)
        if r:
            r.hset(f"post:{post_id}:meta", mapping={
                "author_id": author_id,
                "created_at": ts_epoch
            })
            r.expire(f"post:{post_id}:meta", 30*24*3600)
            r.zadd("active_posts", {post_id: ts_epoch})
            r.expire("active_posts", 7*24*3600)

        # Fanout to followers
        followers = payload.followers or _resolve_followers(author_id)
        updated = 0

        # Prefer pipeline realtime handler if available
        try:
            rt = getattr(pipeline, "rt_handlers", None)
            if rt and hasattr(rt, "on_author_create_post"):
                rt.on_author_create_post(
                    author_id=author_id,
                    post_id=post_id,
                    post_timestamp=ts_epoch,
                    followers=followers
                )
                updated = len(followers or [])
            else:
                # fallback fanout here
                updated = _fanout_following(author_id, post_id, ts_epoch, followers)
        except Exception as e:
            logger.warning(f"rt_handlers.on_author_create_post failed, fallback fanout: {e}")
            updated = _fanout_following(author_id, post_id, ts_epoch, followers)

        # Telemetry (non-blocking)
        tel = getattr(app.state, "telemetry", None)
        if tel:
            try:
                tel.log_event({
                    "event": "post_created",
                    "author_id": author_id,
                    "post_id": post_id,
                    "followers_count": len(followers or []),
                    "timestamp": time.time()
                })
            except Exception:
                pass

        return {
            "success": True,
            "post_id": post_id,
            "author_id": author_id,
            "followers_fanout": updated,
            "created_at_epoch": ts_epoch,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"webhook_post_created error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trending")
async def get_trending(
    limit: int = Query(100, ge=1, le=200, description="Number of trending posts")
):
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
# MAIN (for local testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8010))
    print(f"\n🚀 Starting API server on {host}:{port}\n")
    uvicorn.run("api:app", host=host, port=port, reload=True, log_level="info")
