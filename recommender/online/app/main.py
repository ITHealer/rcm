# # """
# # FastAPI Main Application with Redis Integration

# # This is the main entry point for the recommendation API.
# # Redis is used as the primary data store, synced by Team App from MySQL.
# # """

# # from fastapi import FastAPI, Request, status
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import JSONResponse
# # from contextlib import asynccontextmanager
# # import time
# # import logging

# # # from app.api.v1 import health, recommendation
# # # from app.config import settings
# # # from app.utils.logger import setup_logger
# # # from app.services.cache_service import cache_service

# # from recommender.online.app.api.v1 import health, recommendation
# # from recommender.online.app.config import settings
# # from recommender.online.app.utils.logger import setup_logger
# # from recommender.online.app.services.cache_service import cache_service

# # logger = setup_logger(__name__)

# # # Global recommendation service instance
# # recommendation_service_instance = None


# # def load_models():
# #     """Load all models and data"""
# #     import pickle
# #     import os
# #     from pathlib import Path
    
# #     logger.info("📦 Loading models and data...")
    
# #     # Convert to Path objects (FIX: was string before)
# #     models_dir = Path("models")
# #     data_dir = Path(settings.OFFLINE_PATH) / "dataset"
    
# #     # Find latest model version
# #     latest_version_file = models_dir / "latest.txt"
    
# #     if latest_version_file.exists():
# #         # Read latest version from file
# #         with open(latest_version_file, 'r') as f:
# #             latest_version = f.read().strip()
# #         latest_dir = models_dir / latest_version
# #         logger.info(f"Using model version from latest.txt: {latest_version}")
# #     else:
# #         # Find latest version directory manually
# #         version_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('v_')]
# #         if version_dirs:
# #             latest_dir = max(version_dirs, key=lambda d: d.name)
# #             logger.info(f"Using latest model version: {latest_dir.name}")
# #         else:
# #             raise FileNotFoundError(f"No model versions found in {models_dir}")
    
# #     if not latest_dir.exists():
# #         raise FileNotFoundError(f"Model directory not found: {latest_dir}")
    
# #     logger.info(f"📂 Loading models from: {latest_dir}")
    
# #     models = {}
    
# #     # 1. Load ranking model (check both .pkl and .txt extensions)
# #     model_path = latest_dir / "ranking_model.pkl"
# #     if not model_path.exists():
# #         model_path = latest_dir / "ranking_model.txt"  # Try .txt extension
    
# #     if model_path.exists():
# #         try:
# #             with open(model_path, 'rb') as f:
# #                 models['ranking_model'] = pickle.load(f)
# #             logger.info(f"✅ Loaded ranking model from {model_path.name}")
# #         except Exception as e:
# #             logger.warning(f"⚠️ Failed to load ranking model: {e}")
# #     else:
# #         logger.warning(f"⚠️ Ranking model not found in {latest_dir}")
    
# #     # 2. Load scaler
# #     scaler_path = latest_dir / "ranking_scaler.pkl"
# #     if scaler_path.exists():
# #         with open(scaler_path, 'rb') as f:
# #             models['scaler'] = pickle.load(f)
# #         logger.info("✅ Loaded scaler")
# #     else:
# #         logger.warning(f"⚠️ Scaler not found at {scaler_path}")
    
# #     # 3. Load feature columns
# #     feature_cols_path = latest_dir / "ranking_feature_cols.pkl"
# #     if feature_cols_path.exists():
# #         with open(feature_cols_path, 'rb') as f:
# #             models['feature_cols'] = pickle.load(f)
# #         logger.info(f"✅ Loaded {len(models['feature_cols'])} feature columns")
# #     else:
# #         logger.warning(f"⚠️ Feature columns not found at {feature_cols_path}")
# #         # Fallback: empty list
# #         models['feature_cols'] = []
    
# #     # 4. Load CF model
# #     cf_model_path = latest_dir / "cf_model.pkl"
# #     if cf_model_path.exists():
# #         with open(cf_model_path, 'rb') as f:
# #             models['cf_model'] = pickle.load(f)
# #         logger.info("✅ Loaded CF model")
# #     else:
# #         logger.warning(f"⚠️ CF model not found at {cf_model_path}")
    
# #     # 5. Load embeddings
# #     embeddings_path = latest_dir / "embeddings.pkl"
# #     if embeddings_path.exists():
# #         with open(embeddings_path, 'rb') as f:
# #             embeddings = pickle.load(f)
        
# #         n_users = len(embeddings.get('user', {}))
# #         n_posts = len(embeddings.get('post', {}))
# #         logger.info(f"✅ Loaded embeddings: {n_users} users, {n_posts} posts")
# #     else:
# #         embeddings = {'user': {}, 'post': {}}
# #         logger.warning("⚠️ Embeddings not found, using empty dict")
    
# #     # 6. Load FAISS index (check both in version dir and models root)
# #     faiss_index = None
# #     faiss_post_ids = []
    
# #     try:
# #         import faiss
        
# #         # Try version directory first
# #         faiss_index_path = latest_dir / "faiss_index.bin"
# #         faiss_ids_path = latest_dir / "faiss_post_ids.pkl"
        
# #         # If not in version dir, try models root
# #         if not faiss_index_path.exists():
# #             faiss_index_path = models_dir / "faiss_index.bin"
# #             faiss_ids_path = models_dir / "faiss_post_ids.pkl"
        
# #         if faiss_index_path.exists() and faiss_ids_path.exists():
# #             faiss_index = faiss.read_index(str(faiss_index_path))
            
# #             with open(faiss_ids_path, 'rb') as f:
# #                 faiss_post_ids = pickle.load(f)
            
# #             logger.info(f"✅ Loaded FAISS index: {faiss_index.ntotal} vectors from {faiss_index_path.parent.name}")
# #         else:
# #             logger.warning("⚠️ FAISS index not found, content recall will use brute-force")
    
# #     except ImportError:
# #         logger.warning("⚠️ FAISS not available, content recall will use brute-force")
# #     except Exception as e:
# #         logger.error(f"❌ Error loading FAISS: {e}")
    
# #     # 7. Load data (for recall channels)
# #     data = {}
    
# #     logger.info(f"📂 Loading data from: {data_dir}")
    
# #     # Load posts
# #     posts_path = data_dir / "posts.parquet"
# #     if posts_path.exists():
# #         import pandas as pd
# #         data['posts'] = pd.read_parquet(posts_path)
# #         logger.info(f"✅ Loaded {len(data['posts'])} posts")
# #     else:
# #         logger.warning(f"⚠️ Posts data not found at {posts_path}")
# #         data['posts'] = None
    
# #     # Load reactions
# #     reactions_path = data_dir / "postreaction.parquet"
# #     if reactions_path.exists():
# #         import pandas as pd
# #         data['postreaction'] = pd.read_parquet(reactions_path)
# #         logger.info(f"✅ Loaded {len(data['postreaction'])} reactions")
# #     else:
# #         logger.warning(f"⚠️ Reactions data not found at {reactions_path}")
# #         data['postreaction'] = None
    
# #     # Load friendships
# #     friendships_path = data_dir / "friendships.parquet"
# #     if friendships_path.exists():
# #         import pandas as pd
# #         data['friendships'] = pd.read_parquet(friendships_path)
# #         logger.info(f"✅ Loaded {len(data['friendships'])} friendships")
# #     else:
# #         logger.warning(f"⚠️ Friendships data not found at {friendships_path}")
# #         data['friendships'] = None
    
# #     logger.info("✅ Model loading complete")
# #     return models, embeddings, faiss_index, faiss_post_ids, data


# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     """
# #     Lifespan context manager for startup/shutdown events
    
# #     Startup:
# #     - Connect to Redis
# #     - Load ML models
# #     - Load Faiss index
# #     - Initialize recommendation service
    
# #     Shutdown:
# #     - Close Redis connection
# #     - Cleanup resources
# #     """
# #     global recommendation_service_instance
    
# #     # ==================== STARTUP ====================
# #     logger.info("=" * 60)
# #     logger.info("🚀 Starting Recommendation Service...")
# #     logger.info("=" * 60)
    
# #     try:
# #         # 1. Connect to Redis
# #         logger.info("📡 Connecting to Redis...")
# #         cache_service.connect()
        
# #         if cache_service.is_connected():
# #             logger.info("✅ Redis connected successfully")
            
# #             # Log cache stats
# #             stats = cache_service.get_cache_stats()
# #             logger.info(f"📊 Redis Stats:")
# #             logger.info(f"   - Memory Used: {stats.get('used_memory_human', 'unknown')}")
# #             logger.info(f"   - Hit Rate: {stats.get('hit_rate', 0):.2f}%")
# #             logger.info(f"   - Connected Clients: {stats.get('connected_clients', 0)}")
# #         else:
# #             logger.error("❌ Redis connection failed!")
# #             raise Exception("Failed to connect to Redis")
        
# #         # 2. Load ML models and data
# #         logger.info("🤖 Loading ML models...")
# #         models, embeddings, faiss_index, faiss_post_ids, data = load_models()
# #         logger.info("✅ ML models loaded")
        
# #         # 3. Initialize Feature Engineer
# #         logger.info("🔧 Initializing Feature Engineer...")
# #         from recommender.common.feature_engineer import FeatureEngineer
        
# #         # Build stats dicts
# #         user_stats = {}
# #         author_stats = {}
# #         following_dict = {}
        
# #         if 'friendships' in data:
# #             for user_id, group in data['friendships'].groupby('UserId'):
# #                 following_dict[int(user_id)] = group['FriendId'].astype(int).tolist()
        
# #         feature_engineer = FeatureEngineer(
# #             data=data,
# #             user_stats=user_stats,
# #             author_stats=author_stats,
# #             following_dict=following_dict,
# #             embeddings=embeddings
# #         )
# #         logger.info("✅ Feature Engineer initialized")
        
# #         # 4. Initialize Recommendation Service
# #         logger.info("🎯 Initializing Recommendation Service...")
# #         from recommender.online.app.services.recommendation_service import RecommendationService
        
# #         # Load config
# #         import yaml
# #         config_path = settings.OFFLINE_PATH + "/../configs/config_online.yaml"
# #         try:
# #             with open(config_path, 'r') as f:
# #                 config = yaml.safe_load(f)
# #         except FileNotFoundError:
# #             logger.warning(f"Config file not found: {config_path}, using defaults")
# #             config = {
# #                 'recall': {'channels': {}},
# #                 'ranking': {},
# #                 'reranking': {}
# #             }
        
# #         recommendation_service_instance = RecommendationService(
# #             data=data,
# #             models=models,
# #             embeddings=embeddings,
# #             faiss_index=faiss_index,
# #             faiss_post_ids=faiss_post_ids,
# #             feature_engineer=feature_engineer,
# #             config=config
# #         )
        
# #         # Inject into recommendation API
# #         recommendation.set_recommendation_service(recommendation_service_instance)
        
# #         logger.info("✅ Recommendation Service initialized")
        
# #         logger.info("=" * 60)
# #         logger.info("✅ Recommendation Service Started Successfully!")
# #         logger.info(f"📍 Environment: {settings.ENVIRONMENT}")
# #         logger.info(f"🌐 API URL: http://{settings.HOST}:{settings.PORT}")
# #         logger.info(f"📚 Docs: http://{settings.HOST}:{settings.PORT}/docs")
# #         logger.info("=" * 60)
        
# #     except Exception as e:
# #         logger.error(f"❌ Failed to start service: {e}", exc_info=True)
# #         raise
    
# #     yield
    
# #     # ==================== SHUTDOWN ====================
# #     logger.info("=" * 60)
# #     logger.info("🛑 Shutting down Recommendation Service...")
# #     logger.info("=" * 60)
    
# #     try:
# #         # Close Redis connection
# #         logger.info("📡 Closing Redis connection...")
# #         cache_service.close()
# #         logger.info("✅ Redis connection closed")
        
# #         # Cleanup
# #         recommendation_service_instance = None
        
# #         logger.info("=" * 60)
# #         logger.info("✅ Recommendation Service Shutdown Complete")
# #         logger.info("=" * 60)
        
# #     except Exception as e:
# #         logger.error(f"❌ Error during shutdown: {e}", exc_info=True)


# # # Initialize FastAPI app
# # app = FastAPI(
# #     title=settings.PROJECT_NAME,
# #     description="""
# #     Social Network Recommendation API
    
# #     Architecture:
# #     - Team App syncs data from MySQL to Redis
# #     - AI team reads from Redis for recommendations
# #     - No direct MySQL queries to avoid database overload
    
# #     Features:
# #     - Multi-channel recall (Following, CF, Content-based, Trending)
# #     - ML-based ranking (LightGBM)
# #     - Re-ranking with business rules
# #     - Real-time caching with Redis
# #     """,
# #     version="1.0.0",
# #     lifespan=lifespan,
# #     docs_url="/docs" if settings.DEBUG else None,
# #     redoc_url="/redoc" if settings.DEBUG else None
# # )

# # # ==================== MIDDLEWARES ====================

# # # CORS Middleware
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=settings.CORS_ORIGINS,
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Request Logging Middleware
# # @app.middleware("http")
# # async def log_requests(request: Request, call_next):
# #     """
# #     Log all incoming requests with timing
# #     """
# #     start_time = time.time()
    
# #     # Log request
# #     logger.info(
# #         f"📨 Incoming: {request.method} {request.url.path} "
# #         f"from {request.client.host if request.client else 'unknown'}"
# #     )
    
# #     # Process request
# #     try:
# #         response = await call_next(request)
# #         duration_ms = (time.time() - start_time) * 1000
        
# #         # Log response
# #         logger.info(
# #             f"📤 Response: {request.method} {request.url.path} "
# #             f"status={response.status_code} duration={duration_ms:.2f}ms"
# #         )
        
# #         # Add timing header
# #         response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
        
# #         return response
        
# #     except Exception as e:
# #         duration_ms = (time.time() - start_time) * 1000
# #         logger.error(
# #             f"❌ Error: {request.method} {request.url.path} "
# #             f"duration={duration_ms:.2f}ms error={str(e)}"
# #         )
# #         raise


# # # Timeout Middleware
# # @app.middleware("http")
# # async def timeout_middleware(request: Request, call_next):
# #     """
# #     Enforce request timeout
# #     """
# #     # TODO: Implement proper timeout with asyncio.wait_for
# #     # For now, just pass through
# #     # This will be implemented when we have actual recommendation logic
# #     return await call_next(request)


# # # ==================== EXCEPTION HANDLERS ====================

# # @app.exception_handler(Exception)
# # async def global_exception_handler(request: Request, exc: Exception):
# #     """
# #     Global exception handler for unhandled errors
# #     """
# #     logger.error(
# #         f"❌ Unhandled exception: {exc}",
# #         exc_info=True,
# #         extra={
# #             "method": request.method,
# #             "path": request.url.path,
# #             "client": request.client.host if request.client else None
# #         }
# #     )
    
# #     return JSONResponse(
# #         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
# #         content={
# #             "error": "Internal Server Error",
# #             "message": "An unexpected error occurred",
# #             "detail": str(exc) if settings.DEBUG else None
# #         }
# #     )


# # @app.exception_handler(ValueError)
# # async def value_error_handler(request: Request, exc: ValueError):
# #     """
# #     Handle validation errors
# #     """
# #     logger.warning(f"⚠️ Validation error: {exc}")
    
# #     return JSONResponse(
# #         status_code=status.HTTP_400_BAD_REQUEST,
# #         content={
# #             "error": "Validation Error",
# #             "message": str(exc)
# #         }
# #     )


# # # ==================== INCLUDE ROUTERS ====================

# # # Health check endpoints
# # app.include_router(
# #     health.router,
# #     prefix=settings.API_V1_PREFIX,
# #     tags=["health"]
# # )

# # # Recommendation endpoints
# # app.include_router(
# #     recommendation.router,
# #     prefix=settings.API_V1_PREFIX + "/recommendation",
# #     tags=["recommendation"]
# # )


# # # ==================== ROOT ENDPOINTS ====================

# # @app.get("/")
# # async def root():
# #     """
# #     Root endpoint - API information
# #     """
# #     return {
# #         "service": settings.PROJECT_NAME,
# #         "version": "1.0.0",
# #         "status": "running",
# #         "environment": settings.ENVIRONMENT,
# #         "docs": f"http://{settings.HOST}:{settings.PORT}/docs" if settings.DEBUG else None,
# #         "architecture": {
# #             "data_source": "Redis (synced from MySQL by Team App)",
# #             "cache_layer": "Redis",
# #             "ml_framework": "LightGBM",
# #             "vector_search": "Faiss"
# #         }
# #     }


# # @app.get("/info")
# # async def info():
# #     """
# #     Service information endpoint
# #     """
# #     return {
# #         "service": settings.PROJECT_NAME,
# #         "version": "1.0.0",
# #         "environment": settings.ENVIRONMENT,
# #         "redis": {
# #             "host": settings.REDIS_HOST,
# #             "port": settings.REDIS_PORT,
# #             "connected": cache_service.is_connected()
# #         },
# #         "features": {
# #             "recall_channels": ["following", "cf", "content", "trending"],
# #             "ranking": "LightGBM ML model",
# #             "reranking": "Business rules",
# #             "max_candidates": settings.MAX_RECALL_SIZE,
# #             "final_feed_size": settings.FINAL_FEED_SIZE
# #         },
# #         "performance_targets": {
# #             "p95_latency_ms": 200,
# #             "recall_latency_ms": 50,
# #             "ranking_latency_ms": 50,
# #             "reranking_latency_ms": 20
# #         }
# #     }


# # if __name__ == "__main__":
# #     import uvicorn
    
# #     uvicorn.run(
# #         "main:app",
# #         host=settings.HOST,
# #         port=settings.PORT,
# #         reload=settings.DEBUG,
# #         log_level="info"
# #     )

# """
# FastAPI Main Application with Redis Integration

# This is the main entry point for the recommendation API.
# Redis is used as the primary data store, synced by Team App from MySQL.

# FIXED:
# 1. NULL-safe friendships groupby (line 250)
# 2. FAISS loading supports new structure (embeddings_YYYYMMDD_HHMMSS/)
# 3. sklearn version compatibility warnings suppressed
# """

# from fastapi import FastAPI, Request, status
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from contextlib import asynccontextmanager
# import time
# import logging
# import warnings

# # Suppress sklearn version warnings
# warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# from recommender.online.app.api.v1 import health, recommendation
# from recommender.online.app.config import settings
# from recommender.online.app.utils.logger import setup_logger
# from recommender.online.app.services.cache_service import cache_service

# logger = setup_logger(__name__)

# # Global recommendation service instance
# recommendation_service_instance = None


# def load_models():
#     """Load all models and data"""
#     import pickle
#     import os
#     import pandas as pd
#     from pathlib import Path
    
#     logger.info("📦 Loading models and data...")
    
#     # Convert to Path objects
#     models_dir = Path("models")
#     data_dir = Path(settings.OFFLINE_PATH) / "dataset"
    
#     # Find latest model version
#     latest_version_file = models_dir / "latest.txt"
    
#     if latest_version_file.exists():
#         # Read latest version from file
#         with open(latest_version_file, 'r') as f:
#             latest_version = f.read().strip()
#         latest_dir = models_dir / latest_version
#         logger.info(f"Using model version from latest.txt: {latest_version}")
#     else:
#         # Find latest version directory manually
#         version_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('v_')]
#         if version_dirs:
#             latest_dir = max(version_dirs, key=lambda d: d.name)
#             logger.info(f"Using latest model version: {latest_dir.name}")
#         else:
#             raise FileNotFoundError(f"No model versions found in {models_dir}")
    
#     if not latest_dir.exists():
#         raise FileNotFoundError(f"Model directory not found: {latest_dir}")
    
#     logger.info(f"📂 Loading models from: {latest_dir}")
    
#     models = {}
    
#     # 1. Load ranking model
#     model_path = latest_dir / "ranking_model.pkl"
#     if not model_path.exists():
#         model_path = latest_dir / "ranking_model.txt"
    
#     if model_path.exists():
#         try:
#             # Suppress sklearn warnings
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 with open(model_path, 'rb') as f:
#                     models['ranking_model'] = pickle.load(f)
#             logger.info(f"✅ Loaded ranking model from {model_path.name}")
#         except Exception as e:
#             logger.warning(f"⚠️ Failed to load ranking model: {e}")
#     else:
#         logger.warning(f"⚠️ Ranking model not found in {latest_dir}")
    
#     # 2. Load scaler
#     scaler_path = latest_dir / "ranking_scaler.pkl"
#     if scaler_path.exists():
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             with open(scaler_path, 'rb') as f:
#                 models['scaler'] = pickle.load(f)
#         logger.info("✅ Loaded scaler")
#     else:
#         logger.warning(f"⚠️ Scaler not found at {scaler_path}")
    
#     # 3. Load feature columns
#     feature_cols_path = latest_dir / "ranking_feature_cols.pkl"
#     if feature_cols_path.exists():
#         with open(feature_cols_path, 'rb') as f:
#             models['feature_cols'] = pickle.load(f)
#         logger.info(f"✅ Loaded {len(models['feature_cols'])} feature columns")
#     else:
#         logger.warning(f"⚠️ Feature columns not found at {feature_cols_path}")
#         models['feature_cols'] = []
    
#     # 4. Load CF model
#     cf_model_path = latest_dir / "cf_model.pkl"
#     if cf_model_path.exists():
#         with open(cf_model_path, 'rb') as f:
#             models['cf_model'] = pickle.load(f)
#         logger.info("✅ Loaded CF model")
#     else:
#         logger.warning(f"⚠️ CF model not found at {cf_model_path}")
    
#     # 5. Load embeddings
#     embeddings_path = latest_dir / "embeddings.pkl"
#     if embeddings_path.exists():
#         with open(embeddings_path, 'rb') as f:
#             embeddings = pickle.load(f)
        
#         n_users = len(embeddings.get('user', {}))
#         n_posts = len(embeddings.get('post', {}))
#         logger.info(f"✅ Loaded embeddings: {n_users} users, {n_posts} posts")
#     else:
#         embeddings = {'user': {}, 'post': {}}
#         logger.warning("⚠️ Embeddings not found, using empty dict")
    
#     # 6. Load FAISS index - SUPPORT NEW STRUCTURE
#     faiss_index = None
#     faiss_post_ids = []
    
#     try:
#         import faiss
        
#         # Try new structure first: embeddings_YYYYMMDD_HHMMSS/
#         embedding_dirs = sorted(
#             [d for d in models_dir.iterdir() 
#              if d.is_dir() and d.name.startswith('embeddings_')],
#             reverse=True
#         )
        
#         faiss_loaded = False
        
#         # Try loading from new structure
#         if embedding_dirs:
#             for emb_dir in embedding_dirs:
#                 faiss_index_path = emb_dir / "faiss_index.bin"
#                 faiss_ids_path = emb_dir / "faiss_post_ids.pkl"
                
#                 if faiss_index_path.exists() and faiss_ids_path.exists():
#                     try:
#                         faiss_index = faiss.read_index(str(faiss_index_path))
                        
#                         with open(faiss_ids_path, 'rb') as f:
#                             faiss_post_ids = pickle.load(f)
                        
#                         logger.info(f"✅ Loaded FAISS index: {faiss_index.ntotal} vectors from {emb_dir.name}")
#                         faiss_loaded = True
#                         break
#                     except Exception as e:
#                         logger.warning(f"Failed to load FAISS from {emb_dir.name}: {e}")
#                         continue
        
#         # Fallback to old structure if new not found
#         if not faiss_loaded:
#             faiss_index_path = latest_dir / "faiss_index.bin"
#             faiss_ids_path = latest_dir / "faiss_post_ids.pkl"
            
#             if not faiss_index_path.exists():
#                 faiss_index_path = models_dir / "faiss_index.bin"
#                 faiss_ids_path = models_dir / "faiss_post_ids.pkl"
            
#             if faiss_index_path.exists() and faiss_ids_path.exists():
#                 faiss_index = faiss.read_index(str(faiss_index_path))
                
#                 with open(faiss_ids_path, 'rb') as f:
#                     faiss_post_ids = pickle.load(f)
                
#                 logger.info(f"✅ Loaded FAISS index: {faiss_index.ntotal} vectors from models")
#                 faiss_loaded = True
        
#         if not faiss_loaded:
#             logger.warning("⚠️ FAISS index not found, content recall will use brute-force")
    
#     except ImportError:
#         logger.warning("⚠️ FAISS not available, content recall will use brute-force")
#     except Exception as e:
#         logger.error(f"❌ Error loading FAISS: {e}")
    
#     # 7. Load data (for recall channels)
#     data = {}
    
#     # logger.info(f"📂 Loading data from: {data_dir}")
    
#     # # Load posts
#     # posts_path = data_dir / "posts.parquet"
#     # if posts_path.exists():
#     #     data['posts'] = pd.read_parquet(posts_path)
#     #     logger.info(f"✅ Loaded {len(data['posts'])} posts")
#     # else:
#     #     logger.warning(f"⚠️ Posts data not found at {posts_path}")
#     #     data['posts'] = pd.DataFrame()  # Empty DataFrame instead of None
    
#     # # Load reactions
#     # reactions_path = data_dir / "postreaction.parquet"
#     # if reactions_path.exists():
#     #     data['postreaction'] = pd.read_parquet(reactions_path)
#     #     logger.info(f"✅ Loaded {len(data['postreaction'])} reactions")
#     # else:
#     #     logger.warning(f"⚠️ Reactions data not found at {reactions_path}")
#     #     data['postreaction'] = pd.DataFrame()  # Empty DataFrame instead of None
    
#     # # Load friendships
#     # friendships_path = data_dir / "friendships.parquet"
#     # if friendships_path.exists():
#     #     data['friendships'] = pd.read_parquet(friendships_path)
#     #     logger.info(f"✅ Loaded {len(data['friendships'])} friendships")
#     # else:
#     #     logger.warning(f"⚠️ Friendships data not found at {friendships_path}")
#     #     data['friendships'] = pd.DataFrame()  # Empty DataFrame instead of None

#     mysql_url = (
#             f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
#             f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}"
#         )
#     from sqlalchemy import create_engine, text
#     engine = create_engine(mysql_url, pool_pre_ping=True)
        
#     with engine.connect() as conn:
#         # Load posts
#         try:
#             posts_query = "SELECT * FROM Post LIMIT 10000"  # Add limit for performance
#             data['posts'] = pd.read_sql(posts_query, conn)
#             logger.info(f"✅ Loaded {len(data['posts']):,} posts from MySQL")
#         except Exception as e:
#             logger.warning(f"⚠️ Failed to load posts: {e}")
#             data['posts'] = pd.DataFrame()
        
#         # Load reactions
#         try:
#             reactions_query = "SELECT * FROM PostReaction LIMIT 100000"
#             data['postreaction'] = pd.read_sql(reactions_query, conn)
#             logger.info(f"✅ Loaded {len(data['postreaction']):,} reactions from MySQL")
#         except Exception as e:
#             logger.warning(f"⚠️ Failed to load reactions: {e}")
#             data['postreaction'] = pd.DataFrame()
        
#         # Load friendships  
#         try:
#             friendships_query = "SELECT * FROM Friendship"
#             data['friendships'] = pd.read_sql(friendships_query, conn)
#             logger.info(f"✅ Loaded {len(data['friendships']):,} friendships from MySQL")
#         except Exception as e:
#             logger.warning(f"⚠️ Failed to load friendships: {e}")
#             data['friendships'] = pd.DataFrame()
        
#         # Load users
#         try:
#             users_query = "SELECT * FROM User"
#             data['users'] = pd.read_sql(users_query, conn)
#             logger.info(f"✅ Loaded {len(data['users']):,} users from MySQL")
#         except Exception as e:
#             logger.warning(f"⚠️ Failed to load users: {e}")
#             data['users'] = pd.DataFrame()
    
#     logger.info("✅ Data loading from MySQL complete")
    
#     # Fallback: empty data structures
#     data = {
#         'posts': pd.DataFrame(),
#         'postreaction': pd.DataFrame(),
#         'friendships': pd.DataFrame(),
#         'users': pd.DataFrame()
#     }
    
    
#     logger.info("✅ Model loading complete")
#     return models, embeddings, faiss_index, faiss_post_ids, data


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Lifespan context manager for startup/shutdown events
    
#     Startup:
#     - Connect to Redis
#     - Load ML models
#     - Load Faiss index
#     - Initialize recommendation service
    
#     Shutdown:
#     - Close Redis connection
#     - Cleanup resources
#     """
#     global recommendation_service_instance
    
#     # ==================== STARTUP ====================
#     logger.info("=" * 60)
#     logger.info("🚀 Starting Recommendation Service...")
#     logger.info("=" * 60)
    
#     try:
#         # 1. Connect to Redis
#         logger.info("📡 Connecting to Redis...")
#         cache_service.connect()
        
#         if cache_service.is_connected():
#             logger.info("✅ Redis connected successfully")
            
#             # Log cache stats
#             stats = cache_service.get_cache_stats()
#             logger.info(f"📊 Redis Stats:")
#             logger.info(f"   - Memory Used: {stats.get('used_memory_human', 'unknown')}")
#             logger.info(f"   - Hit Rate: {stats.get('hit_rate', 0):.2f}%")
#             logger.info(f"   - Connected Clients: {stats.get('connected_clients', 0)}")
#         else:
#             logger.error("❌ Redis connection failed!")
#             raise Exception("Failed to connect to Redis")
        
#         # 2. Load ML models and data
#         logger.info("🤖 Loading ML models...")
#         models, embeddings, faiss_index, faiss_post_ids, data = load_models()
#         logger.info("✅ ML models loaded")
        
#         # 3. Initialize Feature Engineer
#         logger.info("🔧 Initializing Feature Engineer...")
#         from recommender.common.feature_engineer import FeatureEngineer
        
#         # Build stats dicts
#         user_stats = {}
#         author_stats = {}
#         following_dict = {}
        
#         # FIX: NULL-SAFE friendships groupby
#         if 'friendships' in data and not data['friendships'].empty:
#             logger.info("📊 Building following dictionary from friendships...")
#             try:
#                 # Ensure correct column names (UserId, FriendId from MySQL schema)
#                 friendships_df = data['friendships']
#                 # Check which column names exist
#                 if 'UserId' in friendships_df.columns and 'FriendId' in friendships_df.columns:
#                     for user_id, group in friendships_df.groupby('UserId'):
#                         following_dict[int(user_id)] = set(group['FriendId'].astype(int).tolist())
#                 elif 'user_id' in friendships_df.columns and 'friend_id' in friendships_df.columns:
#                     for user_id, group in friendships_df.groupby('user_id'):
#                         following_dict[int(user_id)] = set(group['friend_id'].astype(int).tolist())
#                 else:
#                     logger.warning(f"⚠️ Unknown friendships columns: {friendships_df.columns.tolist()}")
                
#                 logger.info(f"✅ Built following dict: {len(following_dict)} users")
#             except Exception as e:
#                 logger.warning(f"⚠️ Failed to build following dict: {e}")
#                 following_dict = {}
#         else:
#             logger.warning("⚠️ No friendships data available - using empty following dict")
#             following_dict = {}
        
#         feature_engineer = FeatureEngineer(
#             data=data,
#             user_stats=user_stats,
#             author_stats=author_stats,
#             following=following_dict,
#             embeddings=embeddings
#         )
#         logger.info("✅ Feature Engineer initialized")
        
#         # 4. Initialize Recommendation Service
#         logger.info("🎯 Initializing Recommendation Service...")
#         from recommender.online.app.services.recommendation_service import RecommendationService
        
#         # Load config
#         import yaml
#         from pathlib import Path
#         docker_config = Path("/app/configs/config_online.yaml")
#         config_path = docker_config
#         try:
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#         except FileNotFoundError:
#             logger.warning(f"Config file not found: {config_path}, using defaults")
#             config = {
#                 'recall': {'channels': {}},
#                 'ranking': {},
#                 'reranking': {}
#             }
        
#         recommendation_service_instance = RecommendationService(
#             data=data,
#             models=models,
#             embeddings=embeddings,
#             faiss_index=faiss_index,
#             faiss_post_ids=faiss_post_ids,
#             feature_engineer=feature_engineer,
#             config=config
#         )
        
#         # Inject into recommendation API
#         recommendation.set_recommendation_service(recommendation_service_instance)
        
#         logger.info("✅ Recommendation Service initialized")
        
#         logger.info("=" * 60)
#         logger.info("✅ Recommendation Service Started Successfully!")
#         logger.info(f"📍 Environment: {settings.ENVIRONMENT}")
#         logger.info(f"🌐 API URL: http://{settings.HOST}:{settings.PORT}")
#         logger.info(f"📚 Docs: http://{settings.HOST}:{settings.PORT}/docs")
#         logger.info("=" * 60)
        
#         yield
        
#     except Exception as e:
#         logger.error(f"❌ Failed to start service: {e}", exc_info=True)
#         raise
#     finally:
#         # ==================== SHUTDOWN ====================
#         logger.info("🛑 Shutting down Recommendation Service...")
        
#         if cache_service:
#             cache_service.close()
#             logger.info("✅ Redis connection closed")
        
#         logger.info("👋 Shutdown complete")


# # Create FastAPI app
# app = FastAPI(
#     title=settings.PROJECT_NAME,
#     version="1.0.0",
#     description="Social News Feed Recommendation System",
#     lifespan=lifespan
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Request middleware for logging
# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     """Log all requests"""
#     start_time = time.time()
    
#     response = await call_next(request)
    
#     process_time = (time.time() - start_time) * 1000
#     logger.info(
#         f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms"
#     )
    
#     return response

# # Exception handlers
# @app.exception_handler(Exception)
# async def global_exception_handler(request: Request, exc: Exception):
#     """Handle all uncaught exceptions"""
#     logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
#     return JSONResponse(
#         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#         content={
#             "error": "Internal server error",
#             "message": str(exc) if settings.DEBUG else "An error occurred"
#         }
#     )

# # Include routers
# app.include_router(
#     health.router,
#     prefix=settings.API_V1_PREFIX,
#     tags=["health"]
# )

# app.include_router(
#     recommendation.router,
#     prefix=settings.API_V1_PREFIX + "/recommendation",
#     tags=["recommendation"]
# )

# # Root endpoint
# @app.get("/")
# async def root():
#     """Root endpoint"""
#     return {
#         "service": settings.PROJECT_NAME,
#         "version": "1.0.0",
#         "status": "running",
#         "environment": settings.ENVIRONMENT,
#         "docs": f"http://{settings.HOST}:{settings.PORT}/docs" if settings.DEBUG else None
#     }


# if __name__ == "__main__":
#     import uvicorn
    
#     uvicorn.run(
#         "main:app",
#         host=settings.HOST,
#         port=settings.PORT,
#         reload=settings.DEBUG
#     )


"""
FastAPI Main Application - COMPLETE FINAL FIX
==============================================

ALL FIXES:
1. ✅ Removed redundant parquet loading
2. ✅ MySQL data loading with NULL checks
3. ✅ FAISS new structure support
4. ✅ sklearn version compatibility
5. ✅ Column name normalization (UserId)
6. ✅ CacheService initialization fix (no parameters)

This is the FINAL working version.
"""

import os
import sys
import logging
import pickle
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import warnings

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import services
from recommender.online.app.services.cache_service import CacheService
from recommender.online.app.services.recommendation_service import RecommendationService
from recommender.online.app.config import settings
from recommender.online.app.api.v1 import health, recommendation

# Global instances
cache_service: Optional[CacheService] = None
recommendation_service_instance: Optional[RecommendationService] = None


# ==================== DATA UTILITIES ====================

def normalize_column_names(df: pd.DataFrame, table_name: str = "") -> pd.DataFrame:
    """
    Normalize column names to PascalCase
    
    MySQL: UserId, FriendId, PostId
    Pandas: may have userid, friendid, postid
    
    This ensures consistent PascalCase
    """
    if df.empty:
        return df
    
    column_mapping = {
        'userid': 'UserId',
        'user_id': 'UserId',
        'UserID': 'UserId',
        'USERID': 'UserId',
        
        'friendid': 'FriendId',
        'friend_id': 'FriendId',
        'FriendID': 'FriendId',
        'FRIENDID': 'FriendId',
        
        'postid': 'PostId',
        'post_id': 'PostId',
        'PostID': 'PostId',
        'POSTID': 'PostId',
        
        'id': 'Id',
        'ID': 'Id',
        'status': 'Status',
        'STATUS': 'Status',
        'createdate': 'CreateDate',
        'create_date': 'CreateDate',
    }
    
    df_normalized = df.copy()
    rename_dict = {}
    
    for col in df_normalized.columns:
        if col in column_mapping:
            rename_dict[col] = column_mapping[col]
        elif col.lower() in column_mapping:
            rename_dict[col] = column_mapping[col.lower()]
    
    if rename_dict:
        df_normalized.rename(columns=rename_dict, inplace=True)
        logger.debug(f"Normalized {len(rename_dict)} columns in {table_name}: {rename_dict}")
    
    return df_normalized


def safe_groupby_to_dict(
    df: pd.DataFrame,
    by_column: str,
    value_column: str,
    table_name: str = ""
) -> Dict:
    """
    Safe groupby that returns a dictionary
    
    Args:
        df: DataFrame to group
        by_column: Column to group by (e.g., 'UserId')
        value_column: Column to collect (e.g., 'FriendId')
        table_name: Name for logging
    
    Returns:
        Dict mapping by_column values to sets of value_column values
    """
    result = {}
    
    if df is None or df.empty:
        logger.warning(f"Cannot groupby on empty {table_name} DataFrame")
        return result
    
    # Normalize columns first
    df = normalize_column_names(df, table_name)
    
    # Check columns exist
    if by_column not in df.columns:
        logger.error(f"Column '{by_column}' not found in {table_name}")
        logger.error(f"Available columns: {df.columns.tolist()}")
        return result
    
    if value_column not in df.columns:
        logger.error(f"Column '{value_column}' not found in {table_name}")
        logger.error(f"Available columns: {df.columns.tolist()}")
        return result
    
    try:
        for key, group in df.groupby(by_column):
            result[int(key)] = set(group[value_column].astype(int).tolist())
        
        logger.info(f"✅ Built {table_name} dict: {len(result)} keys")
        return result
        
    except Exception as e:
        logger.error(f"Error grouping {table_name} by {by_column}: {e}", exc_info=True)
        return result


# ==================== LOAD MODELS AND DATA ====================

def load_models_and_data():
    """
    Load ML models, embeddings, and data
    """
    from sqlalchemy import create_engine, text
    import joblib
    
    models = {}
    embeddings = {'post': {}, 'user': {}}
    faiss_index = None
    faiss_post_ids = []
    data = {}
    
    try:
        # ==================== 1. LOAD ML MODELS ====================
        logger.info("📦 Loading models and data...")
        
        models_dir = Path('models')
        latest_file = models_dir / 'latest.txt'
        
        if latest_file.exists():
            model_version = latest_file.read_text().strip()
            logger.info(f"Using model version: {model_version}")
        else:
            version_dirs = sorted(
                [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('v_')],
                reverse=True
            )
            if version_dirs:
                model_version = version_dirs[0].name
                logger.info(f"Using latest model: {model_version}")
            else:
                raise FileNotFoundError("No model version found")
        
        model_dir = models_dir / model_version
        logger.info(f"📂 Loading from: {model_dir}")
        
        # Load models with sklearn warning suppression
        ranking_model_path = model_dir / 'ranking_model.pkl'
        if ranking_model_path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    models['ranking_model'] = joblib.load(ranking_model_path)
                logger.info("✅ Loaded ranking model")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load ranking model: {e}")
                models['ranking_model'] = None
        
        scaler_path = model_dir / 'scaler.pkl'
        if scaler_path.exists():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                models['scaler'] = joblib.load(scaler_path)
            logger.info("✅ Loaded scaler")
        
        feature_cols_path = model_dir / 'feature_columns.pkl'
        if feature_cols_path.exists():
            with open(feature_cols_path, 'rb') as f:
                models['feature_columns'] = pickle.load(f)
            logger.info(f"✅ Loaded {len(models['feature_columns'])} features")
        
        cf_model_path = model_dir / 'cf_similarities.pkl'
        if cf_model_path.exists():
            with open(cf_model_path, 'rb') as f:
                models['cf_model'] = pickle.load(f)
            logger.info("✅ Loaded CF model")
        
        # ==================== 2. LOAD EMBEDDINGS ====================
        
        embeddings_path = model_dir / 'post_embeddings.pkl'
        if embeddings_path.exists():
            with open(embeddings_path, 'rb') as f:
                loaded_embeddings = pickle.load(f)
                if isinstance(loaded_embeddings, dict):
                    if 'post' in loaded_embeddings and 'user' in loaded_embeddings:
                        embeddings = loaded_embeddings
                    else:
                        embeddings['post'] = loaded_embeddings
            n_users = len(embeddings.get('user', {}))
            n_posts = len(embeddings.get('post', {}))
            logger.info(f"✅ Embeddings: {n_users} users, {n_posts} posts")
        
        # ==================== 3. LOAD FAISS INDEX ====================
        
        embedding_dirs = sorted(
            [d for d in models_dir.iterdir() 
             if d.is_dir() and d.name.startswith('embeddings_')],
            reverse=True
        )
        
        faiss_loaded = False
        
        if embedding_dirs:
            for emb_dir in embedding_dirs:
                faiss_index_path = emb_dir / 'faiss_index.bin'
                faiss_ids_path = emb_dir / 'faiss_post_ids.pkl'
                
                if faiss_index_path.exists() and faiss_ids_path.exists():
                    try:
                        import faiss
                        faiss_index = faiss.read_index(str(faiss_index_path))
                        
                        with open(faiss_ids_path, 'rb') as f:
                            faiss_post_ids = pickle.load(f)
                        
                        logger.info(f"✅ FAISS: {faiss_index.ntotal} vectors from {emb_dir.name}")
                        faiss_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load FAISS from {emb_dir.name}: {e}")
                        continue
        
        if not faiss_loaded:
            faiss_index_path = model_dir / 'faiss_index.bin'
            faiss_ids_path = model_dir / 'faiss_post_ids.pkl'
            
            if faiss_index_path.exists() and faiss_ids_path.exists():
                try:
                    import faiss
                    faiss_index = faiss.read_index(str(faiss_index_path))
                    
                    with open(faiss_ids_path, 'rb') as f:
                        faiss_post_ids = pickle.load(f)
                    
                    logger.info(f"✅ FAISS: {faiss_index.ntotal} vectors")
                    faiss_loaded = True
                except Exception as e:
                    logger.warning(f"Failed to load FAISS: {e}")
        
        if not faiss_loaded:
            logger.warning("⚠️ No FAISS index")
        
        # ==================== 4. LOAD MYSQL DATA ====================
        
        logger.info("📊 Loading data from MySQL...")
        
        mysql_url = (
            f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
            f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}"
        )
        engine = create_engine(mysql_url, pool_pre_ping=True)
        
        with engine.connect() as conn:
            # Load friendships
            try:
                friendships_query = """
                    SELECT UserId, FriendId, Status 
                    FROM Friendship 
                    WHERE Status = 10
                """
                data['friendships'] = pd.read_sql(friendships_query, conn)
                data['friendships'] = normalize_column_names(data['friendships'], 'Friendship')
                
                logger.info(f"✅ Loaded {len(data['friendships']):,} friendships")
                logger.info(f"   Columns: {data['friendships'].columns.tolist()}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load friendships: {e}", exc_info=True)
                data['friendships'] = pd.DataFrame(columns=['UserId', 'FriendId', 'Status'])
            
            # Load recent posts
            try:
                posts_query = """
                    SELECT Id, UserId, CreateDate 
                    FROM Post 
                    WHERE Status = 10 
                    AND CreateDate >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                    LIMIT 10000
                """
                data['posts'] = pd.read_sql(posts_query, conn)
                data['posts'] = normalize_column_names(data['posts'], 'Post')
                logger.info(f"✅ Loaded {len(data['posts']):,} posts")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load posts: {e}")
                data['posts'] = pd.DataFrame()
            
            # Load users
            try:
                users_query = "SELECT Id, CreateDate FROM User WHERE Status = 10 LIMIT 10000"
                data['users'] = pd.read_sql(users_query, conn)
                data['users'] = normalize_column_names(data['users'], 'User')
                logger.info(f"✅ Loaded {len(data['users']):,} users")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load users: {e}")
                data['users'] = pd.DataFrame()
        
        logger.info("✅ Data loading complete")
        
    except Exception as e:
        logger.error(f"❌ Failed to load: {e}", exc_info=True)
        data = {
            'posts': pd.DataFrame(),
            'friendships': pd.DataFrame(columns=['UserId', 'FriendId', 'Status']),
            'users': pd.DataFrame()
        }
    
    return models, embeddings, faiss_index, faiss_post_ids, data


# ==================== LIFESPAN ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager
    
    FIXED: CacheService() no longer takes parameters
    """
    global cache_service, recommendation_service_instance
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Recommendation Service...")
    logger.info("=" * 60)
    
    try:
        # 1. Initialize Redis (NO PARAMETERS - uses settings internally)
        logger.info("📡 Connecting to Redis...")
        cache_service = CacheService()  # ✅ FIX: No parameters!
        cache_service.connect()
        logger.info("✅ Redis connected")
        
        stats = cache_service.get_cache_stats()
        logger.info("📊 Redis Stats:")
        logger.info(f"   - Memory: {stats.get('memory_used', 'N/A')}")
        logger.info(f"   - Hit Rate: {stats.get('hit_rate', 0):.2f}%")
        
        # 2. Load models
        logger.info("🤖 Loading ML models...")
        models, embeddings, faiss_index, faiss_post_ids, data = load_models_and_data()
        logger.info("✅ Models loaded")
        
        # 3. Build following dict
        logger.info("🔧 Building following dictionary...")
        
        following_dict = safe_groupby_to_dict(
            df=data.get('friendships'),
            by_column='UserId',
            value_column='FriendId',
            table_name='Friendship'
        )
        
        if not following_dict:
            logger.warning("⚠️ Following dict is empty")
            following_dict = {}
        else:
            logger.info(f"✅ Following dict: {len(following_dict)} users")
        
        # 4. Initialize Feature Engineer
        logger.info("🔧 Initializing Feature Engineer...")
        from recommender.common.feature_engineer import FeatureEngineer
        
        feature_engineer = FeatureEngineer(
            data=data,
            user_stats={},
            author_stats={},
            following=following_dict,
            embeddings=embeddings
        )
        logger.info("✅ Feature Engineer ready")
        
        # 5. Initialize Recommendation Service
        logger.info("🎯 Initializing Recommendation Service...")
        recommendation_service_instance = RecommendationService(
            cache_service=cache_service,
            models=models,
            embeddings=embeddings,
            faiss_index=faiss_index,
            faiss_post_ids=faiss_post_ids,
            feature_engineer=feature_engineer,
            data=data
        )
        logger.info("✅ Recommendation Service ready")
        
        logger.info("🔌 Injecting service into router...")
        recommendation.set_recommendation_service(recommendation_service_instance)
        logger.info("✅ Service injected into router")
        
        logger.info("=" * 60)
        logger.info("✅ Service started successfully!")
        logger.info("=" * 60)
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start: {e}", exc_info=True)
        raise
    
    finally:
        logger.info("🛑 Shutting down...")
        
        if cache_service:
            cache_service.close()
            logger.info("✅ Redis disconnected")
        
        logger.info("👋 Shutdown complete")


# ==================== CREATE APP ====================

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.3",
    description="Recommendation System - COMPLETE FIX",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== EXCEPTION HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Validation Error", "message": str(exc)}
    )


# ==================== ROUTERS ====================

app.include_router(health.router, prefix=settings.API_V1_PREFIX, tags=["health"])
app.include_router(recommendation.router, prefix=settings.API_V1_PREFIX + "/recommendation", tags=["recommendation"])


# ==================== ROOT ====================

@app.get("/")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": "1.0.3 - COMPLETE FIX",
        "status": "running",
        "all_fixes": [
            "✅ No parquet loading",
            "✅ Column normalization (UserId)",
            "✅ Safe groupby",
            "✅ FAISS new structure",
            "✅ sklearn 1.7.2",
            "✅ CacheService no params"
        ]
    }


@app.get("/info")
async def info():
    return {
        "service": settings.PROJECT_NAME,
        "version": "1.0.3",
        "redis": {
            "connected": cache_service.is_connected() if cache_service else False
        },
        "features": ["following", "cf", "content", "trending"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)