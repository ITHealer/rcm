"""
FASTAPI APPLICATION - RECOMMENDATION API
========================================
RESTful API for recommendation system

Endpoints:
- GET /feed         : Get personalized feed (multi-channel + seen-filter + fallback)
- GET /friends      : Get friend recommendations
- POST /interaction : Log user interaction (+realtime mirrors to Redis; mark seen on view)
- POST /webhook/post_created : BE notifies new post (fanout to followers)
- GET /trending     : Get trending posts
- GET /health       : Health check
- GET /metrics      : System metrics
- GET /debug/*      : Debug endpoints (dev only)
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add project root
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Import pipeline
from recommender.online.online_inference_pipeline import OnlineInferencePipeline
# Async telemetry logger (non-blocking)
from recommender.online.telemetry import AsyncDBLogger

# Realtime backend ingestor (optional)
try:
    from recommender.online.realtime_ingestor import BackendIngestor
except Exception:  # dev fallback
    BackendIngestor = None

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
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
    version="1.0.0",
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
app.state.telemetry = None      # type: Optional[AsyncDBLogger]
app.state.ingestor = None       # type: Optional[Any]

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
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
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
                    (author_id,),
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
        df = getattr(pipeline, "data", {}).get("posts")
        if df is not None and not df.empty:
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
        logger.info("=" * 70)
        logger.info("🚀 STARTING RECOMMENDATION API")
        logger.info("=" * 70)

        config_path = os.getenv("CONFIG_PATH", "configs/config_online.yaml")
        models_dir = os.getenv("MODELS_DIR", "models")
        data_dir = os.getenv("DATA_DIR", "dataset")

        # Nếu dùng config Redis trong YAML thì pipeline sẽ tự đọc;
        # biến env REDIS_HOST chỉ là gợi ý cho môi trường dev.
        use_redis = True

        logger.info(f"Config path: {config_path}")
        logger.info(f"Models dir: {models_dir}")
        logger.info(f"Data dir: {data_dir}")

        pipeline = OnlineInferencePipeline(
            config_path=config_path,
            models_dir=models_dir,
            data_dir=data_dir,
            use_redis=use_redis,
        )
        logger.info("✅ Pipeline initialized successfully!")

        # Start realtime ingestor (optional)
        try:
            if BackendIngestor is not None:
                ingest_cfg = pipeline.config.get("realtime_ingest", {}) or {}
                if ingest_cfg.get("enabled", False):
                    app.state.ingestor = BackendIngestor(
                        cfg=ingest_cfg,
                        mysql_engine=pipeline.db_engine,   # may be Postgres engine
                        redis_client=pipeline.redis,
                        pipeline=pipeline,
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

        logger.info("=" * 70)
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
        "docs": "/docs",
    }

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    components = {
        "pipeline": pipeline is not None,
        "redis": False,
        "models": False,
    }
    if pipeline is not None:
        components["redis"] = pipeline.redis is not None
        components["models"] = pipeline.ranking_model is not None
    status = "healthy" if all(components.values()) else "degraded"
    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        components=components,
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    metrics = pipeline.get_metrics()
    return MetricsResponse(
        total_requests=metrics.get("total_requests", 0),
        avg_latency_ms=metrics.get("avg_latency_ms", 0.0),
        p50_latency_ms=metrics.get("p50_latency_ms", 0.0),
        p95_latency_ms=metrics.get("p95_latency_ms", 0.0),
        p99_latency_ms=metrics.get("p99_latency_ms", 0.0),
        error_rate=0.0,  # TODO: track errors
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
    exclude_seen: Optional[str] = Query(None, description="Comma-separated post IDs to exclude"),
    mark_seen: bool = Query(False, description="Mark returned posts as seen")
):
    """
    Trả về danh sách posts đã rerank. Pipeline đã:
      - multi-channel recall với quota
      - seen-dedupe (Redis ZSET user:{uid}:seen_posts, TTL=7d)
      - fallback (trending→popular→random recent) khi recall rỗng
      - mark seen sau khi trả feed
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        # Parse exclude_seen
        exclude_list = None
        if exclude_seen:
            try:
                exclude_list = [int(x.strip()) for x in exclude_seen.split(",") if x.strip()]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid exclude_seen format")

        t0 = time.time()
        feed = pipeline.generate_feed(user_id=user_id, limit=limit, exclude_seen=exclude_list, mark_seen=mark_seen)
        t1 = time.time()

        # Telemetry (non-blocking)
        tel = getattr(app.state, "telemetry", None)
        if tel:
            try:
                stage_ms = {
                    "recall": int(pipeline.metrics["recall_latency"][-1]) if pipeline.metrics["recall_latency"] else 0,
                    "ranking": int(pipeline.metrics["ranking_latency"][-1]) if pipeline.metrics["ranking_latency"] else 0,
                    "rerank": int(pipeline.metrics["reranking_latency"][-1]) if pipeline.metrics["reranking_latency"] else 0,
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
                    "stage_ms": stage_ms,
                })
            except Exception:
                pass

        # Chuẩn hóa output: pipeline đã trả list[dict] với post_id/score/author_id
        return {
            "user_id": user_id,
            "posts": feed or [],
            "count": len(feed or []),
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/friends")
async def get_friend_recommendations(
    user_id: int = Query(..., description="User ID"),
    limit: int = Query(20, ge=1, le=50, description="Number of recommendations"),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        recommendations = pipeline.recommend_friends(user_id=user_id, k=limit)
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations),
            "timestamp": datetime.now().isoformat(),
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
    + mark seen on 'view'
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        valid_actions = ["like", "comment", "share", "save", "view", "hide", "report"]
        if request.action not in valid_actions:
            raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {', '.join(valid_actions)}")

        # 1) Update user embedding realtime (content-based reacts quickly)
        pipeline.update_user_embedding_realtime(
            user_id=request.user_id,
            post_id=request.post_id,
            action=request.action,
        )

        # 2) Mirror realtime to Redis (+covisit)
        r = getattr(pipeline, "redis", None)
        author_id = _get_author_id_from_pipeline(request.post_id)
        if r is not None:
            try:
                # user 24h counters
                r.hincrby(f"user:{request.user_id}:eng24h", request.action, 1)
                r.expire(f"user:{request.user_id}:eng24h", 24 * 3600)

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
                r.expire(f"user:{request.user_id}:recent_items", 7 * 24 * 3600)
                recent = r.lrange(f"user:{request.user_id}:recent_items", 0, 50) or []
                for rid in recent:
                    try:
                        rid = int(rid)
                        if rid == request.post_id:
                            continue
                        r.zincrby(f"covisit:{rid}", 1.0, request.post_id)
                        r.zincrby(f"covisit:{request.post_id}", 1.0, rid)
                        r.expire(f"covisit:{rid}", 7 * 24 * 3600)
                        r.expire(f"covisit:{request.post_id}", 7 * 24 * 3600)
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Realtime mirror to Redis failed: {e}")

        # 3) Mark seen when 'view'
        if request.action == "view":
            try:
                pipeline._mark_seen_posts(request.user_id, [request.post_id])  # internal but safe
            except Exception:
                pass

        # 4) Telemetry (non-blocking DB logging)
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
                    "meta": request.meta or {},
                })
            except Exception as e:
                logger.debug(f"telemetry log_interaction failed: {e}")

        return {
            "success": True,
            "message": f"Interaction logged: user {request.user_id} {request.action}d post {request.post_id}",
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging interaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/post_created")
async def webhook_post_created(
    payload: PostCreatedWebhook,
    x_webhook_token: Optional[str] = Header(default=None, convert_underscores=False),
):
    """
    Webhook từ BE khi tác giả tạo bài mới.
    Công việc:
      1) Ghi meta & active_posts vào Redis (cho trending/job nền & content-based)
      2) Fanout sorted-set vào cache following của followers
      3) (optional) Telemetry
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
            try:
                r.hset(f"post:{post_id}:meta", mapping={
                    "author_id": author_id,
                    "created_at": ts_epoch,
                })
                r.expire(f"post:{post_id}:meta", 30 * 24 * 3600)
                r.zadd("active_posts", {post_id: ts_epoch})
                r.expire("active_posts", 7 * 24 * 3600)
            except Exception:
                pass

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
                    followers=followers,
                )
                updated = len(followers or [])
            else:
                updated = _fanout_following(author_id, post_id, ts_epoch, followers)
        except Exception as e:
            logger.warning(f"rt_handlers.on_author_create_post failed, fallback fanout: {e}")
            updated = _fanout_following(author_id, post_id, ts_epoch, followers)

        # Optional telemetry (no generic log_event in AsyncDBLogger; skip to avoid errors)

        return {
            "success": True,
            "post_id": post_id,
            "author_id": author_id,
            "followers_fanout": updated,
            "created_at_epoch": ts_epoch,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"webhook_post_created error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trending")
async def get_trending(
    limit: int = Query(100, ge=1, le=200, description="Number of trending posts"),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        trending_posts = pipeline.trending_recall.recall(k=limit)
        return {
            "posts": trending_posts,
            "count": len(trending_posts),
            "timestamp": datetime.now().isoformat(),
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
            "post_count": len(pipeline.embeddings.get("post", {})),
            "user_count": len(pipeline.embeddings.get("user", {})),
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
            "trending": pipeline.trending_recall is not None,
            "covisit": pipeline.covisit_recall is not None,
        },
    }

@app.get("/debug/user/{user_id}")
async def debug_user(user_id: int):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return {
        "user_id": user_id,
        "has_embedding": user_id in pipeline.embeddings.get("user", {}),
        "has_stats": user_id in pipeline.user_stats,
        "following_count": len(pipeline.following_dict.get(user_id, [])),
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
            "timestamp": datetime.now().isoformat(),
        },
    )

# ============================================================================
# MAIN (for local testing)
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8010))
    print(f"\n🚀 Starting API server on {host}:{port}\n")
    uvicorn.run("api:app", host=host, port=port, reload=True, log_level="info")




