"""
Health Check Endpoints

Provides health status for the recommendation service and its dependencies.
"""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import time
from typing import Dict, Any

from recommender.online.app.services.cache_service import cache_service
from recommender.online.app.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint
    
    Returns quick health status without checking dependencies.
    Use this for load balancer health checks.
    """
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "timestamp": time.time()
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with all dependencies
    
    Checks:
    - API status
    - Redis connection
    - Cache statistics
    - TODO: ML model status
    - TODO: Faiss index status
    """
    health_status = {
        "status": "unknown",
        "service": settings.PROJECT_NAME,
        "timestamp": time.time(),
        "environment": settings.ENVIRONMENT,
        "components": {}
    }
    
    # Check API
    health_status["components"]["api"] = {
        "status": "healthy",
        "message": "API is running"
    }
    
    # Check Redis
    redis_healthy = cache_service.health_check()
    
    if redis_healthy:
        health_status["components"]["redis"] = {
            "status": "healthy",
            "message": "Redis is connected and responding",
            "details": {
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
                "db": settings.REDIS_DB
            }
        }
        
        # Get cache statistics
        try:
            stats = cache_service.get_cache_stats()
            health_status["cache_stats"] = {
                "memory_used": stats.get("used_memory_human", "unknown"),
                "memory_peak": stats.get("used_memory_peak_human", "unknown"),
                "hit_rate": stats.get("hit_rate", 0),
                "total_commands": stats.get("total_commands_processed", 0),
                "ops_per_sec": stats.get("instantaneous_ops_per_sec", 0),
                "connected_clients": stats.get("connected_clients", 0),
                "keyspace_hits": stats.get("keyspace_hits", 0),
                "keyspace_misses": stats.get("keyspace_misses", 0),
                "evicted_keys": stats.get("evicted_keys", 0),
                "expired_keys": stats.get("expired_keys", 0)
            }
        except Exception as e:
            health_status["cache_stats"] = {
                "error": str(e)
            }
    else:
        health_status["components"]["redis"] = {
            "status": "unhealthy",
            "message": "Redis is not responding",
            "details": {
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT
            }
        }
    
    # TODO: Check ML model
    health_status["components"]["ml_model"] = {
        "status": "not_loaded",
        "message": "ML model not yet implemented"
    }
    
    # TODO: Check Faiss index
    health_status["components"]["faiss_index"] = {
        "status": "not_loaded",
        "message": "Faiss index not yet implemented"
    }
    
    # Determine overall status
    component_statuses = [
        comp.get("status") for comp in health_status["components"].values()
    ]
    
    if all(s == "healthy" for s in component_statuses if s != "not_loaded"):
        health_status["status"] = "healthy"
    elif "unhealthy" in component_statuses:
        health_status["status"] = "unhealthy"
    else:
        health_status["status"] = "degraded"
    
    # Return appropriate status code
    if health_status["status"] == "healthy":
        return health_status
    elif health_status["status"] == "degraded":
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=health_status
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status
        )


@router.get("/health/ready")
async def readiness_probe():
    """
    Kubernetes readiness probe
    
    Returns 200 if service is ready to receive traffic.
    Returns 503 if service is not ready.
    
    Checks:
    - Redis connection is active
    - TODO: ML model is loaded
    - TODO: Faiss index is loaded
    """
    ready = True
    components = {}
    
    # Check Redis
    redis_ready = cache_service.is_connected()
    components["redis"] = redis_ready
    ready = ready and redis_ready
    
    # TODO: Check ML model
    # model_ready = ranking_service.is_loaded()
    # components["ml_model"] = model_ready
    # ready = ready and model_ready
    
    # TODO: Check Faiss index
    # faiss_ready = content_recall.is_loaded()
    # components["faiss_index"] = faiss_ready
    # ready = ready and faiss_ready
    
    response = {
        "ready": ready,
        "components": components,
        "timestamp": time.time()
    }
    
    if ready:
        return response
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response
        )


@router.get("/health/live")
async def liveness_probe():
    """
    Kubernetes liveness probe
    
    Returns 200 if service is alive (process is running).
    This should almost always return 200 unless the process is deadlocked.
    """
    return {
        "alive": True,
        "timestamp": time.time()
    }


@router.get("/health/cache")
async def cache_health():
    """
    Detailed cache health and statistics
    
    Provides comprehensive information about Redis cache performance.
    """
    if not cache_service.is_connected():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "message": "Cache is not connected",
                "timestamp": time.time()
            }
        )
    
    try:
        stats = cache_service.get_cache_stats()
        
        # Calculate derived metrics
        hit_rate = stats.get("hit_rate", 0)
        
        # Determine health based on metrics
        health = "healthy"
        issues = []
        
        # Check hit rate
        if hit_rate < 50:
            health = "degraded"
            issues.append(f"Low hit rate: {hit_rate:.2f}% (target: >80%)")
        
        # Check evicted keys
        evicted = stats.get("evicted_keys", 0)
        if evicted > 1000:
            health = "degraded"
            issues.append(f"High eviction rate: {evicted} keys evicted")
        
        return {
            "status": health,
            "issues": issues if issues else None,
            "statistics": {
                "memory": {
                    "used": stats.get("used_memory_human", "unknown"),
                    "used_bytes": stats.get("used_memory", 0),
                    "peak": stats.get("used_memory_peak_human", "unknown")
                },
                "performance": {
                    "hit_rate": f"{hit_rate:.2f}%",
                    "hits": stats.get("keyspace_hits", 0),
                    "misses": stats.get("keyspace_misses", 0),
                    "ops_per_sec": stats.get("instantaneous_ops_per_sec", 0)
                },
                "keys": {
                    "evicted": stats.get("evicted_keys", 0),
                    "expired": stats.get("expired_keys", 0)
                },
                "connections": {
                    "clients": stats.get("connected_clients", 0)
                },
                "commands": {
                    "total_processed": stats.get("total_commands_processed", 0)
                }
            },
            "configuration": {
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
                "db": settings.REDIS_DB,
                "ttl_settings": {
                    "user_profile": f"{settings.USER_PROFILE_TTL}s",
                    "post_features": f"{settings.POST_FEATURES_TTL}s",
                    "following_cache": f"{settings.FOLLOWING_CACHE_TTL}s",
                    "trending_cache": f"{settings.TRENDING_CACHE_TTL}s"
                }
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Failed to get cache statistics",
                "error": str(e),
                "timestamp": time.time()
            }
        )