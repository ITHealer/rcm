"""
BASE RECALL CLASS
=================
Abstract base class for all recall channels

All recall channels must inherit from this class and implement:
- recall(user_id, k) -> List[int]
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class BaseRecall(ABC):
    """
    Abstract base class for recall channels
    
    All recall channels implement this interface for consistency
    """
    
    def __init__(
        self,
        redis_client=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize recall channel
        
        Args:
            redis_client: Redis connection (optional)
            config: Configuration dict
        """
        self.redis = redis_client
        self.config = config or {}
        self.metrics = {
            'latency': [],
            'count': [],
            'errors': 0
        }
        self.name = self.__class__.__name__
    
    @abstractmethod
    def recall(self, user_id: int, k: int = 100) -> List[int]:
        """
        Recall candidates for user
        
        Args:
            user_id: Target user ID
            k: Number of candidates to return
            
        Returns:
            List of post IDs
        """
        pass
    
    def recall_with_metrics(self, user_id: int, k: int = 100) -> List[int]:
        """
        Recall with performance tracking
        
        Args:
            user_id: Target user ID
            k: Number of candidates
            
        Returns:
            List of post IDs
        """
        start_time = time.time()
        
        try:
            # Call the implemented recall method
            results = self.recall(user_id, k)
            
            # Track metrics
            latency_ms = (time.time() - start_time) * 1000
            self.metrics['latency'].append(latency_ms)
            self.metrics['count'].append(len(results))
            
            logger.debug(
                f"{self.name}: {len(results)} posts in {latency_ms:.1f}ms"
            )
            
            return results
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"{self.name} error: {e}", exc_info=True)
            return []
    
    def get_metrics(self) -> Dict:
        """
        Get performance metrics
        
        Returns:
            Dict with latency and count statistics
        """
        if not self.metrics['latency']:
            return {
                'name': self.name,
                'calls': 0,
                'avg_latency_ms': 0,
                'avg_count': 0,
                'errors': self.metrics['errors']
            }
        
        return {
            'name': self.name,
            'calls': len(self.metrics['latency']),
            'avg_latency_ms': sum(self.metrics['latency']) / len(self.metrics['latency']),
            'p50_latency_ms': sorted(self.metrics['latency'])[len(self.metrics['latency'])//2],
            'p95_latency_ms': sorted(self.metrics['latency'])[int(len(self.metrics['latency'])*0.95)],
            'avg_count': sum(self.metrics['count']) / len(self.metrics['count']),
            'errors': self.metrics['errors']
        }
    
    def reset_metrics(self):
        """Reset metrics"""
        self.metrics = {
            'latency': [],
            'count': [],
            'errors': 0
        }
    
    def _get_from_cache(self, key: str) -> Optional[any]:
        """
        Get data from Redis cache
        
        Args:
            key: Redis key
            
        Returns:
            Cached data or None
        """
        if self.redis is None:
            return None
        
        try:
            return self.redis.get(key)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None
    
    def _set_to_cache(
        self,
        key: str,
        value: any,
        ttl: int = 3600
    ):
        """
        Set data to Redis cache
        
        Args:
            key: Redis key
            value: Data to cache
            ttl: Time to live in seconds
        """
        if self.redis is None:
            return
        
        try:
            self.redis.setex(key, ttl, value)
        except Exception as e:
            logger.warning(f"Redis set error: {e}")