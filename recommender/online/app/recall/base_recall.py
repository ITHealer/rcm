"""
Base Recall Channel
==================
Abstract base class for all recall channels
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import time
import logging

logger = logging.getLogger(__name__)


class BaseRecall(ABC):
    """
    Abstract base class for recall channels
    
    All recall channels must implement:
    - recall(user_id, k) -> List[int]
    """
    
    def __init__(self, redis_client=None, config: Optional[Dict] = None):
        """
        Initialize base recall
        
        Args:
            redis_client: Redis connection for caching
            config: Channel-specific configuration
        """
        self.redis = redis_client
        self.config = config or {}
        
        # Metrics
        self.metrics = {
            'total_recalls': 0,
            'total_candidates': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time_ms': 0
        }
    
    @abstractmethod
    def recall(self, user_id: int, k: int = 100) -> List[int]:
        """
        Recall candidates for user
        
        Args:
            user_id: User ID
            k: Number of candidates to return
            
        Returns:
            List of post IDs
        """
        pass
    
    def get_metrics(self) -> Dict:
        """Get channel metrics"""
        avg_time = (
            self.metrics['total_time_ms'] / self.metrics['total_recalls']
            if self.metrics['total_recalls'] > 0 else 0
        )
        
        hit_rate = (
            self.metrics['cache_hits'] / 
            (self.metrics['cache_hits'] + self.metrics['cache_misses'])
            if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 
            else 0
        )
        
        return {
            **self.metrics,
            'avg_time_ms': round(avg_time, 2),
            'cache_hit_rate': round(hit_rate * 100, 2)
        }
    
    def reset_metrics(self):
        """Reset metrics"""
        for key in self.metrics:
            self.metrics[key] = 0
    
    def _cache_get(self, key: str) -> Optional[str]:
        """Get from cache"""
        if not self.redis:
            return None
        
        try:
            return self.redis.get(key)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def _cache_set(self, key: str, value: str, ttl: int = 3600):
        """Set to cache"""
        if not self.redis:
            return
        
        try:
            self.redis.setex(key, ttl, value)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")