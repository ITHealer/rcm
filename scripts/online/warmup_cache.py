"""
CACHE WARMUP SCRIPT
===================
Warm up Redis cache on startup

Usage: python scripts/online/warmup_cache.py
"""

import sys
from pathlib import Path
import logging

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from recommender.offline.artifact_manager import ArtifactManager
from recommender.online.cache.redis_cache_manager import RedisCacheManager

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Warm up cache"""
    logger.info("="*70)
    logger.info("CACHE WARMUP")
    logger.info("="*70)
    
    # Initialize
    artifact_mgr = ArtifactManager()
    cache_mgr = RedisCacheManager()
    
    # Load latest artifacts
    version = artifact_mgr.get_latest_version()
    logger.info(f"Loading artifacts from {version}")
    
    artifacts = artifact_mgr.load_artifacts(version)
    
    # Warm up cache
    logger.info("\n🔥 Warming up cache...")
    
    # 1. Embeddings
    cache_mgr.set_embeddings_batch(artifacts['embeddings'])
    
    # 2. CF similarities
    cache_mgr.set_cf_similarities_batch(artifacts['cf_model'])
    
    # 3. Stats
    cache_mgr.set_stats_batch(
        artifacts['user_stats'],
        artifacts['author_stats']
    )
    
    logger.info("✅ Cache warmed up successfully!")
    cache_mgr.print_info()


if __name__ == "__main__":
    main()