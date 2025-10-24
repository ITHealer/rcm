"""
UPDATE REDIS CACHE AFTER OFFLINE TRAINING
==========================================
Run after main_offline_pipeline.py completes

Updates:
- User/Post embeddings
- CF similarities
- User/Author stats
- Model metadata

Usage: python scripts/offline/update_cache.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import argparse

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

# Imports
from recommender.offline.artifact_manager import ArtifactManager
from redis_cache_manager import RedisCacheManager


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main(
    models_dir: str = 'models',
    redis_host: str = 'localhost',
    redis_port: int = 6380,
    redis_db: int = 0,
    redis_password: str = None
):
    """
    Update Redis cache from latest trained artifacts
    
    Args:
        models_dir: Directory with model artifacts
        redis_host: Redis host
        redis_port: Redis port
        redis_db: Redis database
        redis_password: Redis password
    """
    logger.info("="*70)
    logger.info("UPDATE REDIS CACHE FROM ARTIFACTS")
    logger.info("="*70)
    logger.info(f"Started at: {datetime.now()}")
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Redis: {redis_host}:{redis_port}/{redis_db}")
    
    try:
        # ════════════════════════════════════════════════════════════════
        # STEP 1: LOAD LATEST ARTIFACTS
        # ════════════════════════════════════════════════════════════════
        
        logger.info("\n" + "="*70)
        logger.info("STEP 1: LOADING ARTIFACTS")
        logger.info("="*70)
        
        artifact_mgr = ArtifactManager(base_dir=models_dir)
        
        # Get latest version
        version = artifact_mgr.get_latest_version()
        logger.info(f"Latest version: {version}")
        
        # Load artifacts
        artifacts = artifact_mgr.load_artifacts(version)
        
        logger.info("✅ Artifacts loaded successfully")
        logger.info(f"   User embeddings: {len(artifacts['embeddings']['user']):,}")
        logger.info(f"   Post embeddings: {len(artifacts['embeddings']['post']):,}")
        logger.info(f"   CF user similarities: {len(artifacts['cf_model']['user_similarities']):,}")
        logger.info(f"   CF item similarities: {len(artifacts['cf_model']['item_similarities']):,}")
        logger.info(f"   User stats: {len(artifacts['user_stats']):,}")
        logger.info(f"   Author stats: {len(artifacts['author_stats']):,}")
        
        # ════════════════════════════════════════════════════════════════
        # STEP 2: CONNECT TO REDIS
        # ════════════════════════════════════════════════════════════════
        
        logger.info("\n" + "="*70)
        logger.info("STEP 2: CONNECTING TO REDIS")
        logger.info("="*70)
        
        cache = RedisCacheManager(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password
        )
        
        # Print current cache info
        logger.info("\nCurrent cache state:")
        cache.print_info()
        
        # ════════════════════════════════════════════════════════════════
        # STEP 3: UPDATE CACHE
        # ════════════════════════════════════════════════════════════════
        
        logger.info("\n" + "="*70)
        logger.info("STEP 3: UPDATING CACHE")
        logger.info("="*70)
        
        # Update all layers
        cache.update_all_from_artifacts(artifacts)
        
        # ════════════════════════════════════════════════════════════════
        # STEP 4: VERIFY UPDATE
        # ════════════════════════════════════════════════════════════════
        
        logger.info("\n" + "="*70)
        logger.info("STEP 4: VERIFYING UPDATE")
        logger.info("="*70)
        
        # Check model version
        cached_version = cache.get_model_version()
        logger.info(f"Model version in cache: {cached_version}")
        
        if cached_version == version:
            logger.info("✅ Version matches!")
        else:
            logger.warning(f"⚠️  Version mismatch: {cached_version} != {version}")
        
        # Print updated cache info
        logger.info("\nUpdated cache state:")
        cache.print_info()
        
        # Sample checks
        logger.info("\nSample data checks:")
        
        # Check user embedding
        user_ids = list(artifacts['embeddings']['user'].keys())
        if user_ids:
            sample_user_id = user_ids[0]
            cached_emb = cache.get_user_embedding(sample_user_id)
            if cached_emb is not None:
                logger.info(f"   ✅ User {sample_user_id} embedding: cached")
            else:
                logger.warning(f"   ⚠️  User {sample_user_id} embedding: NOT cached")
        
        # Check CF similarity
        if artifacts['cf_model']['user_similarities']:
            sample_user_id = list(artifacts['cf_model']['user_similarities'].keys())[0]
            cached_sims = cache.get_user_similar_users(sample_user_id)
            if cached_sims:
                logger.info(f"   ✅ User {sample_user_id} similarities: {len(cached_sims)} cached")
            else:
                logger.warning(f"   ⚠️  User {sample_user_id} similarities: NOT cached")
        
        # Check stats
        if artifacts['user_stats']:
            sample_user_id = list(artifacts['user_stats'].keys())[0]
            cached_stats = cache.get_user_stats(sample_user_id)
            if cached_stats:
                logger.info(f"   ✅ User {sample_user_id} stats: cached")
            else:
                logger.warning(f"   ⚠️  User {sample_user_id} stats: NOT cached")
        
        # ════════════════════════════════════════════════════════════════
        # COMPLETE
        # ════════════════════════════════════════════════════════════════
        
        logger.info("\n" + "="*70)
        logger.info("✅ CACHE UPDATE COMPLETE!")
        logger.info("="*70)
        logger.info(f"Version: {version}")
        logger.info(f"Completed at: {datetime.now()}")
        logger.info("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ CACHE UPDATE FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    """
    Command-line interface
    """
    parser = argparse.ArgumentParser(description="Update Redis cache from trained artifacts")
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory with model artifacts (default: models)'
    )
    
    parser.add_argument(
        '--redis-host',
        type=str,
        default='localhost',
        help='Redis host (default: localhost)'
    )
    
    parser.add_argument(
        '--redis-port',
        type=int,
        default=6380,
        help='Redis port (default: 6379)'
    )
    
    parser.add_argument(
        '--redis-db',
        type=int,
        default=0,
        help='Redis database number (default: 0)'
    )
    
    parser.add_argument(
        '--redis-password',
        type=str,
        default=None,
        help='Redis password (optional)'
    )
    
    args = parser.parse_args()
    
    # Run
    exit_code = main(
        models_dir=args.models_dir,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        redis_password=args.redis_password
    )
    
    sys.exit(exit_code)