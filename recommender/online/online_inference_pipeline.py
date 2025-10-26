# """
# ONLINE INFERENCE PIPELINE - PRODUCTION READY
# =============================================
# Complete recommendation pipeline integrating all components

# Architecture:
# 1. Load artifacts (models, embeddings, stats)
# 2. Initialize recall channels (Following, CF, Content, Trending)
# 3. Initialize ranker & reranker
# 4. Real-time user embedding updates
# 5. Feed generation with < 200ms latency

# Flow: Recall → Feature Extraction → Ranking → Re-ranking → Output
# """

# import os
# import sys
# import time
# import numpy as np
# import pandas as pd
# from typing import List, Dict, Optional
# from datetime import datetime
# from pathlib import Path
# import logging
# import yaml
# import warnings
# from collections import defaultdict
# from dataclasses import dataclass

# warnings.filterwarnings('ignore')

# # Add project root
# sys.path.append(str(Path(__file__).parent.parent.parent))

# # Core imports
# from recommender.offline.artifact_manager import ArtifactManager
# from recommender.common.feature_engineer import FeatureEngineer

# # Recall channels
# from recommender.online.recall import (
#     FollowingRecall,
#     CFRecall,
#     ContentRecall,
#     TrendingRecall
# )

# # Ranking
# from recommender.online.ranking import MLRanker, Reranker

# # Friend recommendation
# from recommender.online.friend_recommendation import FriendRecommendation

# from recommender.online.recall.cf_realtime import CFRealtimeRecall

# # Try optional imports
# try:
#     import redis
#     REDIS_AVAILABLE = True
# except ImportError:
#     REDIS_AVAILABLE = False
#     print("⚠️  redis not available")

# try:
#     import lightgbm as lgb
#     LIGHTGBM_AVAILABLE = True
# except ImportError:
#     LIGHTGBM_AVAILABLE = False
#     print("⚠️  lightgbm not available")

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# @dataclass
# class RecallDiag:
#     following: int = 0
#     cf: int = 0
#     content: int = 0
#     trending: int = 0
#     total: int = 0
#     notes: Dict[str, str] = None  # optional human-readable notes


# class OnlineInferencePipeline:
#     """
#     Complete online inference pipeline
    
#     Features:
#     - Multi-channel recall (4 channels)
#     - ML ranking with LightGBM
#     - Business rules re-ranking
#     - Real-time user embedding updates
#     - Friend recommendations
#     - Redis caching
#     - < 200ms latency
#     """
    
#     def __init__(
#         self,
#         config_path: str = 'configs/config_online.yaml',
#         models_dir: str = 'models',
#         data_dir: str = 'dataset',
#         use_redis: bool = True
#     ):
#         """
#         Initialize online pipeline
        
#         Args:
#             config_path: Path to online config file
#             models_dir: Directory with model artifacts
#             data_dir: Directory with data (for development)
#             use_redis: Whether to use Redis cache
#         """
#         self.models_dir = Path(models_dir)
#         self.data_dir = Path(data_dir)
        
#         # ════════════════════════════════════════════════════════════════
#         # STEP 1: LOAD CONFIGURATION
#         # ════════════════════════════════════════════════════════════════
        
#         logger.info("\n" + "="*70)
#         logger.info("INITIALIZING ONLINE INFERENCE PIPELINE")
#         logger.info("="*70)
        
#         logger.info("\n📋 Step 1: Loading configuration...")
#         with open(config_path, 'r', encoding='utf-8') as f:
#             self.config = yaml.safe_load(f)
        
#         self.recall_config = self.config.get('recall', {})
#         self.ranking_config = self.config.get('ranking', {})
#         self.reranking_config = self.config.get('reranking', {})
        
#         logger.info("✅ Configuration loaded")
        
#         # ════════════════════════════════════════════════════════════════
#         # STEP 2: CONNECT TO REDIS
#         # ════════════════════════════════════════════════════════════════
        
#         logger.info("\n🔧 Step 2: Connecting to Redis...")
        
#         self.redis = None
#         if use_redis and REDIS_AVAILABLE:
#             try:
#                 redis_config = self.config.get('redis', {})
#                 self.redis = redis.Redis(
#                     host=redis_config.get('host', 'localhost'),
#                     port=redis_config.get('port', 6380),
#                     db=redis_config.get('db', 0),
#                     decode_responses=False,
#                     socket_timeout=redis_config.get('socket_timeout', 5),
#                     max_connections=redis_config.get('max_connections', 50)
#                 )
#                 # Test connection
#                 self.redis.ping()
#                 logger.info("✅ Redis connected")
#             except Exception as e:
#                 logger.warning(f"Redis connection failed: {e}")
#                 self.redis = None
#         else:
#             logger.warning("Redis disabled or not available")
        
#         # ════════════════════════════════════════════════════════════════
#         # STEP 3: LOAD ARTIFACTS
#         # ════════════════════════════════════════════════════════════════
        
#         logger.info("\n📦 Step 3: Loading model artifacts...")
        
#         self.artifact_mgr = ArtifactManager(base_dir=str(self.models_dir))
        
#         # Get version
#         model_version = self.config.get('models', {}).get('version', 'latest')
#         if model_version == 'latest':
#             self.current_version = self.artifact_mgr.get_latest_version()
#         else:
#             self.current_version = model_version
        
#         logger.info(f"Loading version: {self.current_version}")
        
#         # Load all artifacts
#         artifacts = self.artifact_mgr.load_artifacts(self.current_version)
        
#         # Extract components
#         self.embeddings = artifacts['embeddings']
#         self.faiss_index = artifacts.get('faiss_index')
#         self.faiss_post_ids = artifacts.get('faiss_post_ids', [])
#         self.cf_model = artifacts['cf_model']
#         self.ranking_model = artifacts['ranking_model']
#         self.ranking_scaler = artifacts['ranking_scaler']
#         self.ranking_feature_cols = artifacts['ranking_feature_cols']
#         self.user_stats = artifacts['user_stats']
#         self.author_stats = artifacts['author_stats']
#         self.following_dict = artifacts['following_dict']
#         self.metadata = artifacts.get('metadata', {})
        
#         logger.info("✅ Artifacts loaded successfully!")
#         logger.info(f"   Post embeddings: {len(self.embeddings['post']):,}")
#         logger.info(f"   User embeddings: {len(self.embeddings['user']):,}")
#         logger.info(f"   Ranking features: {len(self.ranking_feature_cols)}")
        
#         # ════════════════════════════════════════════════════════════════
#         # STEP 4: LOAD DATA (for development)
#         # ════════════════════════════════════════════════════════════════
        
#         logger.info("\n📊 Step 4: Loading data...")
        
#         from recommender.common.data_loader import load_data
#         self.data = load_data(str(self.data_dir))
        
#         logger.info("✅ Data loaded")
        
#         # ════════════════════════════════════════════════════════════════
#         # STEP 5: INITIALIZE FEATURE ENGINEER
#         # ════════════════════════════════════════════════════════════════
        
#         logger.info("\n🔧 Step 5: Initializing feature engineer...")
        
#         self.feature_engineer = FeatureEngineer(
#             data=self.data,
#             user_stats=self.user_stats,
#             author_stats=self.author_stats,
#             following_dict=self.following_dict,
#             embeddings=self.embeddings
#         )
        
#         logger.info("✅ Feature engineer initialized")
        
#         # ════════════════════════════════════════════════════════════════
#         # STEP 6: INITIALIZE RECALL CHANNELS
#         # ════════════════════════════════════════════════════════════════
        
#         logger.info("\n🔧 Step 6: Initializing recall channels...")
        
#         channels_config = self.recall_config.get('channels', {})
        
#         # Channel 1: Following
#         self.following_recall = FollowingRecall(
#             redis_client=self.redis,
#             data=self.data,
#             following_dict=self.following_dict,
#             config=channels_config.get('following', {})
#         )
        
#         # Channel 2: CF
#         self.cf_recall = CFRecall(
#             redis_client=self.redis,
#             data=self.data,
#             cf_model=self.cf_model,
#             config=channels_config.get('collaborative_filtering', {})
#         )
        
#         # Channel 3: Content
#         self.content_recall = ContentRecall(
#             redis_client=self.redis,
#             embeddings=self.embeddings,
#             faiss_index=self.faiss_index,
#             faiss_post_ids=self.faiss_post_ids,
#             data=self.data,
#             config=channels_config.get('content_based', {})
#         )
        
#         # Channel 4: Trending
#         self.trending_recall = TrendingRecall(
#             redis_client=self.redis,
#             data=self.data,
#             config=channels_config.get('trending', {})
#         )
        
#         logger.info("✅ Recall channels initialized")
        
#         # ════════════════════════════════════════════════════════════════
#         # STEP 7: INITIALIZE RANKER & RERANKER
#         # ════════════════════════════════════════════════════════════════
        
#         logger.info("\n🔧 Step 7: Initializing ranker & reranker...")
        
#         # Ranker
#         self.ranker = MLRanker(
#             model=self.ranking_model,
#             scaler=self.ranking_scaler,
#             feature_cols=self.ranking_feature_cols,
#             feature_engineer=self.feature_engineer,
#             config=self.ranking_config
#         )
        
#         # Reranker
#         self.reranker = Reranker(config=self.reranking_config)
        
#         logger.info("✅ Ranker & Reranker initialized")
        
#         # ════════════════════════════════════════════════════════════════
#         # STEP 8: INITIALIZE FRIEND RECOMMENDATION
#         # ════════════════════════════════════════════════════════════════
        
#         logger.info("\n🔧 Step 8: Initializing friend recommendation...")
        
#         self.friend_recommender = FriendRecommendation(
#             data=self.data,
#             embeddings=self.embeddings,
#             cf_model=self.cf_model,
#             config=self.config.get('friend_recommendation', {})
#         )
        
#         logger.info("✅ Friend recommendation initialized")
        
#         # ════════════════════════════════════════════════════════════════
#         # METRICS TRACKING
#         # ════════════════════════════════════════════════════════════════
        
#         self.metrics = defaultdict(list)

#         self.cf_rt = None
#         rt_cfg = self.recall_config.get("channels", {}).get("cf_realtime", {})
#         if rt_cfg.get("enabled") and self.redis is not None:
#             self.cf_rt = CFRealtimeRecall(self.redis, rt_cfg)
        
#         logger.info("\n✅ ONLINE PIPELINE READY!")
#         logger.info("="*70 + "\n")
    
#     # ════════════════════════════════════════════════════════════════════
#     # MAIN FEED GENERATION
#     # ════════════════════════════════════════════════════════════════════
    
#     def _build_post_metadata(self, post_ids: List[int]) -> Dict[int, Dict]:
#         """
#         Build metadata map {post_id: {...}} serving rerank:
#         - author_id: int | None
#         - created_at: datetime (UTC-naive) | None
#         - status: int | None
#         - content_hash: str | None
#         - title: str | None
#         - content: str | None
#         Safe with missing columns; automatically parse datetime.
#         """
#         meta: Dict[int, Dict] = {}
#         try:
#             post_df = self.data.get("post")
#             if post_df is None or post_df.empty:
#                 return meta

#             # Only get the necessary posts to reduce costs
#             df = post_df[post_df["Id"].isin(post_ids)] if "Id" in post_df.columns else post_df.copy()

#             # Define column names according to existing schema
#             col_id = "Id" if "Id" in df.columns else None
#             col_author = "UserId" if "UserId" in df.columns else None
#             col_created = "CreateDate" if "CreateDate" in df.columns else None
#             col_status = "Status" if "Status" in df.columns else None
#             col_hash = "ContentHash" if "ContentHash" in df.columns else None
#             col_title = "Title" if "Title" in df.columns else None
#             col_content = "Content" if "Content" in df.columns else None

#             if col_id is None:
#                 # No Id ⇒ cannot map metadata by post_id
#                 return meta

#             for _, r in df.iterrows():
#                 try:
#                     pid = int(r[col_id])
#                 except Exception:
#                     continue

#                 # author_id
#                 author_id = None
#                 if col_author and pd.notna(r.get(col_author)):
#                     try:
#                         author_id = int(r[col_author])
#                     except Exception:
#                         author_id = None

#                 # created_at (parse to UTC-naive)
#                 created_at = None
#                 if col_created and pd.notna(r.get(col_created)):
#                     try:
#                         dt = pd.to_datetime(r[col_created], utc=True)
#                         # to naive UTC for quick comparison
#                         created_at = dt.tz_convert(None) if hasattr(dt, "tz_convert") else dt.tz_localize(None)
#                     except Exception:
#                         created_at = None

#                 # status
#                 status = None
#                 if col_status and pd.notna(r.get(col_status)):
#                     try:
#                         status = int(r[col_status])
#                     except Exception:
#                         status = None

#                 # content hash / title / content
#                 content_hash = str(r[col_hash]) if col_hash and pd.notna(r.get(col_hash)) else None
#                 title = str(r[col_title]) if col_title and pd.notna(r.get(col_title)) else None
#                 content = str(r[col_content]) if col_content and pd.notna(r.get(col_content)) else None

#                 meta[pid] = {
#                     "author_id": author_id,
#                     "created_at": created_at,
#                     "status": status,
#                     "content_hash": content_hash,
#                     "title": title,
#                     "content": content,
#                 }

#         except Exception as e:
#             logger.warning(f"Build post_metadata failed: {e}", exc_info=True)

#         return meta

#     def _has_user_embedding(self, user_id: int) -> bool:
#         try:
#             return user_id in self.embeddings.get("user", {})
#         except Exception:
#             return False

#     def _user_following_count(self, user_id: int) -> int:
#         try:
#             flw = self.following_dict.get(user_id, [])
#             return len(flw)
#         except Exception:
#             return 0

#     def _trending_empty_last_hours(self, hours: int) -> bool:
#         try:
#             post_df = self.data.get("post")
#             if post_df is None or post_df.empty or "CreateDate" not in post_df.columns:
#                 return True
#             dt = pd.to_datetime(post_df["CreateDate"], errors="coerce", utc=True)
#             cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=hours)
#             return bool((dt >= cutoff).sum() == 0)
#         except Exception:
#             return True

#     def _status_ok_mismatch_ratio(self, post_ids: List[int], ok_values: set) -> float:
#         """Return fraction of posts NOT in ok status among given ids."""
#         post_df = self.data.get("post")
#         if post_df is None or post_df.empty or "Id" not in post_df.columns or "Status" not in post_df.columns:
#             return 0.0
#         sub = post_df.loc[post_df["Id"].isin(post_ids), ["Id", "Status"]]
#         if sub.empty: 
#             return 0.0
#         bad = (~sub["Status"].isin(ok_values)).sum()
#         return round(bad / len(sub), 3)

#     # def generate_feed(
#     #     self,
#     #     user_id: int,
#     #     limit: int = 50,
#     #     exclude_seen: Optional[List[int]] = None
#     # ) -> List[Dict]:
#     #     """
#     #     Generate personalized feed for user
        
#     #     Pipeline:
#     #     1. Multi-channel recall (~1000 candidates)
#     #     2. Feature extraction (47 features × N posts)
#     #     3. ML ranking (LightGBM)
#     #     4. Re-ranking & business rules
#     #     5. Return top K
        
#     #     Args:
#     #         user_id: Target user ID
#     #         limit: Number of posts to return
#     #         exclude_seen: Post IDs to exclude
            
#     #     Returns:
#     #         List of post dicts with scores
#     #     """
#     #     start_time = time.time()
        
#     #     try:
#     #         # ════════════════════════════════════════════════════════════
#     #         # STAGE 1: MULTI-CHANNEL RECALL
#     #         # ════════════════════════════════════════════════════════════
            
#     #         t1 = time.time()
#     #         candidates = self._recall_candidates(user_id, target_count=1000)
#     #         recall_latency = (time.time() - t1) * 1000
#     #         self.metrics['recall_latency'].append(recall_latency)
            
#     #         if not candidates:
#     #             logger.warning(f"No candidates for user {user_id}")
#     #             return []
            
#     #         # Exclude seen posts
#     #         if exclude_seen:
#     #             candidates = [p for p in candidates if p not in exclude_seen]
            
#     #         logger.debug(f"Recall: {len(candidates)} candidates in {recall_latency:.1f}ms")
            
#     #         # ════════════════════════════════════════════════════════════
#     #         # STAGE 2: ML RANKING
#     #         # ════════════════════════════════════════════════════════════
            
#     #         t2 = time.time()
#     #         ranked_df = self.ranker.rank(user_id, candidates)
#     #         ranking_latency = (time.time() - t2) * 1000
#     #         self.metrics['ranking_latency'].append(ranking_latency)
            
#     #         if ranked_df.empty:
#     #             logger.warning(f"Ranking failed for user {user_id}")
#     #             return []
            
#     #         # Get top 100 for re-ranking
#     #         ranked_df = ranked_df.head(100)
#     #         print(ranked_df.head(10))
#     #         logger.debug(f"Ranking: {len(ranked_df)} posts in {ranking_latency:.1f}ms")
            
#     #         # === NEW: build post_metadata for top-N posts ===
#     #         top_post_ids = ranked_df["post_id"].astype(int).tolist()
#     #         post_metadata = self._build_post_metadata(top_post_ids)
#     #         if not post_metadata:
#     #             logger.debug("post_metadata is empty; freshness/diversity/dedup may be limited.")

#     #         # ════════════════════════════════════════════════════════════
#     #         # STAGE 3: RE-RANKING
#     #         # ════════════════════════════════════════════════════════════
            
#     #         t3 = time.time()
#     #         final_feed = self.reranker.rerank(
#     #             ranked_df=ranked_df,
#     #             post_metadata=post_metadata,  # TODO: Add metadata
#     #             limit=limit
#     #         )
#     #         reranking_latency = (time.time() - t3) * 1000
#     #         self.metrics['reranking_latency'].append(reranking_latency)
            
#     #         logger.debug(f"Reranking: {len(final_feed)} posts in {reranking_latency:.1f}ms")
            
#     #         # ════════════════════════════════════════════════════════════
#     #         # METRICS
#     #         # ════════════════════════════════════════════════════════════
            
#     #         total_latency = (time.time() - start_time) * 1000
#     #         self.metrics['total_latency'].append(total_latency)
            
#     #         logger.info(
#     #             f"Feed generated for user {user_id}: "
#     #             f"{len(final_feed)} posts | "
#     #             f"Latency: {total_latency:.1f}ms "
#     #             f"(R:{recall_latency:.0f} + M:{ranking_latency:.0f} + B:{reranking_latency:.0f})"
#     #         )
            
#     #         return final_feed
            
#     #     except Exception as e:
#     #         logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
#     #         return []
#     def generate_feed(self, user_id: int, limit: int = 50, exclude_seen: Optional[List[int]] = None) -> List[Dict]:
#         start_time = time.time()
#         try:
#             # 1) RECALL
#             t1 = time.time()
#             candidates, recall_diag = self._recall_candidates(user_id, target_count=self.recall_config.get("target_count", 1000))
#             self.metrics['recall_latency'].append((time.time() - t1) * 1000)

#             if not candidates:
#                 reasons = self._diagnose_no_candidates(user_id, recall_diag)
#                 logger.warning("No candidates for user %s | reasons=%s", user_id, reasons)
#                 # (tuỳ chọn) lưu reasons vào nơi API layer đọc để trả về JSON
#                 self._last_no_feed_reasons = reasons
#                 return []

#             # Exclude seen
#             exclude_set = set(exclude_seen) if exclude_seen else None
#             if exclude_set:
#                 candidates = [p for p in candidates if p not in exclude_set]

#             # 2) RANK
#             t2 = time.time()
#             ranked_df = self.ranker.rank(user_id, candidates)
#             self.metrics['ranking_latency'].append((time.time() - t2) * 1000)
#             if ranked_df.empty:
#                 self._last_no_feed_reasons = ["Ranking produced empty scores (check feature engineering/model inputs)"]
#                 return []

#             ranked_df = ranked_df.head(self.ranking_config.get('top_k', 100))

#             # 3) RERANK
#             top_ids = ranked_df["post_id"].astype(int).tolist()
#             post_metadata = self._build_post_metadata(top_ids)

#             # Nếu sau Quality/Dedup/Diversity rỗng, chẩn đoán tiếp
#             t3 = time.time()
#             final_feed = self.reranker.rerank(ranked_df=ranked_df, post_metadata=post_metadata, limit=limit)
#             self.metrics['reranking_latency'].append((time.time() - t3) * 1000)

#             if not final_feed:
#                 reasons = self._diagnose_empty_after_rerank(top_ids, post_metadata)
#                 logger.warning("Empty after rerank for user %s | reasons=%s", user_id, reasons)
#                 self._last_no_feed_reasons = reasons
#                 return []

#             # OK
#             self._last_no_feed_reasons = []
#             return final_feed

#         except Exception as e:
#             logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
#             self._last_no_feed_reasons = [f"Unhandled exception: {type(e).__name__}"]
#             return []

    
#     def _recall_candidates(self, user_id: int, target_count: int = 1000) -> List[int]:
#         """
#         Multi-channel recall
        
#         Channels:
#         1. Following (400) - Posts from followed users
#         2. CF (300) - Posts liked by similar users
#         3. Content (200) - Similar posts via embeddings
#         4. Trending (100) - Popular posts
#         """
#         all_candidates = []

#         # Check acc
#         diag = RecallDiag(notes={})
        
#         # Channel 1: Following
#         # following_posts = self.following_recall.recall(user_id, k=400)
#         # Following
#         try:
#             following_posts = self.following_recall.recall(user_id, k=self.recall_config['channels']['following'].get('count', 400))
#         except Exception as e:
#             logger.warning(f"FollowingRecall failed: {e}")
#             following_posts = []
#         diag.following = len(following_posts); all_candidates.extend(following_posts)
#         print(f"Following posts: {following_posts}")
#         all_candidates.extend(following_posts)
        
#         # Channel 2: CF
#         # cf_posts = self.cf_recall.recall(user_id, k=300)
#         try:
#             cf_posts = self.cf_recall.recall(user_id, k=self.recall_config['channels']['collaborative_filtering'].get('count', 300))
#         except Exception as e:
#             logger.warning(f"CFRecall failed: {e}")
#             cf_posts = []
#         diag.cf = len(cf_posts); all_candidates.extend(cf_posts)
#         print(f"CF posts: {cf_posts}")
#         all_candidates.extend(cf_posts)
        
#         # Channel 3: Content
#         # content_posts = self.content_recall.recall(user_id, k=200)
#         try:
#             content_posts = self.content_recall.recall(user_id, k=self.recall_config['channels']['content_based'].get('count', 200))
#         except Exception as e:
#             logger.warning(f"ContentRecall failed: {e}")
#             content_posts = []
#         diag.content = len(content_posts); all_candidates.extend(content_posts)
#         print(f"Content posts: {content_posts}")
#         all_candidates.extend(content_posts)
        
#         # Channel 4: Trending
#         # trending_posts = self.trending_recall.recall(k=100)
#         try:
#             trending_posts = self.trending_recall.recall(k=self.recall_config['channels']['trending'].get('count', 100))
#         except Exception as e:
#             logger.warning(f"TrendingRecall failed: {e}")
#             trending_posts = []
#         diag.trending = len(trending_posts); all_candidates.extend(trending_posts)
#         print(f"Trending posts: {trending_posts}")
#         all_candidates.extend(trending_posts)
        
#         # Deduplicate (preserve order)
#         unique_candidates = list(dict.fromkeys(all_candidates))
#         diag.total = len(unique_candidates)
#         print(f"Unique candidates: {len(unique_candidates)}")

#         logger.info("RECALL SUMMARY | user=%s | following=%s cf=%s content=%s trending=%s | total=%s",
#                 user_id, diag.following, diag.cf, diag.content, diag.trending, diag.total)
        
#         flw_cnt = self._user_following_count(user_id)
#         if flw_cnt == 0:
#             diag.notes["following"] = "User has no following"

#         if self._trending_empty_last_hours(self.recall_config['channels']['trending'].get('trending_window_hours', 6)):
#             diag.notes["trending"] = "No posts within trending window"

#         if not self._has_user_embedding(user_id):
#             diag.notes["content"] = "User embedding missing (cold-start)"

#         if not unique_candidates:
#             logger.warning("No candidates after standard recall — applying DEV fallback for user %s", user_id)
#             fallback = self._dev_backfill_candidates(user_id, target_count)
#             logger.info("DEV FALLBACK | user=%s | backfill=%s", user_id, len(fallback))
#             return fallback

#         return unique_candidates[:target_count], diag
    
#     def _dev_backfill_candidates(self, user_id: int, target_count: int = 1000) -> List[int]:
#         """
#         Fallback cho DEV/STAGING: nới lỏng điều kiện để đảm bảo có ứng viên.
#         Ưu tiên: Trending rất rộng → ContentRandom → Random recent posts.
#         """
#         posts_df = self.data.get("post")
#         if posts_df is None or posts_df.empty:
#             return []

#         # 1) Nới Trending (bỏ min_engagement / cửa sổ rộng)
#         try:
#             loose_trending = self.trending_recall.recall(
#                 k=min(200, target_count),
#                 override_window_hours=720,          # 30 ngày
#                 override_min_engagement=0
#             )
#         except Exception:
#             loose_trending = []

#         # 2) Content random (nếu có embeddings list)
#         try:
#             content_rand = self.content_recall.recall_random(k=min(300, target_count))
#         except Exception:
#             content_rand = []

#         # 3) Random recent posts (Status hợp lệ nếu có)
#         try:
#             valid_status = {1, 10}   # DEV/PROD
#             df = posts_df.copy()
#             if "Status" in df.columns:
#                 df = df[df["Status"].isin(valid_status)]
#             if "CreateDate" in df.columns:
#                 df["__cd"] = pd.to_datetime(df["CreateDate"], errors="coerce")
#                 df = df.sort_values("__cd", ascending=False)
#             pool = df["Id"].astype(int).tolist()
#             recent_rand = pool[:min(500, len(pool))]
#         except Exception:
#             recent_rand = []

#         merged = list(dict.fromkeys(loose_trending + content_rand + recent_rand))
#         return merged[:target_count]

#     # ════════════════════════════════════════════════════════════════════
#     # FRIEND RECOMMENDATION
#     # ════════════════════════════════════════════════════════════════════
    
#     def recommend_friends(
#         self,
#         user_id: int,
#         k: int = 20
#     ) -> List[Dict]:
#         """
#         Recommend potential friends
        
#         Args:
#             user_id: Target user ID
#             k: Number of recommendations
            
#         Returns:
#             List of friend recommendations
#         """
#         try:
#             recommendations = self.friend_recommender.recommend_friends(
#                 user_id=user_id,
#                 k=k
#             )
            
#             logger.info(f"Friend recommendations for user {user_id}: {len(recommendations)} users")
            
#             return recommendations
            
#         except Exception as e:
#             logger.error(f"Error recommending friends for user {user_id}: {e}", exc_info=True)
#             return []
    
#     # ════════════════════════════════════════════════════════════════════
#     # REAL-TIME USER EMBEDDING UPDATE
#     # ════════════════════════════════════════════════════════════════════
    
#     def update_user_embedding_realtime(
#         self,
#         user_id: int,
#         post_id: int,
#         action: str
#     ):
#         """
#         Update user embedding in real-time after interaction
        
#         Strategy: Incremental weighted average
#         new_embedding = (1 - α) × old_embedding + α × post_embedding
        
#         Args:
#             user_id: User ID
#             post_id: Post ID interacted with
#             action: Action type (like, comment, share, etc.)
#         """
#         # Check if real-time update enabled
#         if not self.config.get('user_embedding', {}).get('real_time_update', {}).get('enabled', False):
#             return
        
#         # Check if action triggers update
#         trigger_actions = self.config.get('user_embedding', {}).get('real_time_update', {}).get('trigger_actions', [])
#         if action not in trigger_actions:
#             return
        
#         try:
#             # Get post embedding
#             if post_id not in self.embeddings['post']:
#                 logger.debug(f"No embedding for post {post_id}")
#                 return
            
#             post_emb = self.embeddings['post'][post_id]
            
#             # Get current user embedding
#             if user_id in self.embeddings['user']:
#                 old_emb = self.embeddings['user'][user_id]
#             else:
#                 # First interaction - use post embedding
#                 old_emb = post_emb
            
#             # Learning rate (alpha)
#             alpha = self.config.get('user_embedding', {}).get('real_time_update', {}).get('incremental', {}).get('learning_rate', 0.1)
            
#             # Action weight
#             action_weights = {
#                 'like': 1.0,
#                 'comment': 1.5,
#                 'share': 2.0,
#                 'save': 1.2
#             }
#             action_weight = action_weights.get(action, 1.0)
            
#             # Weighted alpha
#             weighted_alpha = alpha * action_weight
            
#             # Update embedding (incremental weighted average)
#             new_emb = (1 - weighted_alpha) * old_emb + weighted_alpha * post_emb
            
#             # Normalize
#             new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)
            
#             # Update in memory
#             self.embeddings['user'][user_id] = new_emb.astype(np.float32)
            
#             # Update Redis cache (async)
#             if self.redis is not None:
#                 try:
#                     key = f"user:{user_id}:embedding"
#                     value = new_emb.tobytes()
#                     self.redis.setex(key, 604800, value)  # 7 days TTL
#                 except Exception as e:
#                     logger.warning(f"Redis update failed: {e}")
            
#             logger.debug(f"Updated embedding for user {user_id} after {action} on post {post_id}")
            
#         except Exception as e:
#             logger.error(f"Error updating user embedding: {e}")
    
#     # ════════════════════════════════════════════════════════════════════
#     # METRICS & MONITORING
#     # ════════════════════════════════════════════════════════════════════
    
#     def get_metrics(self) -> Dict:
#         """Get performance metrics"""
#         if not self.metrics['total_latency']:
#             return {
#                 'total_requests': 0,
#                 'avg_latency_ms': 0,
#                 'p50_latency_ms': 0,
#                 'p95_latency_ms': 0,
#                 'p99_latency_ms': 0
#             }
        
#         latencies = self.metrics['total_latency']
        
#         return {
#             'total_requests': len(latencies),
#             'avg_latency_ms': np.mean(latencies),
#             'p50_latency_ms': np.percentile(latencies, 50),
#             'p95_latency_ms': np.percentile(latencies, 95),
#             'p99_latency_ms': np.percentile(latencies, 99),
#             'avg_recall_latency_ms': np.mean(self.metrics['recall_latency']),
#             'avg_ranking_latency_ms': np.mean(self.metrics['ranking_latency']),
#             'avg_reranking_latency_ms': np.mean(self.metrics['reranking_latency'])
#         }
    
#     def _diagnose_no_candidates(self, user_id: int, diag: RecallDiag) -> List[str]:
#         reasons = []
#         # Following
#         if diag.following == 0:
#             flw_cnt = self._user_following_count(user_id)
#             if flw_cnt == 0:
#                 reasons.append("No following (FollowingRecall empty)")
#             else:
#                 reasons.append("FollowingRecall returned 0 (filtering or time-window?)")

#         # Trending
#         if diag.trending == 0:
#             reasons.append("Trending empty (no posts within trending window)")

#         # CF
#         if diag.cf == 0:
#             reasons.append("CF has no similar users/items (no interactions/latent)")

#         # Content
#         if diag.content == 0:
#             if not self._has_user_embedding(user_id):
#                 reasons.append("Content cold-start (no user embedding)")
#             else:
#                 reasons.append("Content recall returned 0 (index/filter?)")

#         if not reasons:
#             reasons.append("All recall channels returned 0 due to strict filters")
#         return reasons

#     def _diagnose_empty_after_rerank(self, post_ids: List[int], post_metadata: Dict[int, Dict]) -> List[str]:
#         reasons = []
#         # Status mismatch check
#         ok_values = set()
#         ok_v = self.reranking_config.get("quality_status_ok_value", 1)
#         ok_values.add(int(ok_v))
#         for v in self.reranking_config.get("quality_status_ok_values", []):
#             try: ok_values.add(int(v))
#             except: pass

#         mismatch_ratio = self._status_ok_mismatch_ratio(post_ids, ok_values)
#         if mismatch_ratio >= 0.9:  # gần như tất cả bị loại bởi Status
#             reasons.append(f"Quality filter removed ~{int(mismatch_ratio*100)}% due to Status not in {sorted(ok_values)}")

#         # Freshness: không boost → không phải lý do rỗng, chỉ là 0-boost
#         # Dedup: nếu có content_hash trùng nhiều, có thể gây rỗng (hiếm)
#         # Diversity: hiếm khi ăn sạch, trừ khi top-K toàn một tác giả với cap total=0 (không hợp lý)

#         if not reasons:
#             reasons.append("Post-filtering removed all candidates (check Status mapping / dedup / diversity caps)")
#         return reasons

#     def print_metrics(self):
#         """Print performance metrics"""
#         metrics = self.get_metrics()
        
#         print("\n" + "="*70)
#         print("PERFORMANCE METRICS")
#         print("="*70)
#         print(f"Total requests: {metrics['total_requests']}")
#         print(f"\nLatency:")
#         print(f"  Avg: {metrics['avg_latency_ms']:.1f}ms")
#         print(f"  P50: {metrics['p50_latency_ms']:.1f}ms")
#         print(f"  P95: {metrics['p95_latency_ms']:.1f}ms")
#         print(f"  P99: {metrics['p99_latency_ms']:.1f}ms")
#         print(f"\nStage breakdown:")
#         print(f"  Recall: {metrics['avg_recall_latency_ms']:.1f}ms")
#         print(f"  Ranking: {metrics['avg_ranking_latency_ms']:.1f}ms")
#         print(f"  Reranking: {metrics['avg_reranking_latency_ms']:.1f}ms")
#         print("="*70 + "\n")


# # ============================================================================
# # EXAMPLE USAGE
# # ============================================================================

# if __name__ == "__main__":
#     """Test online pipeline"""
    
#     # Initialize
#     print("Initializing pipeline...")
#     pipeline = OnlineInferencePipeline(
#         config_path='configs/config_online.yaml',
#         models_dir='models',
#         data_dir='dataset',
#         use_redis=False  # Disable Redis for testing
#     )
    
#     # Test feed generation
#     print("\nTesting feed generation...")
#     feed = pipeline.generate_feed(user_id=1, limit=50)
    
#     print(f"\nGenerated feed: {len(feed)} posts")
#     print(f"Sample: {feed[:5]}")
    
#     # Test friend recommendation
#     print("\nTesting friend recommendation...")
#     friends = pipeline.recommend_friends(user_id=1, k=20)
    
#     print(f"\nRecommended friends: {len(friends)}")
#     print(f"Sample: {friends[:5]}")
    
#     # Print metrics
#     pipeline.print_metrics()



# # """
# # ONLINE INFERENCE PIPELINE - INTEGRATED WITH NEW OFFLINE ARTIFACTS
# # ==================================================================
# # Complete online serving pipeline using versioned artifacts

# # Architecture:
# # - Load models from ArtifactManager (latest version)
# # - Multi-channel recall (Following, CF, Content, Trending)
# # - Feature extraction (47 features)
# # - ML Ranking (LightGBM)
# # - Re-ranking & Business rules

# # Target: < 200ms end-to-end latency
# # """

# # import os
# # import sys
# # import time
# # import numpy as np
# # import pandas as pd
# # from typing import List, Dict, Optional, Tuple
# # from datetime import datetime, timedelta
# # from pathlib import Path
# # import logging
# # import warnings

# # warnings.filterwarnings('ignore')

# # # Add project root
# # sys.path.append(str(Path(__file__).parent.parent.parent))

# # # Try imports
# # try:
# #     from recommender.offline.artifact_manager import ArtifactManager
# #     ARTIFACT_MANAGER_AVAILABLE = True
# # except ImportError:
# #     ARTIFACT_MANAGER_AVAILABLE = False
# #     print("⚠️  ArtifactManager not available")

# # try:
# #     from recommender.common.feature_engineer import FeatureEngineer
# #     FEATURE_ENGINEER_AVAILABLE = True
# # except ImportError:
# #     FEATURE_ENGINEER_AVAILABLE = False
# #     print("⚠️  FeatureEngineer not available")

# # try:
# #     import redis
# #     REDIS_AVAILABLE = True
# # except ImportError:
# #     REDIS_AVAILABLE = False
# #     print("⚠️  redis not available. Install: pip install redis")

# # try:
# #     import lightgbm as lgb
# #     LIGHTGBM_AVAILABLE = True
# # except ImportError:
# #     LIGHTGBM_AVAILABLE = False
# #     print("⚠️  lightgbm not available")

# # try:
# #     import faiss
# #     FAISS_AVAILABLE = True
# # except ImportError:
# #     FAISS_AVAILABLE = False
# #     print("⚠️  faiss not available")


# # from recommender.online.recall.following_recall import FollowingRecall
# # from recommender.online.recall.base_recall import BaseRecall


# # # Setup logging
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format='[%(asctime)s] %(levelname)s - %(message)s',
# #     datefmt='%Y-%m-%d %H:%M:%S'
# # )
# # logger = logging.getLogger(__name__)


# # class OnlineInferencePipeline:
# #     """
# #     Complete online inference pipeline
    
# #     Features:
# #     - Load from versioned artifacts (ArtifactManager)
# #     - Multi-channel recall
# #     - ML ranking with LightGBM
# #     - Redis caching
# #     - < 200ms latency
# #     """
    
# #     def __init__(
# #         self,
# #         models_dir: str = 'models',
# #         redis_host: str = 'localhost',
# #         redis_port: int = 6380,
# #         redis_db: int = 0,
# #         use_redis: bool = True
# #     ):
# #         """
# #         Initialize online pipeline
        
# #         Args:
# #             models_dir: Directory with model artifacts
# #             redis_host: Redis host
# #             redis_port: Redis port
# #             redis_db: Redis database number
# #             use_redis: Enable Redis caching
# #         """
# #         logger.info("="*70)
# #         logger.info("INITIALIZING ONLINE INFERENCE PIPELINE")
# #         logger.info("="*70)
        
# #         self.models_dir = Path(models_dir)
# #         self.use_redis = use_redis and REDIS_AVAILABLE
        
# #         # Performance metrics
# #         self.metrics = {
# #             'recall_latency': [],
# #             'feature_latency': [],
# #             'ranking_latency': [],
# #             'reranking_latency': [],
# #             'total_latency': []
# #         }
        
# #         # ════════════════════════════════════════════════════════════════
# #         # STEP 1: CONNECT TO REDIS
# #         # ════════════════════════════════════════════════════════════════
        
# #         if self.use_redis:
# #             try:
# #                 self.redis_client = redis.Redis(
# #                     host=redis_host,
# #                     port=redis_port,
# #                     db=redis_db,
# #                     decode_responses=False  # For binary data
# #                 )
# #                 # Test connection
# #                 self.redis_client.ping()
# #                 logger.info(f"✅ Connected to Redis: {redis_host}:{redis_port}/{redis_db}")
# #             except Exception as e:
# #                 logger.warning(f"⚠️  Redis connection failed: {e}")
# #                 logger.warning("⚠️  Falling back to no-cache mode")
# #                 self.use_redis = False
# #                 self.redis_client = None
# #         else:
# #             self.redis_client = None
# #             logger.info("ℹ️  Redis disabled (using no-cache mode)")
        
# #         # ════════════════════════════════════════════════════════════════
# #         # STEP 2: LOAD ARTIFACTS FROM LATEST VERSION
# #         # ════════════════════════════════════════════════════════════════
        
# #         if not ARTIFACT_MANAGER_AVAILABLE:
# #             raise ImportError("ArtifactManager required")
        
# #         logger.info("\n📦 Loading artifacts...")
        
# #         self.artifact_mgr = ArtifactManager(base_dir=self.models_dir)
        
# #         # Get latest version
# #         try:
# #             self.current_version = self.artifact_mgr.get_latest_version()
# #             logger.info(f"   Latest version: {self.current_version}")
# #         except FileNotFoundError:
# #             raise FileNotFoundError(
# #                 f"No models found in {self.models_dir}. "
# #                 "Run offline training first: python scripts/offline/main_offline_pipeline.py"
# #             )
        
# #         # Load all artifacts
# #         artifacts = self.artifact_mgr.load_artifacts(self.current_version)
        
# #         # Extract components
# #         self.embeddings = artifacts['embeddings']
# #         self.faiss_index = artifacts['faiss_index']
# #         self.faiss_post_ids = artifacts['faiss_post_ids']
# #         self.cf_model = artifacts['cf_model']
# #         self.ranking_model = artifacts['ranking_model']
# #         self.ranking_scaler = artifacts['ranking_scaler']
# #         self.ranking_feature_cols = artifacts['ranking_feature_cols']
# #         self.user_stats = artifacts['user_stats']
# #         self.author_stats = artifacts['author_stats']
# #         self.following_dict = artifacts['following_dict']
# #         self.metadata = artifacts['metadata']
        
# #         logger.info("✅ Artifacts loaded successfully!")
# #         logger.info(f"   Post embeddings: {len(self.embeddings['post']):,}")
# #         logger.info(f"   User embeddings: {len(self.embeddings['user']):,}")
# #         logger.info(f"   CF users: {len(self.cf_model['user_ids']):,}")
# #         logger.info(f"   Faiss index: {self.faiss_index.ntotal:,} vectors")
# #         logger.info(f"   Ranking features: {len(self.ranking_feature_cols)}")
        
# #         # ════════════════════════════════════════════════════════════════
# #         # STEP 3: INITIALIZE FEATURE ENGINEER
# #         # ════════════════════════════════════════════════════════════════
        
# #         if not FEATURE_ENGINEER_AVAILABLE:
# #             raise ImportError("FeatureEngineer required")
        
# #         # Load data (dummy for online - only need structure)
# #         # In production, this would connect to live database
# #         logger.info("\n🔧 Initializing feature engineer...")
        
# #         # Create dummy data structure (will be replaced by Redis/DB in production)
# #         self.data = {
# #             'user': pd.DataFrame(),
# #             'post': pd.DataFrame(),
# #             'postreaction': pd.DataFrame(),
# #             'friendship': pd.DataFrame()
# #         }
        
# #         self.feature_engineer = FeatureEngineer(
# #             data=self.data,
# #             user_stats=self.user_stats,
# #             author_stats=self.author_stats,
# #             following_dict=self.following_dict,
# #             embeddings=self.embeddings
# #         )
        
# #         logger.info("✅ Feature engineer initialized")
        
# #         # ════════════════════════════════════════════════════════════════
# #         # COMPLETE
# #         # ════════════════════════════════════════════════════════════════
        
# #         logger.info("\n✅ ONLINE PIPELINE READY!")
# #         logger.info("="*70 + "\n")
    
# #     # ════════════════════════════════════════════════════════════════════
# #     # MAIN INFERENCE METHOD
# #     # ════════════════════════════════════════════════════════════════════
    
# #     def get_feed(
# #         self,
# #         user_id: int,
# #         limit: int = 50,
# #         exclude_seen: Optional[set] = None
# #     ) -> List[Dict]:
# #         """
# #         Get personalized feed for user
        
# #         Args:
# #             user_id: Target user ID
# #             limit: Number of posts to return
# #             exclude_seen: Set of post IDs to exclude
        
# #         Returns:
# #             List of dicts with post_id, score, metadata
# #         """
# #         start_time = time.time()
        
# #         try:
# #             # ════════════════════════════════════════════════════════════
# #             # STAGE 1: MULTI-CHANNEL RECALL (~1000 candidates)
# #             # ════════════════════════════════════════════════════════════
            
# #             t1 = time.time()
# #             candidates = self._recall_candidates(user_id, target_count=1000)
# #             recall_latency = (time.time() - t1) * 1000
# #             self.metrics['recall_latency'].append(recall_latency)
            
# #             if not candidates:
# #                 logger.warning(f"No candidates found for user {user_id}")
# #                 return []
            
# #             # Filter out seen posts
# #             if exclude_seen:
# #                 candidates = [p for p in candidates if p not in exclude_seen]
            
# #             logger.debug(f"Recall: {len(candidates)} candidates in {recall_latency:.1f}ms")
            
# #             # ════════════════════════════════════════════════════════════
# #             # STAGE 2: FEATURE EXTRACTION (47 features × N posts)
# #             # ════════════════════════════════════════════════════════════
            
# #             t2 = time.time()
# #             features_df = self._extract_features_batch(user_id, candidates)
# #             feature_latency = (time.time() - t2) * 1000
# #             self.metrics['feature_latency'].append(feature_latency)
            
# #             if features_df.empty:
# #                 logger.warning(f"No features extracted for user {user_id}")
# #                 return []
            
# #             logger.debug(f"Features: {len(features_df)} posts in {feature_latency:.1f}ms")
            
# #             # ════════════════════════════════════════════════════════════
# #             # STAGE 3: ML RANKING (LightGBM)
# #             # ════════════════════════════════════════════════════════════
            
# #             t3 = time.time()
# #             features_df = self._rank_candidates(features_df)
# #             ranking_latency = (time.time() - t3) * 1000
# #             self.metrics['ranking_latency'].append(ranking_latency)
            
# #             # Get top 100 for re-ranking
# #             features_df = features_df.nlargest(100, 'ml_score')
            
# #             logger.debug(f"Ranking: {len(features_df)} posts in {ranking_latency:.1f}ms")
            
# #             # ════════════════════════════════════════════════════════════
# #             # STAGE 4: RE-RANKING & BUSINESS RULES
# #             # ════════════════════════════════════════════════════════════
            
# #             t4 = time.time()
# #             final_feed = self._rerank_with_business_rules(features_df, limit)
# #             reranking_latency = (time.time() - t4) * 1000
# #             self.metrics['reranking_latency'].append(reranking_latency)
            
# #             logger.debug(f"Reranking: {len(final_feed)} posts in {reranking_latency:.1f}ms")
            
# #             # ════════════════════════════════════════════════════════════
# #             # METRICS
# #             # ════════════════════════════════════════════════════════════
            
# #             total_latency = (time.time() - start_time) * 1000
# #             self.metrics['total_latency'].append(total_latency)
            
# #             logger.info(
# #                 f"Feed generated for user {user_id}: "
# #                 f"{len(final_feed)} posts | "
# #                 f"Latency: {total_latency:.1f}ms "
# #                 f"(R:{recall_latency:.0f} + F:{feature_latency:.0f} + "
# #                 f"M:{ranking_latency:.0f} + B:{reranking_latency:.0f})"
# #             )
            
# #             return final_feed
            
# #         except Exception as e:
# #             logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
# #             return []
    
# #     # ════════════════════════════════════════════════════════════════════
# #     # STAGE 1: MULTI-CHANNEL RECALL
# #     # ════════════════════════════════════════════════════════════════════
    
# #     def _recall_candidates(self, user_id: int, target_count: int = 1000) -> List[int]:
# #         """
# #         Multi-channel recall
        
# #         Channels:
# #         1. Following (400) - Posts from followed users
# #         2. CF (300) - Posts liked by similar users
# #         3. Content (200) - Similar posts via embeddings
# #         4. Trending (100) - Popular posts
        
# #         Returns:
# #             List of unique post IDs
# #         """
# #         all_candidates = []
        
# #         # Channel 1: Following
# #         following_posts = self._recall_following(user_id, k=400)
# #         all_candidates.extend(following_posts)
        
# #         # Channel 2: Collaborative Filtering
# #         cf_posts = self._recall_cf(user_id, k=300)
# #         all_candidates.extend(cf_posts)
        
# #         # Channel 3: Content-based (Embeddings)
# #         content_posts = self._recall_content(user_id, k=200)
# #         all_candidates.extend(content_posts)
        
# #         # Channel 4: Trending
# #         trending_posts = self._recall_trending(k=100)
# #         all_candidates.extend(trending_posts)
        
# #         # Deduplicate (preserve order)
# #         unique_candidates = list(dict.fromkeys(all_candidates))
        
# #         return unique_candidates[:target_count]
    
# #     def _recall_following(self, user_id: int, k: int) -> List[int]:
# #         """Recall posts from followed users"""
# #         # Get followed users
# #         followed_users = self.following_dict.get(user_id, [])
        
# #         if not followed_users:
# #             return []
        
# #         # In production: Query Redis sorted set
# #         # following:{user_id}:posts (sorted by timestamp)
        
# #         # For now: Return empty (needs live data)
# #         return []
    
# #     def _recall_cf(self, user_id: int, k: int) -> List[int]:
# #         """Recall via collaborative filtering"""
# #         # Get similar users
# #         user_similarities = self.cf_model['user_similarities']
# #         similar_users = user_similarities.get(user_id, [])
        
# #         if not similar_users:
# #             return []
        
# #         # Get posts liked by similar users
# #         # In production: Query from Redis or database
        
# #         # For now: Return top K most similar items
# #         item_similarities = self.cf_model.get('item_similarities', {})
        
# #         # Return first K items (placeholder)
# #         all_items = list(item_similarities.keys())[:k] if item_similarities else []
        
# #         return all_items
    
# #     def _recall_content(self, user_id: int, k: int) -> List[int]:
# #         """Recall via content similarity (embeddings)"""
# #         # Get user embedding
# #         user_emb = self.embeddings['user'].get(user_id)
        
# #         if user_emb is None or not FAISS_AVAILABLE:
# #             return []
        
# #         # Normalize
# #         user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
        
# #         # Search Faiss
# #         distances, indices = self.faiss_index.search(
# #             user_emb.reshape(1, -1).astype(np.float32),
# #             k * 2  # Over-fetch
# #         )
        
# #         # Convert indices to post IDs
# #         post_ids = [int(self.faiss_post_ids[idx]) for idx in indices[0]]
        
# #         return post_ids[:k]
    
# #     def _recall_trending(self, k: int) -> List[int]:
# #         """Recall trending posts"""
# #         # In production: Query Redis sorted set
# #         # trending:global:6h (sorted by engagement score)
        
# #         # For now: Return empty
# #         return []
    
# #     # ════════════════════════════════════════════════════════════════════
# #     # STAGE 2: FEATURE EXTRACTION
# #     # ════════════════════════════════════════════════════════════════════
    
# #     def _extract_features_batch(
# #         self,
# #         user_id: int,
# #         post_ids: List[int]
# #     ) -> pd.DataFrame:
# #         """Extract 47 features for each (user, post) pair"""
# #         features_list = []
# #         timestamp = datetime.now()
        
# #         for post_id in post_ids:
# #             try:
# #                 features = self.feature_engineer.extract_features(
# #                     user_id,
# #                     post_id,
# #                     timestamp
# #                 )
# #                 features['post_id'] = post_id
# #                 features_list.append(features)
# #             except Exception as e:
# #                 logger.debug(f"Feature extraction failed for post {post_id}: {e}")
# #                 continue
        
# #         if not features_list:
# #             return pd.DataFrame()
        
# #         features_df = pd.DataFrame(features_list)
        
# #         # Ensure all features present
# #         for col in self.ranking_feature_cols:
# #             if col not in features_df.columns:
# #                 features_df[col] = 0.0
        
# #         return features_df
    
# #     # ════════════════════════════════════════════════════════════════════
# #     # STAGE 3: ML RANKING
# #     # ════════════════════════════════════════════════════════════════════
    
# #     def _rank_candidates(self, features_df: pd.DataFrame) -> pd.DataFrame:
# #         """Rank using LightGBM model"""
# #         # Select features
# #         X = features_df[self.ranking_feature_cols].fillna(0)
        
# #         # Scale
# #         X_scaled = self.ranking_scaler.transform(X)
        
# #         # Predict
# #         scores = self.ranking_model.predict(X_scaled)
        
# #         # Add to dataframe
# #         features_df['ml_score'] = scores
        
# #         return features_df
    
# #     # ════════════════════════════════════════════════════════════════════
# #     # STAGE 4: RE-RANKING
# #     # ════════════════════════════════════════════════════════════════════
    
# #     def _rerank_with_business_rules(
# #         self,
# #         ranked_df: pd.DataFrame,
# #         limit: int
# #     ) -> List[Dict]:
# #         """Apply business rules and diversity"""
# #         # Sort by ML score
# #         ranked_df = ranked_df.sort_values('ml_score', ascending=False)
        
# #         final_feed = []
# #         last_author_id = None
# #         same_author_count = 0
        
# #         for _, row in ranked_df.iterrows():
# #             post_id = int(row['post_id'])
# #             score = float(row['ml_score'])
            
# #             # Get author (from embeddings or stats)
# #             # Placeholder: author_id = post_id % 1000
# #             author_id = post_id % 1000
            
# #             # Rule 1: Diversity (max 2 consecutive from same author)
# #             if author_id == last_author_id:
# #                 same_author_count += 1
# #                 if same_author_count >= 2:
# #                     continue
# #             else:
# #                 same_author_count = 0
# #                 last_author_id = author_id
            
# #             # Rule 2: Freshness boost (would need post creation time)
# #             # Placeholder: No boost
            
# #             final_feed.append({
# #                 'post_id': post_id,
# #                 'score': score,
# #                 'author_id': author_id
# #             })
            
# #             if len(final_feed) >= limit:
# #                 break
        
# #         return final_feed
    
# #     # ════════════════════════════════════════════════════════════════════
# #     # UTILITIES
# #     # ════════════════════════════════════════════════════════════════════
    
# #     def get_metrics_summary(self) -> Dict:
# #         """Get performance metrics summary"""
# #         if not self.metrics['total_latency']:
# #             return {}
        
# #         summary = {}
# #         for key, values in self.metrics.items():
# #             if values:
# #                 summary[key] = {
# #                     'mean': np.mean(values),
# #                     'p50': np.percentile(values, 50),
# #                     'p95': np.percentile(values, 95),
# #                     'p99': np.percentile(values, 99),
# #                     'min': np.min(values),
# #                     'max': np.max(values)
# #                 }
        
# #         return summary
    
# #     def print_metrics(self):
# #         """Print performance metrics"""
# #         summary = self.get_metrics_summary()
        
# #         print("\n" + "="*70)
# #         print("ONLINE PIPELINE PERFORMANCE METRICS")
# #         print("="*70)
        
# #         for metric, stats in summary.items():
# #             print(f"\n{metric}:")
# #             print(f"  Mean:  {stats['mean']:.1f}ms")
# #             print(f"  P50:   {stats['p50']:.1f}ms")
# #             print(f"  P95:   {stats['p95']:.1f}ms")
# #             print(f"  P99:   {stats['p99']:.1f}ms")
# #             print(f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]ms")
        
# #         print("="*70 + "\n")


# # # ════════════════════════════════════════════════════════════════════════
# # # MAIN TESTING
# # # ════════════════════════════════════════════════════════════════════════

# # if __name__ == "__main__":
# #     """
# #     Test online inference pipeline
# #     """
# #     # Initialize pipeline
# #     pipeline = OnlineInferencePipeline(
# #         models_dir='models',
# #         use_redis=False  # Disable Redis for testing
# #     )
    
# #     # Test with a few users
# #     test_user_ids = [1, 2, 3, 4, 5]
    
# #     print("\n" + "="*70)
# #     print("TESTING INFERENCE")
# #     print("="*70 + "\n")
    
# #     for user_id in test_user_ids:
# #         feed = pipeline.get_feed(user_id, limit=50)
# #         print(f"User {user_id}: Generated {len(feed)} posts")
    
# #     # Print metrics
# #     pipeline.print_metrics()



# """
# ONLINE INFERENCE PIPELINE - PRODUCTION READY
# ============================================
# Flow: Recall → Feature Extraction → Ranking → Re-ranking → Output
# """

# from __future__ import annotations
# import os
# import sys
# import time
# import logging
# import warnings
# from pathlib import Path
# from typing import List, Dict, Optional, Tuple
# from dataclasses import dataclass
# from collections import defaultdict

# import numpy as np
# import pandas as pd
# import yaml
# from datetime import datetime

# # Add project root
# sys.path.append(str(Path(__file__).parent.parent.parent))

# warnings.filterwarnings('ignore')
# logger = logging.getLogger(__name__)

# # Core imports
# from recommender.offline.artifact_manager import ArtifactManager
# from recommender.common.feature_engineer import FeatureEngineer

# # Recall channels
# from recommender.online.recall import (
#     FollowingRecall,
#     CFRecall,
#     ContentRecall,
#     TrendingRecall
# )

# # Ranking
# from recommender.online.ranking import MLRanker, Reranker

# # Optional caches
# try:
#     import redis
#     REDIS_AVAILABLE = True
# except ImportError:
#     REDIS_AVAILABLE = False
#     logger.warning("redis not available")

# # SQLAlchemy (backend DB optional)
# from sqlalchemy import create_engine, text
# from sqlalchemy.orm import sessionmaker
# from recommender.online.realtime_handlers import RealtimeHandlers
# from recommender.online.realtime_jobs import RealTimeJobs
# from recommender.online.recall.covisit import CovisitRecall

# @dataclass
# class RecallDiag:
#     following: int = 0
#     cf: int = 0
#     content: int = 0
#     trending: int = 0
#     total: int = 0
#     notes: Dict[str, str] = None


# class OnlineInferencePipeline:
#     """
#     Complete online inference pipeline

#     Features:
#     - Multi-channel recall (4 channels)
#     - ML ranking with LightGBM
#     - Business rules re-ranking
#     - Real-time user embedding updates
#     - Redis caching
#     """

#     def __init__(
#         self,
#         config_path: str = 'configs/config_online.yaml',
#         models_dir: str = 'models',
#         data_dir: str = 'dataset',
#         use_redis: bool = True
#     ):
#         self.models_dir = Path(models_dir)
#         self.data_dir = Path(data_dir)

#         # ============== STEP 1: CONFIG ==========================
#         logger.info("\n" + "="*70)
#         logger.info("INITIALIZING ONLINE INFERENCE PIPELINE")
#         logger.info("="*70)

#         with open(config_path, 'r', encoding='utf-8') as f:
#             self.config = yaml.safe_load(f)

#         self.recall_config = self.config.get('recall', {})
#         self.ranking_config = self.config.get('ranking', {})
#         self.reranking_config = self.config.get('reranking', {})

#         # ============== STEP 2: REDIS ===========================
#         self.redis = None
#         if use_redis and REDIS_AVAILABLE:
#             try:
#                 rc = self.config.get('redis', {})
#                 self.redis = redis.Redis(
#                     host=rc.get('host', 'localhost'),
#                     port=rc.get('port', 6381),
#                     db=rc.get('db', 0),
#                     decode_responses=False,
#                     socket_timeout=rc.get('socket_timeout', 5),
#                     max_connections=rc.get('max_connections', 50)
#                 )
#                 self.redis.ping()
#                 logger.info("✅ Redis connected")
#             except Exception as e:
#                 logger.warning(f"Redis connection failed: {e}")
#                 self.redis = None
#         else:
#             logger.warning("Redis disabled or not available")

#         # ============== STEP 2b: BACKEND DB (MySQL optional) ====
#         self.db_engine = None
#         self.db_session_factory = None
#         try:
#             be = self.config.get('backend_db', {}) or {}
#             if be.get('enabled'):
#                 connect_args = be.get('connect_args', {}) or {}
#                 self.db_engine = create_engine(
#                     be['url'],
#                     pool_size=be.get('pool_size', 20),
#                     max_overflow=be.get('max_overflow', 40),
#                     pool_recycle=be.get('pool_recycle', 1800),
#                     pool_pre_ping=be.get('pool_pre_ping', True),
#                     connect_args=connect_args,           # 👈 NEW
#                     future=True,
#                 )
#                 self.db_session_factory = sessionmaker(
#                     bind=self.db_engine, autocommit=False, autoflush=False, future=True
#                 )
#                 # Ping ngay và log chi tiết nếu fail
#                 try:
#                     with self.db_engine.connect() as conn:
#                         conn.execute(text("SELECT 1"))
#                     logger.info("✅ Backend DB connected")
#                 except Exception as e:
#                     logger.warning("Backend DB initial ping failed: %s", e)  # 👈 NEW
#             else:
#                 logger.info("Backend DB disabled in config")
#         except Exception as e:
#             logger.warning(f"Backend DB connection failed: {e}")
#             self.db_engine = None
#             self.db_session_factory = None

#         # ============== STEP 3: ARTIFACTS =======================
#         logger.info("\n📦 Loading model artifacts...")
#         self.artifact_mgr = ArtifactManager(base_dir=str(self.models_dir))

#         model_version = self.config.get('models', {}).get('version', 'latest')
#         self.current_version = self.artifact_mgr.get_latest_version() if model_version == 'latest' else model_version

#         artifacts = self.artifact_mgr.load_artifacts(self.current_version)

#         self.embeddings = artifacts['embeddings']
#         self.faiss_index = artifacts.get('faiss_index')
#         self.faiss_post_ids = artifacts.get('faiss_post_ids', [])
#         self.cf_model = artifacts['cf_model']
#         self.ranking_model = artifacts['ranking_model']
#         self.ranking_scaler = artifacts['ranking_scaler']
#         self.ranking_feature_cols = artifacts['ranking_feature_cols']
#         self.user_stats = artifacts['user_stats']
#         self.author_stats = artifacts['author_stats']
#         self.following_dict = artifacts['following_dict']
#         self.metadata = artifacts.get('metadata', {})

#         logger.info("✅ Artifacts loaded")

#         # ============== STEP 4: DATA (for dev) ==================
#         logger.info("\n📊 Loading data...")
#         from recommender.common.data_loader import load_data
#         self.data = load_data(str(self.data_dir))
#         logger.info("✅ Data loaded")

#         # ============== STEP 5: FEATURE ENGINEER ================
#         self.feature_engineer = FeatureEngineer(
#             data=self.data,
#             user_stats=self.user_stats,
#             author_stats=self.author_stats,
#             following_dict=self.following_dict,
#             embeddings=self.embeddings,
#             redis_client=self.redis
#         )

#         # ============== STEP 6: RECALL CHANNELS =================
#         channels_config = self.recall_config.get('channels', {})
#         self.following_recall = FollowingRecall(
#             redis_client=self.redis,
#             data=self.data,
#             following_dict=self.following_dict,
#             config=channels_config.get('following', {})
#         )
#         self.cf_recall = CFRecall(
#             redis_client=self.redis,
#             data=self.data,
#             cf_model=self.cf_model,
#             config=channels_config.get('collaborative_filtering', {})
#         )
#         self.content_recall = ContentRecall(
#             redis_client=self.redis,
#             embeddings=self.embeddings,
#             faiss_index=self.faiss_index,
#             faiss_post_ids=self.faiss_post_ids,
#             data=self.data,
#             config=channels_config.get('content_based', {})
#         )
#         self.trending_recall = TrendingRecall(
#             redis_client=self.redis,
#             data=self.data,
#             config=channels_config.get('trending', {})
#         )

#         # ============== STEP 7: RANKER & RERANKER ===============
#         self.ranker = MLRanker(
#             model=self.ranking_model,
#             scaler=self.ranking_scaler,
#             feature_cols=self.ranking_feature_cols,
#             feature_engineer=self.feature_engineer,
#             config=self.ranking_config
#         )
#         self.reranker = Reranker(config=self.reranking_config)

#         # ============== METRICS =================================
#         self.metrics = defaultdict(list)
#         self._last_recall_diag: Optional[RecallDiag] = None
#         self._last_no_feed_reasons: List[str] = []


#         # Realtime helpers
#         self.rt_handlers = RealtimeHandlers(self.redis, action_weights=self.config.get("user_embedding", {}).get("real_time_update", {}).get("action_weights", None))
#         self.rt_jobs = RealTimeJobs(self.redis,
#             interval_trending= self.config.get("recall", {}).get("channels", {}).get("trending", {}).get("refresh_interval_seconds", 300),
#             interval_post_feat=900
#         )
#         self.rt_jobs.start()

#         # Covisit recall (optional)
#         self.covisit_recall = CovisitRecall(self.redis, k_per_anchor=25, max_anchors=5)

#         logger.info("\n✅ ONLINE PIPELINE READY!")
#         logger.info("="*70 + "\n")

#     # ------------------------ Public APIs -----------------------

#     def generate_feed(
#         self,
#         user_id: int,
#         limit: int = 50,
#         exclude_seen: Optional[List[int]] = None
#     ) -> List[Dict]:
#         """
#         1) Recall -> 2) Feature Extraction & ML Ranking -> 3) Reranking (business rules)
#         """
#         start_time = time.time()
#         self._last_no_feed_reasons = []
#         try:
#             # 1) Recall
#             t1 = time.time()
#             candidates, recall_diag = self._recall_candidates(user_id, target_count=self.recall_config.get("target_count", 1000))
#             self._last_recall_diag = recall_diag
#             self.metrics['recall_latency'].append((time.time() - t1) * 1000)

#             if not candidates:
#                 logger.warning(f"No candidates for user {user_id}")
#                 return []

#             # exclude seen
#             if exclude_seen:
#                 seen = set(exclude_seen)
#                 candidates = [p for p in candidates if p not in seen]

#             # 2) Ranking
#             t2 = time.time()
#             ranked_df = self.ranker.rank(user_id, candidates)
#             self.metrics['ranking_latency'].append((time.time() - t2) * 1000)
#             if ranked_df.empty:
#                 self._last_no_feed_reasons.append("ranking_empty")
#                 return []

#             # Top-K (for rerank)
#             top_k = int(self.ranking_config.get("top_k", 100))
#             ranked_df = ranked_df.head(top_k)

#             # 3) Rerank (with business rules) — NEED post_metadata
#             t3 = time.time()
#             post_meta = self._build_post_metadata(ranked_df['post_id'].tolist())
#             final_feed = self.reranker.rerank(
#                 ranked_df=ranked_df,
#                 post_metadata=post_meta,
#                 limit=limit
#             )
#             self.metrics['reranking_latency'].append((time.time() - t3) * 1000)

#             self.metrics['total_latency'].append((time.time() - start_time) * 1000)
#             logger.info(
#                 "Feed generated for user %s: %s posts | Latency: %.1fms",
#                 user_id, len(final_feed), self.metrics['total_latency'][-1]
#             )
#             return final_feed

#         except Exception as e:
#             logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
#             return []

#     def recommend_friends(self, user_id: int, k: int = 20) -> List[Dict]:
#         # nếu bạn có FriendRecommendation module, gọi ở đây
#         try:
#             from recommender.online.friend_recommendation import FriendRecommendation
#             fr = FriendRecommendation(
#                 data=self.data,
#                 embeddings=self.embeddings,
#                 cf_model=self.cf_model,
#                 config=self.config.get('friend_recommendation', {})
#             )
#             return fr.recommend_friends(user_id=user_id, k=k)
#         except Exception as e:
#             logger.error(f"Error recommending friends: {e}")
#             return []

#     def update_user_embedding_realtime(self, user_id: int, post_id: int, action: str):
#         """Incremental update of user embedding after interaction"""
#         conf = self.config.get('user_embedding', {}).get('real_time_update', {})
#         if not conf.get('enabled', False):
#             return
#         triggers = conf.get('trigger_actions', [])
#         if action not in triggers:
#             return
#         try:
#             if post_id not in self.embeddings['post']:
#                 return
#             post_emb = self.embeddings['post'][post_id]
#             old_emb = self.embeddings['user'][user_id] if user_id in self.embeddings['user'] else post_emb
#             alpha = conf.get('incremental', {}).get('learning_rate', 0.1)
#             weights = {'like': 1.0, 'comment': 1.5, 'share': 2.0, 'save': 1.2, 'view': 0.5}
#             a = alpha * weights.get(action, 1.0)
#             new_emb = (1 - a) * old_emb + a * post_emb
#             new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)
#             self.embeddings['user'][user_id] = new_emb.astype(np.float32)
#             if self.redis is not None:
#                 try:
#                     self.redis.setex(f"user:{user_id}:embedding", 604800, new_emb.tobytes())
#                 except Exception:
#                     pass
#         except Exception as e:
#             logger.error(f"update_user_embedding_realtime error: {e}")

#     def get_metrics(self) -> Dict:
#         if not self.metrics['total_latency']:
#             return {
#                 'total_requests': 0,
#                 'avg_latency_ms': 0,
#                 'p50_latency_ms': 0,
#                 'p95_latency_ms': 0,
#                 'p99_latency_ms': 0,
#                 'avg_recall_latency_ms': 0,
#                 'avg_ranking_latency_ms': 0,
#                 'avg_reranking_latency_ms': 0
#             }
#         lat = self.metrics['total_latency']
#         return {
#             'total_requests': len(lat),
#             'avg_latency_ms': float(np.mean(lat)),
#             'p50_latency_ms': float(np.percentile(lat, 50)),
#             'p95_latency_ms': float(np.percentile(lat, 95)),
#             'p99_latency_ms': float(np.percentile(lat, 99)),
#             'avg_recall_latency_ms': float(np.mean(self.metrics['recall_latency'])),
#             'avg_ranking_latency_ms': float(np.mean(self.metrics['ranking_latency'])),
#             'avg_reranking_latency_ms': float(np.mean(self.metrics['reranking_latency'])),
#         }

#     # ------------------------ Internals ------------------------

#     # def _recall_candidates(self, user_id: int, target_count: int = 1000) -> Tuple[List[int], RecallDiag]:
#     #     all_candidates: List[int] = []
#     #     diag = RecallDiag(notes={})

#     #     channels_cfg = self.recall_config.get('channels', {})
        
#     #     # Following
#     #     try:
#     #         cnt = self.recall_config['channels']['following'].get('count', 400)
#     #         following_posts = self.following_recall.recall(user_id, k=cnt)
#     #     except Exception as e:
#     #         logger.debug(f"FollowingRecall failed: {e}")
#     #         following_posts = []
#     #     diag.following = len(following_posts); all_candidates.extend(following_posts)

#     #     # CF
#     #     try:
#     #         cnt = self.recall_config['channels']['collaborative_filtering'].get('count', 300)
#     #         cf_posts = self.cf_recall.recall(user_id, k=cnt)
#     #     except Exception as e:
#     #         logger.debug(f"CFRecall failed: {e}")
#     #         cf_posts = []
#     #     diag.cf = len(cf_posts); all_candidates.extend(cf_posts)

#     #     # Content
#     #     try:
#     #         cnt = self.recall_config['channels']['content_based'].get('count', 200)
#     #         content_posts = self.content_recall.recall(user_id, k=cnt)
#     #     except Exception as e:
#     #         logger.debug(f"ContentRecall failed: {e}")
#     #         content_posts = []
#     #     diag.content = len(content_posts); all_candidates.extend(content_posts)

#     #     # Trending
#     #     try:
#     #         cnt = self.recall_config['channels']['trending'].get('count', 100)
#     #         trending_posts = self.trending_recall.recall(k=cnt)
#     #     except Exception as e:
#     #         logger.debug(f"TrendingRecall failed: {e}")
#     #         trending_posts = []
#     #     diag.trending = len(trending_posts); all_candidates.extend(trending_posts)

#     #     # Dedup & cap
#     #     unique_candidates = list(dict.fromkeys(all_candidates))[:target_count]
#     #     diag.total = len(unique_candidates)

#     #     logger.info(
#     #         "RECALL SUMMARY | user=%s | following=%s cf=%s content=%s trending=%s | total=%s",
#     #         user_id, diag.following, diag.cf, diag.content, diag.trending, diag.total
#     #     )

#     #     # basic notes
#     #     flw_cnt = self._user_following_count(user_id)
#     #     if flw_cnt == 0 and diag.following == 0:
#     #         diag.notes["following"] = "User has no following"
#     #     tw = self.recall_config['channels']['trending'].get('trending_window_hours', 6)
#     #     if diag.trending == 0 and self._trending_empty_last_hours(tw):
#     #         diag.notes["trending"] = "No posts within trending window"
#     #     if diag.content == 0 and not self._has_user_embedding(user_id):
#     #         diag.notes["content"] = "User embedding missing (cold-start)"

#     #     return unique_candidates, diag

#     def _recall_candidates(self, user_id: int, target_count: int = 1000) -> list[int]:
#         # 1. Khởi tạo diag và danh sách ứng viên
#         all_candidates: List[int] = []
#         diag = RecallDiag(notes={}) 

#         channels_cfg = self.recall_config.get('channels', {})
        
#         # --- Following ---
#         try:
#             cnt = channels_cfg.get('following', {}).get('count', 400)
#             following_posts = self.following_recall.recall(user_id, k=cnt)
#         except Exception as e:
#             logger.debug(f"FollowingRecall failed for user {user_id}: {e}")
#             following_posts = []
#         diag.following = len(following_posts)
#         all_candidates.extend(following_posts)

#         # --- CF ---
#         try:
#             cnt = channels_cfg.get('collaborative_filtering', {}).get('count', 300)
#             cf_posts = self.cf_recall.recall(user_id, k=cnt)
#         except Exception as e:
#             logger.debug(f"CFRecall failed for user {user_id}: {e}")
#             cf_posts = []
#         diag.cf = len(cf_posts)
#         all_candidates.extend(cf_posts)

#         # --- Content ---
#         try:
#             cnt = channels_cfg.get('content_based', {}).get('count', 200)
#             content_posts = self.content_recall.recall(user_id, k=cnt)
#         except Exception as e:
#             logger.debug(f"ContentRecall failed for user {user_id}: {e}")
#             content_posts = []
#         diag.content = len(content_posts)
#         all_candidates.extend(content_posts)

#         # --- Covisit (MỚI) ---
#         try:
#             # Lấy count từ config, mặc định là 150
#             cnt = channels_cfg.get('covisit', {}).get('count', 150)
#             covisit_posts = self.covisit_recall.recall(user_id, k=cnt)
#         except Exception as e:
#             logger.debug(f"CovisitRecall failed for user {user_id}: {e}")
#             covisit_posts = []
#         # Thêm trường covisit vào diag
#         diag.covisit = len(covisit_posts) 
#         all_candidates.extend(covisit_posts)

#         # --- Trending ---
#         try:
#             cnt = channels_cfg.get('trending', {}).get('count', 100)
#             # Trending recall thường không cần user_id
#             trending_posts = self.trending_recall.recall(k=cnt) 
#         except Exception as e:
#             logger.debug(f"TrendingRecall failed: {e}")
#             trending_posts = []
#         diag.trending = len(trending_posts)
#         all_candidates.extend(trending_posts)

#         # Dedup & cap
#         # Đảm bảo giữ thứ tự ưu tiên của các kênh recall bằng dict.fromkeys
#         unique_candidates = list(dict.fromkeys(all_candidates))[:target_count]
#         diag.total = len(unique_candidates)
        
#         # 2. Ghi logs TỔNG KẾT
#         logger.info(
#             "RECALL SUMMARY | user=%s | following=%s cf=%s content=%s covisit=%s trending=%s | total=%s",
#             user_id, diag.following, diag.cf, diag.content, diag.covisit, diag.trending, diag.total
#         )

#         # basic notes
#         flw_cnt = self._user_following_count(user_id)
#         if flw_cnt == 0 and diag.following == 0:
#             diag.notes["following"] = "User has no following"
#         tw = self.recall_config['channels']['trending'].get('trending_window_hours', 6)
#         if diag.trending == 0 and self._trending_empty_last_hours(tw):
#             diag.notes["trending"] = "No posts within trending window"
#         if diag.content == 0 and not self._has_user_embedding(user_id):
#             diag.notes["content"] = "User embedding missing (cold-start)"

#         return unique_candidates, diag

#     def _safe_text(self, v) -> str:
#         """Return '' if v is NaN/None/non-string; else normalized text."""
#         if v is None:
#             return ""
#         # pandas NA/NaN
#         try:
#             import pandas as pd
#             if pd.isna(v):
#                 return ""
#         except Exception:
#             pass
#         # only accept real strings
#         if not isinstance(v, str):
#             return ""
#         return v.strip()

#     def _build_post_metadata(self, post_ids: List[int]) -> Dict[int, Dict]:
#         """
#         Return per-post metadata for reranker:
#           { post_id: {author_id, status, created_at, title, content, content_hash} }
#         """
#         meta: Dict[int, Dict] = {}

#         # 1) from loaded data (CSV/dev)
#         try:
#             post_df: pd.DataFrame = self.data["post"]
#             sub = post_df[post_df["Id"].isin(post_ids)][["Id", "UserId", "Status", "CreateDate", "Content"]].copy()
#             for _, r in sub.iterrows():
#                 pid = int(r["Id"])
#                 meta[pid] = {
#                     "author_id": int(r.get("UserId")) if not pd.isna(r.get("UserId")) else None,
#                     "status": int(r.get("Status")) if not pd.isna(r.get("Status")) else None,
#                     "created_at": pd.to_datetime(r.get("CreateDate"), errors="coerce", utc=True),
#                     "title": "",
#                     "content": (r.get("Content") or ""),
#                     "content_hash": None
#                 }
#         except Exception:
#             pass

#         # 2) Fallback: query backend DB if missing
#         missing = [pid for pid in post_ids if pid not in meta]
#         if missing and self.db_session_factory:
#             with self.db_session_factory() as s:
#                 rows = s.execute(text("""
#                     SELECT Id, UserId, Status, CreateDate, Content
#                     FROM Post WHERE Id IN :ids
#                 """), {"ids": tuple(missing)}).all()
#                 for row in rows:
#                     pid, uid, st, cd, ct = row
#                     meta[int(pid)] = {
#                         "author_id": int(uid) if uid is not None else None,
#                         "status": int(st) if st is not None else None,
#                         "created_at": pd.to_datetime(cd, errors="coerce", utc=True),
#                         "title": self._safe_text(None),
#                         "content": ct or "",
#                         "content_hash": None
#                     }

#         # 3) Hash for dedup
#         import hashlib
#         for pid, m in meta.items():
#             if not m.get("content_hash"):
#                 title = self._safe_text(m.get("title")).lower()
#                 body  = self._safe_text(m.get("content")).lower()
#                 if title or body:
#                     m["content_hash"] = hashlib.sha1(f"{title}|{body}".encode("utf-8")).hexdigest()

#         return meta

#     # ----- helpers for diag -----

#     def _user_following_count(self, user_id: int) -> int:
#         try:
#             if self.following_dict is None:
#                 return 0
#             flw = self.following_dict.get(user_id, [])
#             return len(flw) if flw is not None else 0
#         except Exception:
#             return 0

#     def _has_user_embedding(self, user_id: int) -> bool:
#         try:
#             return user_id in (self.embeddings.get("user", {}) or {})
#         except Exception:
#             return False

#     def _trending_empty_last_hours(self, hours: int) -> bool:
#         try:
#             post_df = self.data.get("post")
#             if post_df is None or post_df.empty or "CreateDate" not in post_df.columns:
#                 return True
#             dt = pd.to_datetime(post_df["CreateDate"], errors="coerce", utc=True)
#             cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=hours)
#             return bool((dt >= cutoff).sum() == 0)
#         except Exception:
#             return True

#     # ----- backend DB utilities -----

#     def ping_backend_db(self) -> bool:
#         if self.db_engine is None:
#             return False
#         try:
#             with self.db_engine.connect() as conn:
#                 conn.execute(text("SELECT 1"))
#             return True
#         except Exception:
#             return False

#     def backend_db_stats(self) -> dict:
#         if not self.db_session_factory:
#             return {"connected": False}
#         stats = {"connected": self.ping_backend_db()}
#         try:
#             with self.db_session_factory() as s:
#                 posts = s.execute(text("SELECT COUNT(1) FROM Post")).scalar_one()
#                 users = s.execute(text("SELECT COUNT(1) FROM User")).scalar_one()
#                 reactions = s.execute(text("SELECT COUNT(1) FROM PostReaction")).scalar_one()
#                 comments = s.execute(text("SELECT COUNT(1) FROM Comment")).scalar_one()
#                 last_post = s.execute(text("SELECT MAX(CreateDate) FROM Post")).scalar_one()
#                 stats.update({
#                     "post_count": int(posts or 0),
#                     "user_count": int(users or 0),
#                     "reaction_count": int(reactions or 0),
#                     "comment_count": int(comments or 0),
#                     "last_post_at": str(last_post) if last_post else None,
#                 })
#         except Exception as e:
#             stats.update({"error": str(e)})
#         return stats

#     def get_author_id(self, post_id: int) -> Optional[int]:
#         # from dev data
#         try:
#             row = self.data["post"].loc[self.data["post"]["Id"] == post_id]
#             if not row.empty:
#                 return int(row["UserId"].iloc[0])
#         except Exception:
#             pass
#         # fallback DB
#         try:
#             if self.db_session_factory:
#                 with self.db_session_factory() as s:
#                     r = s.execute(text("SELECT UserId FROM Post WHERE Id=:pid"), {"pid": post_id}).first()
#                     return int(r[0]) if r else None
#         except Exception:
#             pass
#         return None


"""
ONLINE INFERENCE PIPELINE - PRODUCTION READY
============================================
Flow: Recall → Feature Extraction → Ranking → Re-ranking → Output

- Tương thích artifact “latest.version” (Windows-safe, không dùng symlink bắt buộc)
- Không truyền following_dict/redis_client vào FeatureEngineer.__init__ (repo hiện tại không nhận arg này)
- Gán following (nếu có) sau khi khởi tạo FeatureEngineer (giữ tương thích 2 phía)
"""

from __future__ import annotations

import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.getLogger("lightgbm").setLevel(logging.WARNING)

# -------------------------------------------------------------------
# Core & Artifacts
# -------------------------------------------------------------------
from recommender.offline.artifact_manager import ArtifactManager, get_latest_version_dir
from recommender.common.feature_engineer import FeatureEngineer

# -------------------------------------------------------------------
# Recall channels
# -------------------------------------------------------------------
from recommender.online.recall import (
    FollowingRecall,
    CFRecall,
    ContentRecall,
    TrendingRecall,
)
from recommender.online.recall.covisit import CovisitRecall

# -------------------------------------------------------------------
# Ranking
# -------------------------------------------------------------------
from recommender.online.ranking import MLRanker, Reranker

# -------------------------------------------------------------------
# Redis (optional)
# -------------------------------------------------------------------
try:
    import redis as _redis
    REDIS_AVAILABLE = True
except Exception:
    _redis = None
    REDIS_AVAILABLE = False
    logger.warning("redis not available")

# -------------------------------------------------------------------
# Backend DB (optional)
# -------------------------------------------------------------------
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Realtime helpers
from recommender.online.realtime_handlers import RealtimeHandlers
from recommender.online.realtime_jobs import RealTimeJobs


@dataclass
class RecallDiag:
    following: int = 0
    cf: int = 0
    content: int = 0
    covisit: int = 0
    trending: int = 0
    total: int = 0
    notes: Dict[str, str] = None


# =========================
# Helper: load ranker artifacts
# =========================
def _load_ranker_artifacts(base_dir: str):
    """
    Load ranker artifacts theo format:
      models/<version>/
        - ranker_model.txt
        - ranker_scaler.pkl
        - ranker_feature_cols.pkl
        - meta.json (optional)
    Ưu tiên đọc latest.version / symlink latest.
    Trả về: (model, scaler, feature_cols, meta_dict, version_dir)
    """
    import json
    import pickle
    from lightgbm import Booster

    base = Path(base_dir)

    # 1) Resolve version dir bằng offline.artifact_manager API
    vdir = get_latest_version_dir(str(base))
    if vdir is None:
        # fallback: nếu config chỉ định thẳng folder
        latest_file = base / "latest.version"
        if latest_file.exists():
            ver = latest_file.read_text(encoding="utf-8").strip()
            vdir = base / ver
        elif (base / "latest").is_symlink():
            try:
                vdir = (base / "latest").resolve(strict=True)
            except Exception:
                vdir = None
    if vdir is None:
        raise FileNotFoundError(
            f"Cannot resolve latest model version under {base_dir}. "
            f"Missing latest.version or latest symlink."
        )

    # 2) File names
    f_model = vdir / "ranker_model.txt"
    f_scaler = vdir / "ranker_scaler.pkl"
    f_cols  = vdir / "ranker_feature_cols.pkl"
    f_meta  = vdir / "meta.json"

    # 3) Load
    if not f_model.exists():
        raise FileNotFoundError(f"Missing model file: {f_model}")
    model = Booster(model_file=str(f_model))

    scaler = None
    if f_scaler.exists():
        with f_scaler.open("rb") as f:
            scaler = pickle.load(f)

    feature_cols = None
    if f_cols.exists():
        with f_cols.open("rb") as f:
            feature_cols = pickle.load(f)

    meta = {}
    if f_meta.exists():
        try:
            meta = json.loads(f_meta.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    return model, scaler, feature_cols, meta, vdir


class OnlineInferencePipeline:
    """
    Complete online inference pipeline

    Features:
    - Multi-channel recall (following / CF / content / covisit / trending)
    - ML ranking with LightGBM (tương thích artifact mới & cũ)
    - Business rules re-ranking
    - Real-time user embedding updates
    - Redis caching (ưu tiên), fallback MySQL
    """

    def __init__(
        self,
        config_path: str = "configs/config_online.yaml",
        models_dir: str = "models",
        data_dir: str = "dataset",
        use_redis: bool = True,
    ):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)

        logger.info("\n" + "=" * 70)
        logger.info("INITIALIZING ONLINE INFERENCE PIPELINE")
        logger.info("=" * 70)

        # ----------------- Step 1: CONFIG ----------------------
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        self.recall_config = self.config.get("recall", {}) or {}
        self.ranking_config = self.config.get("ranking", {}) or {}
        self.reranking_config = self.config.get("reranking", {}) or {}

        # ----------------- Step 2: REDIS -----------------------
        self.redis: Optional[_redis.Redis] = None
        if use_redis and REDIS_AVAILABLE:
            try:
                rc = self.config.get("redis", {}) or {}
                url = rc.get("url")
                if url:
                    self.redis = _redis.from_url(url, decode_responses=True)
                else:
                    self.redis = _redis.Redis(
                        host=rc.get("host", "localhost"),
                        port=rc.get("port", 6379),
                        db=rc.get("db", 0),
                        decode_responses=True,
                        socket_timeout=rc.get("socket_timeout", 5),
                        max_connections=rc.get("max_connections", 50),
                    )
                self.redis.ping()
                logger.info("✅ Redis connected")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis = None
        else:
            logger.warning("Redis disabled or not available")

        # ----------------- Step 2b: BACKEND DB (optional) ------
        self.db_engine = None
        self.db_session_factory = None
        try:
            db = self.config.get("database", {}) or self.config.get("backend_db", {}) or {}
            db_url = db.get("url")
            if db_url:
                self.db_engine = create_engine(
                    db_url,
                    pool_size=db.get("pool_size", 20),
                    max_overflow=db.get("max_overflow", 40),
                    pool_recycle=db.get("pool_recycle", 1800),
                    pool_pre_ping=db.get("pool_pre_ping", True),
                    future=True,
                )
                self.db_session_factory = sessionmaker(
                    bind=self.db_engine, autocommit=False, autoflush=False, future=True
                )
                with self.db_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("✅ Backend DB connected")
            else:
                logger.info("Backend DB not configured")
        except Exception as e:
            logger.warning(f"Backend DB connection failed: {e}")
            self.db_engine = None
            self.db_session_factory = None

        # ----------------- Step 3: ARTIFACTS -------------------
        logger.info("\n📦 Loading model artifacts...")

        # ArtifactManager: chỉ dùng để resolve version dir 'latest'
        self.artifact_mgr = ArtifactManager(artifacts_base_dir=str(self.models_dir))

        req_version = self.config.get("models", {}).get("version", "latest")
        if req_version == "latest":
            vdir = self.artifact_mgr.get_latest_version_dir()
            if not vdir:
                raise RuntimeError("No model version found in models/ (missing latest.version or symlink latest)")
            self.current_version = vdir.name
        else:
            self.current_version = req_version

        # Mặc định các artifact không bắt buộc (embeddings/cf…) — để compatible khi chưa có
        self.embeddings = {}
        self.faiss_index = None
        self.faiss_post_ids = []
        self.cf_model = None
        self.user_stats = {}
        self.author_stats = {}
        self.following_dict = {}
        self.metadata = {}

        # Load ranker_* theo format mới
        (
            self.ranking_model,
            self.ranking_scaler,
            self.ranking_feature_cols,
            meta2,
            ver_dir,
        ) = _load_ranker_artifacts(base_dir=str(self.models_dir))
        self.metadata.update(meta2 or {})
        self._version_dir = ver_dir

        logger.info("✅ Ranker artifacts loaded")

        # ----------------- Step 4: DATA (dev/feature) ----------
        logger.info("\n📊 Loading data...")
        # Lưu ý: feature_engineer.py hiện kỳ vọng keys: users, posts, friendships, post_hashtags
        # Hàm load_data (nếu của bạn trả về user/post/...) thì ta normalize lại.
        from recommender.common.data_loader import load_data
        raw = load_data(str(self.data_dir)) or {}
        self.data = self._normalize_data_keys(raw)
        logger.info("✅ Data loaded")

        # ----------------- Step 5: FEATURE ENGINEER ------------
        # ⚠️ FeatureEngineer trong repo hiện tại KHÔNG nhận following_dict/redis_client
        self.feature_engineer = FeatureEngineer(
            data=self.data,
            user_stats=self.user_stats,
            author_stats=self.author_stats,
            following=self.following_dict,          # mapping đúng với dataclass
            embeddings=self.embeddings,
        )
        # nếu class có set_following_dict() / thuộc tính following: vẫn gán cho chắc
        try:
            if hasattr(self.feature_engineer, "set_following_dict"):
                self.feature_engineer.set_following_dict(self.following_dict or {})
            elif hasattr(self.feature_engineer, "following"):
                setattr(self.feature_engineer, "following", self.following_dict or {})
        except Exception:
            pass

        # ----------------- Step 6: RECALL CHANNELS -------------
        channels_config = self.recall_config.get("channels", {}) or {}
        self.following_recall = FollowingRecall(
            redis_client=self.redis,
            data=self.data,
            following_dict=self.following_dict,
            config=channels_config.get("following", {}) or {},
        )
        self.cf_recall = CFRecall(
            redis_client=self.redis,
            data=self.data,
            cf_model=self.cf_model,
            config=channels_config.get("collaborative_filtering", {}) or {},
        )
        self.content_recall = ContentRecall(
            redis_client=self.redis,
            embeddings=self.embeddings,
            faiss_index=self.faiss_index,
            faiss_post_ids=self.faiss_post_ids,
            data=self.data,
            config=channels_config.get("content_based", {}) or {},
        )
        self.trending_recall = TrendingRecall(
            redis_client=self.redis,
            data=self.data,
            config=channels_config.get("trending", {}) or {},
        )
        self.covisit_recall = CovisitRecall(self.redis, k_per_anchor=25, max_anchors=5)

        # ----------------- Step 7: RANKER & RERANK -------------
        self.ranker = MLRanker(
            model=self.ranking_model,
            scaler=self.ranking_scaler,
            feature_cols=self.ranking_feature_cols,
            feature_engineer=self.feature_engineer,
            config=self.ranking_config,
        )
        self.reranker = Reranker(config=self.reranking_config)

        # ----------------- METRICS -----------------------------
        self.metrics = defaultdict(list)
        self._last_recall_diag: Optional[RecallDiag] = None
        self._last_no_feed_reasons: List[str] = []

        # Realtime helpers
        self.rt_handlers = RealtimeHandlers(
            self.redis,
            action_weights=self.config.get("user_embedding", {})
            .get("real_time_update", {})
            .get("action_weights", None),
        )
        self.rt_jobs = RealTimeJobs(
            self.redis,
            interval_trending=self.recall_config.get("channels", {})
            .get("trending", {})
            .get("refresh_interval_seconds", 300),
            interval_post_feat=900,
        )
        self.rt_jobs.start()

        logger.info("\n✅ ONLINE PIPELINE READY!")
        logger.info("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # PUBLIC APIS
    # ------------------------------------------------------------------
    def generate_feed(
        self, user_id: int, limit: int = 50, exclude_seen: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        1) Recall -> 2) Feature Extraction & ML Ranking -> 3) Reranking
        """
        start_time = time.time()
        self._last_no_feed_reasons = []
        try:
            # 1) Recall
            t1 = time.time()
            candidates, recall_diag = self._recall_candidates(
                user_id, target_count=self.recall_config.get("target_count", 1000)
            )
            self._last_recall_diag = recall_diag
            self.metrics["recall_latency"].append((time.time() - t1) * 1000)

            if not candidates:
                logger.warning(f"No candidates for user {user_id}")
                return []

            # exclude seen
            if exclude_seen:
                seen = set(exclude_seen)
                candidates = [p for p in candidates if p not in seen]

            # 2) Ranking
            t2 = time.time()
            ranked_df = self.ranker.rank(user_id, candidates)
            self.metrics["ranking_latency"].append((time.time() - t2) * 1000)
            if ranked_df.empty:
                self._last_no_feed_reasons.append("ranking_empty")
                return []

            # Top-K (for rerank)
            top_k = int(self.ranking_config.get("top_k", 100))
            ranked_df = ranked_df.head(top_k)

            # 3) Reranking với business rules
            t3 = time.time()
            post_meta = self._build_post_metadata(ranked_df["post_id"].tolist())
            final_feed = self.reranker.rerank(
                ranked_df=ranked_df, post_metadata=post_meta, limit=limit
            )
            self.metrics["reranking_latency"].append((time.time() - t3) * 1000)

            self.metrics["total_latency"].append((time.time() - start_time) * 1000)
            logger.info(
                "Feed generated for user %s: %s posts | Latency: %.1fms",
                user_id,
                len(final_feed),
                self.metrics["total_latency"][-1],
            )
            return final_feed
        except Exception as e:
            logger.error(f"Error generating feed for user {user_id}: {e}", exc_info=True)
            return []

    def recommend_friends(self, user_id: int, k: int = 20) -> List[Dict]:
        try:
            from recommender.online.friend_recommendation import FriendRecommendation
            fr = FriendRecommendation(
                data=self.data,
                embeddings=self.embeddings,
                cf_model=self.cf_model,
                config=self.config.get("friend_recommendation", {}) or {},
            )
            return fr.recommend_friends(user_id=user_id, k=k)
        except Exception as e:
            logger.error(f"Error recommending friends: {e}")
            return []

    def update_user_embedding_realtime(self, user_id: int, post_id: int, action: str):
        """Incremental update of user embedding after interaction"""
        conf = self.config.get("user_embedding", {}).get("real_time_update", {}) or {}
        if not conf.get("enabled", False):
            return
        triggers = conf.get("trigger_actions", []) or []
        if action not in triggers:
            return
        try:
            if post_id not in (self.embeddings.get("post", {}) or {}):
                return
            post_emb = self.embeddings["post"][post_id]
            old_emb = (
                self.embeddings["user"][user_id]
                if user_id in (self.embeddings.get("user", {}) or {})
                else post_emb
            )
            alpha = conf.get("incremental", {}).get("learning_rate", 0.1)
            weights = {"like": 1.0, "comment": 1.5, "share": 2.0, "save": 1.2, "view": 0.5}
            a = alpha * weights.get(action, 1.0)
            new_emb = (1 - a) * old_emb + a * post_emb
            new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)
            self.embeddings.setdefault("user", {})[user_id] = new_emb.astype(np.float32)
            if self.redis is not None:
                try:
                    self.redis.setex(
                        f"user:{user_id}:embedding", 7 * 24 * 3600, new_emb.tobytes()
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"update_user_embedding_realtime error: {e}")

    def get_metrics(self) -> Dict:
        if not self.metrics["total_latency"]:
            return {
                "total_requests": 0,
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "avg_recall_latency_ms": 0,
                "avg_ranking_latency_ms": 0,
                "avg_reranking_latency_ms": 0,
            }
        lat = self.metrics["total_latency"]
        return {
            "total_requests": len(lat),
            "avg_latency_ms": float(np.mean(lat)),
            "p50_latency_ms": float(np.percentile(lat, 50)),
            "p95_latency_ms": float(np.percentile(lat, 95)),
            "p99_latency_ms": float(np.percentile(lat, 99)),
            "avg_recall_latency_ms": float(np.mean(self.metrics["recall_latency"])),
            "avg_ranking_latency_ms": float(np.mean(self.metrics["ranking_latency"])),
            "avg_reranking_latency_ms": float(np.mean(self.metrics["reranking_latency"])),
        }

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------
    def _normalize_data_keys(self, raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Chuẩn hoá keys để khớp FeatureEngineer hiện tại:
          FE kỳ vọng: users, posts, friendships, post_hashtags
          (nếu loader trả: user, post, friendship, post_hashtag thì map lại)
        """
        if raw is None:
            return {}
        data = dict(raw)
        # map phổ biến
        if "users" not in data and "user" in data: data["users"] = data["user"]
        if "posts" not in data and "post" in data: data["posts"] = data["post"]
        if "friendships" not in data and "friendship" in data: data["friendships"] = data["friendship"]
        if "post_hashtags" not in data and "post_hashtag" in data: data["post_hashtags"] = data["post_hashtag"]
        return data

    def _recall_candidates(
        self, user_id: int, target_count: int = 1000
    ) -> Tuple[List[int], RecallDiag]:
        all_candidates: List[int] = []
        diag = RecallDiag(notes={})

        channels_cfg = self.recall_config.get("channels", {}) or {}

        # Following
        try:
            cnt = channels_cfg.get("following", {}).get("count", 400)
            following_posts = self.following_recall.recall(user_id, k=cnt)
        except Exception as e:
            logger.debug(f"FollowingRecall failed for user {user_id}: {e}")
            following_posts = []
        diag.following = len(following_posts)
        all_candidates.extend(following_posts)

        # CF
        try:
            cnt = channels_cfg.get("collaborative_filtering", {}).get("count", 300)
            cf_posts = self.cf_recall.recall(user_id, k=cnt)
        except Exception as e:
            logger.debug(f"CFRecall failed for user {user_id}: {e}")
            cf_posts = []
        diag.cf = len(cf_posts)
        all_candidates.extend(cf_posts)

        # Content
        try:
            cnt = channels_cfg.get("content_based", {}).get("count", 200)
            content_posts = self.content_recall.recall(user_id, k=cnt)
        except Exception as e:
            logger.debug(f"ContentRecall failed for user {user_id}: {e}")
            content_posts = []
        diag.content = len(content_posts)
        all_candidates.extend(content_posts)

        # Covisit
        try:
            cnt = channels_cfg.get("covisit", {}).get("count", 150)
            covisit_posts = self.covisit_recall.recall(user_id, k=cnt)
        except Exception as e:
            logger.debug(f"CovisitRecall failed for user {user_id}: {e}")
            covisit_posts = []
        diag.covisit = len(covisit_posts)
        all_candidates.extend(covisit_posts)

        # Trending
        try:
            cnt = channels_cfg.get("trending", {}).get("count", 100)
            trending_posts = self.trending_recall.recall(k=cnt)
        except Exception as e:
            logger.debug(f"TrendingRecall failed: {e}")
            trending_posts = []
        diag.trending = len(trending_posts)
        all_candidates.extend(trending_posts)

        unique_candidates = list(dict.fromkeys(all_candidates))[:target_count]
        diag.total = len(unique_candidates)

        logger.info(
            "RECALL SUMMARY | user=%s | following=%s cf=%s content=%s covisit=%s trending=%s | total=%s",
            user_id,
            diag.following,
            diag.cf,
            diag.content,
            diag.covisit,
            diag.trending,
            diag.total,
        )

        # Notes
        flw_cnt = self._user_following_count(user_id)
        if flw_cnt == 0 and diag.following == 0:
            diag.notes["following"] = "User has no following"
        tw = channels_cfg.get("trending", {}).get("trending_window_hours", 6)
        if diag.trending == 0 and self._trending_empty_last_hours(tw):
            diag.notes["trending"] = "No posts within trending window"
        if diag.content == 0 and not self._has_user_embedding(user_id):
            diag.notes["content"] = "User embedding missing (cold-start)"

        return unique_candidates, diag

    def _safe_text(self, v) -> str:
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        if not isinstance(v, str):
            return ""
        return v.strip()

    def _build_post_metadata(self, post_ids: List[int]) -> Dict[int, Dict]:
        meta: Dict[int, Dict] = {}

        # 1) từ data đã load (CSV/dev) — chú ý: theo FE keys là 'posts'
        try:
            post_df: pd.DataFrame = self.data["posts"]
            sub = post_df[post_df["Id"].isin(post_ids)][
                ["Id", "UserId", "Status", "CreateDate", "Content"]
            ].copy()
            for _, r in sub.iterrows():
                pid = int(r["Id"])
                meta[pid] = {
                    "author_id": int(r.get("UserId"))
                    if not pd.isna(r.get("UserId"))
                    else None,
                    "status": int(r.get("Status")) if not pd.isna(r.get("Status")) else None,
                    "created_at": pd.to_datetime(r.get("CreateDate"), errors="coerce", utc=True),
                    "title": "",
                    "content": (r.get("Content") or ""),
                    "content_hash": None,
                }
        except Exception:
            pass

        # 2) Fallback DB nếu thiếu
        missing = [pid for pid in post_ids if pid not in meta]
        if missing and self.db_session_factory:
            with self.db_session_factory() as s:
                rows = s.execute(
                    text(
                        "SELECT Id, UserId, Status, CreateDate, Content FROM Post WHERE Id IN :ids"
                    ),
                    {"ids": tuple(missing)},
                ).all()
                for row in rows:
                    pid, uid, st, cd, ct = row
                    meta[int(pid)] = {
                        "author_id": int(uid) if uid is not None else None,
                        "status": int(st) if st is not None else None,
                        "created_at": pd.to_datetime(cd, errors="coerce", utc=True),
                        "title": self._safe_text(None),
                        "content": ct or "",
                        "content_hash": None,
                    }

        # 3) Hash nội dung cho dedup
        import hashlib
        for pid, m in meta.items():
            if not m.get("content_hash"):
                title = self._safe_text(m.get("title")).lower()
                body = self._safe_text(m.get("content")).lower()
                if title or body:
                    m["content_hash"] = hashlib.sha1(f"{title}|{body}".encode("utf-8")).hexdigest()

        return meta

    # ----- helpers for diag -----
    def _user_following_count(self, user_id: int) -> int:
        try:
            if self.following_dict is None:
                return 0
            flw = self.following_dict.get(user_id, [])
            return len(flw) if flw is not None else 0
        except Exception:
            return 0

    def _has_user_embedding(self, user_id: int) -> bool:
        try:
            return user_id in (self.embeddings.get("user", {}) or {})
        except Exception:
            return False

    def _trending_empty_last_hours(self, hours: int) -> bool:
        try:
            post_df = self.data.get("posts")
            if post_df is None or post_df.empty or "CreateDate" not in post_df.columns:
                return True
            dt = pd.to_datetime(post_df["CreateDate"], errors="coerce", utc=True)
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=hours)
            return bool((dt >= cutoff).sum() == 0)
        except Exception:
            return True

    # ----- backend DB utilities -----
    def ping_backend_db(self) -> bool:
        if self.db_engine is None:
            return False
        try:
            with self.db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def backend_db_stats(self) -> dict:
        if not self.db_session_factory:
            return {"connected": False}
        stats = {"connected": self.ping_backend_db()}
        try:
            with self.db_session_factory() as s:
                posts = s.execute(text("SELECT COUNT(1) FROM Post")).scalar_one()
                users = s.execute(text("SELECT COUNT(1) FROM User")).scalar_one()
                reactions = s.execute(text("SELECT COUNT(1) FROM PostReaction")).scalar_one()
                comments = s.execute(text("SELECT COUNT(1) FROM Comment")).scalar_one()
                last_post = s.execute(text("SELECT MAX(CreateDate) FROM Post")).scalar_one()
                stats.update(
                    {
                        "post_count": int(posts or 0),
                        "user_count": int(users or 0),
                        "reaction_count": int(reactions or 0),
                        "comment_count": int(comments or 0),
                        "last_post_at": str(last_post) if last_post else None,
                    }
                )
        except Exception as e:
            stats.update({"error": str(e)})
        return stats

    def get_author_id(self, post_id: int) -> Optional[int]:
        # từ data đã load
        try:
            row = self.data["posts"].loc[self.data["posts"]["Id"] == post_id]
            if not row.empty:
                return int(row["UserId"].iloc[0])
        except Exception:
            pass
        # fallback DB
        try:
            if self.db_session_factory:
                with self.db_session_factory() as s:
                    r = s.execute(
                        text("SELECT UserId FROM Post WHERE Id=:pid"), {"pid": post_id}
                    ).first()
                    return int(r[0]) if r else None
        except Exception:
            pass
        return None
