"""
MULTI-CHANNEL RECALL FOR ONLINE INFERENCE
==========================================
4 Channels: Following, Collaborative Filtering, Content-Based, Trending
Target: ~1000 candidates in < 50ms
"""

import time
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import logging

logger = logging.getLogger(__name__)


class MultiChannelRecall:
    """
    Multi-channel recall system for online serving
    
    Channels:
    1. Following Feed (400 posts) - Posts from users you follow
    2. Collaborative Filtering (300 posts) - Posts liked by similar users
    3. Content-Based (200 posts) - Posts similar to your interests
    4. Trending (100 posts) - Globally trending posts
    
    Total: ~1000 unique candidates
    """
    
    def __init__(
        self,
        redis_client,
        data: Dict[str, pd.DataFrame],
        cf_model: Dict,
        embeddings: Dict,
        faiss_index=None,
        post_id_map: Optional[Dict] = None
    ):
        """
        Initialize recall system
        
        Args:
            redis_client: Redis connection (optional, will use in-memory if None)
            data: Dictionary of DataFrames (user, post, postreaction, friendship)
            cf_model: Collaborative filtering model
            embeddings: Dictionary with 'user' and 'post' embeddings
            faiss_index: FAISS index for fast similarity search (optional)
            post_id_map: Mapping from FAISS index to post_id
        """
        self.redis = redis_client
        self.data = data
        self.cf_model = cf_model
        self.embeddings = embeddings
        self.faiss_index = faiss_index
        self.post_id_map = post_id_map or {}
        
        # Precompute data structures for fast lookup
        self._build_lookup_structures()
        
        # Performance tracking
        self.metrics = defaultdict(list)
    
    def _build_lookup_structures(self):
        """
        Build in-memory data structures for fast lookup
        """
        logger.info("Building lookup structures...")
        
        # 1. Following dict: user_id -> [followed_user_ids]
        self.following_dict = {}
        if 'friendship' in self.data:
            friendships = self.data['friendship']
            # Status=2 means accepted friendship
            accepted = friendships[friendships['Status'] == 2]
            for _, row in accepted.iterrows():
                user1, user2 = row['UserId'], row['FriendId']
                if user1 not in self.following_dict:
                    self.following_dict[user1] = []
                if user2 not in self.following_dict:
                    self.following_dict[user2] = []
                self.following_dict[user1].append(user2)
                self.following_dict[user2].append(user1)  # Bidirectional
        
        # 2. User reactions: user_id -> [post_ids they liked]
        self.user_liked_posts = {}
        if 'postreaction' in self.data:
            reactions = self.data['postreaction']
            # ReactionTypeId 1,2,3,5 = positive (like, love, laugh, care)
            positive = reactions[reactions['ReactionTypeId'].isin([1, 2, 3, 5])]
            for user_id, group in positive.groupby('UserId'):
                self.user_liked_posts[user_id] = group['PostId'].tolist()
        
        # 3. Posts by author: author_id -> [post_ids]
        self.posts_by_author = {}
        if 'post' in self.data:
            posts = self.data['post']
            for author_id, group in posts.groupby('UserId'):
                self.posts_by_author[author_id] = group['Id'].tolist()
        
        # 4. Recent posts (last 48h)
        self.recent_posts = []
        if 'post' in self.data:
            posts = self.data['post']
            cutoff = datetime.now() - timedelta(hours=48)
            recent = posts[pd.to_datetime(posts['CreateDate']) >= cutoff]
            self.recent_posts = recent['Id'].tolist()
        
        logger.info(f"Following relationships: {len(self.following_dict)}")
        logger.info(f"User reactions: {len(self.user_liked_posts)}")
        logger.info(f"Recent posts (48h): {len(self.recent_posts)}")
    
    # ========================================================================
    # MAIN RECALL METHOD
    # ========================================================================
    
    def recall(self, user_id: int, k: int = 1000) -> List[int]:
        """
        Recall candidates from all channels
        
        Args:
            user_id: Target user ID
            k: Target number of candidates (default 1000)
        
        Returns:
            List of unique post IDs
        """
        start_time = time.time()
        
        # Run all channels
        channel_results = {}
        
        # Channel 1: Following (target: 400)
        t1 = time.time()
        channel_results['following'] = self.recall_following(user_id, k=400)
        self.metrics['following_latency'].append((time.time() - t1) * 1000)
        
        # Channel 2: Collaborative Filtering (target: 300)
        t2 = time.time()
        channel_results['cf'] = self.recall_collaborative(user_id, k=300)
        self.metrics['cf_latency'].append((time.time() - t2) * 1000)
        
        # Channel 3: Content-Based (target: 200)
        t3 = time.time()
        channel_results['content'] = self.recall_content_based(user_id, k=200)
        self.metrics['content_latency'].append((time.time() - t3) * 1000)
        
        # Channel 4: Trending (target: 100)
        t4 = time.time()
        channel_results['trending'] = self.recall_trending(k=100)
        self.metrics['trending_latency'].append((time.time() - t4) * 1000)
        
        # Merge and deduplicate
        all_candidates = []
        for channel, posts in channel_results.items():
            all_candidates.extend(posts)
        
        unique_candidates = list(dict.fromkeys(all_candidates))  # Preserve order
        
        # Track metrics
        total_latency = (time.time() - start_time) * 1000
        self.metrics['total_latency'].append(total_latency)
        self.metrics['total_candidates'].append(len(unique_candidates))
        
        logger.info(f"Recall completed in {total_latency:.1f}ms | "
                   f"Candidates: {len(unique_candidates)} | "
                   f"Following: {len(channel_results['following'])} | "
                   f"CF: {len(channel_results['cf'])} | "
                   f"Content: {len(channel_results['content'])} | "
                   f"Trending: {len(channel_results['trending'])}")
        
        return unique_candidates[:k]
    
    # ========================================================================
    # CHANNEL 1: FOLLOWING FEED
    # ========================================================================
    
    def recall_following(self, user_id: int, k: int = 400) -> List[int]:
        """
        Get posts from users that target user follows
        
        Process:
        1. Get list of followed users
        2. Get recent posts from those users (last 48h)
        3. Sort by recency
        4. Return top K
        
        Args:
            user_id: Target user ID
            k: Number of posts to return
            
        Returns:
            List of post IDs
        """
        try:
            # Get followed users
            followed_users = self.following_dict.get(user_id, [])
            
            if not followed_users:
                return []
            
            # Get posts from followed users
            candidate_posts = []
            posts_df = self.data['post']
            
            # Filter: posts by followed users + recent (48h)
            cutoff_time = datetime.now() - timedelta(hours=48)
            
            for author_id in followed_users:
                author_posts = posts_df[
                    (posts_df['UserId'] == author_id) &
                    (pd.to_datetime(posts_df['CreateDate']) >= cutoff_time)
                ]
                
                for _, post in author_posts.iterrows():
                    candidate_posts.append({
                        'post_id': post['Id'],
                        'created_at': pd.to_datetime(post['CreateDate']),
                        'score': 1.0  # All equal priority for now
                    })
            
            # Sort by recency (newest first)
            candidate_posts.sort(key=lambda x: x['created_at'], reverse=True)
            
            # Return top K post IDs
            return [p['post_id'] for p in candidate_posts[:k]]
            
        except Exception as e:
            logger.error(f"Error in following recall: {e}")
            return []
    
    # ========================================================================
    # CHANNEL 2: COLLABORATIVE FILTERING
    # ========================================================================
    
    def recall_collaborative(self, user_id: int, k: int = 300) -> List[int]:
        """
        Get posts liked by similar users
        
        Process:
        1. Get similar users (from CF model)
        2. Get posts they liked (recent 7 days)
        3. Score by similarity
        4. Return top K
        
        Args:
            user_id: Target user ID
            k: Number of posts to return
            
        Returns:
            List of post IDs
        """
        try:
            # Get similar users from CF model
            similar_users = self._get_similar_users(user_id, top_n=50)
            
            if not similar_users:
                return []
            
            # Get posts liked by similar users
            post_scores = defaultdict(float)
            
            for sim_user_id, similarity in similar_users:
                # Get posts this similar user liked (recent 7 days)
                liked_posts = self._get_user_liked_posts_recent(
                    sim_user_id, 
                    days=7
                )
                
                for post_id in liked_posts:
                    post_scores[post_id] += similarity
            
            # Sort by score
            sorted_posts = sorted(
                post_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [post_id for post_id, score in sorted_posts[:k]]
            
        except Exception as e:
            logger.error(f"Error in CF recall: {e}")
            return []
    
    def _get_similar_users(
        self, 
        user_id: int, 
        top_n: int = 50
    ) -> List[Tuple[int, float]]:
        """
        Get similar users from CF model
        
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if 'user_similarity_matrix' not in self.cf_model:
            return []
        
        user_ids = self.cf_model['user_ids']
        similarity_matrix = self.cf_model['user_similarity_matrix']
        
        # Find user index
        try:
            user_idx = list(user_ids).index(user_id)
        except ValueError:
            return []
        
        # Get similarity scores
        similarities = similarity_matrix[user_idx].toarray()[0]
        
        # Get top N similar users (excluding self)
        top_indices = np.argsort(similarities)[::-1][1:top_n+1]
        
        similar_users = [
            (user_ids[idx], similarities[idx])
            for idx in top_indices
            if similarities[idx] > 0
        ]
        
        return similar_users
    
    def _get_user_liked_posts_recent(
        self, 
        user_id: int, 
        days: int = 7
    ) -> List[int]:
        """Get posts user liked in last N days"""
        if 'postreaction' not in self.data:
            return []
        
        reactions = self.data['postreaction']
        cutoff = datetime.now() - timedelta(days=days)
        
        user_reactions = reactions[
            (reactions['UserId'] == user_id) &
            (reactions['ReactionTypeId'].isin([1, 2, 3, 5])) &  # Positive reactions
            (pd.to_datetime(reactions['CreateDate']) >= cutoff)
        ]
        
        return user_reactions['PostId'].tolist()
    
    # ========================================================================
    # CHANNEL 3: CONTENT-BASED
    # ========================================================================
    
    def recall_content_based(self, user_id: int, k: int = 200) -> List[int]:
        """
        Get posts similar to user's interests (embedding-based)
        
        Process:
        1. Get user embedding
        2. Search similar post embeddings
        3. Filter by age (>= 6h old, so embeddings are ready)
        4. Return top K
        
        Args:
            user_id: Target user ID
            k: Number of posts to return
            
        Returns:
            List of post IDs
        """
        try:
            # Check if user has embedding
            if 'user' not in self.embeddings:
                return []
            
            if user_id not in self.embeddings['user']:
                return []
            
            user_emb = self.embeddings['user'][user_id]
            
            # Option 1: Use FAISS (fast)
            if self.faiss_index is not None:
                return self._faiss_search(user_emb, k=k)
            
            # Option 2: Fallback to brute-force cosine similarity
            return self._brute_force_similarity_search(user_emb, k=k)
            
        except Exception as e:
            logger.error(f"Error in content-based recall: {e}")
            return []
    
    def _faiss_search(self, user_emb: np.ndarray, k: int) -> List[int]:
        """Fast similarity search using FAISS"""
        # Normalize embedding
        user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
        
        # Search (over-fetch for filtering)
        distances, indices = self.faiss_index.search(
            user_emb.reshape(1, -1).astype(np.float32),
            k * 3
        )
        
        # Convert indices to post IDs
        candidate_ids = [
            self.post_id_map.get(idx, -1) 
            for idx in indices[0]
        ]
        
        # Filter out invalid IDs
        candidate_ids = [pid for pid in candidate_ids if pid > 0]
        
        # Filter by age (>= 6h old)
        filtered = self._filter_by_min_age(candidate_ids, min_hours=6)
        
        return filtered[:k]
    
    def _brute_force_similarity_search(
        self, 
        user_emb: np.ndarray, 
        k: int
    ) -> List[int]:
        """Fallback: Brute-force cosine similarity"""
        if 'post' not in self.embeddings:
            return []
        
        post_embeddings = self.embeddings['post']
        
        # Normalize user embedding
        user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
        
        # Compute similarities
        similarities = []
        for post_id, post_emb in post_embeddings.items():
            post_emb_norm = post_emb / (np.linalg.norm(post_emb) + 1e-8)
            sim = np.dot(user_emb, post_emb_norm)
            similarities.append((post_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by age
        candidate_ids = [post_id for post_id, sim in similarities[:k*3]]
        filtered = self._filter_by_min_age(candidate_ids, min_hours=6)
        
        return filtered[:k]
    
    def _filter_by_min_age(
        self, 
        post_ids: List[int], 
        min_hours: int = 6
    ) -> List[int]:
        """Filter posts by minimum age (for embedding readiness)"""
        if 'post' not in self.data:
            return post_ids
        
        posts_df = self.data['post']
        cutoff_time = datetime.now() - timedelta(hours=min_hours)
        
        filtered = []
        for post_id in post_ids:
            post = posts_df[posts_df['Id'] == post_id]
            if not post.empty:
                create_time = pd.to_datetime(post.iloc[0]['CreateDate'])
                if create_time <= cutoff_time:
                    filtered.append(post_id)
        
        return filtered
    
    # ========================================================================
    # CHANNEL 4: TRENDING
    # ========================================================================
    
    def recall_trending(self, k: int = 100) -> List[int]:
        """
        Get globally trending posts
        
        Process:
        1. Compute trending score for recent posts (last 24h)
        2. Score = engagement / time_decay
        3. Return top K
        
        Args:
            k: Number of posts to return
            
        Returns:
            List of post IDs
        """
        try:
            # Get recent posts (last 24h)
            posts_df = self.data['post']
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_posts = posts_df[
                pd.to_datetime(posts_df['CreateDate']) >= cutoff_time
            ]
            
            if recent_posts.empty:
                return []
            
            # Compute trending scores
            trending_scores = []
            
            for _, post in recent_posts.iterrows():
                post_id = post['Id']
                created_at = pd.to_datetime(post['CreateDate'])
                
                # Get engagement
                engagement = self._get_post_engagement(post_id)
                
                # Time decay (hours since creation)
                hours_old = (datetime.now() - created_at).total_seconds() / 3600
                time_decay = (hours_old + 2) ** 1.5  # Power law decay
                
                # Trending score
                score = engagement / time_decay
                
                trending_scores.append((post_id, score))
            
            # Sort by score
            trending_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [post_id for post_id, score in trending_scores[:k]]
            
        except Exception as e:
            logger.error(f"Error in trending recall: {e}")
            return []
    
    def _get_post_engagement(self, post_id: int) -> float:
        """
        Compute engagement score for a post
        
        Formula: likes + comments*2 + shares*3
        """
        if 'postreaction' not in self.data:
            return 0.0
        
        reactions = self.data['postreaction']
        post_reactions = reactions[reactions['PostId'] == post_id]
        
        if post_reactions.empty:
            return 0.0
        
        # Count by reaction type
        reaction_counts = post_reactions['ReactionTypeId'].value_counts()
        
        # Weighted sum
        likes = reaction_counts.get(1, 0) + reaction_counts.get(2, 0)  # like + love
        comments = reaction_counts.get(3, 0)  # comment
        shares = reaction_counts.get(5, 0)  # share
        
        engagement = likes + comments * 2 + shares * 3
        
        return float(engagement)
    
    # ========================================================================
    # PERFORMANCE MONITORING
    # ========================================================================
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of performance metrics"""
        if not self.metrics['total_latency']:
            return {}
        
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        return summary
    
    def print_metrics(self):
        """Print performance metrics"""
        summary = self.get_metrics_summary()
        
        print("\n" + "="*60)
        print("RECALL PERFORMANCE METRICS")
        print("="*60)
        
        for metric, stats in summary.items():
            print(f"\n{metric}:")
            print(f"  Mean: {stats['mean']:.1f}")
            print(f"  P50:  {stats['p50']:.1f}")
            print(f"  P95:  {stats['p95']:.1f}")
            print(f"  P99:  {stats['p99']:.1f}")
        
        print("="*60 + "\n")


# ========================================================================
# ASYNC VERSION (For production)
# ========================================================================

class AsyncMultiChannelRecall(MultiChannelRecall):
    """
    Async version for parallel channel execution
    """
    
    async def recall_async(self, user_id: int, k: int = 1000) -> List[int]:
        """
        Async recall with parallel channel execution
        """
        start_time = time.time()
        
        # Run all channels in parallel
        tasks = [
            self._recall_following_async(user_id, 400),
            self._recall_cf_async(user_id, 300),
            self._recall_content_async(user_id, 200),
            self._recall_trending_async(100)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results
        all_candidates = []
        channel_names = ['following', 'cf', 'content', 'trending']
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Channel {channel_names[i]} failed: {result}")
                continue
            all_candidates.extend(result)
        
        unique_candidates = list(dict.fromkeys(all_candidates))
        
        total_latency = (time.time() - start_time) * 1000
        logger.info(f"Async recall completed in {total_latency:.1f}ms | "
                   f"Candidates: {len(unique_candidates)}")
        
        return unique_candidates[:k]
    
    async def _recall_following_async(self, user_id: int, k: int) -> List[int]:
        return await asyncio.to_thread(self.recall_following, user_id, k)
    
    async def _recall_cf_async(self, user_id: int, k: int) -> List[int]:
        return await asyncio.to_thread(self.recall_collaborative, user_id, k)
    
    async def _recall_content_async(self, user_id: int, k: int) -> List[int]:
        return await asyncio.to_thread(self.recall_content_based, user_id, k)
    
    async def _recall_trending_async(self, k: int) -> List[int]:
        return await asyncio.to_thread(self.recall_trending, k)


# ========================================================================
# EXAMPLE USAGE
# ========================================================================

if __name__ == "__main__":
    """
    Example usage of MultiChannelRecall
    """
    import pickle
    from recommender.common.data_loader import load_data
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    print("Loading data...")
    data = load_data('dataset')
    
    # Load models
    print("Loading models...")
    with open(r'models\v20251024_101625\cf_model.pkl', 'rb') as f:
        cf_model = pickle.load(f)
    
    with open(r'models\v20251024_101625\embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    
    # Initialize recall system
    print("Initializing recall system...")
    recall = MultiChannelRecall(
        redis_client=None,  # Will use in-memory
        data=data,
        cf_model=cf_model,
        embeddings=embeddings
    )
    
    # Test recall
    print("\nTesting recall for user 123...")
    candidates = recall.recall(user_id=1, k=1000)
    
    print(f"\nRecalled {len(candidates)} candidates")
    print(f"Sample candidates: {candidates[:10]}")
    
    # Print performance metrics
    recall.print_metrics()