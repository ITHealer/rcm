"""
FRIEND RECOMMENDATION SYSTEM
============================
Recommend potential friends based on:
1. Mutual friends (graph-based)
2. Similar interests (embedding-based)
3. Similar interaction patterns (CF-based)
4. Popular users (fallback)

Output: Ranked list of users to friend
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
import logging
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class FriendRecommendation:
    """
    Friend recommendation system
    
    Combines multiple signals:
    - Social graph (mutual friends)
    - User embeddings (similar interests)
    - Collaborative filtering (similar behavior)
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        embeddings: Dict[str, Dict],
        cf_model: Dict,
        config: Optional[Dict] = None
    ):
        """
        Initialize friend recommender
        
        Args:
            data: Data dictionary with 'friendship' DataFrame
            embeddings: User embeddings
            cf_model: CF similarities
            config: Configuration
        """
        self.data = data
        self.embeddings = embeddings
        self.cf_model = cf_model
        self.config = config or {}
        
        # Build friendship graph
        self.friendship_graph = self._build_friendship_graph()
        
        # Configuration
        self.mutual_friends_weight = self.config.get('mutual_friends_weight', 0.4)
        self.embedding_similarity_weight = self.config.get('embedding_similarity_weight', 0.3)
        self.cf_similarity_weight = self.config.get('cf_similarity_weight', 0.3)
        
        logger.info("FriendRecommendation initialized")
    
    def recommend_friends(
        self,
        user_id: int,
        k: int = 20,
        exclude_existing: bool = True
    ) -> List[Dict]:
        """
        Recommend potential friends
        
        Args:
            user_id: Target user ID
            k: Number of recommendations
            exclude_existing: Exclude existing friends
            
        Returns:
            List of dicts with user_id and score
        """
        start_time = time.time()
        
        # Get existing friends
        existing_friends = self.friendship_graph.get(user_id, set())
        
        # Channel 1: Mutual friends (graph-based)
        mutual_scores = self._score_by_mutual_friends(user_id, existing_friends)
        
        # Channel 2: Similar interests (embedding-based)
        embedding_scores = self._score_by_embeddings(user_id, existing_friends)
        
        # Channel 3: Similar behavior (CF-based)
        cf_scores = self._score_by_cf(user_id, existing_friends)
        
        # Combine scores
        all_candidates = set(mutual_scores.keys()) | set(embedding_scores.keys()) | set(cf_scores.keys())
        
        final_scores = {}
        for candidate_id in all_candidates:
            if exclude_existing and candidate_id in existing_friends:
                continue
            
            if candidate_id == user_id:  # Don't recommend self
                continue
            
            # Weighted combination
            score = (
                mutual_scores.get(candidate_id, 0) * self.mutual_friends_weight +
                embedding_scores.get(candidate_id, 0) * self.embedding_similarity_weight +
                cf_scores.get(candidate_id, 0) * self.cf_similarity_weight
            )
            
            final_scores[candidate_id] = score
        
        # Sort by score
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format output
        recommendations = [
            {
                'user_id': int(user_id),
                'score': float(score),
                'mutual_friends': mutual_scores.get(user_id, 0),
                'embedding_similarity': embedding_scores.get(user_id, 0),
                'cf_similarity': cf_scores.get(user_id, 0)
            }
            for user_id, score in ranked[:k]
        ]
        
        latency = (time.time() - start_time) * 1000
        logger.debug(f"Friend recommendation for user {user_id}: {len(recommendations)} candidates in {latency:.1f}ms")
        
        return recommendations
    
    def _build_friendship_graph(self) -> Dict[int, Set[int]]:
        """
        Build friendship graph from data
        
        Returns:
            Dict mapping user_id to set of friend_ids
        """
        graph = defaultdict(set)
        
        if 'friendship' not in self.data:
            logger.warning("No friendship data available")
            return graph
        
        friendships = self.data['friendship']
        
        # Filter accepted friendships (Status=2)
        accepted = friendships[friendships['Status'] == 2]
        
        for _, row in accepted.iterrows():
            user1 = int(row['UserId'])
            user2 = int(row['FriendId'])
            
            # Bidirectional
            graph[user1].add(user2)
            graph[user2].add(user1)
        
        logger.info(f"Built friendship graph: {len(graph)} users")
        
        return dict(graph)
    
    def _score_by_mutual_friends(
        self,
        user_id: int,
        existing_friends: Set[int]
    ) -> Dict[int, float]:
        """
        Score candidates by number of mutual friends
        
        Strategy: "Friend of friends"
        
        Args:
            user_id: Target user
            existing_friends: Current friends
            
        Returns:
            Dict mapping candidate_id to normalized score
        """
        mutual_counts = defaultdict(int)
        
        # For each friend, count their friends
        for friend_id in existing_friends:
            friend_of_friend = self.friendship_graph.get(friend_id, set())
            
            for candidate_id in friend_of_friend:
                if candidate_id != user_id and candidate_id not in existing_friends:
                    mutual_counts[candidate_id] += 1
        
        # Normalize scores (0-1)
        if not mutual_counts:
            return {}
        
        max_count = max(mutual_counts.values())
        normalized = {
            uid: count / max_count
            for uid, count in mutual_counts.items()
        }
        
        return normalized
    
    def _score_by_embeddings(
        self,
        user_id: int,
        existing_friends: Set[int]
    ) -> Dict[int, float]:
        """
        Score candidates by user embedding similarity
        
        Args:
            user_id: Target user
            existing_friends: Current friends
            
        Returns:
            Dict mapping candidate_id to similarity score
        """
        if 'user' not in self.embeddings:
            return {}
        
        user_embs = self.embeddings['user']
        
        if user_id not in user_embs:
            return {}
        
        target_emb = user_embs[user_id]
        
        # Compute similarities
        similarities = {}
        for candidate_id, candidate_emb in user_embs.items():
            if candidate_id == user_id or candidate_id in existing_friends:
                continue
            
            # Cosine similarity
            sim = np.dot(target_emb, candidate_emb) / (
                np.linalg.norm(target_emb) * np.linalg.norm(candidate_emb) + 1e-8
            )
            
            similarities[candidate_id] = float(max(0, sim))  # Clip to [0, 1]
        
        return similarities
    
    def _score_by_cf(
        self,
        user_id: int,
        existing_friends: Set[int]
    ) -> Dict[int, float]:
        """
        Score candidates by CF similarity
        
        Args:
            user_id: Target user
            existing_friends: Current friends
            
        Returns:
            Dict mapping candidate_id to similarity score
        """
        if 'user_similarities' not in self.cf_model:
            return {}
        
        user_sims = self.cf_model['user_similarities']
        
        if user_id not in user_sims:
            return {}
        
        # Get similar users (already scored)
        similar_users = user_sims[user_id]
        
        # Convert to dict
        scores = {}
        for candidate_id, score in similar_users:
            if candidate_id != user_id and candidate_id not in existing_friends:
                scores[int(candidate_id)] = float(score)
        
        return scores


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_friend_recommendations(
    user_id: int,
    data: Dict,
    embeddings: Dict,
    cf_model: Dict,
    k: int = 20,
    config: Optional[Dict] = None
) -> List[Dict]:
    """
    Convenience function to get friend recommendations
    
    Args:
        user_id: Target user ID
        data: Data dictionary
        embeddings: User embeddings
        cf_model: CF model
        k: Number of recommendations
        config: Configuration
        
    Returns:
        List of friend recommendations
    """
    recommender = FriendRecommendation(
        data=data,
        embeddings=embeddings,
        cf_model=cf_model,
        config=config
    )
    
    return recommender.recommend_friends(user_id, k=k)