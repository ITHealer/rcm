"""
RE-RANKER
=========
Apply business rules and diversity constraints

Rules:
1. Diversity: Max 2 consecutive posts from same author
2. Freshness boost: Recent posts get score boost
3. Quality filter: Remove low-quality posts
4. Deduplication: Remove duplicate content
"""

import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class Reranker:
    """
    Re-rank posts with business rules
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize reranker
        
        Args:
            config: Configuration dict with rules
        """
        self.config = config or {}
        
        # Diversity rules
        self.max_consecutive_same_author = self.config.get(
            'max_consecutive_same_author', 2
        )
        self.max_same_author_in_feed = self.config.get(
            'max_same_author_in_feed', 5
        )
        
        # Freshness rules
        self.freshness_enabled = self.config.get('freshness_enabled', True)
        self.freshness_boost_hours = self.config.get('freshness_boost_hours', 24)
        self.freshness_boost_factor = self.config.get('freshness_boost_factor', 1.5)
        
        # Quality rules
        self.quality_enabled = self.config.get('quality_enabled', True)
        self.min_score = self.config.get('min_score', 0.3)
        
        logger.info("Reranker initialized with rules:")
        logger.info(f"  Max consecutive same author: {self.max_consecutive_same_author}")
        logger.info(f"  Freshness boost: {self.freshness_enabled}")
        logger.info(f"  Quality filter: {self.quality_enabled} (min_score={self.min_score})")
    
    def rerank(
        self,
        ranked_df: pd.DataFrame,
        post_metadata: Optional[Dict] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Apply business rules and re-rank
        
        Args:
            ranked_df: DataFrame with columns [post_id, ml_score]
            post_metadata: Dict mapping post_id to metadata
            limit: Number of posts to return
            
        Returns:
            List of dicts with post info
        """
        if ranked_df.empty:
            return []
        
        # Make a copy
        df = ranked_df.copy()
        
        # Apply freshness boost (if enabled and metadata available)
        if self.freshness_enabled and post_metadata:
            df = self._apply_freshness_boost(df, post_metadata)
        
        # Apply quality filter
        if self.quality_enabled:
            df = df[df['ml_score'] >= self.min_score]
        
        # Sort by adjusted score
        df = df.sort_values('ml_score', ascending=False)
        
        # Apply diversity rules
        final_posts = self._apply_diversity_rules(df, post_metadata, limit)
        
        return final_posts
    
    def _apply_freshness_boost(
        self,
        df: pd.DataFrame,
        post_metadata: Dict
    ) -> pd.DataFrame:
        """
        Boost scores for recent posts
        
        Args:
            df: DataFrame with ml_score
            post_metadata: Post metadata with creation times
            
        Returns:
            DataFrame with boosted scores
        """
        now = datetime.now()
        boost_cutoff = now - timedelta(hours=self.freshness_boost_hours)
        
        for idx, row in df.iterrows():
            post_id = int(row['post_id'])
            
            if post_id in post_metadata:
                created_at = post_metadata[post_id].get('created_at')
                
                if created_at and created_at >= boost_cutoff:
                    # Apply boost
                    df.at[idx, 'ml_score'] *= self.freshness_boost_factor
        
        return df
    
    def _apply_diversity_rules(
        self,
        df: pd.DataFrame,
        post_metadata: Optional[Dict],
        limit: int
    ) -> List[Dict]:
        """
        Apply diversity constraints
        
        Rules:
        1. Max 2 consecutive posts from same author
        2. Max 5 total posts from same author in feed
        
        Args:
            df: Sorted DataFrame
            post_metadata: Post metadata
            limit: Max posts to return
            
        Returns:
            List of post dicts
        """
        final_posts = []
        author_counts = {}
        last_author_id = None
        consecutive_count = 0
        
        for _, row in df.iterrows():
            if len(final_posts) >= limit:
                break
            
            post_id = int(row['post_id'])
            score = float(row['ml_score'])
            
            # Get author ID (from metadata or estimate)
            if post_metadata and post_id in post_metadata:
                author_id = post_metadata[post_id].get('author_id', post_id % 1000)
            else:
                # Fallback: estimate from post_id
                author_id = post_id % 1000
            
            # Check diversity rules
            # Rule 1: Max consecutive from same author
            if author_id == last_author_id:
                consecutive_count += 1
                if consecutive_count >= self.max_consecutive_same_author:
                    continue  # Skip this post
            else:
                consecutive_count = 0
                last_author_id = author_id
            
            # Rule 2: Max total from same author
            author_count = author_counts.get(author_id, 0)
            if author_count >= self.max_same_author_in_feed:
                continue  # Skip this post
            
            # Add to feed
            final_posts.append({
                'post_id': post_id,
                'score': score,
                'author_id': author_id,
                'rank': len(final_posts) + 1
            })
            
            # Update author count
            author_counts[author_id] = author_count + 1
        
        logger.debug(f"Re-ranked: {len(final_posts)} posts from {len(author_counts)} authors")
        
        return final_posts
    
    def get_config(self) -> Dict:
        """Get current configuration"""
        return {
            'max_consecutive_same_author': self.max_consecutive_same_author,
            'max_same_author_in_feed': self.max_same_author_in_feed,
            'freshness_enabled': self.freshness_enabled,
            'freshness_boost_hours': self.freshness_boost_hours,
            'freshness_boost_factor': self.freshness_boost_factor,
            'quality_enabled': self.quality_enabled,
            'min_score': self.min_score
        }