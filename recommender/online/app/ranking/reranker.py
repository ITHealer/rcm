"""
Re-ranking Service
=================
Apply business rules to improve feed quality

Rules:
1. Diversity: No more than 2 consecutive posts from same author
2. Category diversity: Mix categories within sliding window
3. Freshness boost: Recent posts get score boost
4. Ad insertion: Insert ads at specific positions
5. Content safety: Filter unsafe content

Target latency: < 20ms
"""

from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)


class Reranker:
    """
    Apply business rules and final adjustments to ranked feed
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize reranker
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Diversity rules
        self.max_consecutive_same_author = self.config.get('max_consecutive_same_author', 2)
        self.category_diversity_window = self.config.get('category_diversity_window', 5)
        
        # Freshness boost
        freshness_cfg = self.config.get('freshness', {})
        self.freshness_enabled = freshness_cfg.get('enabled', True)
        self.freshness_threshold_hours = freshness_cfg.get('threshold_hours', 3.0)
        self.freshness_boost_factor = freshness_cfg.get('boost_factor', 1.2)
        
        # Ad insertion
        ads_cfg = self.config.get('ads', {})
        self.ads_enabled = ads_cfg.get('enabled', False)
        self.ad_positions = ads_cfg.get('positions', [5, 15, 30])
        
        # Metrics
        self.metrics = {
            'total_reranks': 0,
            'total_items': 0,
            'diversity_swaps': 0,
            'freshness_boosts': 0,
            'ads_inserted': 0,
            'total_time_ms': 0
        }
        
        logger.info(f"Reranker initialized: diversity={self.max_consecutive_same_author}, "
                   f"freshness={self.freshness_enabled}, ads={self.ads_enabled}")
    
    def rerank(
        self,
        ranked_df: pd.DataFrame,
        post_metadata: Dict[int, Dict],
        limit: int = 50
    ) -> List[Dict]:
        """
        Apply reranking rules
        
        Process:
        1. Apply freshness boost
        2. Enforce diversity rules
        3. Insert ads (if enabled)
        4. Return final feed
        
        Args:
            ranked_df: DataFrame with post_id and score (sorted)
            post_metadata: Dict mapping post_id to metadata (author_id, category, created_at, etc.)
            limit: Final feed size
            
        Returns:
            List of post dicts with metadata
        """
        start_time = time.time()
        
        if ranked_df.empty:
            return []
        
        # Make a copy to avoid modifying original
        df = ranked_df.copy()
        
        # Apply freshness boost
        if self.freshness_enabled:
            df = self._apply_freshness_boost(df, post_metadata)
        
        # Re-sort after freshness boost
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        
        # Apply diversity rules
        df = self._apply_diversity(df, post_metadata)
        
        # Limit to target size
        df = df.head(limit)
        
        # Build final feed with metadata
        final_feed = self._build_feed_response(df, post_metadata)
        
        # Insert ads (if enabled)
        if self.ads_enabled:
            final_feed = self._insert_ads(final_feed)
        
        # Update metrics
        self.metrics['total_reranks'] += 1
        self.metrics['total_items'] += len(final_feed)
        self.metrics['total_time_ms'] += (time.time() - start_time) * 1000
        
        logger.debug(f"Reranked to {len(final_feed)} items, "
                    f"diversity_swaps={self.metrics['diversity_swaps']}, "
                    f"freshness_boosts={self.metrics['freshness_boosts']}")
        
        return final_feed
    
    def _apply_freshness_boost(
        self,
        df: pd.DataFrame,
        post_metadata: Dict[int, Dict]
    ) -> pd.DataFrame:
        """
        Boost scores for fresh posts (< 3 hours old)
        
        Args:
            df: DataFrame with post_id and score
            post_metadata: Post metadata
            
        Returns:
            DataFrame with adjusted scores
        """
        now = pd.Timestamp.now()
        threshold = pd.Timedelta(hours=self.freshness_threshold_hours)
        
        boosted_count = 0
        
        for idx, row in df.iterrows():
            post_id = row['post_id']
            metadata = post_metadata.get(post_id, {})
            
            created_at = metadata.get('created_at')
            
            if created_at and isinstance(created_at, (pd.Timestamp, datetime)):
                if not isinstance(created_at, pd.Timestamp):
                    created_at = pd.Timestamp(created_at)
                
                age = now - created_at
                
                if age < threshold:
                    # Apply boost
                    df.at[idx, 'score'] *= self.freshness_boost_factor
                    boosted_count += 1
        
        if boosted_count > 0:
            self.metrics['freshness_boosts'] += boosted_count
            logger.debug(f"Applied freshness boost to {boosted_count} posts")
        
        return df
    
    def _apply_diversity(
        self,
        df: pd.DataFrame,
        post_metadata: Dict[int, Dict]
    ) -> pd.DataFrame:
        """
        Apply diversity rules:
        - No more than N consecutive posts from same author
        - Mix categories within sliding window
        
        Args:
            df: DataFrame with post_id and score (sorted)
            post_metadata: Post metadata
            
        Returns:
            Reordered DataFrame
        """
        if len(df) <= 1:
            return df
        
        posts = df.to_dict('records')
        final_order = []
        remaining = posts.copy()
        
        swap_count = 0
        
        while remaining:
            # Try to find a post that satisfies diversity rules
            found = False
            
            for i, post in enumerate(remaining):
                post_id = post['post_id']
                metadata = post_metadata.get(post_id, {})
                author_id = metadata.get('author_id')
                category = metadata.get('category')
                
                # Check author diversity
                if len(final_order) >= self.max_consecutive_same_author:
                    # Check last N posts
                    recent_authors = [
                        post_metadata.get(p['post_id'], {}).get('author_id')
                        for p in final_order[-self.max_consecutive_same_author:]
                    ]
                    
                    if all(a == author_id for a in recent_authors if a is not None):
                        # Too many consecutive from same author, skip
                        continue
                
                # Check category diversity (within window)
                if self.category_diversity_window > 0 and len(final_order) >= 2:
                    window_size = min(self.category_diversity_window, len(final_order))
                    recent_categories = [
                        post_metadata.get(p['post_id'], {}).get('category')
                        for p in final_order[-window_size:]
                    ]
                    
                    # Allow if: different category OR not all recent are same category
                    if category and category in recent_categories:
                        if all(c == category for c in recent_categories if c is not None):
                            # Window is saturated with this category, skip
                            continue
                
                # This post satisfies diversity rules
                final_order.append(post)
                remaining.pop(i)
                
                if i > 0:
                    swap_count += 1
                
                found = True
                break
            
            # If no post satisfies rules, take the best remaining
            if not found:
                final_order.append(remaining.pop(0))
        
        if swap_count > 0:
            self.metrics['diversity_swaps'] += swap_count
            logger.debug(f"Made {swap_count} swaps for diversity")
        
        # Convert back to DataFrame
        result_df = pd.DataFrame(final_order)
        
        return result_df
    
    def _build_feed_response(
        self,
        df: pd.DataFrame,
        post_metadata: Dict[int, Dict]
    ) -> List[Dict]:
        """
        Build final feed response with metadata
        
        Args:
            df: Final DataFrame with post_id and score
            post_metadata: Post metadata
            
        Returns:
            List of post dicts
        """
        feed = []
        
        for idx, row in df.iterrows():
            post_id = row['post_id']
            score = row['score']
            metadata = post_metadata.get(post_id, {})
            
            feed.append({
                'post_id': int(post_id),
                'score': float(score),
                'rank': idx + 1,
                'author_id': metadata.get('author_id'),
                'category': metadata.get('category'),
                'created_at': str(metadata.get('created_at')) if metadata.get('created_at') else None,
                **metadata  # Include all metadata
            })
        
        return feed
    
    def _insert_ads(self, feed: List[Dict]) -> List[Dict]:
        """
        Insert ads at predefined positions
        
        Args:
            feed: List of post dicts
            
        Returns:
            Feed with ads inserted
        """
        # TODO: Implement actual ad fetching logic
        # For now, just mark positions where ads should go
        
        result = []
        ads_inserted = 0
        
        for idx, post in enumerate(feed):
            result.append(post)
            
            # Check if we should insert ad after this position
            if (idx + 1) in self.ad_positions:
                # Insert ad placeholder
                result.append({
                    'is_ad': True,
                    'ad_id': f"ad_{ads_inserted + 1}",
                    'rank': idx + 1.5  # Between posts
                })
                ads_inserted += 1
        
        if ads_inserted > 0:
            self.metrics['ads_inserted'] += ads_inserted
            logger.debug(f"Inserted {ads_inserted} ads")
        
        return result
    
    def get_metrics(self) -> Dict:
        """Get reranking metrics"""
        avg_time = (
            self.metrics['total_time_ms'] / self.metrics['total_reranks']
            if self.metrics['total_reranks'] > 0 else 0
        )
        
        return {
            **self.metrics,
            'avg_time_ms': round(avg_time, 2)
        }
    
    def reset_metrics(self):
        """Reset metrics"""
        for key in self.metrics:
            self.metrics[key] = 0