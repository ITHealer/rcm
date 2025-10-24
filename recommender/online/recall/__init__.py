"""
RECALL PACKAGE
==============
Multi-channel recall system

Available channels:
- FollowingRecall: Posts from followed users
- CFRecall: Collaborative filtering based
- ContentRecall: Content-based (embedding similarity)
- TrendingRecall: Trending/popular posts
"""

from .base_recall import BaseRecall
from .following_recall import FollowingRecall
from .cf_recall import CFRecall
from .content_recall import ContentRecall
from .trending_recall import TrendingRecall

__all__ = [
    'BaseRecall',
    'FollowingRecall',
    'CFRecall',
    'ContentRecall',
    'TrendingRecall',
]