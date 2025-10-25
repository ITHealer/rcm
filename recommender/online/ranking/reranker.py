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

import hashlib
import pandas as pd
from collections import Counter
from typing import List, Dict, Optional, Tuple
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
        self.freshness_boost_hours = self.config.get('freshness_boost_hours', 6) # 24, < 6h, +10% (1.1)
        self.freshness_boost_factor = self.config.get('freshness_boost_factor', 1.1) # 1.5
        
        # Quality rules
        self.quality_enabled = self.config.get('quality_enabled', True)
        ## On/Off score threshold ML (Default off to avoid disqualification due to low score)
        self.use_min_score = self.config.get('use_min_score', False)
        self.min_score = self.config.get('min_score', 0.3) 

        # Valid Value Status (1 = active)
        self.quality_status_ok_value = self.config.get('quality_status_ok_value', 10)

        # Content dedup
        self.content_dedup_enabled = self.config.get('content_dedup_enabled', True)

        # Audit logging config
        audit_cfg = self.config.get('audit_logging', {})
        self.audit_enabled = audit_cfg.get('enabled', True)
        self.audit_sample = int(audit_cfg.get('sample_posts', 5))
        
        logger.info("Reranker initialized with rules:")
        logger.info(f"  Max consecutive same author: {self.max_consecutive_same_author}")
        logger.info(f"  Max total per author: {self.max_same_author_in_feed}")
        logger.info(f"  Freshness boost: {self.freshness_enabled}")
        logger.info(f"  Quality filter: {self.quality_enabled} (min_score={self.min_score})"
                    f"status_ok={self.quality_status_ok_value}, "
                    f"use_min_score={self.use_min_score} (min_score={self.min_score})")
        logger.info(f"  Content dedup: {self.content_dedup_enabled}")
    
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
                post_metadata: Dict mapping post_id -> {
                    'author_id': int,
                    'created_at': datetime or str,
                    'status': int,
                    'content_hash': str,
                    'title': str,
                    'content': str,
                }
            limit: Number of posts to return
            
        Returns:
            List of dicts with post info
        """
        if ranked_df is None or ranked_df.empty:
            return []
        
        # Make a copy
        df = ranked_df.copy()
        
        # ---------------------------
        # AUDIT: pre snapshot
        # ---------------------------
        audit = {
            "pre": {
                "count": int(len(df)),
                "unique_authors": None,
                "top_author_share": None,
            },
            "freshness": {"boosted": 0},
            "quality": {"removed_status": 0, "removed_min_score": 0},
            "dedup": {"removed": 0, "examples": []},
            "diversity": {"removed_consecutive": 0, "removed_cap": 0},
            "post": {"count": None, "unique_authors": None, "top_author_share": None}
        }

        if post_metadata:
            pre_authors = self._extract_authors(df, post_metadata)
            audit["pre"]["unique_authors"] = len(set(pre_authors)) if pre_authors else 0
            audit["pre"]["top_author_share"] = self._top_author_share(pre_authors) if pre_authors else 0.0

        
        # 1) Apply freshness boost (if enabled and metadata available)
        # if self.freshness_enabled and post_metadata:
        #     df = self._apply_freshness_boost(df, post_metadata)

        # Debug
        if self.freshness_enabled and post_metadata:
            before_scores = df["ml_score"].copy()
            df = self._apply_freshness_boost(df, post_metadata, audit)
            # (Optionally) compute avg boost %
            try:
                delta = (df["ml_score"] - before_scores).clip(lower=0.0)
                boosted_any = (delta > 0).sum()
                audit["freshness"]["boosted"] = int(boosted_any)
            except Exception:
                pass
        
        # 2) Apply quality filter
        # if self.quality_enabled:
        #     df = df[df['ml_score'] >= self.min_score]

        # if self.quality_enabled and post_metadata:
        #     df = self._apply_quality_filter(df, post_metadata)
        # if self.use_min_score:
        #     df = df[df['ml_score'] >= self.min_score]

        # Debug
        if self.quality_enabled and post_metadata:
            n_before = len(df)
            df = self._apply_quality_filter(df, post_metadata, audit=audit)
            audit["quality"]["removed_status"] += int(n_before - len(df))
        if self.use_min_score:
            n_before = len(df)
            df = df[df['ml_score'] >= self.min_score]
            audit["quality"]["removed_min_score"] += int(n_before - len(df))

        if df.empty:
            self._emit_audit(audit)
            return []

        # 3) Dedup content
        # if self.content_dedup_enabled and post_metadata:
        #     df = self._apply_content_dedup(df, post_metadata)
        #     if df.empty:
        #         return []

        # Debug
        if self.content_dedup_enabled and post_metadata:
            n_before = len(df)
            df, dedup_examples = self._apply_content_dedup(df, post_metadata)
            audit["dedup"]["removed"] += int(n_before - len(df))
            audit["dedup"]["examples"] = dedup_examples[: self.audit_sample]
            if df is None:
                self._emit_audit(audit)
                return []
            
        # Sort by adjusted score
        df = df.sort_values('ml_score', ascending=False)
        
        # 4) Apply diversity rules (consecutive + cap per author)
        # final_posts = self._apply_diversity_rules(df, post_metadata, limit)
        
        # Debug
        final_posts, removed_counts = self._apply_diversity_rules(df, post_metadata, limit)
        audit["diversity"]["removed_consecutive"] += removed_counts.get("consecutive", 0)
        audit["diversity"]["removed_cap"] += removed_counts.get("cap", 0)

        # AUDIT: post snapshot
        audit["post"]["count"] = int(len(final_posts))
        if post_metadata and final_posts:
            post_authors = [p["author_id"] for p in final_posts if p.get("author_id") is not None]
            audit["post"]["unique_authors"] = len(set(post_authors)) if post_authors else 0
            audit["post"]["top_author_share"] = self._top_author_share(post_authors) if post_authors else 0.0

        self._emit_audit(audit, sample=final_posts[: self.audit_sample])

        return final_posts
    
    def _to_naive_utc_ts(self, dt_val) -> Optional[pd.Timestamp]:
        """
        Chuẩn hóa bất kỳ giá trị thời gian nào về pd.Timestamp naive-UTC
        Trả về None nếu không parse được.
        """
        if dt_val is None:
            return None

        try:
            # String -> parse về tz-aware UTC rồi drop tz
            if isinstance(dt_val, str):
                ts = pd.to_datetime(dt_val, utc=True, errors="coerce")
                if ts is None or pd.isna(ts):
                    return None
                # tz-aware -> drop tz => naive-UTC
                return ts.tz_convert(None) if hasattr(ts, "tz_convert") else ts.tz_localize(None)

            # pandas.Timestamp
            if isinstance(dt_val, pd.Timestamp):
                if getattr(dt_val, "tzinfo", None) is not None or getattr(dt_val, "tz", None) is not None:
                    # tz-aware -> drop tz
                    return dt_val.tz_convert(None)
                # đã naive
                return dt_val

            # Python datetime
            if isinstance(dt_val, datetime):
                # đưa về pandas.Timestamp rồi xử lý tz giống trên
                ts = pd.Timestamp(dt_val)
                if getattr(ts, "tzinfo", None) is not None or getattr(ts, "tz", None) is not None:
                    return ts.tz_convert(None)
                return ts

            # fallback: cố parse bằng pandas
            ts = pd.to_datetime(dt_val, utc=True, errors="coerce")
            if ts is None or pd.isna(ts):
                return None
            return ts.tz_convert(None) if hasattr(ts, "tz_convert") else ts.tz_localize(None)
        except Exception:
            return None
    
    def _apply_freshness_boost(
        self,
        df: pd.DataFrame,
        post_metadata: Dict, 
        audit: Dict
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
        boosted = 0
        boost_cutoff = now - timedelta(hours=self.freshness_boost_hours)
        
        for idx, row in df.iterrows():
            post_id = int(row['post_id'])
            meta = post_metadata.get(post_id, {})
            created_at = meta.get('created_at')
            
        #     if post_id in post_metadata:
        #         created_at = post_metadata[post_id].get('created_at')
                
        #         if created_at and created_at >= boost_cutoff:
        #             # Apply boost
        #             df.at[idx, 'ml_score'] *= self.freshness_boost_factor

            # if created_at:
            #     if isinstance(created_at, str):
            #         try:
            #             created_at_dt = pd.to_datetime(created_at, utc=True)
            #             created_at_dt = created_at_dt.tz_convert(None) if hasattr(created_at_dt, "tz_convert") else created_at_dt.tz_localize(None)
            #         except Exception:
            #             created_at_dt = None
            #     else:
            #         created_at_dt = created_at

            #     if isinstance(created_at_dt, datetime) and created_at_dt >= boost_cutoff:
            #         df.at[idx, 'ml_score'] = float(row['ml_score']) * float(self.freshness_boost_factor)
            #         boosted += 1

            # Chuẩn hóa created_at về naive-UTC
            created_at_dt = self._to_naive_utc_ts(created_at)
            if created_at_dt is None:
                continue

            # So sánh 2 naive-UTC timestamps => không bị TypeError
            if created_at_dt >= boost_cutoff:
                df.at[idx, 'ml_score'] = float(row['ml_score']) * float(self.freshness_boost_factor)
                boosted += 1

        audit["freshness"]["boosted"] = int(boosted)
        return df
        
    
    def _apply_quality_filter(
        self,
        df: pd.DataFrame,
        post_metadata: Dict, 
        audit: Dict
    ) -> pd.DataFrame:
        """Remove low-quality posts, e.g., Status != quality_status_ok_value"""
        keep_idx = []
        removed_status = 0

        for idx, row in df.iterrows():
            post_id = int(row['post_id'])
            meta = post_metadata.get(post_id, {})
            status = meta.get('status', None)
            if status is None:
                # if there is no status, hold it
                keep_idx.append(idx)
            else:
                try:
                    if int(status) == int(self.quality_status_ok_value):
                        keep_idx.append(idx)
                    else:
                        removed_status += 1
                except Exception:
                    keep_idx.append(idx)

        audit["quality"]["removed_status"] += int(removed_status)
        return df.loc[keep_idx]
    
    # Hàm đúng tạm chưa mở vì content bài post chưa ổn.
    # def _apply_content_dedup(
    #     self,
    #     df: pd.DataFrame,
    #     post_metadata: Dict
    # ) -> pd.DataFrame:
    #     """Remove duplicate content using content_hash or fallback title|content hash"""
    #     seen = set()
    #     keep_idx = []
    #     examples = []

    #     for idx, row in df.iterrows():
    #         post_id = int(row['post_id'])
    #         meta = post_metadata.get(post_id, {})
    #         ch = meta.get('content_hash')

    #         if not ch:
    #             title = (meta.get('title') or '').strip().lower()
    #             body = (meta.get('content') or '').strip().lower()
    #             if title or body:
    #                 ch = hashlib.sha1(f'{title}|{body}'.encode('utf-8')).hexdigest()

    #         if ch:
    #             if ch in seen:
    #                 if len(examples) < 3:
    #                     examples.append({"post_id": post_id, "hash": ch})
    #                 continue
    #             seen.add(ch)

    #         keep_idx.append(idx)

    #     return df.loc[keep_idx], examples

    def _safe_text(self, v) -> str:
        if v is None:
            return ""
        try:
            import pandas as pd
            if pd.isna(v):
                return ""
        except Exception:
            pass
        if not isinstance(v, str):
            return ""
        return v.strip()

    def _apply_content_dedup(
        self,
        df: pd.DataFrame,
        post_metadata: Dict
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Remove duplicate content: nới lỏng để tránh quét sạch dev data.
        Trả về (df_lọc, dedup_examples)
        """
        seen = set()
        keep_idx = []
        dedup_examples: List[Dict] = []

        min_len = 10  # text quá ngắn => không dedup

        for idx, row in df.iterrows():
            post_id = int(row['post_id'])
            meta = post_metadata.get(post_id, {}) or {}
            author_id = meta.get('author_id')

            # safe text
            title = self._safe_text(meta.get('title')).lower()
            body  = self._safe_text(meta.get('content')).lower()

            if len(title) < min_len and len(body) < min_len:
                keep_idx.append(idx)
                continue

            ch = meta.get('content_hash')
            if not ch:
                import hashlib
                ch = hashlib.sha1(f"{title}|{body}".encode("utf-8")).hexdigest()

            key = (author_id, ch)
            if key in seen:
                if len(dedup_examples) < 5:
                    dedup_examples.append({"post_id": post_id, "author_id": author_id})
                continue

            seen.add(key)
            keep_idx.append(idx)

        df2 = df.loc[keep_idx]
        return df2, dedup_examples


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
        removed_consecutive = 0
        removed_cap = 0

        for _, row in df.iterrows():
            if len(final_posts) >= limit:
                break
            
            post_id = int(row['post_id'])
            score = float(row['ml_score'])
            
            # Get author ID (from metadata or estimate)
            if post_metadata and post_id in post_metadata:
                # author_id = post_metadata[post_id].get('author_id', post_id % 1000)
                author_id = post_metadata[post_id].get('author_id')
                if author_id is None:
                    author_id = post_id % 1000  # fallback temp
            else:
                # Fallback: estimate from post_id
                author_id = post_id % 1000
            
            # Check diversity rules
            # Rule 1: Max consecutive from same author
            if author_id == last_author_id:
                consecutive_count += 1
                if consecutive_count >= self.max_consecutive_same_author:
                    removed_consecutive += 1
                    continue  # Skip this post
            else:
                consecutive_count = 1 # 0
                last_author_id = author_id
            
            # Rule 2: Max total from same author
            author_count = author_counts.get(author_id, 0)
            if author_count >= self.max_same_author_in_feed:
                removed_cap += 1
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
        
        return final_posts, {"consecutive": removed_consecutive, "cap": removed_cap}
    
    # --------- small utilities ---------

    def _extract_authors(self, df: pd.DataFrame, post_metadata: Dict) -> List[int]:
        authors = []
        for _, r in df.iterrows():
            pid = int(r["post_id"])
            a = post_metadata.get(pid, {}).get("author_id")
            if a is not None:
                authors.append(a)
        return authors

    def _top_author_share(self, authors: List[int]) -> float:
        if not authors:
            return 0.0
        c = Counter(authors)
        top = c.most_common(1)[0][1]
        return round(top / len(authors), 3)

    # def _emit_audit(self, audit: Dict, sample: Optional[List[Dict]] = None):
    #     if not self.audit_enabled:
    #         return
    #     # Tóm tắt ngắn gọn, đủ “bằng chứng”
    #     logger.info(
    #         "RERANK AUDIT | pre.count=%s pre.authors=%s pre.top_author_share=%.3f | "
    #         "fresh.boosted=%s | quality.removed_status=%s min_score.removed=%s | "
    #         "dedup.removed=%s | diversity.removed_consecutive=%s removed_cap=%s | "
    #         "post.count=%s post.authors=%s post.top_author_share=%.3f",
    #         audit["pre"]["count"], audit["pre"]["unique_authors"], (audit["pre"]["top_author_share"] or 0.0),
    #         audit["freshness"]["boosted"],
    #         audit["quality"]["removed_status"], audit["quality"]["removed_min_score"],
    #         audit["dedup"]["removed"],
    #         audit["diversity"]["removed_consecutive"], audit["diversity"]["removed_cap"],
    #         audit["post"]["count"], audit["post"]["unique_authors"], (audit["post"]["top_author_share"] or 0.0)
    #     )
    #     if sample:
    #         logger.debug("RERANK AUDIT SAMPLE (first %d): %s", len(sample), sample)
    #     if audit["dedup"]["examples"]:
    #         logger.debug("RERANK DEDUP EXAMPLES: %s", audit["dedup"]["examples"])
    def _emit_audit(self, audit: Dict, sample: Optional[List[Dict]] = None):
        """
        Log 'RERANK AUDIT' ở dạng vừa súc tích vừa có 'legend' (giải thích ý nghĩa).
        """

        # --- Tính các giá trị mặc định an toàn ---
        pre_count = audit.get("pre", {}).get("count", 0) or 0
        pre_authors = audit.get("pre", {}).get("unique_authors", 0) or 0
        pre_top_share = audit.get("pre", {}).get("top_author_share", 0.0) or 0.0

        fresh_boosted = audit.get("freshness", {}).get("boosted", 0) or 0

        q_removed_status = audit.get("quality", {}).get("removed_status", 0) or 0
        q_removed_min_score = audit.get("quality", {}).get("removed_min_score", 0) or 0

        dedup_removed = audit.get("dedup", {}).get("removed", 0) or 0

        div_removed_consec = audit.get("diversity", {}).get("removed_consecutive", 0) or 0
        div_removed_cap = audit.get("diversity", {}).get("removed_cap", 0) or 0

        post_count = audit.get("post", {}).get("count", 0) or 0
        post_authors = audit.get("post", {}).get("unique_authors", 0) or 0
        post_top_share = audit.get("post", {}).get("top_author_share", 0.0) or 0.0

        # --- Log súc tích (machine-friendly) ---
        logger.info(
            "RERANK AUDIT | "
            "pre.count=%s pre.authors=%s pre.top_author_share=%.3f | "
            "fresh.boosted=%s | "
            "quality.removed_status=%s min_score.removed=%s | "
            "dedup.removed=%s | "
            "diversity.removed_consecutive=%s removed_cap=%s | "
            "post.count=%s post.authors=%s post.top_author_share=%.3f",
            pre_count, pre_authors, pre_top_share,
            fresh_boosted,
            q_removed_status, q_removed_min_score,
            dedup_removed,
            div_removed_consec, div_removed_cap,
            post_count, post_authors, post_top_share
        )

        # --- Log giải thích (human-friendly legend) ---
        # (INFO để thấy trên production; đổi sang DEBUG nếu bạn muốn gọn hơn)
        logger.info(
            "RERANK LEGEND |\n"
            "  pre.count               : số ứng viên đầu vào reranker (top-K sau ML)\n"
            "  pre.authors             : số tác giả duy nhất trong ứng viên\n"
            "  pre.top_author_share    : tỷ lệ lớn nhất của một tác giả trong ứng viên (càng thấp càng đa dạng)\n"
            "  fresh.boosted           : số bài được cộng điểm do mới (trong %sh)\n"
            "  quality.removed_status  : số bài bị loại vì Status không hợp lệ\n"
            "  min_score.removed       : số bài bị loại vì điểm ML dưới ngưỡng (nếu bật)\n"
            "  dedup.removed           : số bài bị loại vì nội dung trùng lặp\n"
            "  diversity.removed_consecutive : số bài bị loại vì vượt giới hạn bài liên tiếp cùng tác giả (max %s)\n"
            "  diversity.removed_cap   : số bài bị loại vì vượt tổng số bài cho một tác giả trong feed (max %s)\n"
            "  post.count              : số bài cuối cùng sau rerank\n"
            "  post.authors            : số tác giả duy nhất sau rerank\n"
            "  post.top_author_share   : tỷ lệ lớn nhất của một tác giả sau rerank (càng thấp càng đa dạng)",
            self.freshness_boost_hours,
            self.max_consecutive_same_author,
            self.max_same_author_in_feed
        )


        # --- Sample / ví dụ ---
        if sample:
            logger.debug("RERANK AUDIT SAMPLE (first %d): %s", len(sample), sample)

        # --- Ví dụ các bài bị dedup (nếu có) ---
        if audit.get("dedup", {}).get("examples"):
            logger.debug("RERANK DEDUP EXAMPLES: %s", audit["dedup"]["examples"])
            
    def get_config(self) -> Dict:
        """Get current configuration"""
        return {
            'max_consecutive_same_author': self.max_consecutive_same_author,
            'max_same_author_in_feed': self.max_same_author_in_feed,
            'freshness_enabled': self.freshness_enabled,
            'freshness_boost_hours': self.freshness_boost_hours,
            'freshness_boost_factor': self.freshness_boost_factor,
            'quality_enabled': self.quality_enabled,
            'use_min_score': self.use_min_score,
            'min_score': self.min_score,
            'quality_status_ok_value': self.quality_status_ok_value,
            'content_dedup_enabled': self.content_dedup_enabled
        }