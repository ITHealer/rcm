

# """
# Post Embeddings + FAISS (MySQL)
# ===============================

# - Đọc Post/Hashtag/Reaction/View từ MySQL (schema như ảnh bạn cung cấp).
# - Sinh embedding bằng SentenceTransformers (GPU nếu có).
# - Lưu vào bảng MySQL `post_embeddings` (LONGBLOB).
# - Build FAISS index (cosine qua IndexFlatIP + normalize).

# Usage:
#   python -m recommender.offline.post_embeddings --mode full
#   python -m recommender.offline.post_embeddings --mode incremental --since-days 7
# """

# from __future__ import annotations

# import os
# import sys
# import math
# import json
# import time
# import pickle
# import logging
# from pathlib import Path
# from typing import List, Dict, Tuple, Optional

# import numpy as np
# import pandas as pd

# # Project root
# ROOT = Path(__file__).resolve().parents[2]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))

# # 3rd libs
# from sqlalchemy import create_engine, text
# from sqlalchemy.engine import Engine
# from sqlalchemy.exc import ProgrammingError
# from sqlalchemy.pool import NullPool

# try:
#     import faiss
#     FAISS_OK = True
# except Exception:
#     FAISS_OK = False

# try:
#     import torch
#     from sentence_transformers import SentenceTransformer
#     ST_OK = True
# except Exception:
#     ST_OK = False

# logging.basicConfig(
#     level=os.getenv("LOG_LEVEL", "INFO"),
#     format="[%(asctime)s] %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# logger = logging.getLogger("post_embeddings")

# MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
# MODELS_DIR.mkdir(parents=True, exist_ok=True)


# # --------------------------------------------------------------------------------------
# # DB helpers
# # --------------------------------------------------------------------------------------
# def make_engine_from_env() -> Engine:
#     url = "mysql+pymysql://way_root:YmhNWpppahN92AtJotFDoHnCoW38keDp@14.225.220.56:15479/wayjet_system"
#     if not url:
#         raise RuntimeError("Missing BACKEND_DB_URL (e.g. mysql+pymysql://user:pass@host:3306/dbname)")
#     # dùng NullPool để script batch không giữ connection lâu
#     engine = create_engine(url, poolclass=NullPool, future=True)
#     with engine.connect() as conn:
#         conn.execute(text("SELECT 1"))
#     return engine


# def _is_mysql(engine: Engine) -> bool:
#     try:
#         name = engine.dialect.name.lower()
#         return "mysql" in name
#     except Exception:
#         return False


# # def ensure_post_embeddings_table(engine: Engine):
# #     """
# #     Tạo bảng lưu embeddings cho MySQL. Không dùng cú pháp Postgres.
# #     """
# #     ddl_mysql = """
# #     CREATE TABLE IF NOT EXISTS post_embeddings (
# #       PostId BIGINT PRIMARY KEY,
# #       Embedding LONGBLOB NOT NULL,
# #       UpdatedAt TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
# #     ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
# #     """
# #     with engine.begin() as conn:
# #         if _is_mysql(engine):
# #             conn.exec_driver_sql(ddl_mysql)
# #         else:
# #             # Fallback generic (SQLite, v.v.) – không dùng ON UPDATE
# #             ddl_generic = """
# #             CREATE TABLE IF NOT EXISTS post_embeddings (
# #               PostId INTEGER PRIMARY KEY,
# #               Embedding BLOB NOT NULL,
# #               UpdatedAt TEXT NOT NULL
# #             );
# #             """
# #             conn.exec_driver_sql(ddl_generic)


# # --------------------------------------------------------------------------------------
# # Fetch data from your schema
# # --------------------------------------------------------------------------------------
# def fetch_posts(engine: Engine, mode: str = "incremental", since_days: int = 7) -> pd.DataFrame:
#     """
#     Đọc Post + hashtags (nếu có) theo schema của bạn:
#       - Post(Id, UserId, Content, Status, CreateDate, ...)
#       - PostHashtag(PostId, HashtagId, Status, ...)
#       - (Optional) Hashtag(Id, Name) — nếu không có, dùng HashtagId làm tag.
#     Chỉ lấy Status hợp lệ (10) và Content không rỗng.
#     """
#     where_time = ""
#     params = {}
#     if mode in ("incremental",) and since_days is not None:
#         where_time = "AND p.CreateDate >= DATE_SUB(NOW(), INTERVAL :days DAY)"
#         params["days"] = int(since_days)

#     # Kiểm tra có bảng Hashtag(Name) hay không
#     has_hashtag_name = False
#     with engine.connect() as conn:
#         try:
#             conn.execute(text("SELECT 1 FROM Hashtag LIMIT 1"))
#             # thử cột Name
#             conn.execute(text("SELECT Name FROM Hashtag LIMIT 1"))
#             has_hashtag_name = True
#         except Exception:
#             has_hashtag_name = False

#     # Query MySQL: group_concat hashtags
#     if has_hashtag_name:
#         tag_expr = "GROUP_CONCAT(DISTINCT CONCAT('#', h.Name) SEPARATOR ' ')"
#         join_tag = """
#             LEFT JOIN PostHashtag ph ON ph.PostId = p.Id AND (ph.Status IS NULL OR ph.Status = 10)
#             LEFT JOIN Hashtag h ON h.Id = ph.HashtagId
#         """
#     else:
#         tag_expr = "GROUP_CONCAT(DISTINCT CONCAT('#', ph.HashtagId) SEPARATOR ' ')"
#         join_tag = "LEFT JOIN PostHashtag ph ON ph.PostId = p.Id AND (ph.Status IS NULL OR ph.Status = 10)"

#     sql = f"""
#     SELECT
#       p.Id           AS PostId,
#       p.UserId       AS AuthorId,
#       p.Content      AS Content,
#       p.Status       AS Status,
#       p.CreateDate   AS CreateDate,
#       {tag_expr}     AS Hashtags
#     FROM Post p
#     {join_tag}
#     WHERE
#       (p.Status IS NULL OR p.Status = 10)
#       AND p.Content IS NOT NULL AND LENGTH(TRIM(p.Content)) > 0
#       {where_time}
#     GROUP BY p.Id, p.UserId, p.Content, p.Status, p.CreateDate
#     ORDER BY p.CreateDate DESC
#     """

#     with engine.connect() as conn:
#         try:
#             rows = conn.execute(text(sql), params).mappings().all()
#         except ProgrammingError as e:
#             logger.error("SQL error when reading posts: %s", e)
#             raise

#     df = pd.DataFrame(rows)
#     if df.empty:
#         logger.warning("No posts returned from DB.")
#         return df

#     # Chuẩn hóa cột/kiểu
#     df["CreateDate"] = pd.to_datetime(df["CreateDate"], errors="coerce", utc=True)
#     df["Hashtags"] = df["Hashtags"].fillna("")

#     # Loại những bài quá cũ nếu chạy full mà DB quá lớn? (giữ nguyên cho bạn chủ động)
#     logger.info("Fetched %s posts from DB.", len(df))
#     return df


# # --------------------------------------------------------------------------------------
# # Embedding runner
# # --------------------------------------------------------------------------------------
# class MySQLEmbedder:
#     def __init__(
#         self,
#         engine: Engine,
#         model_name: str = None,
#         batch_size: int = 512,
#     ):
#         if not ST_OK:
#             raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")

#         self.engine = engine
#         self.batch = int(batch_size)
#         self.model_name = model_name or os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         logger.info("Init embedder | model=%s | device=%s", self.model_name, device)
#         self.model = SentenceTransformer(self.model_name, device=device)
#         self.dim = self.model.get_sentence_embedding_dimension()

#         # ensure_post_embeddings_table(self.engine)

#     # --------------- Text prep ----------------
#     @staticmethod
#     def _prep_text(content: str, hashtags: str) -> str:
#         content = (content or "").strip()
#         if hashtags and hashtags.strip():
#             return f"{content} [HASHTAGS: {hashtags.strip()}]"
#         return content or "empty post"

#     # --------------- Already embedded? -------
#     def _get_existing_ids(self) -> set[int]:
#         sql = "SELECT PostId FROM post_embeddings"
#         with self.engine.connect() as conn:
#             try:
#                 rows = conn.execute(text(sql)).fetchall()
#             except Exception:
#                 return set()
#         return set(int(r[0]) for r in rows)

#     # --------------- Save embeddings ---------
#     # def _save_embeddings(self, post_ids: List[int], embs: np.ndarray):
#     #     rows = []
#     #     now = time.strftime("%Y-%m-%d %H:%M:%S")
#     #     # store as float32 bytes
#     #     for pid, vec in zip(post_ids, embs):
#     #         rows.append({"PostId": int(pid), "Embedding": vec.astype(np.float32).tobytes(), "UpdatedAt": now})

#     #     if _is_mysql(self.engine):
#     #         # MySQL upsert
#     #         sql = text("""
#     #             INSERT INTO post_embeddings (PostId, Embedding, UpdatedAt)
#     #             VALUES (:PostId, :Embedding, :UpdatedAt)
#     #             ON DUPLICATE KEY UPDATE
#     #                 Embedding = VALUES(Embedding),
#     #                 UpdatedAt = VALUES(UpdatedAt)
#     #         """)
#     #     else:
#     #         # Generic replace (delete+insert)
#     #         sql_del = text("DELETE FROM post_embeddings WHERE PostId = :pid")
#     #         sql_ins = text("INSERT INTO post_embeddings (PostId, Embedding, UpdatedAt) VALUES (:PostId, :Embedding, :UpdatedAt)")
#     #         with self.engine.begin() as conn:
#     #             for r in rows:
#     #                 conn.execute(sql_del, {"pid": r["PostId"]})
#     #                 conn.execute(sql_ins, r)
#     #         logger.info("Saved %d embeddings (generic).", len(rows))
#     #         return

#     #     # batch insert
#     #     with self.engine.begin() as conn:
#     #         for i in range(0, len(rows), 1000):
#     #             conn.execute(sql, rows[i:i+1000])
#     #     logger.info("Saved %d embeddings to MySQL.", len(rows))

#     # --------------- Build FAISS ------------
#     def build_faiss(self):
#         if not FAISS_OK:
#             logger.warning("faiss not available. Skip index build.")
#             return
#         sql = "SELECT PostId, Embedding FROM post_embeddings"
#         with self.engine.connect() as conn:
#             rows = conn.execute(text(sql)).fetchall()

#         if not rows:
#             logger.warning("No vectors to build index.")
#             return

#         post_ids: List[int] = []
#         vecs: List[np.ndarray] = []
#         for pid, blob in rows:
#             v = np.frombuffer(blob, dtype=np.float32)
#             post_ids.append(int(pid))
#             vecs.append(v)

#         X = np.vstack(vecs)
#         # normalize for cosine
#         X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
#         index = faiss.IndexFlatIP(X.shape[1])
#         index.add(X.astype(np.float32))

#         faiss.write_index(index, str(MODELS_DIR / "faiss_index.bin"))
#         with open(MODELS_DIR / "faiss_post_ids.pkl", "wb") as f:
#             pickle.dump(post_ids, f)

#         logger.info("FAISS built: %s vectors -> %s", index.ntotal, MODELS_DIR / "faiss_index.bin")

#     # --------------- Run --------------------
#     def run(self, mode: str = "incremental", since_days: int = 7, save: bool = True):
#         """
#         mode = full | incremental
#         """
#         posts = fetch_posts(self.engine, mode=mode, since_days=since_days)
#         if posts.empty:
#             logger.info("No posts to process.")
#             return

#         existing = self._get_existing_ids()
#         # nếu full → chỉ bỏ qua id đã có; incremental → đã filter theo thời gian rồi, vẫn skip id đã có.
#         posts = posts[~posts["PostId"].astype(int).isin(existing)].reset_index(drop=True)
#         if posts.empty:
#             logger.info("All posts already embedded (nothing new).")
#             self.build_faiss()
#             return

#         logger.info("To-embed: %d", len(posts))

#         texts: List[str] = []
#         ids: List[int] = []
#         for _, r in posts.iterrows():
#             texts.append(self._prep_text(str(r["Content"]), str(r.get("Hashtags", ""))))
#             ids.append(int(r["PostId"]))

#         # batch encode
#         all_vecs: List[np.ndarray] = []
#         total = len(texts)
#         bs = self.batch
#         n_batches = math.ceil(total / bs)
#         for i in range(n_batches):
#             s, e = i * bs, min((i + 1) * bs, total)
#             t0 = time.time()
#             vec = self.model.encode(
#                 texts[s:e],
#                 batch_size=bs,
#                 show_progress_bar=False,
#                 convert_to_numpy=True
#             )
#             all_vecs.append(vec)
#             logger.info("Encoded batch %d/%d (%d items) in %.1fs", i+1, n_batches, e - s, time.time()-t0)

#         X = np.vstack(all_vecs)
#         # if save:
#         #     self._save_embeddings(ids, X)
#         self.build_faiss()
#         logger.info("Done. Embedded %d posts.", len(ids))


# # --------------------------------------------------------------------------------------
# # CLI
# # --------------------------------------------------------------------------------------
# def _parse_args():
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument("--mode", choices=["full", "incremental"], default="incremental")
#     p.add_argument("--since-days", type=int, default=7, help="Only used for incremental")
#     p.add_argument("--batch-size", type=int, default=int(os.getenv("EMB_BATCH", "512")))
#     p.add_argument("--model", type=str, default=os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
#     return p.parse_args()


# def main():
#     args = _parse_args()
#     engine = make_engine_from_env()
#     emb = MySQLEmbedder(engine=engine, model_name=args.model, batch_size=args.batch_size)
#     emb.run(mode=args.mode, since_days=args.since_days, save=True)


# if __name__ == "__main__":
#     main()


"""
Post Embeddings + FAISS (No MySQL Storage)
==========================================

Generate embeddings and build FAISS index WITHOUT storing embeddings in MySQL.
Embeddings are computed on-the-fly and only FAISS index is saved to disk.

Usage:
  python -m recommender.offline.post_embeddings --mode full
  python -m recommender.offline.post_embeddings --mode full --version v_20251027_144158
"""

from __future__ import annotations

import os
import sys
import math
import json
import time
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# SQLAlchemy
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.pool import NullPool

# Optional imports
try:
    import faiss
    FAISS_OK = True
except Exception:
    FAISS_OK = False

try:
    import torch
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except Exception:
    ST_OK = False

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("post_embeddings")

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Artifact Manager Integration
# --------------------------------------------------------------------------------------
def write_latest_pointer(base_dir: str, version_name: str):
    """Update latest pointer (Windows-safe)"""
    base_path = Path(base_dir)
    latest_file = base_path / "latest.version"
    
    with open(latest_file, 'w') as f:
        f.write(version_name)
    
    logger.info(f"Updated latest pointer: {base_dir}/latest → {version_name}")


# --------------------------------------------------------------------------------------
# DB helpers
# --------------------------------------------------------------------------------------
def make_engine_from_env() -> Engine:
    """Create database engine from environment variable"""
    url = os.getenv("BACKEND_DB_URL")
    if not url:
        url = "mysql+pymysql://way_root:YmhNWpppahN92AtJotFDoHnCoW38keDp@14.225.220.56:15479/wayjet_system"
        logger.warning("Using hardcoded DB URL. Set BACKEND_DB_URL environment variable!")
    
    engine = create_engine(url, poolclass=NullPool, future=True)
    
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    
    logger.info(f"✅ Connected to database: {engine.url.database}")
    
    return engine


def _is_mysql(engine: Engine) -> bool:
    """Check if engine is MySQL"""
    try:
        return "mysql" in engine.dialect.name.lower()
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# Fetch data from your schema
# --------------------------------------------------------------------------------------
def fetch_posts(
    engine: Engine,
    mode: str = "full",
    since_days: int = 7,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch posts from database
    
    Args:
        engine: Database engine
        mode: 'full' or 'incremental'
        since_days: For incremental mode, days to look back
        limit: Optional limit (for testing)
    
    Returns:
        DataFrame with columns: PostId, AuthorId, Content, Status, CreateDate, Hashtags
    """
    where_time = ""
    params = {}
    
    if mode == "incremental" and since_days is not None:
        where_time = "AND p.CreateDate >= DATE_SUB(NOW(), INTERVAL :days DAY)"
        params["days"] = int(since_days)
    
    # Check if Hashtag table has Name column
    has_hashtag_name = False
    with engine.connect() as conn:
        try:
            conn.execute(text("SELECT Name FROM Hashtag LIMIT 1"))
            has_hashtag_name = True
        except Exception:
            has_hashtag_name = False
    
    # Build query
    if has_hashtag_name:
        tag_expr = "GROUP_CONCAT(DISTINCT CONCAT('#', h.Name) SEPARATOR ' ')"
        join_tag = """
            LEFT JOIN PostHashtag ph ON ph.PostId = p.Id AND (ph.Status IS NULL OR ph.Status = 10)
            LEFT JOIN Hashtag h ON h.Id = ph.HashtagId
        """
    else:
        tag_expr = "GROUP_CONCAT(DISTINCT CONCAT('#', ph.HashtagId) SEPARATOR ' ')"
        join_tag = """
            LEFT JOIN PostHashtag ph ON ph.PostId = p.Id AND (ph.Status IS NULL OR ph.Status = 10)
        """
    
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    
    sql = f"""
    SELECT
      p.Id           AS PostId,
      p.UserId       AS AuthorId,
      p.Content      AS Content,
      p.Status       AS Status,
      p.CreateDate   AS CreateDate,
      {tag_expr}     AS Hashtags
    FROM Post p
    {join_tag}
    WHERE
      (p.Status IS NULL OR p.Status = 10)
      AND p.Content IS NOT NULL 
      AND LENGTH(TRIM(p.Content)) > 0
      {where_time}
    GROUP BY p.Id, p.UserId, p.Content, p.Status, p.CreateDate
    ORDER BY p.CreateDate DESC
    {limit_clause}
    """
    
    with engine.connect() as conn:
        try:
            rows = conn.execute(text(sql), params).mappings().all()
        except ProgrammingError as e:
            logger.error(f"SQL error: {e}")
            raise
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        logger.warning("No posts returned from database")
        return df
    
    df["CreateDate"] = pd.to_datetime(df["CreateDate"], errors="coerce", utc=True)
    df["Hashtags"] = df["Hashtags"].fillna("")
    
    logger.info(f"Fetched {len(df):,} posts from database")
    
    return df


# --------------------------------------------------------------------------------------
# Embedding generator
# --------------------------------------------------------------------------------------
class MySQLEmbedder:
    """
    Generate post embeddings and build FAISS index
    
    Features:
    - GPU acceleration (if available)
    - Batch processing
    - In-memory embeddings (no MySQL storage)
    - FAISS index building
    """
    
    def __init__(
        self,
        engine: Engine,
        model_name: str = None,
        batch_size: int = 512,
    ):
        if not ST_OK:
            raise RuntimeError("sentence-transformers required: pip install sentence-transformers")
        
        self.engine = engine
        self.batch = int(batch_size)
        self.model_name = model_name or os.getenv(
            "EMB_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing embedder | model={self.model_name} | device={device}")
        
        self.model = SentenceTransformer(self.model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded | dimension={self.dim}")
    
    @staticmethod
    def _prep_text(content: str, hashtags: str) -> str:
        """Prepare text for embedding"""
        content = (content or "").strip()
        
        if hashtags and hashtags.strip():
            return f"{content} [HASHTAGS: {hashtags.strip()}]"
        
        return content or "empty post"
    
    def build_faiss(
        self,
        post_ids: List[int],
        embeddings: np.ndarray,
        version_name: str = None
    ) -> str:
        """
        Build FAISS index from in-memory embeddings
        
        Args:
            post_ids: List of post IDs
            embeddings: numpy array (N, D) of embeddings
            version_name: Version name (e.g. 'v_20251027_144158')
        
        Returns:
            Version name used
        """
        if not FAISS_OK:
            logger.warning("⚠️  FAISS not available. Skipping index build.")
            return None
        
        logger.info("\n" + "="*70)
        logger.info("BUILDING FAISS INDEX")
        logger.info("="*70)
        
        # Determine output directory
        if version_name:
            output_dir = MODELS_DIR / version_name
        else:
            version_name = f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir = MODELS_DIR / version_name
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Posts: {len(post_ids):,}")
        logger.info(f"Shape: {embeddings.shape}")
        logger.info(f"Dimension: {embeddings.shape[1]}")
        
        # Normalize for cosine similarity
        logger.info("Normalizing vectors...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        X_normalized = embeddings / (norms + 1e-8)
        
        avg_norm = np.linalg.norm(X_normalized, axis=1).mean()
        logger.info(f"Average norm: {avg_norm:.6f} (should be ~1.0)")
        
        # Create FAISS index
        logger.info(f"Creating FAISS IndexFlatIP (dimension={embeddings.shape[1]})...")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(X_normalized.astype(np.float32))
        
        logger.info(f"✅ Index built with {index.ntotal:,} vectors")
        
        # Save index files
        index_path = output_dir / "faiss_index.bin"
        ids_path = output_dir / "faiss_post_ids.pkl"
        
        faiss.write_index(index, str(index_path))
        logger.info(f"Saved FAISS index: {index_path}")
        
        with open(ids_path, "wb") as f:
            pickle.dump(post_ids, f)
        logger.info(f"Saved post IDs mapping: {ids_path}")
        
        # Save metadata
        metadata = {
            "version": version_name,
            "created_at": datetime.now().isoformat(),
            "n_vectors": int(index.ntotal),
            "embedding_dim": int(embeddings.shape[1]),
            "model_name": self.model_name,
            "index_type": "IndexFlatIP",
            "normalized": True,
            "storage": "faiss_only",
            "component": "faiss_embeddings"
        }
        
        metadata_path = output_dir / "faiss_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata: {metadata_path}")
        
        # Update latest pointer
        write_latest_pointer(str(MODELS_DIR), version_name)
        
        logger.info("\n" + "="*70)
        logger.info("✅ FAISS INDEX BUILD COMPLETE")
        logger.info("="*70)
        logger.info(f"Version: {version_name}")
        logger.info(f"Vectors: {index.ntotal:,}")
        logger.info(f"Output: {output_dir}/")
        
        return version_name
    
    def run(
        self,
        mode: str = "full",
        since_days: int = 7,
        version_name: str = None
    ) -> Optional[str]:
        """
        Run embedding generation and FAISS building
        
        Args:
            mode: 'full' or 'incremental'
            since_days: For incremental mode
            version_name: Optional version name (to match ranking model)
        
        Returns:
            Version name used
        """
        logger.info("\n" + "="*70)
        logger.info("POST EMBEDDINGS + FAISS PIPELINE")
        logger.info("="*70)
        logger.info(f"Mode: {mode}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Batch size: {self.batch}")
        logger.info(f"Storage: FAISS only (no MySQL)")
        
        # Fetch posts
        posts = fetch_posts(self.engine, mode=mode, since_days=since_days)
        
        if posts.empty:
            logger.warning("No posts to process.")
            return None
        
        logger.info(f"To embed: {len(posts):,} posts")
        
        # Prepare texts
        texts: List[str] = []
        ids: List[int] = []
        
        for _, row in posts.iterrows():
            texts.append(self._prep_text(str(row["Content"]), str(row.get("Hashtags", ""))))
            ids.append(int(row["PostId"]))
        
        # Batch encode
        logger.info("\n" + "="*70)
        logger.info("ENCODING POSTS")
        logger.info("="*70)
        
        all_vecs: List[np.ndarray] = []
        total = len(texts)
        bs = self.batch
        n_batches = math.ceil(total / bs)
        
        logger.info(f"Encoding {total:,} posts in {n_batches} batches...")
        
        for i in range(n_batches):
            s, e = i * bs, min((i + 1) * bs, total)
            
            t0 = time.time()
            vec = self.model.encode(
                texts[s:e],
                batch_size=bs,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_vecs.append(vec)
            
            elapsed = time.time() - t0
            logger.info(f"  Batch {i+1}/{n_batches}: {e-s} posts in {elapsed:.1f}s")
        
        X = np.vstack(all_vecs)
        
        logger.info(f"✅ Encoded {len(ids):,} posts")
        logger.info(f"Shape: {X.shape}")
        
        # Build FAISS from in-memory embeddings
        final_version = self.build_faiss(
            post_ids=ids,
            embeddings=X,
            version_name=version_name
        )
        
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("="*70)
        logger.info(f"Processed: {len(ids):,} posts")
        logger.info(f"Version: {final_version}")
        
        return final_version


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _parse_args():
    import argparse
    
    p = argparse.ArgumentParser(
        description="Generate post embeddings and build FAISS index (no MySQL storage)"
    )
    
    p.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help="Full (all posts) or incremental (recent posts only)"
    )
    
    p.add_argument(
        "--since-days",
        type=int,
        default=7,
        help="For incremental mode: days to look back (default: 7)"
    )
    
    p.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("EMB_BATCH", "512")),
        help="Batch size for encoding (default: 512)"
    )
    
    p.add_argument(
        "--model",
        type=str,
        default=os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        help="Sentence-BERT model name"
    )
    
    p.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version name (to match ranking model version)"
    )
    
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of posts (for testing)"
    )
    
    return p.parse_args()


def main():
    """Main entry point"""
    args = _parse_args()
    
    logger.info("="*70)
    logger.info("POST EMBEDDINGS + FAISS BUILDER")
    logger.info("="*70)
    logger.info(f"Started: {datetime.now()}")
    
    try:
        engine = make_engine_from_env()
        
        embedder = MySQLEmbedder(
            engine=engine,
            model_name=args.model,
            batch_size=args.batch_size
        )
        
        version = embedder.run(
            mode=args.mode,
            since_days=args.since_days,
            version_name=args.version
        )
        
        logger.info("="*70)
        logger.info("✅ SUCCESS")
        logger.info("="*70)
        logger.info(f"Version: {version}")
        logger.info(f"Completed: {datetime.now()}")
        
    except Exception as e:
        logger.error("="*70)
        logger.error("❌ FAILED")
        logger.error("="*70)
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()