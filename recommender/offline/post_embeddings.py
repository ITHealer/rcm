"""
POST EMBEDDINGS BATCH PROCESSOR
================================
Generate embeddings for new posts and maintain Faiss index

Features:
- Batch processing (512 posts/batch)
- GPU support (CUDA if available)
- Incremental updates (only new posts)
- Checkpoint recovery
- Faiss index maintenance

Run: python scripts/offline/post_embeddings.py
Schedule: Cron job every 6 hours (00:00, 06:00, 12:00, 18:00)
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Try imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available. Install: pip install sentence-transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  faiss not available. Install: pip install faiss-cpu (or faiss-gpu)")

try:
    import psycopg2
    from psycopg2.extras import execute_batch
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("⚠️  psycopg2 not available. Install: pip install psycopg2-binary")


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class BatchPostEmbedder:
    """
    Batch post embedding generator
    
    Features:
    - Incremental embedding (only new posts)
    - GPU support
    - Checkpointing
    - Faiss index maintenance
    """
    
    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        batch_size: int = 512,
        device: str = 'auto',
        checkpoint_dir: str = 'checkpoints',
        models_dir: str = 'models'
    ):
        """
        Initialize embedder
        
        Args:
            model_name: SentenceTransformer model name
            batch_size: Batch size for encoding
            device: 'auto', 'cuda', or 'cpu'
            checkpoint_dir: Directory for checkpoints
            models_dir: Directory for models and Faiss index
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available")
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.models_dir = Path(models_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing BatchPostEmbedder...")
        logger.info(f"Model: {model_name}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Device: {self.device}")
        
        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
        # Database connection (will be set by set_db_connection)
        self.db_connection = None
        
        # CSV mode (fallback when no database)
        self.use_csv = False
        self.csv_dir = Path('dataset')
    
    def set_db_connection(self, db_connection):
        """Set PostgreSQL database connection"""
        self.db_connection = db_connection
        self.use_csv = False
        logger.info("Database connection set")
    
    def enable_csv_mode(self, csv_dir: str = 'dataset'):
        """Enable CSV mode (for development without database)"""
        self.use_csv = True
        self.csv_dir = Path(csv_dir)
        logger.info(f"CSV mode enabled. Data directory: {csv_dir}")
    
    # ========================================================================
    # TEXT PREPARATION
    # ========================================================================
    
    def prepare_text(self, post: Dict) -> str:
        """
        Prepare post text for embedding
        
        Format: "{content} [HASHTAGS: {hashtags}]"
        
        Args:
            post: Dict with 'Content' and optionally 'Hashtags'
        
        Returns:
            Prepared text string
        """
        # Get content
        content = str(post.get('Content', '')).strip()
        
        # Get hashtags
        hashtags = post.get('Hashtags', '')
        
        if hashtags and isinstance(hashtags, str) and hashtags.strip():
            # Parse hashtags (comma-separated or space-separated)
            tags = [tag.strip() for tag in hashtags.replace(',', ' ').split() if tag.strip()]
            if tags:
                hashtag_text = ' '.join([f'#{tag}' if not tag.startswith('#') else tag for tag in tags])
                content = f"{content} [HASHTAGS: {hashtag_text}]"
        
        return content if content else "empty post"
    
    # ========================================================================
    # FIND NEW POSTS
    # ========================================================================
    
    def find_new_posts(self, since_hours: int = 6) -> pd.DataFrame:
        """
        Find posts without embeddings
        
        Args:
            since_hours: Only posts created in last X hours
        
        Returns:
            DataFrame with post_id, Content, Hashtags, CreateDate
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"FINDING NEW POSTS (last {since_hours} hours)")
        logger.info(f"{'='*70}")
        
        if self.use_csv:
            return self._find_new_posts_csv(since_hours)
        else:
            return self._find_new_posts_db(since_hours)
    
    def _find_new_posts_csv(self, since_hours: int) -> pd.DataFrame:
        """Find new posts from CSV"""
        # Load posts
        posts_df = pd.read_csv(self.csv_dir / 'Post.csv')
        
        # Parse dates
        posts_df['CreateDate'] = pd.to_datetime(posts_df['CreateDate'])
        
        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=since_hours)
        new_posts = posts_df[posts_df['CreateDate'] >= cutoff_time].copy()
        
        # Check if embeddings exist
        embeddings_file = self.models_dir / 'post_embeddings.pkl'
        
        if embeddings_file.exists():
            with open(embeddings_file, 'rb') as f:
                existing_embeddings = pickle.load(f)
            
            # Filter out posts that already have embeddings
            existing_post_ids = set(existing_embeddings.keys())
            new_posts = new_posts[~new_posts['Id'].isin(existing_post_ids)]
        
        logger.info(f"Found {len(new_posts):,} new posts without embeddings")
        
        return new_posts
    
    def _find_new_posts_db(self, since_hours: int) -> pd.DataFrame:
        """Find new posts from database"""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 not available for database mode")
        
        if self.db_connection is None:
            raise ValueError("Database connection not set. Call set_db_connection() first")
        
        # Query for new posts without embeddings
        query = """
        SELECT 
            p.post_id,
            p.content,
            p.hashtags,
            p.created_at
        FROM posts p
        LEFT JOIN post_embeddings e ON p.post_id = e.post_id
        WHERE 
            e.post_id IS NULL
            AND p.created_at >= NOW() - INTERVAL '%s hours'
        ORDER BY p.created_at DESC
        """
        
        cursor = self.db_connection.cursor()
        cursor.execute(query, (since_hours,))
        
        rows = cursor.fetchall()
        
        df = pd.DataFrame(rows, columns=['post_id', 'Content', 'Hashtags', 'CreateDate'])
        
        logger.info(f"Found {len(df):,} new posts without embeddings")
        
        return df
    
    # ========================================================================
    # BATCH EMBEDDING
    # ========================================================================
    
    def embed_new_posts(
        self,
        since_hours: int = 6,
        save_checkpoint_every: int = 5
    ) -> int:
        """
        Embed posts created in last X hours
        
        Args:
            since_hours: Only posts created in last X hours
            save_checkpoint_every: Save checkpoint every N batches
        
        Returns:
            Number of posts embedded
        """
        start_time = datetime.now()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING BATCH EMBEDDING")
        logger.info(f"{'='*70}")
        logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Find new posts
        new_posts = self.find_new_posts(since_hours)
        
        if len(new_posts) == 0:
            logger.info("No new posts to embed. Exiting.")
            return 0
        
        # Prepare texts
        logger.info("\n📝 Preparing texts...")
        texts = []
        post_ids = []
        
        for idx, row in new_posts.iterrows():
            post_dict = row.to_dict()
            text = self.prepare_text(post_dict)
            texts.append(text)
            
            # Handle both column name formats
            post_id = row.get('Id', row.get('post_id'))
            post_ids.append(post_id)
        
        logger.info(f"Prepared {len(texts):,} texts for embedding")
        
        # Batch encode
        logger.info("\n🔧 Batch encoding...")
        
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        all_embeddings = []
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, len(texts))
            
            batch_texts = texts[batch_start:batch_end]
            
            # Encode
            batch_start_time = datetime.now()
            
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            batch_time = (datetime.now() - batch_start_time).total_seconds()
            
            all_embeddings.append(batch_embeddings)
            
            logger.info(
                f"Batch {batch_idx+1}/{n_batches}: "
                f"Encoded {len(batch_texts)} posts ({batch_time:.1f}s)"
            )
            
            # Save checkpoint
            if (batch_idx + 1) % save_checkpoint_every == 0:
                self._save_checkpoint(
                    post_ids[:batch_end],
                    np.vstack(all_embeddings)
                )
        
        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)
        
        logger.info(f"\n✅ Encoding complete!")
        logger.info(f"   Shape: {embeddings.shape}")
        logger.info(f"   Dtype: {embeddings.dtype}")
        
        # Save embeddings
        logger.info("\n💾 Saving embeddings...")
        self.save_embeddings(post_ids, embeddings)
        
        # Rebuild Faiss index
        if FAISS_AVAILABLE:
            logger.info("\n🔧 Rebuilding Faiss index...")
            self.rebuild_faiss_index()
        else:
            logger.warning("⚠️  Faiss not available. Skipping index rebuild.")
        
        # Summary
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ BATCH EMBEDDING COMPLETE!")
        logger.info(f"{'='*70}")
        logger.info(f"Posts embedded: {len(post_ids):,}")
        logger.info(f"Total time: {total_time/60:.1f}m {total_time%60:.0f}s")
        logger.info(f"Throughput: {len(post_ids) / total_time * 60:.0f} posts/minute")
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return len(post_ids)
    
    # ========================================================================
    # SAVE EMBEDDINGS
    # ========================================================================
    
    def save_embeddings(
        self,
        post_ids: List[int],
        embeddings: np.ndarray
    ):
        """
        Save embeddings to storage
        
        Args:
            post_ids: List of post IDs
            embeddings: Numpy array of embeddings (N, embedding_dim)
        """
        if self.use_csv:
            self._save_embeddings_file(post_ids, embeddings)
        else:
            self._save_embeddings_db(post_ids, embeddings)
    
    def _save_embeddings_file(
        self,
        post_ids: List[int],
        embeddings: np.ndarray
    ):
        """Save embeddings to pickle file"""
        embeddings_file = self.models_dir / 'post_embeddings.pkl'
        
        # Load existing embeddings
        if embeddings_file.exists():
            with open(embeddings_file, 'rb') as f:
                existing_embeddings = pickle.load(f)
        else:
            existing_embeddings = {}
        
        # Add new embeddings
        for post_id, embedding in zip(post_ids, embeddings):
            existing_embeddings[int(post_id)] = embedding
        
        # Save
        with open(embeddings_file, 'wb') as f:
            pickle.dump(existing_embeddings, f)
        
        logger.info(f"Saved {len(post_ids):,} embeddings to {embeddings_file}")
        logger.info(f"Total embeddings in storage: {len(existing_embeddings):,}")
    
    def _save_embeddings_db(
        self,
        post_ids: List[int],
        embeddings: np.ndarray
    ):
        """Save embeddings to PostgreSQL"""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 not available for database mode")
        
        cursor = self.db_connection.cursor()
        
        # Prepare data
        data = []
        for post_id, embedding in zip(post_ids, embeddings):
            # Convert to bytes
            embedding_bytes = embedding.astype(np.float32).tobytes()
            data.append((post_id, embedding_bytes, datetime.now()))
        
        # Batch insert with ON CONFLICT
        insert_query = """
        INSERT INTO post_embeddings (post_id, embedding, created_at)
        VALUES (%s, %s, %s)
        ON CONFLICT (post_id) DO UPDATE
        SET embedding = EXCLUDED.embedding, created_at = EXCLUDED.created_at
        """
        
        execute_batch(cursor, insert_query, data, page_size=1000)
        self.db_connection.commit()
        
        logger.info(f"Saved {len(post_ids):,} embeddings to database")
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM post_embeddings")
        total_count = cursor.fetchone()[0]
        logger.info(f"Total embeddings in database: {total_count:,}")
    
    # ========================================================================
    # FAISS INDEX
    # ========================================================================
    
    def rebuild_faiss_index(self):
        """
        Rebuild Faiss IndexFlatIP from all embeddings
        Save to models/faiss_index.bin
        """
        if not FAISS_AVAILABLE:
            logger.warning("Faiss not available. Skipping index rebuild.")
            return
        
        logger.info("\n🔧 Rebuilding Faiss index...")
        
        # Load all embeddings
        if self.use_csv:
            embeddings_dict = self._load_all_embeddings_file()
        else:
            embeddings_dict = self._load_all_embeddings_db()
        
        if len(embeddings_dict) == 0:
            logger.warning("No embeddings found. Cannot build index.")
            return
        
        # Convert to arrays
        post_ids = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[pid] for pid in post_ids])
        
        logger.info(f"Building index with {len(post_ids):,} vectors...")
        
        # Normalize embeddings for cosine similarity (IndexFlatIP)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / (norms + 1e-8)
        
        # Create Faiss index
        index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add vectors
        index.add(embeddings_normalized.astype(np.float32))
        
        # Save index
        index_path = self.models_dir / 'faiss_index.bin'
        faiss.write_index(index, str(index_path))
        
        # Save post_ids mapping
        mapping_path = self.models_dir / 'faiss_post_ids.pkl'
        with open(mapping_path, 'wb') as f:
            pickle.dump(post_ids, f)
        
        logger.info(f"✅ Faiss index rebuilt and saved")
        logger.info(f"   Index: {index_path}")
        logger.info(f"   Mapping: {mapping_path}")
        logger.info(f"   Vectors: {index.ntotal:,}")
    
    def _load_all_embeddings_file(self) -> Dict[int, np.ndarray]:
        """Load all embeddings from file"""
        embeddings_file = self.models_dir / 'post_embeddings.pkl'
        
        if not embeddings_file.exists():
            return {}
        
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        
        logger.info(f"Loaded {len(embeddings):,} embeddings from file")
        
        return embeddings
    
    def _load_all_embeddings_db(self) -> Dict[int, np.ndarray]:
        """Load all embeddings from database"""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 not available")
        
        cursor = self.db_connection.cursor()
        
        query = "SELECT post_id, embedding FROM post_embeddings"
        cursor.execute(query)
        
        rows = cursor.fetchall()
        
        embeddings = {}
        for post_id, embedding_bytes in rows:
            # Convert bytes back to numpy array
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(-1)
            embeddings[post_id] = embedding
        
        logger.info(f"Loaded {len(embeddings):,} embeddings from database")
        
        return embeddings
    
    # ========================================================================
    # CHECKPOINT
    # ========================================================================
    
    def _save_checkpoint(
        self,
        post_ids: List[int],
        embeddings: np.ndarray
    ):
        """Save checkpoint"""
        checkpoint_file = self.checkpoint_dir / f'checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        
        checkpoint = {
            'post_ids': post_ids,
            'embeddings': embeddings,
            'timestamp': datetime.now()
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_file}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[List[int], np.ndarray]:
        """Load checkpoint"""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"   Posts: {len(checkpoint['post_ids']):,}")
        logger.info(f"   Timestamp: {checkpoint['timestamp']}")
        
        return checkpoint['post_ids'], checkpoint['embeddings']


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    
    print("="*70)
    print("POST EMBEDDINGS BATCH PROCESSOR")
    print("="*70)
    
    # Initialize embedder
    embedder = BatchPostEmbedder(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        batch_size=512,
        device='auto',
        checkpoint_dir='checkpoints',
        models_dir='models'
    )
    
    # Enable CSV mode (for development)
    embedder.enable_csv_mode(csv_dir='dataset')
    
    # Run embedding
    n_embedded = embedder.embed_new_posts(
        since_hours=24 * 365,  # All posts (for initial run)
        save_checkpoint_every=5
    )
    
    print(f"\n✅ Embedded {n_embedded:,} posts successfully!")


if __name__ == "__main__":
    main()