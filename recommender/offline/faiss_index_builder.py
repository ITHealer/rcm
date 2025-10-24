"""
FAISS INDEX BUILDER
===================
Build Faiss index from post embeddings for vector similarity search

Features:
- IndexFlatIP (Inner Product for cosine similarity)
- Batch loading from PostgreSQL
- Normalization validation
- GPU acceleration (optional)
- Performance benchmarking

Update: After each batch embedding (every 6 hours)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pickle
import json
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.error("❌ faiss not available. Install: pip install faiss-cpu (or faiss-gpu)")

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.warning("⚠️  psycopg2 not available. Using CSV mode only.")


class FaissIndexBuilder:
    """
    Build and manage Faiss index for post embeddings
    
    Features:
    - IndexFlatIP (Inner Product)
    - Normalization for cosine similarity
    - GPU support (optional)
    - Benchmark search performance
    """
    
    def __init__(
        self,
        dimension: int = 384,
        db_connection = None,
        models_dir: str = 'models',
        use_gpu: bool = False
    ):
        """
        Initialize Faiss index builder
        
        Args:
            dimension: Embedding dimension
            db_connection: PostgreSQL connection (optional)
            models_dir: Directory for models
            use_gpu: Use GPU acceleration
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu or faiss-gpu required")
        
        self.dimension = dimension
        self.db_connection = db_connection
        self.models_dir = Path(models_dir)
        self.use_gpu = use_gpu
        
        self.models_dir.mkdir(exist_ok=True)
        
        self.index = None
        self.post_id_map = {}  # {index: post_id}
        self.post_ids = []
        
        # GPU resources
        self.gpu_resources = None
        if use_gpu and faiss.get_num_gpus() > 0:
            logger.info(f"GPU available: {faiss.get_num_gpus()} GPUs")
            self.gpu_resources = faiss.StandardGpuResources()
        elif use_gpu:
            logger.warning("GPU requested but not available. Using CPU.")
            self.use_gpu = False
        
        logger.info(f"Initialized FaissIndexBuilder")
        logger.info(f"Dimension: {dimension}")
        logger.info(f"Models dir: {self.models_dir}")
        logger.info(f"GPU: {self.use_gpu}")
        
        # Statistics
        self.stats = {
            'n_vectors': 0,
            'build_time': 0,
            'index_size_mb': 0,
            'avg_search_time_ms': 0
        }
    
    # ========================================================================
    # LOAD EMBEDDINGS
    # ========================================================================
    
    def load_embeddings_from_db(self) -> Dict[int, np.ndarray]:
        """
        Load all post embeddings from PostgreSQL
        
        Returns:
            {post_id: embedding, ...}
        """
        if not PSYCOPG2_AVAILABLE or self.db_connection is None:
            logger.error("PostgreSQL not available")
            return {}
        
        logger.info(f"\n📦 Loading embeddings from database...")
        
        start_time = time.time()
        
        query = """
        SELECT post_id, embedding
        FROM post_embeddings
        WHERE embedding IS NOT NULL
        ORDER BY post_id
        """
        
        cursor = self.db_connection.cursor()
        cursor.execute(query)
        
        rows = cursor.fetchall()
        
        logger.info(f"Fetched {len(rows):,} embeddings from database")
        
        # Convert to dict
        embeddings = {}
        for post_id, embedding_bytes in rows:
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings[post_id] = embedding
        
        load_time = time.time() - start_time
        
        logger.info(f"✅ Loaded {len(embeddings):,} embeddings in {load_time:.1f}s")
        
        return embeddings
    
    def load_embeddings_from_pickle(
        self,
        filepath: str = 'models/post_embeddings.pkl'
    ) -> Dict[int, np.ndarray]:
        """
        Load embeddings from pickle file
        
        Args:
            filepath: Path to pickle file
        
        Returns:
            {post_id: embedding, ...}
        """
        logger.info(f"\n📦 Loading embeddings from pickle...")
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return {}
        
        start_time = time.time()
        
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        
        load_time = time.time() - start_time
        
        logger.info(f"✅ Loaded {len(embeddings):,} embeddings in {load_time:.1f}s")
        
        return embeddings
    
    # ========================================================================
    # BUILD INDEX
    # ========================================================================
    
    def build_index(
        self,
        embeddings_dict: Dict[int, np.ndarray],
        validate: bool = True
    ):
        """
        Build Faiss index from embeddings
        
        Args:
            embeddings_dict: {post_id: embedding_vector, ...}
            validate: Validate normalization
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"BUILDING FAISS INDEX")
        logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        if len(embeddings_dict) == 0:
            logger.error("No embeddings to index!")
            return
        
        # Sort by post_id for consistency
        post_ids = sorted(embeddings_dict.keys())
        vectors = np.array([embeddings_dict[pid] for pid in post_ids], dtype=np.float32)
        
        logger.info(f"Embeddings: {len(post_ids):,}")
        logger.info(f"Dimension: {vectors.shape[1]}")
        logger.info(f"Shape: {vectors.shape}")
        logger.info(f"Dtype: {vectors.dtype}")
        
        # Normalize vectors (CRITICAL for cosine similarity)
        logger.info(f"\n🔧 Normalizing vectors...")
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_normalized = vectors / (norms + 1e-8)
        
        # Validate normalization
        if validate:
            self._validate_normalization(vectors_normalized)
        
        # Create index
        logger.info(f"\n🔧 Creating Faiss index...")
        logger.info(f"Index type: IndexFlatIP (Inner Product)")
        
        index = faiss.IndexFlatIP(self.dimension)
        
        # GPU acceleration
        if self.use_gpu and self.gpu_resources is not None:
            logger.info(f"Moving index to GPU...")
            index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
        
        # Add vectors
        logger.info(f"\n🔧 Adding vectors to index...")
        
        index.add(vectors_normalized)
        
        logger.info(f"✅ Added {index.ntotal:,} vectors")
        
        # Store mappings
        self.index = index
        self.post_ids = post_ids
        self.post_id_map = {idx: post_id for idx, post_id in enumerate(post_ids)}
        
        # Statistics
        build_time = time.time() - start_time
        
        self.stats['n_vectors'] = index.ntotal
        self.stats['build_time'] = build_time
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ INDEX BUILT")
        logger.info(f"{'='*70}")
        logger.info(f"Vectors: {index.ntotal:,}")
        logger.info(f"Build time: {build_time:.1f}s")
    
    def _validate_normalization(self, vectors: np.ndarray):
        """
        Validate that vectors are normalized
        
        Args:
            vectors: (N, D) normalized vectors
        """
        logger.info(f"Validating normalization...")
        
        norms = np.linalg.norm(vectors, axis=1)
        
        avg_norm = norms.mean()
        min_norm = norms.min()
        max_norm = norms.max()
        std_norm = norms.std()
        
        logger.info(f"  Average norm: {avg_norm:.6f}")
        logger.info(f"  Min norm: {min_norm:.6f}")
        logger.info(f"  Max norm: {max_norm:.6f}")
        logger.info(f"  Std norm: {std_norm:.6f}")
        
        # Check if normalized (should be ~1.0)
        if abs(avg_norm - 1.0) > 0.01:
            logger.warning(f"⚠️  Vectors may not be normalized! Avg norm: {avg_norm:.6f}")
        else:
            logger.info(f"✅ Vectors are normalized")
    
    # ========================================================================
    # SEARCH
    # ========================================================================
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 200
    ) -> Tuple[List[int], List[float]]:
        """
        Search similar vectors
        
        Args:
            query_vector: (D,) query vector
            k: Number of results
        
        Returns:
            post_ids: List of post IDs
            distances: Similarity scores (cosine similarity)
        """
        if self.index is None:
            logger.error("Index not built!")
            return [], []
        
        # Ensure 2D and normalized
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Normalize
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        # Ensure float32
        query_vector = query_vector.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        # Convert indices to post IDs
        post_ids = [self.post_id_map[idx] for idx in indices[0]]
        distances_list = distances[0].tolist()
        
        return post_ids, distances_list
    
    def benchmark_search(
        self,
        n_queries: int = 100,
        k: int = 200
    ):
        """
        Benchmark search performance
        
        Args:
            n_queries: Number of random queries
            k: Top K results
        """
        logger.info(f"\n📊 Benchmarking search performance...")
        logger.info(f"Queries: {n_queries}")
        logger.info(f"K: {k}")
        
        if self.index is None:
            logger.error("Index not built!")
            return
        
        # Generate random queries
        np.random.seed(42)
        queries = np.random.randn(n_queries, self.dimension).astype(np.float32)
        
        # Normalize
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / norms
        
        # Warm-up
        _ = self.index.search(queries[:10], k)
        
        # Benchmark
        start_time = time.time()
        
        distances, indices = self.index.search(queries, k)
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / n_queries) * 1000
        
        self.stats['avg_search_time_ms'] = avg_time_ms
        
        logger.info(f"\n📊 Benchmark Results:")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average time: {avg_time_ms:.2f}ms per query")
        logger.info(f"Throughput: {n_queries / total_time:.0f} queries/second")
    
    # ========================================================================
    # SAVE/LOAD INDEX
    # ========================================================================
    
    def save_index(
        self,
        index_path: Optional[str] = None,
        map_path: Optional[str] = None
    ):
        """
        Save index to disk
        
        Args:
            index_path: Path to save index
            map_path: Path to save post_id_map
        """
        if self.index is None:
            logger.error("Index not built!")
            return
        
        logger.info(f"\n💾 Saving index...")
        
        # Default paths
        if index_path is None:
            index_path = self.models_dir / 'faiss_index.bin'
        else:
            index_path = Path(index_path)
        
        if map_path is None:
            map_path = self.models_dir / 'post_id_map.json'
        else:
            map_path = Path(map_path)
        
        # Convert GPU index to CPU before saving
        if self.use_gpu:
            logger.info(f"Moving index to CPU for saving...")
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
        
        # Save index
        faiss.write_index(cpu_index, str(index_path))
        logger.info(f"Saved index to {index_path}")
        
        # Get file size
        index_size_mb = index_path.stat().st_size / (1024 * 1024)
        self.stats['index_size_mb'] = index_size_mb
        logger.info(f"Index size: {index_size_mb:.1f} MB")
        
        # Save post_id_map
        # Convert keys to strings for JSON
        post_id_map_serializable = {str(k): v for k, v in self.post_id_map.items()}
        
        with open(map_path, 'w') as f:
            json.dump(post_id_map_serializable, f)
        
        logger.info(f"Saved post_id_map to {map_path}")
        
        # Save stats
        stats_path = self.models_dir / 'faiss_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Saved stats to {stats_path}")
    
    def load_index(
        self,
        index_path: Optional[str] = None,
        map_path: Optional[str] = None
    ):
        """
        Load index from disk
        
        Args:
            index_path: Path to index file
            map_path: Path to post_id_map file
        """
        logger.info(f"\n📦 Loading index...")
        
        # Default paths
        if index_path is None:
            index_path = self.models_dir / 'faiss_index.bin'
        else:
            index_path = Path(index_path)
        
        if map_path is None:
            map_path = self.models_dir / 'post_id_map.json'
        else:
            map_path = Path(map_path)
        
        # Check files exist
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        if not map_path.exists():
            raise FileNotFoundError(f"Post ID map not found: {map_path}")
        
        # Load index
        index = faiss.read_index(str(index_path))
        
        logger.info(f"Loaded index from {index_path}")
        logger.info(f"Vectors: {index.ntotal:,}")
        
        # Move to GPU if requested
        if self.use_gpu and self.gpu_resources is not None:
            logger.info(f"Moving index to GPU...")
            index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
        
        self.index = index
        
        # Load post_id_map
        with open(map_path, 'r') as f:
            post_id_map_str = json.load(f)
        
        # Convert keys back to int
        self.post_id_map = {int(k): v for k, v in post_id_map_str.items()}
        self.post_ids = [self.post_id_map[i] for i in range(len(self.post_id_map))]
        
        logger.info(f"Loaded post_id_map from {map_path}")
        logger.info(f"✅ Index loaded successfully")
    
    # ========================================================================
    # UTILITY
    # ========================================================================
    
    def get_info(self) -> Dict:
        """Get index information"""
        if self.index is None:
            return {}
        
        info = {
            'n_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'is_trained': self.index.is_trained,
            'gpu': self.use_gpu,
            **self.stats
        }
        
        return info
    
    def print_info(self):
        """Print index information"""
        info = self.get_info()
        
        logger.info(f"\n📊 Index Information:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    
    logger.info(f"{'='*70}")
    logger.info(f"FAISS INDEX BUILDER")
    logger.info(f"{'='*70}")
    
    start_time = time.time()
    
    # Initialize
    builder = FaissIndexBuilder(
        dimension=384,
        use_gpu=False  # Change to True if GPU available
    )
    
    # Load embeddings (from pickle)
    embeddings = builder.load_embeddings_from_pickle('models/post_embeddings.pkl')
    
    if len(embeddings) == 0:
        logger.error("No embeddings found!")
        return
    
    # Build index
    builder.build_index(embeddings, validate=True)
    
    # Benchmark
    builder.benchmark_search(n_queries=100, k=200)
    
    # Print info
    builder.print_info()
    
    # Save
    builder.save_index()
    
    # Test search
    logger.info(f"\n🔍 Testing search...")
    
    # Random query
    query = np.random.randn(384).astype(np.float32)
    post_ids, distances = builder.search(query, k=10)
    
    logger.info(f"Query returned {len(post_ids)} results")
    logger.info(f"Top 5 results:")
    for i, (post_id, dist) in enumerate(zip(post_ids[:5], distances[:5])):
        logger.info(f"  {i+1}. Post {post_id}: similarity = {dist:.4f}")
    
    # Total time
    total_time = time.time() - start_time
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ FAISS INDEX BUILD COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total time: {total_time/60:.1f}m {total_time%60:.0f}s")


if __name__ == "__main__":
    main()