

# # recommender/offline/artifact_manager.py
# from __future__ import annotations

# import platform
# import pickle
# import shutil
# from pathlib import Path
# from typing import Optional, Dict, Any

# def _is_windows() -> bool:
#     return platform.system().lower().startswith("win")

# def _ensure_dir(path: Path) -> None:
#     path.mkdir(parents=True, exist_ok=True)

# def write_latest_pointer(artifacts_base_dir: str, version_name: str) -> None:
#     base = Path(artifacts_base_dir)
#     _ensure_dir(base)
#     pointer = base / "latest.version"
#     pointer.write_text(version_name, encoding="utf-8")

# def read_latest_pointer(artifacts_base_dir: str) -> Optional[str]:
#     pointer = Path(artifacts_base_dir) / "latest.version"
#     if not pointer.exists():
#         return None
#     return pointer.read_text(encoding="utf-8").strip()

# def _symlink_latest(artifacts_base_dir: str, version_dir: Path) -> None:
#     latest_link = Path(artifacts_base_dir) / "latest"
#     try:
#         if latest_link.exists() or latest_link.is_symlink():
#             if latest_link.is_dir() and not latest_link.is_symlink():
#                 shutil.rmtree(latest_link)
#             else:
#                 latest_link.unlink(missing_ok=True)
#         latest_link.symlink_to(version_dir, target_is_directory=True)
#     except Exception:
#         write_latest_pointer(artifacts_base_dir, version_dir.name)

# def save_artifacts(
#     version_name: str,
#     model: Any,
#     meta: Optional[Dict[str, Any]] = None,
#     artifacts_base_dir: str = "models",
#     extra_files: Optional[Dict[str, bytes]] = None,
# ) -> Path:
#     """
#     Save model + meta vào models/{version}/
#       - model.pkl
#       - meta.json
#       - extra_files (tuỳ chọn)
#     Cập nhật 'latest' (Linux: symlink; Windows: pointer file).
#     """
#     import json

#     base = Path(artifacts_base_dir)
#     version_dir = base / version_name
#     _ensure_dir(version_dir)

#     model_path = version_dir / "model.pkl"
#     with model_path.open("wb") as f:
#         pickle.dump(model, f)

#     if meta is not None:
#         meta_path = version_dir / "meta.json"
#         meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

#     if extra_files:
#         for rel, content in extra_files.items():
#             out_path = version_dir / rel
#             _ensure_dir(out_path.parent)
#             with out_path.open("wb") as f:
#                 f.write(content)

#     if _is_windows():
#         try:
#             _symlink_latest(artifacts_base_dir, version_dir)
#         except Exception:
#             write_latest_pointer(artifacts_base_dir, version_name)
#     else:
#         _symlink_latest(artifacts_base_dir, version_dir)

#     return version_dir

# def get_latest_version_dir(artifacts_base_dir: str = "models") -> Optional[Path]:
#     base = Path(artifacts_base_dir)
#     link = base / "latest"
#     if link.is_symlink():
#         try:
#             target = link.resolve(strict=True)
#             return target if target.exists() else None
#         except Exception:
#             pass

#     name = read_latest_pointer(artifacts_base_dir)
#     if not name:
#         return None
#     candidate = base / name
#     return candidate if candidate.exists() else None

# class ArtifactManager:
#     def __init__(self, artifacts_base_dir: str = "models"):
#         self.artifacts_base_dir = artifacts_base_dir

#     def save_artifacts(
#         self,
#         version_name: str,
#         model: Any,
#         meta: Optional[Dict[str, Any]] = None,
#         extra_files: Optional[Dict[str, bytes]] = None,
#     ) -> Path:
#         return save_artifacts(
#             version_name=version_name,
#             model=model,
#             meta=meta,
#             artifacts_base_dir=self.artifacts_base_dir,
#             extra_files=extra_files,
#         )

#     def save(
#         self,
#         version_name: str,
#         model: Any,
#         meta: Optional[Dict[str, Any]] = None,
#         extra_files: Optional[Dict[str, bytes]] = None,
#     ) -> Path:
#         return self.save_artifacts(version_name, model, meta, extra_files)

#     def get_latest_version_dir(self) -> Optional[Path]:
#         return get_latest_version_dir(self.artifacts_base_dir)

#     def write_latest_pointer(self, version_name: str) -> None:
#         write_latest_pointer(self.artifacts_base_dir, version_name)


## Version cập nhật 27/10 - Claude

"""
ARTIFACT MANAGER
================
Manage model artifacts with versioning

Features:
- Version management (timestamp-based)
- Save/load all artifacts (embeddings, CF, model, stats)
- Cleanup old versions
- Latest version tracking
- Metadata management

Usage:
    mgr = ArtifactManager()
    
    # Save
    mgr.save_artifacts(
        version='2025-01-27_v1',
        embeddings=embeddings,
        cf_model=cf_model,
        ranking_model=ranking_model,
        ...
    )
    
    # Load
    artifacts = mgr.load_artifacts('2025-01-27_v1')
    # or load latest
    artifacts = mgr.load_artifacts(mgr.get_latest_version())
"""

import os
import sys
import json
import pickle
import shutil
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

# Try import lightgbm for model saving
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("lightgbm not available")


class ArtifactManager:
    """
    Manage model artifacts with versioning
    
    Directory structure:
    models/
    ├── v_2025-01-27_120000/
    │   ├── embeddings.pkl
    │   ├── cf_model.pkl
    │   ├── ranking_model.txt
    │   ├── ranking_scaler.pkl
    │   ├── ranking_feature_cols.pkl
    │   ├── user_stats.pkl
    │   ├── author_stats.pkl
    │   ├── following_dict.pkl
    │   └── metadata.json
    ├── v_2025-01-28_120000/
    │   └── ...
    └── latest.txt (pointer to latest version)
    """
    
    def __init__(self, base_dir: str = 'models'):
        """
        Initialize Artifact Manager
        
        Args:
            base_dir: Base directory for storing artifacts
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ArtifactManager initialized:")
        logger.info(f"   Base directory: {self.base_dir}")
    
    # ========================================================================
    # VERSION MANAGEMENT
    # ========================================================================
    
    def generate_version_name(self) -> str:
        """
        Generate version name based on current timestamp
        
        Returns:
            Version name (e.g., 'v_2025-01-27_120530')
        """
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        return f"v_{timestamp}"
    
    def get_all_versions(self) -> List[str]:
        """
        Get list of all available versions (sorted by name)
        
        Returns:
            List of version names
        """
        if not self.base_dir.exists():
            return []
        
        versions = [
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and d.name.startswith('v_')
        ]
        
        return sorted(versions, reverse=True)  # Latest first
    
    def get_latest_version(self) -> Optional[str]:
        """
        Get latest version name
        
        Strategy:
        1. Check latest.txt pointer file
        2. If not exists, return most recent version by name
        
        Returns:
            Latest version name or None
        """
        # Try pointer file first
        pointer_file = self.base_dir / 'latest.txt'
        
        if pointer_file.exists():
            try:
                version = pointer_file.read_text().strip()
                version_dir = self.base_dir / version
                
                if version_dir.exists():
                    return version
                else:
                    logger.warning(f"Pointer file references non-existent version: {version}")
            except Exception as e:
                logger.warning(f"Error reading pointer file: {e}")
        
        # Fall back to most recent by name
        versions = self.get_all_versions()
        
        if versions:
            return versions[0]  # Already sorted, latest first
        
        return None
    
    def set_latest_version(self, version: str):
        """
        Update latest version pointer
        
        Args:
            version: Version name to set as latest
        """
        pointer_file = self.base_dir / 'latest.txt'
        pointer_file.write_text(version)
        
        logger.info(f"✅ Latest version pointer updated: {version}")
    
    def get_version_dir(self, version: str) -> Path:
        """
        Get path to version directory
        
        Args:
            version: Version name
        
        Returns:
            Path to version directory
        """
        return self.base_dir / version
    
    # ========================================================================
    # SAVE ARTIFACTS
    # ========================================================================
    
    def save_artifacts(
        self,
        version: Optional[str] = None,
        embeddings: Optional[Dict] = None,
        cf_model: Optional[Dict] = None,
        ranking_model: Optional[Any] = None,
        ranking_scaler: Optional[Any] = None,
        ranking_feature_cols: Optional[List[str]] = None,
        user_stats: Optional[Dict] = None,
        author_stats: Optional[Dict] = None,
        following_dict: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        faiss_index: Optional[Any] = None,
        set_as_latest: bool = True
    ) -> str:
        """
        Save all artifacts to versioned directory
        
        Args:
            version: Version name (auto-generated if None)
            embeddings: Dict {'post': {...}, 'user': {...}}
            cf_model: CF model dict
            ranking_model: LightGBM model
            ranking_scaler: Feature scaler
            ranking_feature_cols: List of feature column names
            user_stats: User statistics dict
            author_stats: Author statistics dict
            following_dict: Following relationships dict
            metadata: Metadata dict
            faiss_index: Optional FAISS index
            set_as_latest: Whether to set this version as latest
        
        Returns:
            Version name
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"SAVING ARTIFACTS")
        logger.info(f"{'='*70}")
        
        # Generate version if not provided
        if version is None:
            version = self.generate_version_name()
        
        version_dir = self.get_version_dir(version)
        version_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Version: {version}")
        logger.info(f"Directory: {version_dir}")
        
        artifacts_saved = []
        
        # ────────────────────────────────────────────────────────────────────
        # Save embeddings
        # ────────────────────────────────────────────────────────────────────
        
        if embeddings is not None:
            embeddings_path = version_dir / 'embeddings.pkl'
            with open(embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size_mb = embeddings_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Saved embeddings.pkl ({file_size_mb:.1f} MB)")
            artifacts_saved.append('embeddings')
        
        # ────────────────────────────────────────────────────────────────────
        # Save CF model
        # ────────────────────────────────────────────────────────────────────
        
        if cf_model is not None:
            cf_model_path = version_dir / 'cf_model.pkl'
            with open(cf_model_path, 'wb') as f:
                pickle.dump(cf_model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size_mb = cf_model_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Saved cf_model.pkl ({file_size_mb:.1f} MB)")
            artifacts_saved.append('cf_model')
        
        # ────────────────────────────────────────────────────────────────────
        # Save ranking model (LightGBM)
        # ────────────────────────────────────────────────────────────────────
        
        if ranking_model is not None:
            if LIGHTGBM_AVAILABLE and isinstance(ranking_model, lgb.Booster):
                # Save as .txt (LightGBM native format)
                ranking_model_path = version_dir / 'ranking_model.txt'
                ranking_model.save_model(str(ranking_model_path))
                
                file_size_mb = ranking_model_path.stat().st_size / (1024 * 1024)
                logger.info(f"✓ Saved ranking_model.txt ({file_size_mb:.1f} MB)")
            else:
                # Fallback to pickle
                ranking_model_path = version_dir / 'ranking_model.pkl'
                with open(ranking_model_path, 'wb') as f:
                    pickle.dump(ranking_model, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                file_size_mb = ranking_model_path.stat().st_size / (1024 * 1024)
                logger.info(f"✓ Saved ranking_model.pkl ({file_size_mb:.1f} MB)")
            
            artifacts_saved.append('ranking_model')
        
        # ────────────────────────────────────────────────────────────────────
        # Save ranking scaler
        # ────────────────────────────────────────────────────────────────────
        
        if ranking_scaler is not None:
            scaler_path = version_dir / 'ranking_scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(ranking_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"✓ Saved ranking_scaler.pkl")
            artifacts_saved.append('ranking_scaler')
        
        # ────────────────────────────────────────────────────────────────────
        # Save ranking feature columns
        # ────────────────────────────────────────────────────────────────────
        
        if ranking_feature_cols is not None:
            feature_cols_path = version_dir / 'ranking_feature_cols.pkl'
            with open(feature_cols_path, 'wb') as f:
                pickle.dump(ranking_feature_cols, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"✓ Saved ranking_feature_cols.pkl ({len(ranking_feature_cols)} features)")
            artifacts_saved.append('ranking_feature_cols')
        
        # ────────────────────────────────────────────────────────────────────
        # Save statistics
        # ────────────────────────────────────────────────────────────────────
        
        if user_stats is not None:
            user_stats_path = version_dir / 'user_stats.pkl'
            with open(user_stats_path, 'wb') as f:
                pickle.dump(user_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"✓ Saved user_stats.pkl ({len(user_stats):,} users)")
            artifacts_saved.append('user_stats')
        
        if author_stats is not None:
            author_stats_path = version_dir / 'author_stats.pkl'
            with open(author_stats_path, 'wb') as f:
                pickle.dump(author_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"✓ Saved author_stats.pkl ({len(author_stats):,} authors)")
            artifacts_saved.append('author_stats')
        
        if following_dict is not None:
            following_dict_path = version_dir / 'following_dict.pkl'
            with open(following_dict_path, 'wb') as f:
                pickle.dump(following_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"✓ Saved following_dict.pkl ({len(following_dict):,} users)")
            artifacts_saved.append('following_dict')
        
        # ────────────────────────────────────────────────────────────────────
        # Save FAISS index (optional)
        # ────────────────────────────────────────────────────────────────────
        
        if faiss_index is not None:
            try:
                import faiss
                faiss_path = version_dir / 'faiss_index.bin'
                faiss.write_index(faiss_index, str(faiss_path))
                
                file_size_mb = faiss_path.stat().st_size / (1024 * 1024)
                logger.info(f"✓ Saved faiss_index.bin ({file_size_mb:.1f} MB)")
                artifacts_saved.append('faiss_index')
            except ImportError:
                logger.warning("FAISS not available, skipping index save")
            except Exception as e:
                logger.warning(f"Error saving FAISS index: {e}")
        
        # ────────────────────────────────────────────────────────────────────
        # Save metadata
        # ────────────────────────────────────────────────────────────────────
        
        if metadata is None:
            metadata = {}
        
        metadata['version'] = version
        metadata['timestamp'] = datetime.now().isoformat()
        metadata['artifacts_saved'] = artifacts_saved
        
        metadata_path = version_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"✓ Saved metadata.json")
        
        # ────────────────────────────────────────────────────────────────────
        # Update latest pointer
        # ────────────────────────────────────────────────────────────────────
        
        if set_as_latest:
            self.set_latest_version(version)
        
        # ────────────────────────────────────────────────────────────────────
        # Summary
        # ────────────────────────────────────────────────────────────────────
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ ARTIFACTS SAVED SUCCESSFULLY")
        logger.info(f"{'='*70}")
        logger.info(f"   Version: {version}")
        logger.info(f"   Directory: {version_dir}")
        logger.info(f"   Artifacts: {', '.join(artifacts_saved)}")
        
        return version
    
    # ========================================================================
    # LOAD ARTIFACTS
    # ========================================================================
    
    def load_artifacts(self, version: Optional[str] = None) -> Dict:
        """
        Load all artifacts from version
        
        Args:
            version: Version to load (loads latest if None)
        
        Returns:
            Dict containing all loaded artifacts
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"LOADING ARTIFACTS")
        logger.info(f"{'='*70}")
        
        # Use latest version if not specified
        if version is None:
            version = self.get_latest_version()
            
            if version is None:
                logger.error("No versions available!")
                return {}
        
        version_dir = self.get_version_dir(version)
        
        if not version_dir.exists():
            logger.error(f"Version directory not found: {version_dir}")
            return {}
        
        logger.info(f"Version: {version}")
        logger.info(f"Directory: {version_dir}")
        
        artifacts = {}
        
        # ────────────────────────────────────────────────────────────────────
        # Load embeddings
        # ────────────────────────────────────────────────────────────────────
        
        embeddings_path = version_dir / 'embeddings.pkl'
        if embeddings_path.exists():
            with open(embeddings_path, 'rb') as f:
                artifacts['embeddings'] = pickle.load(f)
            
            n_posts = len(artifacts['embeddings'].get('post', {}))
            n_users = len(artifacts['embeddings'].get('user', {}))
            logger.info(f"✓ Loaded embeddings (posts: {n_posts:,}, users: {n_users:,})")
        
        # ────────────────────────────────────────────────────────────────────
        # Load CF model
        # ────────────────────────────────────────────────────────────────────
        
        cf_model_path = version_dir / 'cf_model.pkl'
        if cf_model_path.exists():
            with open(cf_model_path, 'rb') as f:
                artifacts['cf_model'] = pickle.load(f)
            
            logger.info(f"✓ Loaded cf_model")
        
        # ────────────────────────────────────────────────────────────────────
        # Load ranking model
        # ────────────────────────────────────────────────────────────────────
        
        # Try .txt first (LightGBM native)
        ranking_model_txt_path = version_dir / 'ranking_model.txt'
        ranking_model_pkl_path = version_dir / 'ranking_model.pkl'
        
        if ranking_model_txt_path.exists() and LIGHTGBM_AVAILABLE:
            artifacts['ranking_model'] = lgb.Booster(model_file=str(ranking_model_txt_path))
            logger.info(f"✓ Loaded ranking_model (LightGBM)")
        elif ranking_model_pkl_path.exists():
            with open(ranking_model_pkl_path, 'rb') as f:
                artifacts['ranking_model'] = pickle.load(f)
            logger.info(f"✓ Loaded ranking_model (pickle)")
        
        # ────────────────────────────────────────────────────────────────────
        # Load ranking scaler
        # ────────────────────────────────────────────────────────────────────
        
        scaler_path = version_dir / 'ranking_scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                artifacts['ranking_scaler'] = pickle.load(f)
            logger.info(f"✓ Loaded ranking_scaler")
        
        # ────────────────────────────────────────────────────────────────────
        # Load ranking feature columns
        # ────────────────────────────────────────────────────────────────────
        
        feature_cols_path = version_dir / 'ranking_feature_cols.pkl'
        if feature_cols_path.exists():
            with open(feature_cols_path, 'rb') as f:
                artifacts['ranking_feature_cols'] = pickle.load(f)
            logger.info(f"✓ Loaded ranking_feature_cols ({len(artifacts['ranking_feature_cols'])} features)")
        
        # ────────────────────────────────────────────────────────────────────
        # Load statistics
        # ────────────────────────────────────────────────────────────────────
        
        user_stats_path = version_dir / 'user_stats.pkl'
        if user_stats_path.exists():
            with open(user_stats_path, 'rb') as f:
                artifacts['user_stats'] = pickle.load(f)
            logger.info(f"✓ Loaded user_stats ({len(artifacts['user_stats']):,} users)")
        
        author_stats_path = version_dir / 'author_stats.pkl'
        if author_stats_path.exists():
            with open(author_stats_path, 'rb') as f:
                artifacts['author_stats'] = pickle.load(f)
            logger.info(f"✓ Loaded author_stats ({len(artifacts['author_stats']):,} authors)")
        
        following_dict_path = version_dir / 'following_dict.pkl'
        if following_dict_path.exists():
            with open(following_dict_path, 'rb') as f:
                artifacts['following_dict'] = pickle.load(f)
            logger.info(f"✓ Loaded following_dict ({len(artifacts['following_dict']):,} users)")
        
        # ────────────────────────────────────────────────────────────────────
        # Load FAISS index (optional)
        # ────────────────────────────────────────────────────────────────────
        
        faiss_path = version_dir / 'faiss_index.bin'
        if faiss_path.exists():
            try:
                import faiss
                artifacts['faiss_index'] = faiss.read_index(str(faiss_path))
                logger.info(f"✓ Loaded faiss_index")
            except ImportError:
                logger.warning("FAISS not available, skipping index load")
            except Exception as e:
                logger.warning(f"Error loading FAISS index: {e}")
        
        # ────────────────────────────────────────────────────────────────────
        # Load metadata
        # ────────────────────────────────────────────────────────────────────
        
        metadata_path = version_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                artifacts['metadata'] = json.load(f)
            logger.info(f"✓ Loaded metadata")
        
        logger.info(f"\n✅ Artifacts loaded: {len(artifacts)} components")
        
        return artifacts
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    
    def cleanup_old_versions(self, keep_n: int = 5):
        """
        Delete old versions, keeping only the N most recent
        
        Args:
            keep_n: Number of recent versions to keep
        """
        logger.info(f"\n🧹 Cleaning up old versions (keeping {keep_n})...")
        
        versions = self.get_all_versions()
        
        if len(versions) <= keep_n:
            logger.info(f"   No cleanup needed ({len(versions)} versions)")
            return
        
        versions_to_delete = versions[keep_n:]  # Already sorted, latest first
        
        for version in versions_to_delete:
            version_dir = self.get_version_dir(version)
            
            try:
                shutil.rmtree(version_dir)
                logger.info(f"   Deleted: {version}")
            except Exception as e:
                logger.warning(f"   Failed to delete {version}: {e}")
        
        logger.info(f"✅ Cleanup complete (deleted {len(versions_to_delete)} versions)")
    
    def get_info(self) -> Dict:
        """Get artifact manager information"""
        versions = self.get_all_versions()
        latest = self.get_latest_version()
        
        return {
            'base_dir': str(self.base_dir),
            'total_versions': len(versions),
            'latest_version': latest,
            'all_versions': versions
        }


# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

def main():
    """Test Artifact Manager"""
    
    logger.info(f"{'='*70}")
    logger.info(f"ARTIFACT MANAGER TEST")
    logger.info(f"{'='*70}")
    
    # Initialize
    mgr = ArtifactManager(base_dir='test_models')
    
    # Create dummy artifacts
    import numpy as np
    
    embeddings = {
        'post': {i: np.random.randn(128) for i in range(10)},
        'user': {i: np.random.randn(128) for i in range(5)}
    }
    
    cf_model = {
        'user_similarities': {i: [(j, 0.8) for j in range(3)] for i in range(5)},
        'item_similarities': {i: [(j, 0.7) for j in range(3)] for i in range(10)},
        'metadata': {'n_users': 5, 'n_posts': 10}
    }
    
    metadata = {
        'test_metrics': {'auc': 0.85, 'precision@10': 0.65},
        'training_date': datetime.now().isoformat()
    }
    
    # Save artifacts
    version = mgr.save_artifacts(
        embeddings=embeddings,
        cf_model=cf_model,
        metadata=metadata
    )
    
    logger.info(f"\nSaved version: {version}")
    
    # Load artifacts
    loaded_artifacts = mgr.load_artifacts(version)
    
    logger.info(f"\n✅ TEST COMPLETE")
    logger.info(f"   Loaded {len(loaded_artifacts)} artifacts")
    
    # Info
    info = mgr.get_info()
    logger.info(f"\nArtifact Manager Info:")
    for key, value in info.items():
        logger.info(f"   {key}: {value}")


if __name__ == "__main__":
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    main()