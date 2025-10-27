# """
# Artifact Management - Version Control cho Models
# Purpose: Quản lý versions của models, metadata
# """

# import os
# import json
# import shutil
# from datetime import datetime
# from pathlib import Path
# from typing import Dict
# import pickle
# import faiss
# import platform

# class ArtifactManager:
#     """
#     Manage model artifacts với versioning
#     Cross-platform compatible (Windows + Linux)
#     """
    
#     def __init__(self, base_dir: str = 'models'):
#         self.base_dir = Path(base_dir)
#         self.base_dir.mkdir(exist_ok=True)
#         self.is_windows = platform.system() == 'Windows'
    
#     def create_version(self) -> str:
#         """Create new version directory"""
#         version = datetime.now().strftime('v%Y%m%d_%H%M%S')
#         version_dir = self.base_dir / version
#         version_dir.mkdir(parents=True, exist_ok=True)
#         return version
    
#     def save_artifacts(
#         self,
#         version: str,
#         embeddings: Dict,
#         cf_model: Dict,
#         ranking_model,
#         ranking_scaler,
#         ranking_feature_cols: list,
#         user_stats: Dict,
#         author_stats: Dict,
#         following_dict: Dict,
#         metadata: Dict = None
#     ):
#         """
#         Save all artifacts for a version
#         """
#         version_dir = self.base_dir / version
        
#         print(f"\n💾 Saving artifacts to version: {version}")
        
#         # 1. Save embeddings
#         print("   Saving embeddings...")
#         with open(version_dir / 'embeddings.pkl', 'wb') as f:
#             pickle.dump(embeddings, f)
        
#         # 2. Build and save FAISS index
#         print("   Building FAISS index...")
#         faiss_index, post_ids = self._build_faiss_index(embeddings['post'])
#         faiss.write_index(faiss_index, str(version_dir / 'faiss_index.bin'))
        
#         with open(version_dir / 'faiss_post_ids.pkl', 'wb') as f:
#             pickle.dump(post_ids, f)
        
#         # 3. Save CF model
#         print("   Saving CF model...")
#         with open(version_dir / 'cf_model.pkl', 'wb') as f:
#             pickle.dump(cf_model, f)
        
#         # 4. Save ranking model
#         print("   Saving ranking model...")
#         ranking_model.save_model(str(version_dir / 'ranking_model.txt'))
        
#         with open(version_dir / 'ranking_scaler.pkl', 'wb') as f:
#             pickle.dump(ranking_scaler, f)
        
#         with open(version_dir / 'ranking_feature_cols.pkl', 'wb') as f:
#             pickle.dump(ranking_feature_cols, f)
        
#         # 5. Save stats
#         print("   Saving statistics...")
#         with open(version_dir / 'user_stats.pkl', 'wb') as f:
#             pickle.dump(user_stats, f)
        
#         with open(version_dir / 'author_stats.pkl', 'wb') as f:
#             pickle.dump(author_stats, f)
        
#         with open(version_dir / 'following_dict.pkl', 'wb') as f:
#             pickle.dump(following_dict, f)
        
#         # 6. Save metadata
#         metadata = metadata or {}
#         metadata.update({
#             'version': version,
#             'created_at': datetime.now().isoformat(),
#             'artifacts': [
#                 'embeddings.pkl',
#                 'faiss_index.bin',
#                 'cf_model.pkl',
#                 'ranking_model.txt',
#                 'user_stats.pkl',
#                 'author_stats.pkl',
#                 'following_dict.pkl'
#             ]
#         })
        
#         with open(version_dir / 'metadata.json', 'w') as f:
#             json.dump(metadata, f, indent=2, default=str)
        
#         # 7. Update 'latest' pointer (cross-platform)
#         self._update_latest_pointer(version)
        
#         print(f"✅ All artifacts saved!")
#         print(f"✅ Latest version: {version}")
    
#     def _update_latest_pointer(self, version: str):
#         """
#         Update 'latest' pointer to current version
#         Cross-platform compatible:
#         - Linux/Mac: Use symlink
#         - Windows: Write version to text file
#         """
#         latest_path = self.base_dir / 'latest'
        
#         if self.is_windows:
#             # ============================================================
#             # WINDOWS: Use text file instead of symlink
#             # ============================================================
#             print("   Creating latest pointer (Windows mode)...")
            
#             # Remove old pointer if exists
#             if latest_path.exists():
#                 if latest_path.is_file():
#                     latest_path.unlink()
#                 elif latest_path.is_dir():
#                     shutil.rmtree(latest_path)
            
#             # Write version to text file
#             with open(latest_path, 'w') as f:
#                 f.write(version)
            
#             print(f"   ✅ Latest pointer: {version}")
            
#         else:
#             # ============================================================
#             # LINUX/MAC: Use symlink
#             # ============================================================
#             print("   Creating latest symlink (Linux mode)...")
            
#             # Remove old symlink if exists
#             if latest_path.exists() or latest_path.is_symlink():
#                 latest_path.unlink()
            
#             # Create new symlink
#             latest_path.symlink_to(version, target_is_directory=True)
            
#             print(f"   ✅ Latest symlink: {version}")
    
#     def get_latest_version(self) -> str:
#         """
#         Get latest version
#         Cross-platform compatible
#         """
#         latest_path = self.base_dir / 'latest'
        
#         if not latest_path.exists():
#             raise FileNotFoundError("No 'latest' version found")
        
#         if self.is_windows:
#             # ============================================================
#             # WINDOWS: Read from text file
#             # ============================================================
#             if latest_path.is_file():
#                 with open(latest_path, 'r') as f:
#                     version = f.read().strip()
#                 return version
#             else:
#                 raise ValueError("Latest pointer is not a file on Windows")
        
#         else:
#             # ============================================================
#             # LINUX/MAC: Read symlink
#             # ============================================================
#             if latest_path.is_symlink():
#                 return os.readlink(latest_path)
#             else:
#                 raise ValueError("Latest pointer is not a symlink on Linux")
    
#     def load_latest_artifacts(self) -> Dict:
#         """
#         Load artifacts from latest version
#         """
#         latest_version = self.get_latest_version()
#         return self.load_artifacts(latest_version)
    
#     def load_artifacts(self, version: str) -> Dict:
#         """
#         Load all artifacts for a specific version
#         """
#         version_dir = self.base_dir / version
        
#         if not version_dir.exists():
#             raise FileNotFoundError(f"Version not found: {version}")
        
#         print(f"\n📦 Loading artifacts from version: {version}")
        
#         artifacts = {}
        
#         # Load embeddings
#         with open(version_dir / 'embeddings.pkl', 'rb') as f:
#             artifacts['embeddings'] = pickle.load(f)
        
#         # Load FAISS index
#         artifacts['faiss_index'] = faiss.read_index(str(version_dir / 'faiss_index.bin'))
        
#         with open(version_dir / 'faiss_post_ids.pkl', 'rb') as f:
#             artifacts['faiss_post_ids'] = pickle.load(f)
        
#         # Load CF model
#         with open(version_dir / 'cf_model.pkl', 'rb') as f:
#             artifacts['cf_model'] = pickle.load(f)
        
#         # Load ranking model
#         import lightgbm as lgb
#         artifacts['ranking_model'] = lgb.Booster(
#             model_file=str(version_dir / 'ranking_model.txt')
#         )
        
#         with open(version_dir / 'ranking_scaler.pkl', 'rb') as f:
#             artifacts['ranking_scaler'] = pickle.load(f)
        
#         with open(version_dir / 'ranking_feature_cols.pkl', 'rb') as f:
#             artifacts['ranking_feature_cols'] = pickle.load(f)
        
#         # Load stats
#         with open(version_dir / 'user_stats.pkl', 'rb') as f:
#             artifacts['user_stats'] = pickle.load(f)
        
#         with open(version_dir / 'author_stats.pkl', 'rb') as f:
#             artifacts['author_stats'] = pickle.load(f)
        
#         with open(version_dir / 'following_dict.pkl', 'rb') as f:
#             artifacts['following_dict'] = pickle.load(f)
        
#         # Load metadata
#         with open(version_dir / 'metadata.json', 'r') as f:
#             artifacts['metadata'] = json.load(f)
        
#         print(f"✅ Artifacts loaded successfully")
        
#         return artifacts
    
#     def _build_faiss_index(self, post_embeddings: Dict) -> tuple:
#         """Build FAISS index for fast similarity search"""
#         import numpy as np
        
#         post_ids = list(post_embeddings.keys())
#         embeddings_matrix = np.array([post_embeddings[pid] for pid in post_ids])
        
#         # Normalize for cosine similarity
#         faiss.normalize_L2(embeddings_matrix)
        
#         # Build index
#         dimension = embeddings_matrix.shape[1]
#         index = faiss.IndexFlatIP(dimension)  # Inner Product
#         index.add(embeddings_matrix.astype('float32'))
        
#         print(f"      FAISS index built: {index.ntotal} vectors, {dimension} dims")
        
#         return index, np.array(post_ids)
    
#     def list_versions(self) -> list:
#         """List all versions sorted by date (newest first)"""
#         versions = [
#             d.name for d in self.base_dir.iterdir()
#             if d.is_dir() and d.name.startswith('v')
#         ]
#         return sorted(versions, reverse=True)
    
#     def cleanup_old_versions(self, keep_n: int = 5):
#         """Delete old versions, keep last N"""
#         versions = self.list_versions()
        
#         if len(versions) <= keep_n:
#             print(f"Only {len(versions)} versions, nothing to delete")
#             return
        
#         to_delete = versions[keep_n:]
        
#         print(f"\n🗑️  Cleaning up old versions (keeping last {keep_n}):")
#         for version in to_delete:
#             version_dir = self.base_dir / version
#             print(f"   Deleting: {version}")
#             shutil.rmtree(version_dir)
        
#         print(f"✅ Cleanup complete")
    
#     def get_version_info(self, version: str = None) -> Dict:
#         """Get metadata for a specific version or latest"""
#         if version is None:
#             version = self.get_latest_version()
        
#         metadata_path = self.base_dir / version / 'metadata.json'
        
#         if metadata_path.exists():
#             with open(metadata_path, 'r') as f:
#                 return json.load(f)
        
#         return {}



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
#     """
#     Windows-safe fallback: write pointer file 'latest.version' with version name.
#     """
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
#     """
#     Create/refresh symlink 'latest' -> version_dir.
#     On Windows this usually requires admin privileges; we fallback to pointer file.
#     """
#     latest_link = Path(artifacts_base_dir) / "latest"
#     try:
#         if latest_link.exists() or latest_link.is_symlink():
#             # Xoá link/thư mục cũ
#             if latest_link.is_dir() and not latest_link.is_symlink():
#                 shutil.rmtree(latest_link)
#             else:
#                 latest_link.unlink(missing_ok=True)
#         latest_link.symlink_to(version_dir, target_is_directory=True)
#     except Exception:
#         # Fallback pointer file cho Windows hoặc môi trường hạn chế
#         write_latest_pointer(artifacts_base_dir, version_dir.name)


# def save_artifacts(
#     version_name: str,
#     model: Any,
#     meta: Optional[Dict[str, Any]] = None,
#     artifacts_base_dir: str = "models",
#     extra_files: Optional[Dict[str, bytes]] = None,
# ) -> Path:
#     """
#     Save model + meta into models/{version_name}/
#       - model.pkl
#       - meta.json (nếu có)
#       - extra_files (dict: relative_path -> bytes)
#     Cập nhật 'latest' (Linux: symlink; Windows: pointer file nếu symlink fail).
#     """
#     import json

#     base = Path(artifacts_base_dir)
#     version_dir = base / version_name
#     _ensure_dir(version_dir)

#     # Save model
#     model_path = version_dir / "model.pkl"
#     with model_path.open("wb") as f:
#         pickle.dump(model, f)

#     # Save meta
#     if meta is not None:
#         meta_path = version_dir / "meta.json"
#         meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

#     # Extra files
#     if extra_files:
#         for rel, content in extra_files.items():
#             out_path = version_dir / rel
#             _ensure_dir(out_path.parent)
#             with out_path.open("wb") as f:
#                 f.write(content)

#     # Update latest
#     if _is_windows():
#         try:
#             _symlink_latest(artifacts_base_dir, version_dir)
#         except Exception:
#             write_latest_pointer(artifacts_base_dir, version_name)
#     else:
#         _symlink_latest(artifacts_base_dir, version_dir)

#     return version_dir


# def get_latest_version_dir(artifacts_base_dir: str = "models") -> Optional[Path]:
#     """
#     Resolve latest version directory, supporting both symlink and pointer file.
#     """
#     base = Path(artifacts_base_dir)
#     link = base / "latest"
#     if link.is_symlink():
#         try:
#             target = link.resolve(strict=True)
#             return target if target.exists() else None
#         except Exception:
#             pass

#     # Fallback pointer
#     name = read_latest_pointer(artifacts_base_dir)
#     if not name:
#         return None
#     candidate = base / name
#     return candidate if candidate.exists() else None


# # ---------------------------------------------------------------------------
# # Backward-compatible OO wrapper (để code cũ import ArtifactManager vẫn chạy)
# # ---------------------------------------------------------------------------
# class ArtifactManager:
#     """
#     Wrapper class cho API dựa trên các hàm ở trên.
#     Giữ code cũ: from recommender.offline.artifact_manager import ArtifactManager
#     """

#     def __init__(self, artifacts_base_dir: str = "models"):
#         self.artifacts_base_dir = artifacts_base_dir

#     # Một số code cũ gọi .save_artifacts(...)
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

#     # Một số code khác có thể gọi .save(...)
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

#     # Cho phép cập nhật "latest" thủ công nếu cần
#     def write_latest_pointer(self, version_name: str) -> None:
#         write_latest_pointer(self.artifacts_base_dir, version_name)



# recommender/offline/artifact_manager.py
from __future__ import annotations

import platform
import pickle
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

def _is_windows() -> bool:
    return platform.system().lower().startswith("win")

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def write_latest_pointer(artifacts_base_dir: str, version_name: str) -> None:
    base = Path(artifacts_base_dir)
    _ensure_dir(base)
    pointer = base / "latest.version"
    pointer.write_text(version_name, encoding="utf-8")

def read_latest_pointer(artifacts_base_dir: str) -> Optional[str]:
    pointer = Path(artifacts_base_dir) / "latest.version"
    if not pointer.exists():
        return None
    return pointer.read_text(encoding="utf-8").strip()

def _symlink_latest(artifacts_base_dir: str, version_dir: Path) -> None:
    latest_link = Path(artifacts_base_dir) / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            if latest_link.is_dir() and not latest_link.is_symlink():
                shutil.rmtree(latest_link)
            else:
                latest_link.unlink(missing_ok=True)
        latest_link.symlink_to(version_dir, target_is_directory=True)
    except Exception:
        write_latest_pointer(artifacts_base_dir, version_dir.name)

def save_artifacts(
    version_name: str,
    model: Any,
    meta: Optional[Dict[str, Any]] = None,
    artifacts_base_dir: str = "models",
    extra_files: Optional[Dict[str, bytes]] = None,
) -> Path:
    """
    Save model + meta vào models/{version}/
      - model.pkl
      - meta.json
      - extra_files (tuỳ chọn)
    Cập nhật 'latest' (Linux: symlink; Windows: pointer file).
    """
    import json

    base = Path(artifacts_base_dir)
    version_dir = base / version_name
    _ensure_dir(version_dir)

    model_path = version_dir / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    if meta is not None:
        meta_path = version_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if extra_files:
        for rel, content in extra_files.items():
            out_path = version_dir / rel
            _ensure_dir(out_path.parent)
            with out_path.open("wb") as f:
                f.write(content)

    if _is_windows():
        try:
            _symlink_latest(artifacts_base_dir, version_dir)
        except Exception:
            write_latest_pointer(artifacts_base_dir, version_name)
    else:
        _symlink_latest(artifacts_base_dir, version_dir)

    return version_dir

def get_latest_version_dir(artifacts_base_dir: str = "models") -> Optional[Path]:
    base = Path(artifacts_base_dir)
    link = base / "latest"
    if link.is_symlink():
        try:
            target = link.resolve(strict=True)
            return target if target.exists() else None
        except Exception:
            pass

    name = read_latest_pointer(artifacts_base_dir)
    if not name:
        return None
    candidate = base / name
    return candidate if candidate.exists() else None

class ArtifactManager:
    def __init__(self, artifacts_base_dir: str = "models"):
        self.artifacts_base_dir = artifacts_base_dir

    def save_artifacts(
        self,
        version_name: str,
        model: Any,
        meta: Optional[Dict[str, Any]] = None,
        extra_files: Optional[Dict[str, bytes]] = None,
    ) -> Path:
        return save_artifacts(
            version_name=version_name,
            model=model,
            meta=meta,
            artifacts_base_dir=self.artifacts_base_dir,
            extra_files=extra_files,
        )

    def save(
        self,
        version_name: str,
        model: Any,
        meta: Optional[Dict[str, Any]] = None,
        extra_files: Optional[Dict[str, bytes]] = None,
    ) -> Path:
        return self.save_artifacts(version_name, model, meta, extra_files)

    def get_latest_version_dir(self) -> Optional[Path]:
        return get_latest_version_dir(self.artifacts_base_dir)

    def write_latest_pointer(self, version_name: str) -> None:
        write_latest_pointer(self.artifacts_base_dir, version_name)
