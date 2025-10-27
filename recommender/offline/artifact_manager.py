

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
