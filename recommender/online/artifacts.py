# recommender/online/artifacts.py
from __future__ import annotations
import json, pickle
from pathlib import Path
from typing import Tuple, List, Any, Optional
import lightgbm as lgb

def _latest_dir(base_dir: str = "models") -> Path:
    base = Path(base_dir)
    link = base / "latest"
    if link.is_symlink():
        try:
            return link.resolve(strict=True)
        except Exception:
            pass
    ptr = base / "latest.version"
    if ptr.exists():
        p = base / ptr.read_text(encoding="utf-8").strip()
        if p.exists():
            return p
    # fallback: newest vYYYYMMDD_*
    vers = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("v")]
    if not vers:
        raise FileNotFoundError(f"No artifacts in {base_dir}")
    return max(vers, key=lambda p: p.name)

def _load_pickle(p: Path):
    with p.open("rb") as f:
        return pickle.load(f)

def load_ranker_artifacts(base_dir: str = "models") -> Tuple[Any, Any, List[str], dict, Path]:
    """
    Returns: (model, scaler, feature_cols, meta, version_dir)
    Hỗ trợ cả tên file cũ và mới.
    """
    ver = _latest_dir(base_dir)
    meta = {}
    mp = ver / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))

    # feature cols
    fcols = ver / "ranker_feature_cols.pkl"
    if not fcols.exists():
        fcols = ver / "feature_cols.pkl"
    feature_cols = _load_pickle(fcols)

    # scaler
    scaler = None
    sp = ver / "ranker_scaler.pkl"
    if not sp.exists():
        sp = ver / "scaler.pkl"
    if sp.exists():
        scaler = _load_pickle(sp)

    # model
    model = None
    txt = ver / "ranker_model.txt"
    if txt.exists():
        model = lgb.Booster(model_file=str(txt))
    else:
        pkl = ver / "model.pkl"
        if pkl.exists():
            model = _load_pickle(pkl)
        else:
            raise FileNotFoundError(f"No model file in {ver}")

    return model, scaler, feature_cols, meta, ver
