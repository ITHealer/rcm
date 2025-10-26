# recommender/offline/training_state.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class TrainingState:
    last_train_end: Optional[str] = None  # ISO 8601 UTC string


class TrainingStateManager:
    def __init__(self, state_path: str):
        self.path = Path(state_path)

    def load(self) -> TrainingState:
        if not self.path.exists():
            return TrainingState(last_train_end=None)
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return TrainingState(**data)

    def save(self, state: TrainingState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(state.__dict__, f, ensure_ascii=False, indent=2)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
