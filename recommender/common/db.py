# recommender/common/db.py
from __future__ import annotations
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

_engine = None
_Session = None

def get_engine(mysql_url: str, **kw):
    global _engine
    if _engine is None:
        _engine = create_engine(mysql_url, future=True, **kw)
    return _engine

def get_session_factory(mysql_url: str, **kw) -> sessionmaker:
    global _Session
    if _Session is None:
        eng = get_engine(mysql_url, **kw)
        _Session = sessionmaker(bind=eng, autocommit=False, autoflush=False, future=True)
    return _Session

def create_sync_engine(db_url: str, pool_size: int = 5, max_overflow: int = 5) -> Engine:
    """
    Create a SQLAlchemy sync engine with sensible production defaults.
    """
    if not db_url:
        raise ValueError("Database URL is empty. Please set configs.database.url")
    eng = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        future=True,
    )
    return eng