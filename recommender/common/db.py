# recommender/common/db.py
from __future__ import annotations
from sqlalchemy import create_engine
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
