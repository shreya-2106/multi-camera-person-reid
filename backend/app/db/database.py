from __future__ import annotations

import os
import threading
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker


load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL", "").strip() or None

engine: Optional[Engine] = None
_engine_lock = threading.Lock()

SessionLocal = sessionmaker(
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    """Base class for all database models."""


def _create_mysql_engine(database_url: str) -> Engine:
    try:
        parsed_url = make_url(database_url)
    except Exception as exc:
        raise RuntimeError("DATABASE_URL is not a valid SQLAlchemy URL.") from exc
    if parsed_url.drivername != "mysql+pymysql":
        raise RuntimeError(
            "DATABASE_URL must use MySQL with PyMySQL, for example "
            "mysql+pymysql://USERNAME:PASSWORD@HOST:3306/multicamera_reid."
        )
    return create_engine(database_url, pool_pre_ping=True)


def get_engine() -> Engine:
    """Return the configured MySQL engine, requiring an explicit URL."""
    global DATABASE_URL, engine

    configured_url = os.getenv("DATABASE_URL", "").strip()
    if not configured_url:
        raise RuntimeError(
            "DATABASE_URL is required. Set it to a MySQL URL such as "
            "mysql+pymysql://USERNAME:PASSWORD@HOST:3306/multicamera_reid."
        )

    with _engine_lock:
        if engine is None or str(engine.url) != configured_url:
            if engine is not None:
                engine.dispose()
            engine = _create_mysql_engine(configured_url)
            SessionLocal.configure(bind=engine)
        DATABASE_URL = configured_url
        return engine


def get_db():
    """Provide a database session for FastAPI dependencies."""
    get_engine()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
