"""
Database layer — async PostgreSQL via SQLAlchemy 2.0 + asyncpg.

Provides:
    • Async engine and session factory
    • Dependency injection for FastAPI routes
    • Connection pool management
    • Base model for ORM entities

Usage:
    from backend.app.core.database import get_db, Base

    class Disaster(Base):
        __tablename__ = "disasters"
        id = Column(Integer, primary_key=True)

    @router.get("/disasters")
    async def list_disasters(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(Disaster))
        return result.scalars().all()
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


# ── Engine ──
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    echo=settings.DATABASE_ECHO,
    future=True,
)

# ── Session Factory ──
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── ORM Base ──
class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# ── Dependency ──
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Lifecycle ──
async def init_db() -> None:
    """Create all tables (dev/test only — use Alembic in production)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialised")


async def close_db() -> None:
    """Dispose engine connections."""
    await engine.dispose()
    logger.info("Database connections closed")
