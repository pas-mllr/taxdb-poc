"""
Database connection and session management for the TaxDB-POC application.

This module provides utilities for connecting to the database and managing sessions
with proper connection pooling and error handling.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any, Callable, TypeVar, cast

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, DBAPIError

from src import settings
from src.models import Base

# Configure logging
logger = logging.getLogger("taxdb.db")

# Type variables
T = TypeVar('T')

# Convert PostgreSQL connection string to async format
def get_async_connection_string(conn_str: str) -> str:
    """
    Convert a synchronous PostgreSQL connection string to an async one.
    
    Args:
        conn_str: Synchronous PostgreSQL connection string
        
    Returns:
        Async PostgreSQL connection string
    """
    if conn_str.startswith("postgresql://"):
        return conn_str.replace("postgresql://", "postgresql+asyncpg://")
    return conn_str


class DatabaseManager:
    """
    Database connection manager for the TaxDB-POC application.
    
    This class manages database connections and sessions with proper connection
    pooling and error handling.
    """
    
    _instance: Optional["DatabaseManager"] = None
    _engine: Optional[AsyncEngine] = None
    _sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None
    
    def __new__(cls) -> "DatabaseManager":
        """
        Singleton pattern to ensure only one database manager instance exists.
        
        Returns:
            DatabaseManager instance
        """
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize the database manager."""
        # Convert connection string to async format
        async_conn_str = get_async_connection_string(settings.PG_CONNSTR)
        
        # Configure connection pool settings
        pool_size = 5
        max_overflow = 10
        pool_timeout = 30
        pool_recycle = 1800  # 30 minutes
        
        # Create async engine with connection pooling
        self._engine = create_async_engine(
            async_conn_str,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            pool_pre_ping=True,  # Check connection validity before using
            echo=False,  # Set to True for debugging SQL queries
        )
        
        # Create session factory
        self._sessionmaker = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        
        # Log initialization
        logger.info(f"Database manager initialized with connection pool (size={pool_size}, max_overflow={max_overflow})")
    
    @property
    def engine(self) -> AsyncEngine:
        """
        Get the SQLAlchemy async engine.
        
        Returns:
            AsyncEngine instance
        """
        if self._engine is None:
            self._initialize()
        return cast(AsyncEngine, self._engine)
    
    @property
    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        """
        Get the SQLAlchemy async session maker.
        
        Returns:
            async_sessionmaker instance
        """
        if self._sessionmaker is None:
            self._initialize()
        return cast(async_sessionmaker[AsyncSession], self._sessionmaker)
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session as an async context manager.
        
        Yields:
            AsyncSession: Database session
            
        Example:
            ```python
            async with db_manager.session() as session:
                result = await session.execute(select(Document))
                documents = result.scalars().all()
            ```
        """
        session = self.sessionmaker()
        try:
            yield session
        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            await session.close()
    
    async def create_tables(self) -> None:
        """
        Create all tables defined in the models.
        
        This method should be called during application startup.
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created")
    
    async def close(self) -> None:
        """
        Close the database connection pool.
        
        This method should be called during application shutdown.
        """
        if self._engine is not None:
            await self._engine.dispose()
            logger.info("Database connection pool closed")
            self._engine = None
            self._sessionmaker = None


# Create a global database manager instance
db_manager = DatabaseManager()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session as an async generator.
    
    This function is intended to be used as a FastAPI dependency.
    
    Yields:
        AsyncSession: Database session
        
    Example:
        ```python
        @app.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            result = await session.execute(select(Item))
            return result.scalars().all()
        ```
    """
    async with db_manager.session() as session:
        yield session


async def execute_with_retry(
    session: AsyncSession,
    operation: Callable[[], T],
    max_retries: int = 3
) -> T:
    """
    Execute a database operation with retry logic.
    
    Args:
        session: Database session
        operation: Callable that performs the database operation
        max_retries: Maximum number of retries
        
    Returns:
        Result of the operation
        
    Raises:
        SQLAlchemyError: If the operation fails after all retries
    """
    retries = 0
    last_error: Exception = SQLAlchemyError("Unknown database error")
    
    while retries <= max_retries:
        try:
            return operation()
        except DBAPIError as e:
            # Only retry on connection-related errors
            if e.connection_invalidated:
                retries += 1
                last_error = e
                logger.warning(f"Database connection error, retrying ({retries}/{max_retries}): {str(e)}")
                await session.rollback()
                continue
            raise
        except SQLAlchemyError as e:
            # Don't retry on other SQLAlchemy errors
            await session.rollback()
            raise
    
    # If we get here, all retries failed
    logger.error(f"Database operation failed after {max_retries} retries: {str(last_error)}")
    raise last_error