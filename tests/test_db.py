"""
Tests for database connection and session management.

This module tests the database connection functionality, including
connection pooling, session management, and retry logic.
"""

import os

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from sqlalchemy import text, select
from sqlalchemy.exc import SQLAlchemyError, DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import (
    db_manager,
    get_session,
    get_async_connection_string,
    execute_with_retry,
    DatabaseManager
)
from src.models import Document


@pytest.mark.asyncio
async def test_db_manager_singleton():
    """Test that DatabaseManager is a singleton."""
    # Get two instances of DatabaseManager
    manager1 = DatabaseManager()
    manager2 = DatabaseManager()
    
    # Check that they are the same instance
    assert manager1 is manager2
    assert id(manager1) == id(manager2)
    
    # Check that the engine is the same
    assert manager1.engine is manager2.engine
    assert id(manager1.engine) == id(manager2.engine)
    
    # Check that the sessionmaker is the same
    assert manager1.sessionmaker is manager2.sessionmaker
    assert id(manager1.sessionmaker) == id(manager2.sessionmaker)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.unit
async def test_get_async_connection_string():
    """Test conversion of PostgreSQL connection string to async format."""
    # Test with postgresql:// prefix
    sync_conn_str = "postgresql://user:pass@localhost:5432/db"
    async_conn_str = get_async_connection_string(sync_conn_str)
    assert async_conn_str == "postgresql+asyncpg://user:pass@localhost:5432/db"
    
    # Test with already async connection string
    async_conn_str = "postgresql+asyncpg://user:pass@localhost:5432/db"
    result = get_async_connection_string(async_conn_str)
    assert result == async_conn_str
    
    # Test with other format
    other_conn_str = "sqlite:///test.db"
    result = get_async_connection_string(other_conn_str)
    assert result == other_conn_str


@pytest.mark.asyncio
async def test_db_manager_session(async_session):
    """Test database manager session context manager."""
    async with db_manager.session() as session:
        # Simple query to test connection
        result = await session.execute(select(1))
        assert result.scalar() == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_get_session():
    """Test get_session dependency."""
    # Use get_session as an async generator
    session_gen = get_session()
    session = await session_gen.__anext__()
    
    try:
        # Test that session is an AsyncSession
        assert isinstance(session, AsyncSession)
        
        # Test that session can execute queries
        result = await session.execute(select(1))
        assert result.scalar() == 1
    finally:
        # Clean up
        try:
            await session_gen.__anext__()
        except StopAsyncIteration:
            pass


@pytest.mark.asyncio
@pytest.mark.unit
async def test_db_manager_create_tables():
    """Test database manager create_tables method."""
    # Mock the connection and run_sync method
    with patch.object(db_manager.engine, 'begin') as mock_begin:
        mock_conn = AsyncMock()
        mock_begin.return_value.__aenter__.return_value = mock_conn
        
        # Call create_tables
        await db_manager.create_tables()
        
        # Check that run_sync was called with Base.metadata.create_all
        mock_conn.run_sync.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.unit
async def test_db_manager_close():
    """Test database manager close method."""
    # Create a new instance with a mock engine
    manager = DatabaseManager()
    manager._engine = MagicMock()
    manager._engine.dispose = AsyncMock()
    
    # Call close
    await manager.close()
    
    # Check that dispose was called
    manager._engine.dispose.assert_called_once()
    
    # Check that engine and sessionmaker are None
    assert manager._engine is None
    assert manager._sessionmaker is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_execute_with_retry_success(async_session):
    """Test execute_with_retry with successful operation."""
    # Define a successful operation
    async def successful_operation():
        return "success"
    
    # Execute with retry
    result = await execute_with_retry(async_session, successful_operation)
    
    # Check result
    assert result == "success"


@pytest.mark.asyncio
async def test_execute_with_retry_non_connection_error(async_session):
    """Test execute_with_retry with non-connection error."""
    # Define an operation that raises a non-connection SQLAlchemyError
    async def failing_operation():
        raise SQLAlchemyError("Test error")
    
    # Execute with retry and expect exception
    with pytest.raises(SQLAlchemyError, match="Test error"):
        await execute_with_retry(async_session, failing_operation)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_execute_with_retry_connection_error(async_session):
    """Test execute_with_retry with connection error."""
    # Create a DBAPIError with connection_invalidated=True
    error = DBAPIError("Test error", None, None, None)
    error.connection_invalidated = True
    
    # Define an operation that raises the error
    call_count = 0
    async def failing_operation():
        nonlocal call_count
        call_count += 1
        if call_count < 3:  # Fail twice, succeed on third try
            raise error
        return "success after retry"
    
    # Mock session.rollback
    async_session.rollback = AsyncMock()
    
    # Execute with retry
    result = await execute_with_retry(async_session, failing_operation, max_retries=3)
    
    # Check result and rollback calls
    assert result == "success after retry"
    assert call_count == 3
    assert async_session.rollback.call_count == 2


@pytest.mark.asyncio
@pytest.mark.integration
async def test_execute_with_retry_max_retries_exceeded(async_session):
    """Test execute_with_retry with max retries exceeded."""
    # Create a DBAPIError with connection_invalidated=True
    error = DBAPIError("Test error", None, None, None)
    error.connection_invalidated = True
    
    # Define an operation that always raises the error
    async def failing_operation():
        raise error
    
    # Mock session.rollback
    async_session.rollback = AsyncMock()
    
    # Execute with retry and expect exception
    with pytest.raises(DBAPIError, match="Test error"):
        await execute_with_retry(async_session, failing_operation, max_retries=2)
    
    # Check rollback calls (should be called max_retries times)
    assert async_session.rollback.call_count == 2


@pytest.mark.asyncio
@pytest.mark.integration
async def test_db_real_connection(async_engine):
    """Test real database connection."""
    # This test requires a real database connection
    async with async_engine.begin() as conn:
        # Test simple query
        result = await conn.execute(text("SELECT 1"))
        assert result.scalar() == 1
        
        # Test that we can create and query a table
        await conn.execute(text("CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY)"))
        await conn.execute(text("INSERT INTO test_table (id) VALUES (1), (2), (3)"))
        
        result = await conn.execute(text("SELECT COUNT(*) FROM test_table"))
        count = result.scalar()
        assert count >= 3  # May be more if table already existed
        
        # Clean up
        await conn.execute(text("DROP TABLE IF EXISTS test_table"))


@pytest.mark.asyncio
async def test_db_transaction_commit(async_session):
    """Test database transaction commit."""
    # Create a test table
    await async_session.execute(text("CREATE TABLE IF NOT EXISTS test_transaction (id INTEGER PRIMARY KEY)"))
    
    try:
        # Insert data and commit
        await async_session.execute(text("INSERT INTO test_transaction (id) VALUES (1)"))
        await async_session.commit()
        
        # Check that data was committed
        result = await async_session.execute(text("SELECT COUNT(*) FROM test_transaction"))
        assert result.scalar() == 1
    finally:
        # Clean up
        await async_session.execute(text("DROP TABLE IF EXISTS test_transaction"))
        await async_session.commit()


@pytest.mark.asyncio
async def test_db_transaction_rollback(async_session):
    """Test database transaction rollback."""
    # Create a test table
    await async_session.execute(text("CREATE TABLE IF NOT EXISTS test_transaction (id INTEGER PRIMARY KEY)"))
    await async_session.commit()
    
    try:
        # Insert data
        await async_session.execute(text("INSERT INTO test_transaction (id) VALUES (1)"))
        
        # Rollback
        await async_session.rollback()
        
        # Check that data was not committed
        result = await async_session.execute(text("SELECT COUNT(*) FROM test_transaction"))
        assert result.scalar() == 0
    finally:
        # Clean up
        await async_session.execute(text("DROP TABLE IF EXISTS test_transaction"))
        await async_session.commit()


@pytest.mark.asyncio
async def test_db_connection_pool(async_engine):
    """Test database connection pooling."""
    # Create multiple connections concurrently
    async def get_connection():
        async with async_engine.connect() as conn:
            # Simulate some work
            await asyncio.sleep(0.1)
            result = await conn.execute(text("SELECT 1"))
            return result.scalar()
    
    # Create 10 concurrent connections
    tasks = [get_connection() for _ in range(10)]
    results = await asyncio.gather(*tasks)
    
    # Check that all connections worked
    assert all(result == 1 for result in results)