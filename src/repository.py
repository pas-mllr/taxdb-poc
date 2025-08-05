"""
Repository classes for document storage and retrieval.

This module provides repository classes for interacting with the database,
including vector similarity search functionality.
"""

import logging
from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union, cast

from pgvector.sqlalchemy import Vector
from sqlalchemy import func, select, or_, and_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql.expression import Select

from src.db import db_manager, execute_with_retry
from src.models import Document, Base

# Configure logging
logger = logging.getLogger("taxdb.repository")

# Type variables
T = TypeVar('T', bound=Base)
ModelType = TypeVar('ModelType', bound=Base)


class RepositoryError(Exception):
    """Base exception for repository-related errors."""
    pass


class EntityNotFoundError(RepositoryError):
    """Exception raised when an entity is not found."""
    pass


class PaginationParams:
    """Parameters for pagination."""
    
    def __init__(self, page: int = 1, page_size: int = 10):
        """
        Initialize pagination parameters.
        
        Args:
            page: Page number (1-based)
            page_size: Number of items per page
        """
        self.page = max(1, page)  # Ensure page is at least 1
        self.page_size = min(max(1, page_size), 100)  # Ensure page_size is between 1 and 100
    
    @property
    def offset(self) -> int:
        """
        Get the offset for pagination.
        
        Returns:
            Offset value
        """
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        """
        Get the limit for pagination.
        
        Returns:
            Limit value
        """
        return self.page_size


class SortParams:
    """Parameters for sorting."""
    
    def __init__(self, sort_by: Optional[str] = None, sort_order: str = "desc"):
        """
        Initialize sort parameters.
        
        Args:
            sort_by: Field to sort by
            sort_order: Sort order ("asc" or "desc")
        """
        self.sort_by = sort_by
        self.sort_order = sort_order.lower()
        
        # Validate sort_order
        if self.sort_order not in ["asc", "desc"]:
            self.sort_order = "desc"


class BaseRepository(Generic[T], ABC):
    """
    Base repository class for database operations.
    
    This class provides common CRUD operations for database entities.
    """
    
    def __init__(self, model: Type[T]):
        """
        Initialize the repository.
        
        Args:
            model: SQLAlchemy model class
        """
        self.model = model
    
    async def get_by_id(self, session: AsyncSession, id: Any) -> T:
        """
        Get an entity by ID.
        
        Args:
            session: Database session
            id: Entity ID
            
        Returns:
            Entity instance
            
        Raises:
            EntityNotFoundError: If the entity is not found
        """
        stmt = select(self.model).where(self.model.id == id)
        result = await session.execute(stmt)
        entity = result.scalars().first()
        
        if entity is None:
            raise EntityNotFoundError(f"{self.model.__name__} with ID {id} not found")
        
        return entity
    
    async def get_all(
        self,
        session: AsyncSession,
        pagination: Optional[PaginationParams] = None,
        sort: Optional[SortParams] = None
    ) -> Tuple[List[T], int]:
        """
        Get all entities with pagination and sorting.
        
        Args:
            session: Database session
            pagination: Pagination parameters
            sort: Sort parameters
            
        Returns:
            Tuple of (entities, total_count)
        """
        # Create base query
        stmt = select(self.model)
        
        # Apply sorting if provided
        if sort and sort.sort_by:
            try:
                sort_column = getattr(self.model, sort.sort_by)
                if sort.sort_order == "asc":
                    stmt = stmt.order_by(asc(sort_column))
                else:
                    stmt = stmt.order_by(desc(sort_column))
            except AttributeError:
                logger.warning(f"Sort column '{sort.sort_by}' not found in {self.model.__name__}")
        
        # Get total count
        count_stmt = select(func.count()).select_from(self.model)
        result = await session.execute(count_stmt)
        total_count = result.scalar() or 0
        
        # Apply pagination if provided
        if pagination:
            stmt = stmt.offset(pagination.offset).limit(pagination.limit)
        
        # Execute query
        result = await session.execute(stmt)
        entities = result.scalars().all()
        
        return list(entities), total_count
    
    async def create(self, session: AsyncSession, data: Dict[str, Any]) -> T:
        """
        Create a new entity.
        
        Args:
            session: Database session
            data: Entity data
            
        Returns:
            Created entity
        """
        entity = self.model(**data)
        session.add(entity)
        await session.flush()
        await session.refresh(entity)
        return entity
    
    async def update(self, session: AsyncSession, id: Any, data: Dict[str, Any]) -> T:
        """
        Update an existing entity.
        
        Args:
            session: Database session
            id: Entity ID
            data: Updated entity data
            
        Returns:
            Updated entity
            
        Raises:
            EntityNotFoundError: If the entity is not found
        """
        entity = await self.get_by_id(session, id)
        
        for key, value in data.items():
            if hasattr(entity, key):
                setattr(entity, key, value)
        
        await session.flush()
        await session.refresh(entity)
        return entity
    
    async def delete(self, session: AsyncSession, id: Any) -> None:
        """
        Delete an entity.
        
        Args:
            session: Database session
            id: Entity ID
            
        Raises:
            EntityNotFoundError: If the entity is not found
        """
        entity = await self.get_by_id(session, id)
        await session.delete(entity)
        await session.flush()


class DocumentRepository(BaseRepository[Document]):
    """
    Repository for document operations.
    
    This class provides methods for document storage, retrieval, and vector similarity search.
    """
    
    def __init__(self):
        """Initialize the document repository."""
        super().__init__(Document)
    
    async def search_by_text(
        self,
        session: AsyncSession,
        query: str,
        jurisdiction: Optional[str] = None,
        pagination: Optional[PaginationParams] = None,
        sort: Optional[SortParams] = None
    ) -> Tuple[List[Document], int]:
        """
        Search documents by text.
        
        Args:
            session: Database session
            query: Search query
            jurisdiction: Optional jurisdiction filter
            pagination: Pagination parameters
            sort: Sort parameters
            
        Returns:
            Tuple of (documents, total_count)
        """
        # Create base query
        stmt = select(Document)
        
        # Apply text search
        stmt = stmt.filter(
            or_(
                Document.title.ilike(f"%{query}%"),
                Document.summary.ilike(f"%{query}%")
            )
        )
        
        # Apply jurisdiction filter if provided
        if jurisdiction:
            stmt = stmt.filter(Document.jurisdiction == jurisdiction)
        
        # Get total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        result = await session.execute(count_stmt)
        total_count = result.scalar() or 0
        
        # Apply sorting if provided
        if sort and sort.sort_by:
            try:
                sort_column = getattr(Document, sort.sort_by)
                if sort.sort_order == "asc":
                    stmt = stmt.order_by(asc(sort_column))
                else:
                    stmt = stmt.order_by(desc(sort_column))
            except AttributeError:
                logger.warning(f"Sort column '{sort.sort_by}' not found in Document")
                # Default to sorting by issue_date
                stmt = stmt.order_by(desc(Document.issue_date))
        else:
            # Default to sorting by issue_date
            stmt = stmt.order_by(desc(Document.issue_date))
        
        # Apply pagination if provided
        if pagination:
            stmt = stmt.offset(pagination.offset).limit(pagination.limit)
        
        # Execute query
        result = await session.execute(stmt)
        documents = result.scalars().all()
        
        return list(documents), total_count
    
    async def search_by_vector(
        self,
        session: AsyncSession,
        query_vector: List[float],
        jurisdiction: Optional[str] = None,
        pagination: Optional[PaginationParams] = None,
        max_distance: Optional[float] = None
    ) -> Tuple[List[Tuple[Document, float]], int]:
        """
        Search documents by vector similarity.
        
        Args:
            session: Database session
            query_vector: Query vector
            jurisdiction: Optional jurisdiction filter
            pagination: Pagination parameters
            max_distance: Maximum distance threshold
            
        Returns:
            Tuple of (documents with distances, total_count)
        """
        # Convert query vector to PostgreSQL vector
        vector = Vector(query_vector)
        
        # Create base query with distance calculation
        stmt = select(
            Document,
            func.l2_distance(Document.vector, vector).label("distance")
        ).filter(Document.vector.is_not(None))
        
        # Apply jurisdiction filter if provided
        if jurisdiction:
            stmt = stmt.filter(Document.jurisdiction == jurisdiction)
        
        # Apply distance threshold if provided
        if max_distance is not None:
            stmt = stmt.filter(func.l2_distance(Document.vector, vector) < max_distance)
        
        # Order by distance (closest first)
        stmt = stmt.order_by(func.l2_distance(Document.vector, vector))
        
        # Get total count
        count_stmt = select(func.count()).select_from(
            select(Document.id).filter(Document.vector.is_not(None))
        )
        if jurisdiction:
            count_stmt = count_stmt.filter(Document.jurisdiction == jurisdiction)
        if max_distance is not None:
            count_stmt = count_stmt.filter(func.l2_distance(Document.vector, vector) < max_distance)
        
        result = await session.execute(count_stmt)
        total_count = result.scalar() or 0
        
        # Apply pagination if provided
        if pagination:
            stmt = stmt.offset(pagination.offset).limit(pagination.limit)
        
        # Execute query
        result = await session.execute(stmt)
        documents_with_distances = [(doc, distance) for doc, distance in result.all()]
        
        return documents_with_distances, total_count
    
    async def search_hybrid(
        self,
        session: AsyncSession,
        query: str,
        query_vector: List[float],
        jurisdiction: Optional[str] = None,
        pagination: Optional[PaginationParams] = None,
        vector_weight: float = 0.7,
        text_weight: float = 0.3
    ) -> Tuple[List[Tuple[Document, float]], int]:
        """
        Perform hybrid search combining vector similarity and text search.
        
        Args:
            session: Database session
            query: Text query
            query_vector: Query vector
            jurisdiction: Optional jurisdiction filter
            pagination: Pagination parameters
            vector_weight: Weight for vector similarity (0.0 to 1.0)
            text_weight: Weight for text similarity (0.0 to 1.0)
            
        Returns:
            Tuple of (documents with scores, total_count)
        """
        # Normalize weights
        total_weight = vector_weight + text_weight
        vector_weight = vector_weight / total_weight
        text_weight = text_weight / total_weight
        
        # Convert query vector to PostgreSQL vector
        vector = Vector(query_vector)
        
        # Create text match expression
        text_match = func.greatest(
            func.similarity(Document.title, query),
            func.similarity(Document.summary, query) if Document.summary.is_not(None) else 0.0
        )
        
        # Create combined score expression
        # For vector: convert distance to similarity score (1 / (1 + distance))
        # For text: use text similarity directly
        combined_score = (
            vector_weight * (1 / (1 + func.l2_distance(Document.vector, vector))) +
            text_weight * text_match
        ).label("score")
        
        # Create base query
        stmt = select(
            Document,
            combined_score
        ).filter(Document.vector.is_not(None))
        
        # Apply jurisdiction filter if provided
        if jurisdiction:
            stmt = stmt.filter(Document.jurisdiction == jurisdiction)
        
        # Order by combined score (highest first)
        stmt = stmt.order_by(desc(combined_score))
        
        # Get total count
        count_stmt = select(func.count()).select_from(
            select(Document.id).filter(Document.vector.is_not(None))
        )
        if jurisdiction:
            count_stmt = count_stmt.filter(Document.jurisdiction == jurisdiction)
        
        result = await session.execute(count_stmt)
        total_count = result.scalar() or 0
        
        # Apply pagination if provided
        if pagination:
            stmt = stmt.offset(pagination.offset).limit(pagination.limit)
        
        # Execute query
        result = await session.execute(stmt)
        documents_with_scores = [(doc, float(score)) for doc, score in result.all()]
        
        return documents_with_scores, total_count
    
    async def filter_by_jurisdiction(
        self,
        session: AsyncSession,
        jurisdiction: str,
        pagination: Optional[PaginationParams] = None,
        sort: Optional[SortParams] = None
    ) -> Tuple[List[Document], int]:
        """
        Filter documents by jurisdiction.
        
        Args:
            session: Database session
            jurisdiction: Jurisdiction code
            pagination: Pagination parameters
            sort: Sort parameters
            
        Returns:
            Tuple of (documents, total_count)
        """
        # Create base query
        stmt = select(Document).filter(Document.jurisdiction == jurisdiction)
        
        # Get total count
        count_stmt = select(func.count()).select_from(Document).filter(Document.jurisdiction == jurisdiction)
        result = await session.execute(count_stmt)
        total_count = result.scalar() or 0
        
        # Apply sorting if provided
        if sort and sort.sort_by:
            try:
                sort_column = getattr(Document, sort.sort_by)
                if sort.sort_order == "asc":
                    stmt = stmt.order_by(asc(sort_column))
                else:
                    stmt = stmt.order_by(desc(sort_column))
            except AttributeError:
                logger.warning(f"Sort column '{sort.sort_by}' not found in Document")
                # Default to sorting by issue_date
                stmt = stmt.order_by(desc(Document.issue_date))
        else:
            # Default to sorting by issue_date
            stmt = stmt.order_by(desc(Document.issue_date))
        
        # Apply pagination if provided
        if pagination:
            stmt = stmt.offset(pagination.offset).limit(pagination.limit)
        
        # Execute query
        result = await session.execute(stmt)
        documents = result.scalars().all()
        
        return list(documents), total_count
    
    async def filter_by_date_range(
        self,
        session: AsyncSession,
        start_date: date,
        end_date: date,
        jurisdiction: Optional[str] = None,
        pagination: Optional[PaginationParams] = None,
        sort: Optional[SortParams] = None
    ) -> Tuple[List[Document], int]:
        """
        Filter documents by date range.
        
        Args:
            session: Database session
            start_date: Start date
            end_date: End date
            jurisdiction: Optional jurisdiction filter
            pagination: Pagination parameters
            sort: Sort parameters
            
        Returns:
            Tuple of (documents, total_count)
        """
        # Create base query
        stmt = select(Document).filter(
            and_(
                Document.issue_date >= start_date,
                Document.issue_date <= end_date
            )
        )
        
        # Apply jurisdiction filter if provided
        if jurisdiction:
            stmt = stmt.filter(Document.jurisdiction == jurisdiction)
        
        # Get total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        result = await session.execute(count_stmt)
        total_count = result.scalar() or 0
        
        # Apply sorting if provided
        if sort and sort.sort_by:
            try:
                sort_column = getattr(Document, sort.sort_by)
                if sort.sort_order == "asc":
                    stmt = stmt.order_by(asc(sort_column))
                else:
                    stmt = stmt.order_by(desc(sort_column))
            except AttributeError:
                logger.warning(f"Sort column '{sort.sort_by}' not found in Document")
                # Default to sorting by issue_date
                stmt = stmt.order_by(desc(Document.issue_date))
        else:
            # Default to sorting by issue_date
            stmt = stmt.order_by(desc(Document.issue_date))
        
        # Apply pagination if provided
        if pagination:
            stmt = stmt.offset(pagination.offset).limit(pagination.limit)
        
        # Execute query
        result = await session.execute(stmt)
        documents = result.scalars().all()
        
        return list(documents), total_count
    
    async def save_document(
        self,
        session: AsyncSession,
        document_data: Dict[str, Any]
    ) -> Document:
        """
        Save a document to the database.
        
        If the document already exists (by ID), it will be updated.
        Otherwise, a new document will be created.
        
        Args:
            session: Database session
            document_data: Document data
            
        Returns:
            Saved document
        """
        # Check if document already exists
        document_id = document_data.get("id")
        if document_id:
            try:
                existing_document = await self.get_by_id(session, document_id)
                # Update existing document
                return await self.update(session, document_id, document_data)
            except EntityNotFoundError:
                # Create new document
                return await self.create(session, document_data)
        else:
            # Create new document
            return await self.create(session, document_data)
    
    async def get_similar_documents(
        self,
        session: AsyncSession,
        document_id: str,
        limit: int = 5,
        jurisdiction: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """
        Get documents similar to a given document.
        
        Args:
            session: Database session
            document_id: Document ID
            limit: Maximum number of similar documents to return
            jurisdiction: Optional jurisdiction filter
            
        Returns:
            List of similar documents with similarity scores
            
        Raises:
            EntityNotFoundError: If the document is not found
        """
        # Get the source document
        source_document = await self.get_by_id(session, document_id)
        
        # Check if the document has a vector embedding
        if source_document.vector is None:
            return []
        
        # Create query to find similar documents
        vector = Vector(source_document.vector)
        
        stmt = select(
            Document,
            func.l2_distance(Document.vector, vector).label("distance")
        ).filter(
            and_(
                Document.id != document_id,
                Document.vector.is_not(None)
            )
        )
        
        # Apply jurisdiction filter if provided
        if jurisdiction:
            stmt = stmt.filter(Document.jurisdiction == jurisdiction)
        
        # Order by distance (closest first) and limit results
        stmt = stmt.order_by(func.l2_distance(Document.vector, vector)).limit(limit)
        
        # Execute query
        result = await session.execute(stmt)
        similar_documents = [(doc, distance) for doc, distance in result.all()]
        
        return similar_documents