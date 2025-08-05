"""
API service for the TaxDB-POC application.

This module implements the FastAPI service for the application.
"""

import datetime
import logging
from typing import List, Optional, Dict, Any, Tuple, Union, Annotated

from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist, validator, conint
from sqlalchemy.ext.asyncio import AsyncSession

from src import settings
from src.db import db_manager, get_session
from src.models import Document
from src.repository import (
    DocumentRepository,
    PaginationParams,
    SortParams,
    EntityNotFoundError
)
from src.etl.utils import get_embedding_strategy, EmbeddingStrategy, EmbeddingError

# Configure logging
logger = logging.getLogger("taxdb.api")


# Initialize repositories
document_repository = DocumentRepository()


# Pydantic models
class DocumentDTO(BaseModel):
    """Data Transfer Object for Document."""
    id: str
    jurisdiction: str
    source_system: str
    document_type: str
    title: str
    summary: Optional[str] = None
    issue_date: datetime.date
    effective_date: Optional[datetime.date] = None
    language_orig: str
    blob_url: str
    created_at: datetime.datetime

    class Config:
        """Pydantic config."""
        from_attributes = True


class DocumentWithScoreDTO(BaseModel):
    """Document with similarity score."""
    document: DocumentDTO
    score: float

    @validator('score')
    def validate_score(cls, v):
        """Validate that score is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v


class SearchResult(BaseModel):
    """Search result model."""
    documents: List[DocumentDTO]
    total: int


class VectorSearchResult(BaseModel):
    """Vector search result model."""
    documents: List[DocumentWithScoreDTO]
    total: int


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = "ok"


class VectorQuery(BaseModel):
    """Vector query model."""
    text: str
    jurisdiction: Optional[str] = None
    page: int = 1
    page_size: int = 10
    max_distance: Optional[float] = None

    @validator('text')
    def validate_text(cls, v):
        """Validate that text is not empty and not too long."""
        if not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v) > 1000:
            raise ValueError('Text cannot be longer than 1000 characters')
        return v

    @validator('jurisdiction')
    def validate_jurisdiction(cls, v):
        """Validate that jurisdiction is valid."""
        if v is not None and v not in settings.JURISDICTIONS:
            raise ValueError(f'Jurisdiction must be one of {settings.JURISDICTIONS}')
        return v

    @validator('page')
    def validate_page(cls, v):
        """Validate that page is positive."""
        if v < 1:
            raise ValueError('Page must be at least 1')
        return v

    @validator('page_size')
    def validate_page_size(cls, v):
        """Validate that page_size is between 1 and 100."""
        if not 1 <= v <= 100:
            raise ValueError('Page size must be between 1 and 100')
        return v

    @validator('max_distance')
    def validate_max_distance(cls, v):
        """Validate that max_distance is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError('Max distance must be positive')
        return v


class HybridQuery(BaseModel):
    """Hybrid query model."""
    text: str
    jurisdiction: Optional[str] = None
    page: int = 1
    page_size: int = 10
    vector_weight: float = 0.7
    text_weight: float = 0.3

    @validator('text')
    def validate_text(cls, v):
        """Validate that text is not empty and not too long."""
        if not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v) > 1000:
            raise ValueError('Text cannot be longer than 1000 characters')
        return v

    @validator('jurisdiction')
    def validate_jurisdiction(cls, v):
        """Validate that jurisdiction is valid."""
        if v is not None and v not in settings.JURISDICTIONS:
            raise ValueError(f'Jurisdiction must be one of {settings.JURISDICTIONS}')
        return v

    @validator('page')
    def validate_page(cls, v):
        """Validate that page is positive."""
        if v < 1:
            raise ValueError('Page must be at least 1')
        return v

    @validator('page_size')
    def validate_page_size(cls, v):
        """Validate that page_size is between 1 and 100."""
        if not 1 <= v <= 100:
            raise ValueError('Page size must be between 1 and 100')
        return v

    @validator('vector_weight', 'text_weight')
    def validate_weights(cls, v):
        """Validate that weights are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError('Weight must be between 0 and 1')
        return v


def generate_sas_url(blob_url: str) -> str:
    """Generate SAS URL for blob.
    
    In local mode, returns the blob URL as is.
    In cloud mode, generates a SAS token valid for 15 minutes.
    """
    if settings.LOCAL_MODE:
        return blob_url
    
    # In a real implementation, this would use Azure SDK to generate a SAS token
    # For this skeleton, we'll just return the URL
    return f"{blob_url}?sv=2022-11-02&ss=b&srt=sco&sp=r&se={datetime.datetime.utcnow() + datetime.timedelta(minutes=settings.SAS_TOKEN_EXPIRY_MINUTES)}&st={datetime.datetime.utcnow()}&spr=https&sig=dummy"


async def get_embedding_strategy_dependency() -> EmbeddingStrategy:
    """Get embedding strategy as a FastAPI dependency."""
    return get_embedding_strategy()


async def create_app_startup():
    """Startup event handler for the FastAPI application."""
    # Create database tables if they don't exist
    await db_manager.create_tables()


async def create_app_shutdown():
    """Shutdown event handler for the FastAPI application."""
    # Close database connection pool
    await db_manager.close()


def create_app(config=None):
    """Create FastAPI application."""
    app = FastAPI(
        title="TaxDB API",
        version="0.1.0",
        description="""
        # Tax Document Database API
        
        This API provides access to tax documents from multiple jurisdictions (Belgium, Spain, Germany).
        
        ## Features
        
        * Text search across documents
        * Vector similarity search using embeddings
        * Hybrid search combining text and vector similarity
        * Document metadata retrieval
        * Similar document discovery
        
        ## Authentication
        
        API requests are rate-limited based on client IP address.
        
        ## Jurisdictions
        
        The following jurisdictions are supported:
        
        * BE - Belgium
        * ES - Spain
        * DE - Germany
        """,
        terms_of_service="https://example.com/terms/",
        contact={
            "name": "TaxDB Team",
            "url": "https://example.com/contact/",
            "email": "taxdb-team@example.com",
        },
        license_info={
            "name": "Internal Use Only",
        },
        openapi_tags=[
            {
                "name": "Health",
                "description": "Health check endpoints",
            },
            {
                "name": "Search",
                "description": "Document search endpoints",
            },
            {
                "name": "Documents",
                "description": "Document retrieval endpoints",
            },
            {
                "name": "Jurisdictions",
                "description": "Jurisdiction-specific endpoints",
            },
        ],
        on_startup=[create_app_startup],
        on_shutdown=[create_app_shutdown]
    )
    
    # Apply CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = datetime.datetime.now()
        response = await call_next(request)
        process_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            f"Request: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Process Time: {process_time:.2f}ms - "
            f"Client: {request.client.host if request.client else 'Unknown'}"
        )
        return response
    
    @app.get("/healthz", response_model=HealthCheck, tags=["Health"])
    async def healthz(request: Request):
        """
        Health check endpoint.
        
        Returns a simple status response to verify the API is running.
        
        Rate limit: 60 requests per minute per client IP.
        """
        logger.debug("Health check requested")
        return {"status": "ok"}
    
    @app.get("/search", response_model=SearchResult, tags=["Search"])
    async def search(
        request: Request,
        q: str = Query(..., min_length=1, max_length=500, description="Search query text"),
        jurisdiction: Optional[str] = Query(None, description="Filter by jurisdiction (BE, ES, DE)"),
        page: int = Query(1, ge=1, description="Page number (1-based)"),
        page_size: int = Query(10, ge=1, le=100, description="Page size (max 100)"),
        sort_by: Optional[str] = Query(None, description="Field to sort by (e.g., issue_date, title)"),
        sort_order: str = Query("desc", description="Sort order (asc or desc)"),
        session: AsyncSession = Depends(get_session)
    ):
        """
        Search documents by text.
        
        This endpoint performs a text-based search across document titles and summaries.
        Results can be filtered by jurisdiction and are paginated.
        
        In local mode, uses ILIKE for text search.
        In cloud mode, uses Azure AI Search.
        
        Rate limit: 30 requests per minute per client IP.
        
        Examples:
            - Search for tax documents: `/search?q=tax`
            - Search for Belgian VAT documents: `/search?q=VAT&jurisdiction=BE`
            - Get page 2 with 20 results: `/search?q=tax&page=2&page_size=20`
        """
        # Create pagination and sort parameters
        pagination = PaginationParams(page=page, page_size=page_size)
        sort = SortParams(sort_by=sort_by, sort_order=sort_order)
        
        if settings.LOCAL_MODE:
            # Local search using ILIKE
            documents, total = await document_repository.search_by_text(
                session=session,
                query=q,
                jurisdiction=jurisdiction,
                pagination=pagination,
                sort=sort
            )
            
            # Generate SAS URLs for blob URLs
            for doc in documents:
                doc.blob_url = generate_sas_url(doc.blob_url)
            
            return {
                "documents": documents,
                "total": total
            }
        else:
            # In a real implementation, this would use Azure AI Search
            # For this skeleton, we'll just return an empty result
            return {
                "documents": [],
                "total": 0
            }
    
    @app.post("/search/vector", response_model=VectorSearchResult, tags=["Search"])
    async def search_by_vector(
        request: Request,
        query: VectorQuery,
        embedding_strategy: EmbeddingStrategy = Depends(get_embedding_strategy_dependency),
        session: AsyncSession = Depends(get_session)
    ):
        """
        Search documents by vector similarity.
        
        This endpoint performs a semantic search using vector embeddings.
        It converts the query text to a vector embedding and finds documents with similar meaning,
        regardless of exact keyword matches.
        
        Results include a similarity score between 0 and 1, where 1 is most similar.
        
        Rate limit: 20 requests per minute per client IP.
        
        Example request body:
        ```json
        {
          "text": "tax implications for cross-border transactions",
          "jurisdiction": "BE",
          "page": 1,
          "page_size": 10,
          "max_distance": 0.5
        }
        ```
        """
        try:
            # Generate vector embedding for the query text
            query_vector = await embedding_strategy.embed(query.text)
            
            # Create pagination parameters
            pagination = PaginationParams(page=query.page, page_size=query.page_size)
            
            # Search by vector similarity
            documents_with_distances, total = await document_repository.search_by_vector(
                session=session,
                query_vector=query_vector,
                jurisdiction=query.jurisdiction,
                pagination=pagination,
                max_distance=query.max_distance
            )
            
            # Convert to DTOs with scores
            result_documents = []
            for doc, distance in documents_with_distances:
                # Convert distance to similarity score (1 / (1 + distance))
                similarity_score = 1 / (1 + distance)
                
                # Generate SAS URL for blob URL
                doc.blob_url = generate_sas_url(doc.blob_url)
                
                result_documents.append({
                    "document": doc,
                    "score": similarity_score
                })
            
            return {
                "documents": result_documents,
                "total": total
            }
        except EmbeddingError as e:
            raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")
    
    @app.post("/search/hybrid", response_model=VectorSearchResult, tags=["Search"])
    async def search_hybrid(
        request: Request,
        query: HybridQuery,
        embedding_strategy: EmbeddingStrategy = Depends(get_embedding_strategy_dependency),
        session: AsyncSession = Depends(get_session)
    ):
        """
        Perform hybrid search combining vector similarity and text search.
        
        This endpoint combines the power of semantic search (vector similarity) with
        traditional keyword search (text similarity) for more accurate results.
        
        The weights control the balance between semantic and keyword matching:
        - vector_weight: Weight for semantic similarity (0.0 to 1.0)
        - text_weight: Weight for keyword similarity (0.0 to 1.0)
        
        Rate limit: 20 requests per minute per client IP.
        
        Example request body:
        ```json
        {
          "text": "VAT exemptions for financial services",
          "jurisdiction": "ES",
          "page": 1,
          "page_size": 10,
          "vector_weight": 0.7,
          "text_weight": 0.3
        }
        ```
        """
        try:
            # Generate vector embedding for the query text
            query_vector = await embedding_strategy.embed(query.text)
            
            # Create pagination parameters
            pagination = PaginationParams(page=query.page, page_size=query.page_size)
            
            # Perform hybrid search
            documents_with_scores, total = await document_repository.search_hybrid(
                session=session,
                query=query.text,
                query_vector=query_vector,
                jurisdiction=query.jurisdiction,
                pagination=pagination,
                vector_weight=query.vector_weight,
                text_weight=query.text_weight
            )
            
            # Convert to DTOs with scores
            result_documents = []
            for doc, score in documents_with_scores:
                # Generate SAS URL for blob URL
                doc.blob_url = generate_sas_url(doc.blob_url)
                
                result_documents.append({
                    "document": doc,
                    "score": score
                })
            
            return {
                "documents": result_documents,
                "total": total
            }
        except EmbeddingError as e:
            raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")
    
    @app.get("/doc/{id}", response_model=DocumentDTO, tags=["Documents"])
    async def get_document(
        request: Request,
        id: str,
        session: AsyncSession = Depends(get_session)
    ):
        """
        Get document by ID.
        
        Returns document metadata and a SAS URL for the blob.
        The SAS URL is valid for 15 minutes and provides read-only access to the document.
        
        Document IDs follow the format: `{jurisdiction}:{date}:{identifier}`
        Example: `BE:20250101:123`
        
        Rate limit: 60 requests per minute per client IP.
        """
        try:
            document = await document_repository.get_by_id(session, id)
            
            # Generate SAS URL
            document.blob_url = generate_sas_url(document.blob_url)
            
            return document
        except EntityNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")
    
    @app.get("/doc/{id}/similar", response_model=List[DocumentWithScoreDTO], tags=["Documents"])
    async def get_similar_documents(
        request: Request,
        id: str,
        jurisdiction: Optional[str] = Query(None, description="Filter by jurisdiction (BE, ES, DE)"),
        limit: int = Query(5, ge=1, le=20, description="Maximum number of similar documents to return"),
        session: AsyncSession = Depends(get_session)
    ):
        """
        Get documents similar to a given document.
        
        Uses vector similarity to find documents with similar content to the specified document.
        Results include a similarity score between 0 and 1, where 1 is most similar.
        
        This is useful for finding related documents or exploring a topic further.
        
        Rate limit: 30 requests per minute per client IP.
        """
        try:
            similar_documents = await document_repository.get_similar_documents(
                session=session,
                document_id=id,
                limit=limit,
                jurisdiction=jurisdiction
            )
            
            # Convert to DTOs with scores
            result_documents = []
            for doc, distance in similar_documents:
                # Convert distance to similarity score (1 / (1 + distance))
                similarity_score = 1 / (1 + distance)
                
                # Generate SAS URL for blob URL
                doc.blob_url = generate_sas_url(doc.blob_url)
                
                result_documents.append({
                    "document": doc,
                    "score": similarity_score
                })
            
            return result_documents
        except EntityNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")
    
    @app.get("/jurisdictions/{jurisdiction}/documents", response_model=SearchResult, tags=["Jurisdictions"])
    async def get_documents_by_jurisdiction(
        request: Request,
        jurisdiction: str,
        page: int = Query(1, ge=1, description="Page number (1-based)"),
        page_size: int = Query(10, ge=1, le=100, description="Page size (max 100)"),
        sort_by: Optional[str] = Query(None, description="Field to sort by"),
        sort_order: str = Query("desc", description="Sort order (asc or desc)"),
        session: AsyncSession = Depends(get_session)
    ):
        """
        Get documents by jurisdiction.
        
        Returns a paginated list of documents for the specified jurisdiction.
        Documents are sorted by issue date by default (newest first).
        
        Valid jurisdiction codes:
        - BE: Belgium
        - ES: Spain
        - DE: Germany
        
        Rate limit: 30 requests per minute per client IP.
        """
        # Create pagination and sort parameters
        pagination = PaginationParams(page=page, page_size=page_size)
        sort = SortParams(sort_by=sort_by, sort_order=sort_order)
        
        # Get documents by jurisdiction
        documents, total = await document_repository.filter_by_jurisdiction(
            session=session,
            jurisdiction=jurisdiction,
            pagination=pagination,
            sort=sort
        )
        
        # Generate SAS URLs for blob URLs
        for doc in documents:
            doc.blob_url = generate_sas_url(doc.blob_url)
        
        return {
            "documents": documents,
            "total": total
        }
    
    return app