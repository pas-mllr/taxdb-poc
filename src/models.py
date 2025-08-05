"""
Database models for the TaxDB-POC application.

This module defines the SQLAlchemy models for the application.
"""

from sqlalchemy import (
    Column, Date, DateTime, Enum, Float, String, Text, func, Index
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Document(Base):
    """Document model for storing tax documents."""
    __tablename__ = "documents"

    id = Column(String, primary_key=True)  # "BE:20250804:AR-123"
    jurisdiction = Column(Enum("BE", "ES", "DE", name="jurisd"))
    source_system = Column(String, nullable=False)
    document_type = Column(String, nullable=False)
    title = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    issue_date = Column(Date, nullable=False)
    effective_date = Column(Date, nullable=True)
    language_orig = Column(String(2), nullable=False)
    blob_url = Column(Text, nullable=False)
    checksum = Column(String(64), unique=True, nullable=False)
    vector = Column(Vector(1536))
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_jurisdiction_date", "jurisdiction", "issue_date"),
        Index("ix_vector", "vector", postgresql_using="ivfflat"),
    )

    def __repr__(self) -> str:
        """Return string representation of the document."""
        return f"<Document(id='{self.id}', title='{self.title[:30]}...', jurisdiction='{self.jurisdiction}')>"