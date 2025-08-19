"""
AI Search module for Resume Parser application.

This module provides AI-powered search functionality using vector embeddings
and similarity search for finding relevant resumes based on natural language queries.
"""

__version__ = "1.0.0"
__author__ = "Resume Parser Team"

# Import main components
from .models import (
    SearchRequest, SearchResponse, EmbeddingStatusResponse
)
from .embedding_service import EmbeddingService
from .search_service import SearchService
from .controller import router

__all__ = [
    "SearchRequest",
    "SearchResponse",
    "EmbeddingStatusResponse",
    "EmbeddingService",
    "SearchService",
    "router"
]
