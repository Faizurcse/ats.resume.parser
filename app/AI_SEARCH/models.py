"""
AI Search models and schemas for vector embeddings and search functionality.
Embeddings are created automatically during resume upload.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class SearchRequest(BaseModel):
    """Request model for AI search."""
    
    query: str = Field(description="Search query (e.g., 'Java developer')")
    limit: int = Field(default=10, description="Maximum number of results to return")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity score (0.0 to 1.0)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Java developer",
                "limit": 10,
                "similarity_threshold": 0.7
            }
        }

class SearchResult(BaseModel):
    """Individual search result."""
    
    resume_id: int = Field(description="Resume ID")
    candidate_name: str = Field(description="Candidate name")
    candidate_email: str = Field(description="Candidate email")
    filename: str = Field(description="Resume filename")
    similarity_score: float = Field(description="Similarity score (0.0 to 1.0)")
    similarity_percentage: float = Field(description="Similarity percentage")
    parsed_data: Dict[str, Any] = Field(description="Parsed resume data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "resume_id": 1,
                "candidate_name": "John Doe",
                "candidate_email": "john@example.com",
                "filename": "resume.pdf",
                "similarity_score": 0.85,
                "similarity_percentage": 85.0,
                "parsed_data": {
                    "Name": "John Doe",
                    "Skills": ["Java", "Spring", "MySQL"]
                }
            }
        }

class SearchResponse(BaseModel):
    """Response model for AI search."""
    
    query: str = Field(description="Original search query")
    total_results: int = Field(description="Total number of results found")
    results: List[SearchResult] = Field(description="List of search results")
    search_time: float = Field(description="Time taken for search in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Java developer",
                "total_results": 3,
                "results": [],
                "search_time": 0.15
            }
        }

class EmbeddingStatusResponse(BaseModel):
    """Response model for embedding status."""
    
    total_resumes: int = Field(description="Total number of resumes in database")
    embedded_resumes: int = Field(description="Number of resumes with embeddings")
    pending_resumes: int = Field(description="Number of resumes without embeddings")
    embedding_percentage: float = Field(description="Percentage of resumes with embeddings")
    last_updated: Optional[datetime] = Field(description="Last time embeddings were updated")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_resumes": 100,
                "embedded_resumes": 75,
                "pending_resumes": 25,
                "embedding_percentage": 75.0,
                "last_updated": "2024-01-01T12:00.0Z"
            }
        }
