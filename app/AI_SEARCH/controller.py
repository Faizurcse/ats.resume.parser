"""
AI Search controller for handling API endpoints related to search and status.
Embeddings are created automatically during resume upload.
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from app.AI_SEARCH.models import (
    SearchRequest, SearchResponse, EmbeddingStatusResponse
)
from app.AI_SEARCH.search_service import SearchService
from app.AI_SEARCH.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# Create router for AI search endpoints
router = APIRouter(prefix="/ai-search", tags=["AI Search"])

# Initialize services
search_service = SearchService()
embedding_service = EmbeddingService()

@router.post("/search", response_model=SearchResponse)
async def search_resumes(request: SearchRequest):
    """
    Search resumes using AI-powered vector similarity.
    
    Args:
        request (SearchRequest): Search request with query and parameters
        
    Returns:
        SearchResponse: Search results with similarity scores
    """
    try:
        logger.info(f"AI search request: {request.query}")
        
        # Validate similarity threshold
        if not 0.0 <= request.similarity_threshold <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="Similarity threshold must be between 0.0 and 1.0"
            )
        
        # Perform search
        search_result = await search_service.search_resumes(
            query=request.query,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        
        # Convert to response model
        results = []
        for result in search_result.get('results', []):
            results.append({
                "resume_id": result['resume_id'],
                "candidate_name": result['candidate_name'],
                "candidate_email": result['candidate_email'],
                "filename": result['filename'],
                "similarity_score": result['similarity_score'],
                "similarity_percentage": result['similarity_percentage'],
                "parsed_data": result['parsed_data']
            })
        
        return SearchResponse(
            query=search_result['query'],
            total_results=search_result['total_results'],
            results=results,
            search_time=search_result['search_time']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in AI search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/embedding-status", response_model=EmbeddingStatusResponse)
async def get_embedding_status():
    """
    Get the current status of embeddings in the database.
    
    Returns:
        EmbeddingStatusResponse: Embedding status information
    """
    try:
        logger.info("Getting embedding status")
        
        status = await embedding_service.get_embedding_status()
        
        return EmbeddingStatusResponse(
            total_resumes=status['total_resumes'],
            embedded_resumes=status['embedded_resumes'],
            pending_resumes=status['pending_resumes'],
            embedding_percentage=status['embedding_percentage'],
            last_updated=status['last_updated']
        )
        
    except Exception as e:
        logger.error(f"Error getting embedding status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
