"""
Resume controller for handling API endpoints.
Simplified version with only essential endpoints.
"""

import time
import logging
from typing import Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from app.models.schemas import (
    ResumeParseResponse, 
    ErrorResponse, 
    HealthResponse
)
from app.services.file_processor import FileProcessor
from app.services.openai_service import OpenAIService
from app.services.database_service import DatabaseService
from app.config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["resume"])

# Initialize services
file_processor = FileProcessor()
openai_service = OpenAIService()
database_service = DatabaseService()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify application status.
    
    Returns:
        HealthResponse: Application health status
    """
    try:
        from datetime import datetime
        
        return HealthResponse(
            status="healthy",
            version=settings.APP_VERSION,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )

@router.post("/parse-resume", response_model=ResumeParseResponse)
async def parse_resume(file: UploadFile = File(...)):
    """
    Parse resume from uploaded file and save to database.
    
    Args:
        file (UploadFile): The resume file to parse
        
    Returns:
        ResumeParseResponse: Parsed resume data
        
    Raises:
        HTTPException: If file processing or parsing fails
    """
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        # Check file extension
        import os
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format. Supported formats: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Check file size
        file_content = await file.read()
        file_size = len(file_content)
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Process file and extract text
        logger.info(f"Processing file: {file.filename}")
        extracted_text = await file_processor.process_file(file_content, file.filename)
        
        if not extracted_text or not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from the file"
            )
        
        # Parse resume with AI
        logger.info("Parsing resume with AI")
        parsed_data = await openai_service.parse_resume_text(extracted_text)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Determine file type
        file_type = file_extension.lstrip('.')
        
        # Save to database
        logger.info("Saving parsed data to database")
        record_id = await database_service.save_resume_data(
            filename=file.filename,
            file_type=file_type,
            file_size=file_size,
            processing_time=processing_time,
            parsed_data=parsed_data
        )
        
        logger.info(f"Successfully parsed resume in {processing_time:.2f} seconds and saved with ID: {record_id}")
        
        return ResumeParseResponse(
            parsed_data=parsed_data,
            file_type=file_type,
            processing_time=processing_time
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error parsing resume: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse resume: {str(e)}"
        )

@router.get("/resumes/{resume_id}")
async def get_resume(resume_id: int):
    """
    Get resume data by ID.
    
    Args:
        resume_id (int): Resume record ID
        
    Returns:
        Dict: Resume data
        
    Raises:
        HTTPException: If resume not found or database error
    """
    try:
        resume_data = await database_service.get_resume_by_id(resume_id)
        
        if not resume_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resume with ID {resume_id} not found"
            )
        
        return resume_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting resume {resume_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resume: {str(e)}"
        )

@router.get("/resumes")
async def get_all_resumes(limit: int = 100, offset: int = 0):
    """
    Get all resume records with pagination.
    
    Args:
        limit (int): Number of records to return (default: 100)
        offset (int): Number of records to skip (default: 0)
        
    Returns:
        List[Dict]: List of resume records
    """
    try:
        resumes = await database_service.get_all_resumes(limit=limit, offset=offset)
        return {"resumes": resumes, "total": len(resumes)}
        
    except Exception as e:
        logger.error(f"Error getting resumes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resumes: {str(e)}"
        )

@router.get("/resumes/search/{search_term}")
async def search_resumes(search_term: str):
    """
    Search resumes by candidate name or email.
    
    Args:
        search_term (str): Search term
        
    Returns:
        List[Dict]: List of matching resume records
    """
    try:
        resumes = await database_service.search_resumes(search_term)
        return {"resumes": resumes, "search_term": search_term, "total": len(resumes)}
        
    except Exception as e:
        logger.error(f"Error searching resumes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search resumes: {str(e)}"
        )

@router.delete("/resumes/{resume_id}")
async def delete_resume(resume_id: int):
    """
    Delete resume record by ID.
    
    Args:
        resume_id (int): Resume record ID
        
    Returns:
        Dict: Deletion status
        
    Raises:
        HTTPException: If resume not found or database error
    """
    try:
        deleted = await database_service.delete_resume(resume_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resume with ID {resume_id} not found"
            )
        
        return {"message": f"Resume with ID {resume_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting resume {resume_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete resume: {str(e)}"
        )
