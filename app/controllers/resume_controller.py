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
from app.config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["resume"])

# Initialize services
file_processor = FileProcessor()
openai_service = OpenAIService()

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
    Parse resume from uploaded file.
    
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
        if len(file_content) > settings.MAX_FILE_SIZE:
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
        
        logger.info(f"Successfully parsed resume in {processing_time:.2f} seconds")
        
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
