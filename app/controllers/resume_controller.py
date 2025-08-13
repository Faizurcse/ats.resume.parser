"""
Resume controller for handling API endpoints.
Simplified version with only essential endpoints.
"""

import time
import logging
import uuid
import os
from typing import Dict, Any, List

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from app.models.schemas import (
    BatchResumeParseResponse,
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

@router.post("/parse-resume", response_model=BatchResumeParseResponse)
async def parse_resume(files: List[UploadFile] = File(...)):
    """
    Parse resumes from uploaded files and save to database.
    Supports both single and multiple file uploads.
    Supports various file formats: PDF, DOCX, DOC, TXT, RTF, PNG, JPG, JPEG, WEBP
    
    Args:
        files (List[UploadFile]): List of resume files to parse (can be 1 or multiple)
        
    Returns:
        BatchResumeParseResponse: Batch processing results with individual file results
        
    Raises:
        HTTPException: If file processing or parsing fails
    """
    start_time = time.time()
    
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {settings.MAX_BATCH_SIZE} files allowed per request"
        )
    
    results = []
    successful_files = 0
    failed_files = 0
    batch_data_to_save = []
    total_batch_size = 0
    
    try:
        for file in files:
            file_start_time = time.time()
            file_result = {
                "filename": file.filename,
                "status": "failed",
                "error": None,
                "parsed_data": None,
                "file_type": None,
                "processing_time": 0
            }
            
            try:
                # Validate file
                if not file.filename:
                    file_result["error"] = "No filename provided"
                    results.append(file_result)
                    failed_files += 1
                    continue
                
                # Check file extension
                import os
                file_extension = os.path.splitext(file.filename)[1].lower()
                if file_extension not in settings.ALLOWED_EXTENSIONS:
                    file_result["error"] = f"Unsupported file format. Supported formats: {', '.join(settings.ALLOWED_EXTENSIONS)}"
                    file_result["file_type"] = file_extension.lstrip('.')
                    results.append(file_result)
                    failed_files += 1
                    continue
                
                # Check file size
                file_content = await file.read()
                file_size = len(file_content)
                if file_size > settings.MAX_FILE_SIZE:
                    file_result["error"] = f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE} bytes"
                    file_result["file_type"] = file_extension.lstrip('.')
                    results.append(file_result)
                    failed_files += 1
                    continue
                
                # Check total batch size
                total_batch_size += file_size
                if total_batch_size > settings.MAX_TOTAL_BATCH_SIZE:
                    file_result["error"] = f"Total batch size exceeds maximum limit of {settings.MAX_TOTAL_BATCH_SIZE} bytes"
                    file_result["file_type"] = file_extension.lstrip('.')
                    results.append(file_result)
                    failed_files += 1
                    continue
                
                # Save file to upload folder
                
                # Create upload folder if it doesn't exist
                os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
                
                # Generate unique filename to avoid conflicts
                file_uuid = str(uuid.uuid4())
                file_extension_with_dot = os.path.splitext(file.filename)[1]
                unique_filename = f"{file_uuid}{file_extension_with_dot}"
                file_path = os.path.join(settings.UPLOAD_FOLDER, unique_filename)
                
                # Save file to disk
                with open(file_path, "wb") as f:
                    f.write(file_content)
                
                logger.info(f"File saved to upload folder: {file_path}")
                
                # Process file and extract text
                logger.info(f"Processing file: {file.filename}")
                extracted_text = await file_processor.process_file(file_content, file.filename)
                
                if not extracted_text or not extracted_text.strip():
                    file_result["error"] = "No text could be extracted from the file"
                    file_result["file_type"] = file_extension.lstrip('.')
                    results.append(file_result)
                    failed_files += 1
                    continue
                
                # Parse resume with AI
                logger.info(f"Parsing resume with AI: {file.filename}")
                parsed_data = await openai_service.parse_resume_text(extracted_text)
                
                # Calculate processing time
                file_processing_time = time.time() - file_start_time
                
                # Determine file type
                file_type = file_extension.lstrip('.')
                
                # Update result for successful processing
                file_result.update({
                    "status": "success",
                    "parsed_data": parsed_data,
                    "file_type": file_type,
                    "processing_time": file_processing_time
                })
                
                # Add to batch save data
                batch_data_to_save.append({
                    "filename": file.filename,
                    "file_path": file_path,
                    "file_type": file_type,
                    "file_size": file_size,
                    "processing_time": file_processing_time,
                    "parsed_data": parsed_data
                })
                
                successful_files += 1
                logger.info(f"Successfully parsed resume: {file.filename} in {file_processing_time:.2f} seconds")
                
            except Exception as e:
                file_processing_time = time.time() - file_start_time
                file_result.update({
                    "error": f"Failed to process file: {str(e)}",
                    "processing_time": file_processing_time
                })
                failed_files += 1
                logger.error(f"Error processing file {file.filename}: {str(e)}")
            
            results.append(file_result)
        
        # Save successful files to database in batch
        if batch_data_to_save:
            try:
                record_ids = await database_service.save_batch_resume_data(batch_data_to_save)
                logger.info(f"Successfully saved {len(record_ids)} resume records to database")
            except Exception as e:
                logger.error(f"Error saving batch data to database: {str(e)}")
                # Don't fail the entire request if database save fails
                # The parsed data is still returned to the user
        
        total_processing_time = time.time() - start_time
        
        return BatchResumeParseResponse(
            total_files=len(files),
            successful_files=successful_files,
            failed_files=failed_files,
            total_processing_time=total_processing_time,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Error in batch resume parsing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process resumes: {str(e)}"
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
    Get ALL resume records with pagination (including duplicates).
    This endpoint returns every resume that was uploaded, regardless of duplicates.
    
    Args:
        limit (int): Number of records to return (default: 100)
        offset (int): Number of records to skip (default: 0)
        
    Returns:
        List[Dict]: List of ALL resume records (including duplicates)
    """
    try:
        # Use the method that explicitly returns ALL resumes including duplicates
        resumes = await database_service.get_all_resumes_including_duplicates(limit=limit, offset=offset)
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
