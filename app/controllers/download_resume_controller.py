"""
Download Resume controller for handling resume download and retrieval endpoints.
"""

import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, status, Request
from fastapi.responses import FileResponse
import os

from app.services.database_service import DatabaseService
from app.config.settings import settings
from app.middleware.auth_middleware import get_company_id

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/download", tags=["download-resume"])

# Initialize services
database_service = DatabaseService()

@router.get("/resumes")
async def get_all_resumes_for_download(request: Request):
    """
    Get only unique resumes with essential information: name, email, and upload date.
    This endpoint returns only unique resumes based on candidate email for the authenticated company.
    
    Returns:
        List[Dict]: List of unique resume records with only essential info
    """
    try:
        # Get company ID from JWT token
        company_id = get_company_id(request)
        resumes = await database_service.get_unique_resumes_for_download(company_id)
        
        # Format the response to show only essential information
        formatted_resumes = []
        for resume in resumes:
            formatted_resumes.append({
                "candidate_name": resume["candidate_name"] or "N/A",
                "candidate_email": resume["candidate_email"] or "N/A",
                "upload_date": resume["created_at"] or "N/A"
            })
        
        return {
            "resumes": formatted_resumes,
            "total": len(formatted_resumes),
            "message": "Retrieved all unique resumes successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting resumes for download: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resumes: {str(e)}"
        )

@router.get("/resumes/with-files")
async def get_unique_resumes_with_files(request: Request):
    """
    Get all unique resumes with candidate name, email, upload date/time, and file download links.
    This endpoint returns only unique resumes based on candidate email for the authenticated company.
    
    Returns:
        List[Dict]: List of unique resume records with file download information
    """
    try:
        # Get company ID from JWT token
        company_id = get_company_id(request)
        resumes = await database_service.get_unique_resumes_with_files(company_id)
        
        # Format the response to include file download information
        formatted_resumes = []
        for resume in resumes:
            formatted_resumes.append({
                "id": resume["id"],
                "candidate_name": resume["candidate_name"] or "N/A",
                "candidate_email": resume["candidate_email"] or "N/A",
                "upload_date": resume["created_at"] or "N/A",
                "file_type": resume["file_type"] or "N/A",
                "download_url": f"/api/v1/download/resume/{resume['id']}",
                "filename": resume["filename"] or "N/A"
            })
        
        return {
            "resumes": formatted_resumes,
            "total": len(resumes),
            "message": "Retrieved all unique resumes with file download information successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting resumes with files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resumes with files: {str(e)}"
        )

@router.get("/resume/{resume_id}")
async def download_resume_file(resume_id: int):
    """
    Download a specific resume file by ID.
    
    Args:
        resume_id (int): Resume record ID
        
    Returns:
        FileResponse: The resume file for download
        
    Raises:
        HTTPException: If resume not found or file doesn't exist
    """
    try:
        # Get resume data from database
        resume_data = await database_service.get_resume_by_id(resume_id)
        
        if not resume_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resume with ID {resume_id} not found"
            )
        
        # Get the actual file path from database (UUID filename) and original filename for display
        actual_file_path = resume_data.get('file_path', '')
        original_filename = resume_data['filename']
        
        # Debug logging
        logger.info(f"Download request for resume ID {resume_id}")
        logger.info(f"Original filename: {original_filename}")
        logger.info(f"Actual file path: {actual_file_path}")
        logger.info(f"Upload folder: {settings.UPLOAD_FOLDER}")
        
        # Try to find the file using multiple strategies
        file_path = None
        
        # Strategy 1: Use file_path from database if available and valid
        if actual_file_path and os.path.exists(actual_file_path):
            file_path = actual_file_path
            logger.info(f"Using file_path from database: {file_path}")
        
        # Strategy 2: Try to find file by searching uploads folder for matching extension
        if not file_path:
            upload_folder = settings.UPLOAD_FOLDER
            if os.path.exists(upload_folder):
                file_extension = os.path.splitext(original_filename)[1]
                files_in_folder = [f for f in os.listdir(upload_folder) if f.endswith(file_extension)]
                
                if files_in_folder:
                    # Use the first matching file (this is a best-effort approach)
                    actual_filename = files_in_folder[0]
                    file_path = os.path.join(upload_folder, actual_filename)
                    logger.info(f"Found file by extension search: {file_path}")
                    
                    # Update the database with the found file path for future use
                    try:
                        await database_service.update_file_path(resume_id, file_path)
                        logger.info(f"Updated database with file_path: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not update database with file_path: {str(e)}")
                else:
                    logger.warning(f"No files with extension {file_extension} found in uploads folder")
            else:
                logger.error(f"Upload folder does not exist: {upload_folder}")
        
        # Strategy 3: Fall back to original filename (legacy support)
        if not file_path:
            file_path = os.path.join(settings.UPLOAD_FOLDER, original_filename)
            logger.warning(f"Using fallback path with original filename: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found at path: {file_path}")
            # List files in upload folder for debugging
            try:
                if os.path.exists(settings.UPLOAD_FOLDER):
                    files_in_folder = os.listdir(settings.UPLOAD_FOLDER)
                    logger.info(f"Files in upload folder: {files_in_folder}")
                else:
                    logger.error(f"Upload folder does not exist: {settings.UPLOAD_FOLDER}")
            except Exception as e:
                logger.error(f"Error listing upload folder: {str(e)}")
            
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resume file not found on disk. Tried: {file_path}"
            )
        
        # Return file for download with original filename
        return FileResponse(
            path=file_path,
            filename=original_filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading resume {resume_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download resume: {str(e)}"
        )

@router.get("/resume/{resume_id}/info")
async def get_resume_info(resume_id: int):
    """
    Get resume information without downloading the file.
    
    Args:
        resume_id (int): Resume record ID
        
    Returns:
        Dict: Resume information
        
    Raises:
        HTTPException: If resume not found
    """
    try:
        resume_data = await database_service.get_resume_by_id(resume_id)
        
        if not resume_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resume with ID {resume_id} not found"
            )
        
        # Return only the information, not the file
        return {
            "id": resume_data['id'],
            "filename": resume_data['filename'],
            "file_type": resume_data['file_type'],
            "file_size": resume_data['file_size'],
            "candidate_name": resume_data['candidate_name'],
            "candidate_email": resume_data['candidate_email'],
            "candidate_phone": resume_data['candidate_phone'],
            "total_experience": resume_data['total_experience'],
            "created_at": resume_data['created_at'],
            "parsed_data": resume_data['parsed_data']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting resume info {resume_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resume info: {str(e)}"
        )

@router.get("/debug/file-paths")
async def debug_file_paths():
    """
    Debug endpoint to check file paths and upload folder contents.
    """
    try:
        import os
        
        debug_info = {
            "upload_folder": settings.UPLOAD_FOLDER,
            "upload_folder_exists": os.path.exists(settings.UPLOAD_FOLDER),
            "files_in_uploads": [],
            "database_records": []
        }
        
        # Check uploads folder
        if os.path.exists(settings.UPLOAD_FOLDER):
            try:
                files_in_folder = os.listdir(settings.UPLOAD_FOLDER)
                debug_info["files_in_uploads"] = files_in_folder[:10]  # First 10 files
            except Exception as e:
                debug_info["files_in_uploads_error"] = str(e)
        
        # Check database records
        try:
            resumes = await database_service.get_all_resumes_including_duplicates(limit=5)
            debug_info["database_records"] = [
                {
                    "id": r["id"],
                    "filename": r["filename"],
                    "file_path": r.get("file_path", "NOT_FOUND"),
                    "file_type": r["file_type"]
                }
                for r in resumes
            ]
        except Exception as e:
            debug_info["database_error"] = str(e)
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debug endpoint failed: {str(e)}"
        )

@router.get("/resumes/all")
async def get_all_resumes_including_duplicates():
    """
    Get ALL resumes including duplicates for admin purposes.
    This endpoint returns all resumes without filtering for uniqueness.
    
    Returns:
        List[Dict]: List of all resume records (including duplicates)
    """
    try:
        resumes = await database_service.get_all_resumes_including_duplicates()
        
        # Format the response to show all resumes
        formatted_resumes = []
        for resume in resumes:
            formatted_resumes.append({
                "id": resume["id"],
                "candidate_name": resume["candidate_name"] or "N/A",
                "candidate_email": resume["candidate_email"] or "N/A",
                "upload_date": resume["created_at"] or "N/A",
                "file_type": resume["file_type"] or "N/A",
                "filename": resume["filename"] or "N/A"
            })
        
        return {
            "resumes": formatted_resumes,
            "total": len(resumes),
            "message": "Retrieved all resumes (including duplicates) successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting all resumes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get all resumes: {str(e)}"
        )
