"""
Resume controller for handling API endpoints.
Simplified version with only essential endpoints.
"""

import time
import logging
import uuid
import os
import json
from typing import Dict, Any, List

from fastapi import APIRouter, File, UploadFile, HTTPException, status, BackgroundTasks
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

@router.get("/resume-embeddings-status")
async def get_resume_embeddings_status():
    """
    Get the status of resume embeddings - how many resumes have embeddings and show all resumes with embedding data.
    
    Returns:
        Dict: Resume embedding status and data
    """
    try:
        # Get all resumes
        all_resumes = await database_service.get_all_resumes(limit=1000)
        
        # Count resumes with and without embeddings
        total_resumes = len(all_resumes)
        resumes_with_embeddings = 0
        resumes_without_embeddings = 0
        
        # Process each resume to check embedding status
        resume_details = []
        
        for resume in all_resumes:
            parsed_data = resume.get('parsed_data', {})
            
            # Handle parsed_data that might be a JSON string
            if isinstance(parsed_data, str):
                try:
                    import json
                    parsed_data = json.loads(parsed_data)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Resume {resume['id']}: Could not parse parsed_data JSON string")
                    parsed_data = {}
            
            # Ensure parsed_data is a dictionary
            if not isinstance(parsed_data, dict):
                logger.warning(f"Resume {resume['id']}: parsed_data is not a dictionary, type: {type(parsed_data)}")
                parsed_data = {}
            
            # Check if resume has embedding data
            has_embedding = False
            embedding_dimensions = 0
            
            # Check separate embedding column first
            if resume.get('embedding'):
                has_embedding = True
                embedding = resume['embedding']
                if isinstance(embedding, list):
                    embedding_dimensions = len(embedding)
                elif isinstance(embedding, str):
                    try:
                        import json
                        embedding_list = json.loads(embedding)
                        if isinstance(embedding_list, list):
                            embedding_dimensions = len(embedding_list)
                    except:
                        pass
            
            # Fallback: check parsed_data for backward compatibility
            if not has_embedding and parsed_data:
                # Look for embedding in parsed data
                if 'embedding' in parsed_data and parsed_data['embedding']:
                    has_embedding = True
                    if isinstance(parsed_data['embedding'], list):
                        embedding_dimensions = len(parsed_data['embedding'])
                    elif isinstance(parsed_data['embedding'], str):
                        try:
                            import json
                            embedding_list = json.loads(parsed_data['embedding'])
                            if isinstance(embedding_list, list):
                                embedding_dimensions = len(embedding_list)
                        except:
                            pass
            
            if has_embedding:
                resumes_with_embeddings += 1
            else:
                resumes_without_embeddings += 1
            
            # Add resume details
            resume_details.append({
                "resume_id": resume['id'],
                "filename": resume.get('filename', 'Unknown'),
                "candidate_name": resume.get('candidate_name', 'Unknown'),
                "candidate_email": resume.get('candidate_email', ''),
                "total_experience": resume.get('total_experience', ''),
                "has_embedding": has_embedding,
                "embedding_dimensions": embedding_dimensions,
                "created_at": resume.get('created_at', ''),
                "parsed_data_keys": list(parsed_data.keys()) if parsed_data else [],
                "skills": parsed_data.get('Skills', []) if parsed_data else [],
                "summary": parsed_data.get('Summary', '') if parsed_data else '',
                "work_experience_count": len(parsed_data.get('WorkExperience', [])) if parsed_data and parsed_data.get('WorkExperience') else 0,
                "parsed_data_type": str(type(parsed_data)),
                "parsed_data_sample": str(parsed_data)[:200] + "..." if parsed_data else "No parsed data"
            })
        
        return {
            "success": True,
            "total_resumes": total_resumes,
            "resumes_with_embeddings": resumes_with_embeddings,
            "resumes_without_embeddings": resumes_without_embeddings,
            "embedding_coverage_percentage": round((resumes_with_embeddings / total_resumes * 100) if total_resumes > 0 else 0, 2),
            "resume_details": resume_details,
            "message": f"Found {total_resumes} total resumes. {resumes_with_embeddings} have embeddings ({resumes_without_embeddings} without embeddings)."
        }
        
    except Exception as e:
        logger.error(f"Error getting resume embeddings status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resume embeddings status: {str(e)}"
        )

@router.post("/parse-resume", response_model=BatchResumeParseResponse)
async def parse_resume(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
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

@router.post("/generate-resume-embeddings")
async def generate_resume_embeddings():
    """
    Generate embeddings for all resumes that don't have them.
    This will process resumes in batches to avoid overwhelming the system.
    
    Returns:
        Dict: Status of embedding generation process
    """
    try:
        from app.services.openai_service import OpenAIService
        from app.services.database_service import DatabaseService
        
        # Initialize services
        openai_service = OpenAIService()
        database_service = DatabaseService()
        
        # Get all resumes without embeddings
        all_resumes = await database_service.get_all_resumes(limit=1000)
        resumes_without_embeddings = []
        
        for resume in all_resumes:
            # Check if resume already has embedding in separate column
            has_embedding = False
            
            # Check separate embedding column first
            if resume.get('embedding'):
                has_embedding = True
            
            # Fallback: check parsed_data for backward compatibility
            if not has_embedding:
                parsed_data = resume.get('parsed_data', {})
                
                # Handle parsed_data that might be a JSON string
                if isinstance(parsed_data, str):
                    try:
                        import json
                        parsed_data = json.loads(parsed_data)
                    except (json.JSONDecodeError, TypeError):
                        continue
                
                if parsed_data and isinstance(parsed_data, dict):
                    if 'embedding' in parsed_data and parsed_data['embedding']:
                        has_embedding = True
            
            if not has_embedding:
                resumes_without_embeddings.append(resume)
        
        if not resumes_without_embeddings:
            return {
                "success": True,
                "message": "All resumes already have embeddings!",
                "total_processed": 0,
                "embeddings_generated": 0,
                "failed_resumes": []
            }
        
        # Process resumes in batches
        batch_size = 5  # Process 5 resumes at a time to avoid rate limits
        total_processed = 0
        embeddings_generated = 0
        failed_resumes = []
        
        for i in range(0, len(resumes_without_embeddings), batch_size):
            batch = resumes_without_embeddings[i:i + batch_size]
            
            for resume in batch:
                try:
                    # Generate semantic text representation
                    parsed_data = resume.get('parsed_data', {})
                    if isinstance(parsed_data, str):
                        try:
                            import json
                            parsed_data = json.loads(parsed_data)
                        except:
                            continue
                    
                    if not isinstance(parsed_data, dict):
                        continue
                    
                    # Create structured text for embedding
                    candidate_name = parsed_data.get('Name', '')
                    skills = parsed_data.get('Skills', [])
                    summary = parsed_data.get('Summary', '')
                    experience = parsed_data.get('Experience', [])
                    education = parsed_data.get('Education', [])
                    projects = parsed_data.get('Projects', [])
                    
                    # Build comprehensive text representation
                    text_parts = []
                    if candidate_name:
                        text_parts.append(f"Name: {candidate_name}")
                    
                    if skills:
                        skills_text = ", ".join(skills) if isinstance(skills, list) else str(skills)
                        text_parts.append(f"Skills: {skills_text}")
                    
                    if summary:
                        text_parts.append(f"Summary: {summary}")
                    
                    if experience:
                        if isinstance(experience, list):
                            exp_text = "; ".join([str(exp) for exp in experience])
                        else:
                            exp_text = str(experience)
                        text_parts.append(f"Experience: {exp_text}")
                    
                    if education:
                        if isinstance(education, list):
                            edu_text = "; ".join([str(edu) for edu in education])
                        else:
                            edu_text = str(education)
                        text_parts.append(f"Education: {edu_text}")
                    
                    if projects:
                        if isinstance(projects, list):
                            proj_text = "; ".join([str(proj) for proj in projects])
                        else:
                            proj_text = str(projects)
                        text_parts.append(f"Projects: {proj_text}")
                    
                    # Combine all text
                    combined_text = "\n".join(text_parts)
                    
                    if not combined_text.strip():
                        logger.warning(f"Resume {resume['id']}: No meaningful text content found")
                        continue
                    
                    # Generate embedding
                    embedding = await openai_service.generate_embedding(combined_text)
                    
                    if embedding:
                        # Update resume with embedding in separate column
                        update_success = await database_service.update_resume_embedding_column(resume['id'], embedding)
                        
                        if update_success:
                            embeddings_generated += 1
                            logger.info(f"Generated embedding for resume {resume['id']}: {candidate_name}")
                        else:
                            failed_resumes.append({
                                "resume_id": resume['id'],
                                "filename": resume.get('filename', 'Unknown'),
                                "error": "Failed to update database"
                            })
                            logger.error(f"Failed to update database for resume {resume['id']}")
                    else:
                        failed_resumes.append({
                            "resume_id": resume['id'],
                            "filename": resume.get('filename', 'Unknown'),
                            "error": "Failed to generate embedding"
                        })
                        logger.error(f"Failed to generate embedding for resume {resume['id']}")
                    
                    total_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing resume {resume['id']}: {str(e)}")
                    failed_resumes.append({
                        "resume_id": resume['id'],
                        "filename": resume.get('filename', 'Unknown'),
                        "error": str(e)
                    })
                    total_processed += 1
                
                # Small delay to avoid rate limits
                import asyncio
                await asyncio.sleep(0.1)
            
            # Delay between batches
            await asyncio.sleep(1)
        
        return {
            "success": True,
            "message": f"Embedding generation completed! Processed {total_processed} resumes.",
            "total_processed": total_processed,
            "embeddings_generated": embeddings_generated,
            "failed_resumes": failed_resumes,
            "success_rate": round((embeddings_generated / total_processed * 100) if total_processed > 0 else 0, 2)
        }
        
    except Exception as e:
        logger.error(f"Error generating resume embeddings: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate resume embeddings: {str(e)}"
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

@router.get("/resumes/{resume_id}/parsed-data")
async def get_resume_parsed_data(resume_id: int):
    """
    Get parsed resume data by ID (including embeddings, skills, experience, etc.).
    
    Args:
        resume_id (int): Resume record ID
        
    Returns:
        Dict: Parsed resume data with full details
        
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
        
        # Extract parsed_data from resume
        parsed_data = resume_data.get('parsed_data', {})
        
        # Handle parsed_data that might be a JSON string
        if isinstance(parsed_data, str):
            try:
                import json
                parsed_data = json.loads(parsed_data)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Resume {resume_id}: Could not parse parsed_data JSON string")
                parsed_data = {}
        
        # Ensure parsed_data is a dictionary
        if not isinstance(parsed_data, dict):
            logger.warning(f"Resume {resume_id}: parsed_data is not a dictionary, type: {type(parsed_data)}")
            parsed_data = {}
        
        # Return the parsed data with resume metadata
        return {
            "resume_id": resume_id,
            "candidate_name": resume_data.get('candidate_name', 'Unknown'),
            "candidate_email": resume_data.get('candidate_email', ''),
            "filename": resume_data.get('filename', ''),
            "upload_date": resume_data.get('created_at', ''),
            "parsed_data": parsed_data,
            "message": f"Retrieved parsed data for resume {resume_id} successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting resume parsed data {resume_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resume parsed data: {str(e)}"
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
