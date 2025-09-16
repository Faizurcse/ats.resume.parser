"""
Resume controller for handling API endpoints.
Simplified version with only essential endpoints.
"""

import time
import logging
import uuid
import os
import json
import zipfile
import tempfile
import shutil
import asyncio
from typing import Dict, Any, List

from fastapi import APIRouter, File, UploadFile, HTTPException, status, BackgroundTasks, Request, Query
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.models.schemas import (
    BatchResumeParseResponse,
    ErrorResponse, 
    HealthResponse
)
from app.services.file_processor import FileProcessor
from app.services.openai_service import OpenAIService
from app.services.database_service import DatabaseService
from app.controllers._process_single_file import _process_single_file

from app.config.settings import settings

# Global tracking for bulk processing jobs
bulk_processing_jobs = {}

# Configure logging
logger = logging.getLogger(__name__)

def _normalize_skills(skills_text: str) -> str:
    """
    Pass through skills text without normalization.
    Let GPT Text-Embedding-3-Small handle all variations naturally.
    """
    if not skills_text:
        return ""
    
    # No normalization - trust GPT's natural understanding
    return skills_text

async def _extract_resume_files_from_zip(zip_file: UploadFile) -> List[Dict[str, Any]]:
    """
    Extract resume files from a zip file containing folders.
    
    Args:
        zip_file: Uploaded zip file
        
    Returns:
        List of file data and metadata
    """
    extracted_files = []
    
    try:
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save zip file to temp directory
            zip_path = os.path.join(temp_dir, zip_file.filename)
            with open(zip_path, "wb") as f:
                content = await zip_file.read()
                f.write(content)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Recursively find all resume files
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_extension = os.path.splitext(file)[1].lower()
                    
                    # Check if it's a supported resume file
                    if file_extension in settings.ALLOWED_EXTENSIONS:
                        # Read file content
                        with open(file_path, "rb") as f:
                            file_content = f.read()
                        
                        # Get relative path from zip root
                        rel_path = os.path.relpath(file_path, temp_dir)
                        
                        extracted_files.append({
                            "filename": rel_path,  # Keep folder structure in filename
                            "content": file_content,
                            "size": len(file_content),
                            "extension": file_extension
                        })
            
            logger.info(f"Extracted {len(extracted_files)} resume files from zip: {zip_file.filename}")
            return extracted_files
            
    except Exception as e:
        logger.error(f"Error extracting zip file {zip_file.filename}: {str(e)}")
        raise Exception(f"Failed to extract zip file: {str(e)}")

# Create router
router = APIRouter(prefix="/api/v1", tags=["resume"])

# Initialize rate limiter for 1000+ users
limiter = Limiter(key_func=get_remote_address)

# Initialize services
file_processor = FileProcessor()
openai_service = OpenAIService()
database_service = DatabaseService()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for resume service.
    
    Returns:
        HealthResponse: Service health status
    """
    try:
        return HealthResponse(
            status="healthy",
            version=settings.APP_VERSION,
            timestamp=str(time.time())
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/resume-embeddings-status")
async def get_resume_embeddings_status():
    """
    Get comprehensive status of resume embeddings in the database.
    Shows embedding progress, modes, and detailed statistics.
    
    Returns:
        Dict: Comprehensive status information about resume embeddings
    """
    try:
        # Get total resumes
        all_resumes = await database_service.get_all_resumes(limit=1000)
        total_resumes = len(all_resumes)
        
        # Count resumes with embeddings and analyze embedding sources
        resumes_with_embeddings = 0
        automatic_embeddings = 0
        manual_embeddings = 0
        embedding_errors = 0
        recent_embeddings = 0
        current_time = time.time()
        
        # Handle case where no resumes exist
        if not all_resumes:
            return {
                "total_resumes": 0,
                "resumes_with_embeddings": 0,
                "resumes_without_embeddings": 0,
                "embedding_percentage": 0,
                "embedding_breakdown": {
                    "automatic_embeddings": 0,
                    "manual_embeddings": 0,
                    "automatic_percentage": 0,
                    "manual_percentage": 0
                },
                "recent_activity": {
                    "recent_embeddings_24h": 0,
                    "embedding_errors": 0
                },
                "embedding_modes": {
                    "automatic_mode": "✅ Active - Embeddings generated automatically when resumes are uploaded",
                    "manual_mode": "✅ Available - Use POST /api/v1/generate-resume-embeddings as backup",
                    "status": "operational"
                },
                "recommendations": {
                    "use_automatic": "Primary method - Embeddings generated automatically",
                    "use_manual": "Backup method - Call /api/v1/generate-resume-embeddings if needed",
                    "check_errors": "No resumes found"
                }
            }
        
        for resume in all_resumes:
            has_embedding = False
            embedding_source = "none"
            
            # Check separate embedding column first (automatic)
            if resume.get('embedding'):
                has_embedding = True
                embedding_source = "automatic"
                automatic_embeddings += 1
                
                # Check if embedding was created recently (last 24 hours)
                created_at = resume.get('created_at', 0)
                try:
                    # Convert to float if it's a string
                    if isinstance(created_at, str):
                        created_at = float(created_at)
                    elif created_at is None:
                        created_at = 0
                    
                    if current_time - created_at < 86400:  # 24 hours
                        recent_embeddings += 1
                except (ValueError, TypeError):
                    # Skip if created_at is not a valid number
                    pass
            
            # Fallback: check parsed_data for backward compatibility (manual)
            if not has_embedding:
                parsed_data = resume.get('parsed_data', {})
                
                # Handle parsed_data that might be a JSON string
                if isinstance(parsed_data, str):
                    try:
                        parsed_data = json.loads(parsed_data)
                    except (json.JSONDecodeError, TypeError):
                        continue
                
                if parsed_data and isinstance(parsed_data, dict):
                    if 'embedding' in parsed_data and parsed_data['embedding']:
                        has_embedding = True
                        embedding_source = "manual"
                        manual_embeddings += 1
            
            if has_embedding:
                resumes_with_embeddings += 1
            else:
                # Check if there was an embedding error
                if resume.get('embedding_error'):
                    embedding_errors += 1
        
        # Calculate statistics
        resumes_without_embeddings = total_resumes - resumes_with_embeddings
        embedding_percentage = (resumes_with_embeddings / total_resumes * 100) if total_resumes > 0 else 0
        automatic_percentage = (automatic_embeddings / total_resumes * 100) if total_resumes > 0 else 0
        manual_percentage = (manual_embeddings / total_resumes * 100) if total_resumes > 0 else 0
        
        return {
            "total_resumes": total_resumes,
            "resumes_with_embeddings": resumes_with_embeddings,
            "resumes_without_embeddings": resumes_without_embeddings,
            "embedding_percentage": round(embedding_percentage, 2),
            "embedding_breakdown": {
                "automatic_embeddings": automatic_embeddings,
                "manual_embeddings": manual_embeddings,
                "automatic_percentage": round(automatic_percentage, 2),
                "manual_percentage": round(manual_percentage, 2)
            },
            "recent_activity": {
                "recent_embeddings_24h": recent_embeddings,
                "embedding_errors": embedding_errors
            },
            "embedding_modes": {
                "automatic_mode": "✅ Active - Embeddings generated automatically when resumes are uploaded",
                "manual_mode": "✅ Available - Use POST /api/v1/generate-resume-embeddings as backup",
                "status": "operational"
            },
            "recommendations": {
                "use_automatic": "Primary method - Embeddings generated automatically",
                "use_manual": "Backup method - Call /api/v1/generate-resume-embeddings if needed",
                "check_errors": f"Found {embedding_errors} resumes with embedding errors"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting resume embeddings status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resume embeddings status: {str(e)}"
        )

@router.post("/parse-resume", response_model=BatchResumeParseResponse)
@limiter.limit("50/minute")  # Rate limit: 50 requests per minute per IP for 1000+ users
async def parse_resume(request: Request, file: UploadFile = File(...), background_tasks: BackgroundTasks = None, company_id: int = Query(None, description="Company ID for data isolation")):
    """
    Parse a single resume file with high-scale processing support for 1000+ concurrent users.
    Supports single file upload (1 resume per request) with async queue processing.
    Handles 1000+ users simultaneously with Redis queue and background workers.
    Supports various file formats: PDF, DOCX, DOC, TXT, RTF, PNG, JPG, JPEG, WEBP
    
    Args:
        file (UploadFile): Single resume file to parse (1 file only)
        
    Returns:
        BatchResumeParseResponse: Processing results with job tracking
        
    Raises:
        HTTPException: If file processing or parsing fails
    """
    start_time = time.time()
    
    # Debug logging
    logger.info(f"🔍 Resume Parse Debug - File received: {file.filename if file else 'None'}")
    logger.info(f"🔍 Resume Parse Debug - File size: {file.size if file else 'None'}")
    logger.info(f"🔍 Resume Parse Debug - Content type: {file.content_type if file else 'None'}")
    
    if not file:
        logger.error("❌ No file provided")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Check if queue system is available for high-scale processing
    try:
        from app.services.queue_service import queue_service
        
        # If Redis is available, use queue-based processing for high-scale
        if queue_service.redis_client:
            return await _process_with_queue([file], start_time)
        else:
            logger.warning("⚠️ Queue system not available, falling back to synchronous processing")
    except Exception as e:
        logger.warning(f"⚠️ Queue system error: {str(e)}, falling back to synchronous processing")
    
    # Fallback to synchronous processing for low-scale usage
    results = []
    successful_files = 0
    failed_files = 0
    batch_data_to_save = []
    
    try:
        # Process single file
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
                failed_files += 1
            else:
                # Check file extension
                file_extension = os.path.splitext(file.filename)[1].lower()
                if file_extension not in settings.ALLOWED_EXTENSIONS:
                    file_result["error"] = f"Unsupported file format. Supported formats: {', '.join(settings.ALLOWED_EXTENSIONS)}"
                    file_result["file_type"] = file_extension.lstrip('.')
                    failed_files += 1
                else:
                    # Check file size
                    file_content = await file.read()
                    file_size = len(file_content)
                    if file_size > settings.MAX_FILE_SIZE:
                        file_result["error"] = f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE} bytes"
                        file_result["file_type"] = file_extension.lstrip('.')
                        failed_files += 1
                    else:
                        # Process the file with company isolation
                        await _process_single_file(file, file_content, file_size, file_extension, file_result, batch_data_to_save, results, successful_files, failed_files, company_id)
        
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
                # Get company_id from request query parameters
                company_id = request.query_params.get('company_id')
                if company_id:
                    company_id = int(company_id)
                else:
                    company_id = None
                
                record_ids = await database_service.save_batch_resume_data(batch_data_to_save, company_id)
                logger.info(f"Successfully saved {len(record_ids)} resume records to database for company: {company_id}")
                
                # Save embeddings immediately for resumes that have them
                for i, (resume_data, db_id) in enumerate(zip(batch_data_to_save, record_ids)):
                    if 'embedding' in resume_data and resume_data['embedding']:
                        try:
                            await database_service.update_resume_embedding_column(
                                db_id, 
                                resume_data['embedding']
                            )
                            logger.info(f"✅ Embedding saved immediately for resume {i+1}/{len(batch_data_to_save)} (DB ID: {db_id})")
                        except Exception as e:
                            logger.error(f"❌ Error saving embedding for resume {i+1} (DB ID: {db_id}): {str(e)}")
                
            except Exception as e:
                logger.error(f"Error saving batch data to database: {str(e)}")
        
        total_processing_time = time.time() - start_time
        
        return BatchResumeParseResponse(
            total_files=1,
            successful_files=successful_files,
            failed_files=failed_files,
            total_processing_time=total_processing_time,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Error in resume parsing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process resume: {str(e)}"
        )

async def _check_resume_uniqueness(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if resume is unique based on name, email, phone, and skills.
    
    Args:
        parsed_data: Parsed resume data
        
    Returns:
        Dict with uniqueness status and details
    """
    try:
        # Extract key fields for uniqueness check
        name = parsed_data.get('Name', '').strip().lower()
        email = parsed_data.get('Email', '').strip().lower()
        phone = parsed_data.get('Phone', '').strip()
        skills = parsed_data.get('Skills', [])
        
        # Normalize skills
        if isinstance(skills, list):
            skills_normalized = [skill.strip().lower() for skill in skills if skill.strip()]
        else:
            skills_normalized = [str(skills).strip().lower()] if str(skills).strip() else []
        
        # Check if required fields are present
        missing_fields = []
        if not name:
            missing_fields.append("Name")
        if not email:
            missing_fields.append("Email")
        if not phone:
            missing_fields.append("Phone")
        if not skills_normalized:
            missing_fields.append("Skills")
        
        if missing_fields:
            return {
                "is_unique": False,
                "reason": "missing_required_fields",
                "missing_fields": missing_fields,
                "error": f"Resume parsing failed: Missing required fields: {', '.join(missing_fields)}"
            }
        
        # Check against existing resumes in database
        existing_resumes = await database_service.get_all_resumes(limit=10000)
        
        for existing_resume in existing_resumes:
            existing_parsed_data = existing_resume.get('parsed_data', {})
            if isinstance(existing_parsed_data, str):
                try:
                    existing_parsed_data = json.loads(existing_parsed_data)
                except:
                    continue
            
            if not isinstance(existing_parsed_data, dict):
                continue
            
            # Check name match
            existing_name = existing_parsed_data.get('Name', '').strip().lower()
            if name and existing_name and name == existing_name:
                return {
                    "is_unique": False,
                    "reason": "duplicate_name",
                    "duplicate_field": "Name",
                    "duplicate_value": name,
                    "error": f"Duplicate resume found with same name: {name}"
                }
            
            # Check email match
            existing_email = existing_parsed_data.get('Email', '').strip().lower()
            if email and existing_email and email == existing_email:
                return {
                    "is_unique": False,
                    "reason": "duplicate_email",
                    "duplicate_field": "Email",
                    "duplicate_value": email,
                    "error": f"Duplicate resume found with same email: {email}"
                }
            
            # Check phone match
            existing_phone = existing_parsed_data.get('Phone', '').strip()
            if phone and existing_phone and phone == existing_phone:
                return {
                    "is_unique": False,
                    "reason": "duplicate_phone",
                    "duplicate_field": "Phone",
                    "duplicate_value": phone,
                    "error": f"Duplicate resume found with same phone: {phone}"
                }
            
            # Check skills match (80% similarity)
            existing_skills = existing_parsed_data.get('Skills', [])
            if isinstance(existing_skills, list):
                existing_skills_normalized = [skill.strip().lower() for skill in existing_skills if skill.strip()]
            else:
                existing_skills_normalized = [str(existing_skills).strip().lower()] if str(existing_skills).strip() else []
            
            if skills_normalized and existing_skills_normalized:
                # Calculate skills similarity
                common_skills = set(skills_normalized) & set(existing_skills_normalized)
                total_skills = len(set(skills_normalized) | set(existing_skills_normalized))
                similarity = len(common_skills) / total_skills if total_skills > 0 else 0
                
                if similarity >= 0.8:  # 80% similarity threshold
                    return {
                        "is_unique": False,
                        "reason": "duplicate_skills",
                        "duplicate_field": "Skills",
                        "similarity_percentage": round(similarity * 100, 2),
                        "error": f"Duplicate resume found with similar skills ({round(similarity * 100, 2)}% match)"
                    }
        
        return {
            "is_unique": True,
            "reason": "unique_resume",
            "extracted_fields": {
                "name": name,
                "email": email,
                "phone": phone,
                "skills_count": len(skills_normalized)
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking resume uniqueness: {str(e)}")
        return {
            "is_unique": False,
            "reason": "uniqueness_check_failed",
            "error": f"Failed to check uniqueness: {str(e)}"
        }

async def _retry_failed_embeddings():
    """Retry failed embeddings in the background."""
    try:
        # Get resumes with embedding errors
        all_resumes = await database_service.get_all_resumes(limit=1000)
        failed_embeddings = []
        
        for resume in all_resumes:
            if resume.get('embedding_error') and not resume.get('embedding'):
                failed_embeddings.append(resume)
        
        if failed_embeddings:
            logger.info(f"🔄 Found {len(failed_embeddings)} resumes with failed embeddings, retrying...")
            
            for resume in failed_embeddings:
                resume_id = resume.get('id')
                parsed_data = resume.get('parsed_data', {})
                
                if isinstance(parsed_data, str):
                    try:
                        parsed_data = json.loads(parsed_data)
                    except (json.JSONDecodeError, TypeError):
                        continue
                
                if parsed_data and isinstance(parsed_data, dict):
                    # Retry embedding generation
                    await _generate_embedding_background(parsed_data, resume_id, 0, 3)
                    await asyncio.sleep(5)  # Wait 5 seconds between retries
        
    except Exception as e:
        logger.error(f"Error in retry failed embeddings: {str(e)}")

async def _generate_embedding_background(parsed_data: Dict[str, Any], resume_id: str, retry_count: int = 0, max_retries: int = 3):
    """Generate embedding for a resume in the background with automatic retry."""
    try:
        # Generate embedding for the resume
        embedding = await openai_service.generate_resume_embedding(parsed_data)
        
        if embedding:
            # Update the resume with embedding in separate column
            await database_service.update_resume_embedding_column(resume_id, embedding)
            logger.info(f"✅ Embedding generated successfully for resume {resume_id}")
            return True
        else:
            logger.warning(f"⚠️ Failed to generate embedding for resume {resume_id} (attempt {retry_count + 1})")
            
            # Retry if we haven't exceeded max retries
            if retry_count < max_retries:
                logger.info(f"🔄 Retrying embedding generation for resume {resume_id} in 30 seconds...")
                await asyncio.sleep(30)  # Wait 30 seconds before retry
                return await _generate_embedding_background(parsed_data, resume_id, retry_count + 1, max_retries)
            else:
                # Mark as failed after max retries
                await database_service.update_resume_embedding_error(resume_id, f"Failed after {max_retries} attempts")
                logger.error(f"❌ Embedding generation failed permanently for resume {resume_id} after {max_retries} attempts")
                return False
            
    except Exception as e:
        logger.error(f"❌ Error generating embedding for resume {resume_id} (attempt {retry_count + 1}): {str(e)}")
        
        # Retry if we haven't exceeded max retries
        if retry_count < max_retries:
            logger.info(f"🔄 Retrying embedding generation for resume {resume_id} in 30 seconds...")
            await asyncio.sleep(30)  # Wait 30 seconds before retry
            return await _generate_embedding_background(parsed_data, resume_id, retry_count + 1, max_retries)
        else:
            # Mark as failed after max retries
            await database_service.update_resume_embedding_error(resume_id, f"Error after {max_retries} attempts: {str(e)}")
            logger.error(f"❌ Embedding generation failed permanently for resume {resume_id} after {max_retries} attempts")
            return False

async def _process_single_file_from_data(file_data: Dict[str, Any], file_result: Dict[str, Any], batch_data_to_save: List[Dict], results: List[Dict], successful_files: int, failed_files: int, duplicate_files: int):
    """Process a single file from extracted data (zip or regular file) with uniqueness check."""
    file_start_time = time.time()  # Start timing the file processing
    try:
        # Create upload and failed folders
        os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(os.path.join(settings.UPLOAD_FOLDER, "failed"), exist_ok=True)
        
        # Process file and extract text
        extracted_text = await file_processor.process_file(file_data["content"], file_data["filename"])
        
        if not extracted_text or not extracted_text.strip():
            file_result["error"] = "No text could be extracted from the file"
            file_result["file_type"] = file_data["extension"].lstrip('.')
            file_result["processing_time"] = time.time() - file_start_time
            failed_files[0] += 1
            return
        
        # Parse resume with AI
        parsed_data = await openai_service.parse_resume_text(extracted_text)
        
        # Generate embedding immediately after parsing
        embedding = None
        try:
            embedding = await openai_service.generate_embedding(str(parsed_data))
            if embedding:
                logger.info(f"✅ Embedding generated immediately for resume: {file_data['filename']}")
            else:
                logger.warning(f"⚠️ Failed to generate embedding for resume: {file_data['filename']}")
        except Exception as e:
            logger.error(f"❌ Error generating embedding for resume {file_data['filename']}: {str(e)}")
        
        # Check uniqueness for bulk parsing API only
        uniqueness_check = await _check_resume_uniqueness(parsed_data)
        
        if not uniqueness_check["is_unique"]:
            # Save to failed folder
            file_uuid = str(uuid.uuid4())
            unique_filename = f"{file_uuid}{file_data['extension']}"
            failed_file_path = os.path.join(settings.UPLOAD_FOLDER, "failed", unique_filename)
            
            with open(failed_file_path, "wb") as f:
                f.write(file_data["content"])
            
            # Save failure metadata
            metadata = {
                "failure_reason": uniqueness_check["error"],
                "failure_type": uniqueness_check["reason"],
                "original_filename": file_data["filename"],
                "failed_at": time.time(),
                "uniqueness_check": uniqueness_check
            }
            
            metadata_file = os.path.join(settings.UPLOAD_FOLDER, "failed", f"{file_uuid}.metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            file_result.update({
                "status": "duplicate",
                "error": uniqueness_check["error"],
                "file_type": file_data["extension"].lstrip('.'),
                "processing_time": time.time() - file_start_time,
                "uniqueness_check": uniqueness_check,
                "failed_file_path": failed_file_path,
                "resume_id": file_uuid,
                "failure_reason": uniqueness_check["error"],
                "failure_type": uniqueness_check["reason"]
            })
            duplicate_files[0] += 1
            logger.warning(f"Duplicate resume rejected: {file_data['filename']} - {uniqueness_check['error']}")
            return
        
        # Resume is unique, save to upload folder
        file_uuid = str(uuid.uuid4())
        unique_filename = f"{file_uuid}{file_data['extension']}"
        file_path = os.path.join(settings.UPLOAD_FOLDER, unique_filename)
        
        with open(file_path, "wb") as f:
            f.write(file_data["content"])
        
        # Calculate processing time
        file_processing_time = time.time() - file_start_time
        
        # Update result for successful processing
        file_result.update({
            "status": "success",
            "parsed_data": parsed_data,
            "file_type": file_data["extension"].lstrip('.'),
            "processing_time": file_processing_time,
            "uniqueness_check": uniqueness_check,
            "embedding_status": "completed" if embedding else "failed",
            "embedding_generated": embedding is not None
        })
        
        # Add to batch save data
        batch_data_to_save.append({
            "filename": file_data["filename"],
            "file_path": file_path,
            "file_type": file_data["extension"].lstrip('.'),
            "file_size": file_data["size"],
            "processing_time": file_processing_time,
            "parsed_data": parsed_data,
            "resume_id": file_uuid,
            "embedding": embedding  # Include embedding in batch data
        })
        
        successful_files[0] += 1
        logger.info(f"Successfully parsed unique resume: {file_data['filename']}")
        
    except Exception as e:
        file_result.update({
            "error": f"Failed to process file: {str(e)}"
        })
        failed_files[0] += 1
        logger.error(f"Error processing file {file_data['filename']}: {str(e)}")

async def _process_with_queue(files: List[UploadFile], start_time: float) -> BatchResumeParseResponse:
    """
    Process resumes using queue system for high-scale processing.
    
    Args:
        files: List of resume files
        start_time: Processing start time
        
    Returns:
        BatchResumeParseResponse: Queue processing results
    """
    from app.services.queue_service import queue_service
    
    results = []
    job_ids = []
    
    try:
        for file in files:
            # Validate file
            if not file.filename:
                results.append({
                    "filename": "unknown",
                    "status": "failed",
                    "error": "No filename provided",
                    "parsed_data": None,
                    "file_type": None,
                    "processing_time": 0,
                    "job_id": None
                })
                continue
            
            # Check file size
            file_content = await file.read()
            if len(file_content) > settings.MAX_FILE_SIZE:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes",
                    "parsed_data": None,
                    "file_type": file.filename.split('.')[-1],
                    "processing_time": 0,
                    "job_id": None
                })
                continue
            
            # Check file extension
            file_extension = os.path.splitext(file.filename)[1].lower()
            if file_extension not in settings.ALLOWED_EXTENSIONS:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": f"Unsupported file format. Supported formats: {', '.join(settings.ALLOWED_EXTENSIONS)}",
                    "parsed_data": None,
                    "file_type": file_extension.lstrip('.'),
                    "processing_time": 0,
                    "job_id": None
                })
                continue
            
            # Add to queue
            try:
                job_id = await queue_service.add_resume_job(
                    file_data=file_content,
                    filename=file.filename
                )
                
                results.append({
                    "filename": file.filename,
                    "status": "queued",
                    "error": None,
                    "parsed_data": None,
                    "file_type": file_extension.lstrip('.'),
                    "processing_time": 0,
                    "job_id": job_id,
                    "message": "Resume queued for processing. Use job_id to check status."
                })
                job_ids.append(job_id)
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": f"Failed to queue resume: {str(e)}",
                    "parsed_data": None,
                    "file_type": file_extension.lstrip('.'),
                    "processing_time": 0,
                    "job_id": None
                })
        
        total_processing_time = time.time() - start_time
        successful_files = len([r for r in results if r["status"] == "queued"])
        failed_files = len([r for r in results if r["status"] == "failed"])
        
        return BatchResumeParseResponse(
            total_files=len(files),
            successful_files=successful_files,
            failed_files=failed_files,
            total_processing_time=total_processing_time,
            results=results,
            queue_info={
                "processing_mode": "async_queue",
                "job_ids": job_ids,
                "status_endpoint": "/api/v1/job-status/{job_id}",
                "estimated_processing_time": "2-5 minutes"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in queue processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process resumes with queue: {str(e)}"
        )

@router.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get processing status for a queued resume job.
    
    Args:
        job_id: Job identifier from queue processing
        
    Returns:
        Dict: Job status and results
    """
    try:
        from app.services.queue_service import queue_service
        status_info = await queue_service.get_job_status(job_id)
        
        if "error" in status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=status_info["error"]
            )
        
        return status_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )

@router.get("/candaidte-job-appliction-reume-queue-status")
async def get_queue_status():
    """
    Get current queue status and statistics.
    
    Returns:
        Dict: Queue statistics
    """
    try:
        from app.services.queue_service import queue_service
        
        # Check if Redis is available
        if not queue_service.redis_client:
            return {
                "queue_length": 0,
                "status": "redis_unavailable",
                "max_concurrent_jobs": 1000,
                "estimated_processing_time": "N/A - Redis not available",
                "processing_mode": "synchronous_fallback",
                "message": "Redis server not available. Using synchronous processing fallback.",
                "redis_status": "disconnected"
            }
        
        queue_length = await queue_service.get_queue_length()
        
        return {
            "queue_length": queue_length,
            "status": "operational",
            "max_concurrent_jobs": 1000,
            "estimated_processing_time": f"{queue_length * 2} minutes",
            "processing_mode": "async_queue",
            "redis_status": "connected"
        }
        
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        return {
            "queue_length": 0,
            "status": "error",
            "max_concurrent_jobs": 1000,
            "estimated_processing_time": "N/A",
            "processing_mode": "synchronous_fallback",
            "error": str(e),
            "redis_status": "error"
        }

@router.post("/bulk-parse-resumes")
@limiter.limit("5/minute")  # Rate limit: 5 bulk requests per minute per IP
async def bulk_parse_resumes(request: Request, files: List[UploadFile] = File(...)):
    """
    Bulk parse multiple resume files with high-scale processing support for 1000+ users.
    Supports unlimited file uploads (1000, 20000+ resumes) with async queue processing.
    Handles folder uploads with nested directories and various file formats.
    Supports: PDF, DOCX, DOC, TXT, RTF, PNG, JPG, JPEG, WEBP
    
    IMPORTANT: For folder uploads, you need to:
    1. Zip the folder containing resume files
    2. Upload the zip file as a single file
    3. The system will extract and process all resume files inside
    
    FOR RE-UPLOADING FAILED RESUMES:
    - Use the separate /api/v1/re-upload-failed-resumes endpoint
    - This endpoint only handles new file uploads
    
    Args:
        files: Multiple resume files to parse (unlimited count) OR zip files containing folders
        
    Returns:
        Dict: Bulk processing results with job tracking
        
    Raises:
        HTTPException: If file processing or parsing fails
    """
    start_time = time.time()
    
    # Create a bulk processing job ID
    bulk_job_id = str(uuid.uuid4())
    user_id = getattr(request.client, 'host', 'unknown') if hasattr(request, 'client') else 'unknown'
    
    # Track the bulk processing job
    bulk_processing_jobs[bulk_job_id] = {
        "job_id": bulk_job_id,
        "status": "processing",
        "user_id": user_id,
        "created_at": time.time(),
        "updated_at": time.time(),
        "total_files": 0,
        "processed_files": 0,
        "successful_files": 0,
        "failed_files": 0,
        "duplicate_files": 0,
        "progress": "0"
    }
    
    # Debug logging
    logger.info(f"Received request - files: {files}")
    logger.info(f"Files type: {type(files)}")
    
    # Check if we have files
    if not files or len(files) == 0:
        logger.error("No files provided")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    # Validate file count
    if len(files) > 50000:  # Reasonable limit for bulk processing
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many files. Maximum 50,000 files per bulk request."
        )
    
    # Check if queue system is available for high-scale processing
    try:
        from app.services.queue_service import queue_service
        
        # If Redis is available, use queue-based processing for high-scale
        if queue_service.redis_client:
            return await _process_bulk_with_queue(files, start_time)
        else:
            logger.warning("⚠️ Queue system not available, falling back to synchronous processing")
    except Exception as e:
        logger.warning(f"⚠️ Queue system error: {str(e)}, falling back to synchronous processing")
    
    # Fallback to synchronous processing for low-scale usage
    results = []
    successful_files = [0]  # Use list to make it mutable
    failed_files = [0]      # Use list to make it mutable
    duplicate_files = [0]   # Use list to make it mutable
    batch_data_to_save = []
    all_files_to_process = []
    
    try:
        # Extract files from zip files and collect all files to process
        for file in files:
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            if file_extension == '.zip':
                # Extract resume files from zip
                try:
                    extracted_files = await _extract_resume_files_from_zip(file)
                    for extracted_file in extracted_files:
                        all_files_to_process.append({
                            "filename": extracted_file["filename"],
                            "content": extracted_file["content"],
                            "size": extracted_file["size"],
                            "extension": extracted_file["extension"],
                            "is_from_zip": True,
                            "original_zip": file.filename
                        })
                except Exception as e:
                    logger.error(f"Error processing zip file {file.filename}: {str(e)}")
                    results.append({
                        "filename": file.filename,
                        "status": "failed",
                        "error": f"Failed to extract zip file: {str(e)}",
                        "parsed_data": None,
                        "file_type": "zip",
                        "processing_time": 0,
                        "file_index": len(results) + 1
                    })
                    failed_files[0] += 1
            else:
                # Regular file
                file_content = await file.read()
                file_size = len(file_content)
                file_ext = os.path.splitext(file.filename)[1].lower()
                
                all_files_to_process.append({
                    "filename": file.filename,
                    "content": file_content,
                    "size": file_size,
                    "extension": file_ext,
                    "is_from_zip": False,
                    "original_zip": None
                })
        
        # Update job status with total files count
        if bulk_job_id in bulk_processing_jobs:
            bulk_processing_jobs[bulk_job_id].update({
                "total_files": len(all_files_to_process),
                "updated_at": time.time()
            })
        
        # Process all collected files
        for i, file_data in enumerate(all_files_to_process):
            # Check if job was cancelled
            if bulk_job_id in bulk_processing_jobs and bulk_processing_jobs[bulk_job_id].get("status") == "cancelled":
                logger.info(f"Job {bulk_job_id} was cancelled, stopping processing")
                break
                
            file_start_time = time.time()
            file_result = {
                "filename": file_data["filename"],
                "status": "failed",
                "error": None,
                "parsed_data": None,
                "file_type": None,
                "processing_time": 0,
                "file_index": i + 1,
                "is_from_zip": file_data["is_from_zip"],
                "original_zip": file_data["original_zip"]
            }
            
            try:
                # Validate file
                        if not file_data["filename"]:
                            file_result["error"] = "No filename provided"
                            failed_files[0] += 1
                        else:
                            # Check file extension
                            file_extension = file_data["extension"]
                            if file_extension not in settings.ALLOWED_EXTENSIONS:
                                file_result["error"] = f"Unsupported file format. Supported formats: {', '.join(settings.ALLOWED_EXTENSIONS)}"
                                file_result["file_type"] = file_extension.lstrip('.')
                                failed_files[0] += 1
                            else:
                                # Check file size
                                file_size = file_data["size"]
                                if file_size > settings.MAX_FILE_SIZE:
                                    file_result["error"] = f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE} bytes"
                                    file_result["file_type"] = file_extension.lstrip('.')
                                    failed_files[0] += 1
                                else:
                                    # Check if job was cancelled before processing
                                    if bulk_job_id in bulk_processing_jobs and bulk_processing_jobs[bulk_job_id].get("status") == "cancelled":
                                        logger.info(f"Job {bulk_job_id} was cancelled, skipping file processing")
                                        file_result["error"] = "Processing cancelled by user"
                                        failed_files[0] += 1
                                    else:
                                        # Process the file
                                        await _process_single_file_from_data(file_data, file_result, batch_data_to_save, results, successful_files, failed_files, duplicate_files)
            
            except Exception as e:
                file_processing_time = time.time() - file_start_time
                file_result.update({
                    "error": f"Failed to process file: {str(e)}",
                    "processing_time": file_processing_time
                })
                failed_files[0] += 1
                logger.error(f"Error processing file {file.filename}: {str(e)}")
            
            results.append(file_result)
            
            # Update job progress and store results incrementally
            if bulk_job_id in bulk_processing_jobs:
                progress_percentage = round(((i + 1) / len(all_files_to_process)) * 100, 2)
                bulk_processing_jobs[bulk_job_id].update({
                    "processed_files": i + 1,
                    "successful_files": successful_files[0],
                    "failed_files": failed_files[0],
                    "duplicate_files": duplicate_files[0],
                    "progress": str(progress_percentage),
                    "updated_at": time.time(),
                    "results": results.copy()  # Store current results incrementally
                })
            
            # Progress update every 100 files
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(all_files_to_process)} files...")
        
        # Save successful files to database in batch
        if batch_data_to_save:
            try:
                # Get company_id from request query parameters
                company_id = request.query_params.get('company_id')
                if company_id:
                    company_id = int(company_id)
                else:
                    company_id = None
                
                record_ids = await database_service.save_batch_resume_data(batch_data_to_save, company_id)
                logger.info(f"Successfully saved {len(record_ids)} resume records to database for company: {company_id}")
                
                # Save embeddings immediately for resumes that have them
                for i, (resume_data, db_id) in enumerate(zip(batch_data_to_save, record_ids)):
                    if 'embedding' in resume_data and resume_data['embedding']:
                        try:
                            await database_service.update_resume_embedding_column(
                                db_id, 
                                resume_data['embedding']
                            )
                            logger.info(f"✅ Embedding saved immediately for resume {i+1}/{len(batch_data_to_save)} (DB ID: {db_id})")
                        except Exception as e:
                            logger.error(f"❌ Error saving embedding for resume {i+1} (DB ID: {db_id}): {str(e)}")
                
            except Exception as e:
                logger.error(f"Error saving batch data to database: {str(e)}")
        
        total_processing_time = time.time() - start_time
        
        # Update job status to completed
        if bulk_job_id in bulk_processing_jobs:
            bulk_processing_jobs[bulk_job_id].update({
                "status": "completed",
                "updated_at": time.time(),
                "total_files": len(all_files_to_process),
                "processed_files": len(all_files_to_process),
                "successful_files": successful_files[0],
                "failed_files": failed_files[0],
                "duplicate_files": duplicate_files[0],
                "progress": "100",
                "total_processing_time": total_processing_time,
                "results": results
            })
        
        return {
            "total_files": len(all_files_to_process),
            "successful_files": successful_files[0],
            "failed_files": failed_files[0],
            "duplicate_files": duplicate_files[0],
            "total_processing_time": total_processing_time,
            "results": results,
            "processing_mode": "synchronous_bulk",
            "success_rate": round((successful_files[0] / len(all_files_to_process) * 100) if len(all_files_to_process) > 0 else 0, 2),
            "duplicate_rate": round((duplicate_files[0] / len(all_files_to_process) * 100) if len(all_files_to_process) > 0 else 0, 2),
            "zip_files_processed": len([f for f in files if f.filename.endswith('.zip')]),
            "extracted_files_count": len(all_files_to_process),
            "bulk_job_id": bulk_job_id
        }
        
    except Exception as e:
        logger.error(f"Error in bulk resume parsing: {str(e)}")
        
        # Update job status to failed
        if bulk_job_id in bulk_processing_jobs:
            bulk_processing_jobs[bulk_job_id].update({
                "status": "failed",
                "updated_at": time.time(),
                "error": str(e),
                "progress": "0"
            })
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process bulk resumes: {str(e)}"
        )

async def _process_bulk_with_queue(files: List[UploadFile], start_time: float) -> Dict[str, Any]:
    """
    Process bulk resumes using queue system for high-scale processing.
    
    Args:
        files: List of resume files
        start_time: Processing start time
        
    Returns:
        Dict: Queue processing results
    """
    from app.services.queue_service import queue_service
    
    results = []
    job_ids = []
    successful_queued = 0
    failed_files = 0
    
    try:
        for i, file in enumerate(files):
            # Validate file
            if not file.filename:
                results.append({
                    "filename": "unknown",
                    "status": "failed",
                    "error": "No filename provided",
                    "parsed_data": None,
                    "file_type": None,
                    "processing_time": 0,
                    "file_index": i + 1,
                    "job_id": None
                })
                failed_files += 1
                continue
            
            # Check file size
            file_content = await file.read()
            if len(file_content) > settings.MAX_FILE_SIZE:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes",
                    "parsed_data": None,
                    "file_type": file.filename.split('.')[-1],
                    "processing_time": 0,
                    "file_index": i + 1,
                    "job_id": None
                })
                failed_files += 1
                continue
            
            # Check file extension
            file_extension = os.path.splitext(file.filename)[1].lower()
            if file_extension not in settings.ALLOWED_EXTENSIONS:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": f"Unsupported file format. Supported formats: {', '.join(settings.ALLOWED_EXTENSIONS)}",
                    "parsed_data": None,
                    "file_type": file_extension.lstrip('.'),
                    "processing_time": 0,
                    "file_index": i + 1,
                    "job_id": None
                })
                failed_files += 1
                continue
            
            # Add to queue
            try:
                job_id = await queue_service.add_resume_job(
                    file_data=file_content,
                    filename=file.filename
                )
                
                results.append({
                    "filename": file.filename,
                    "status": "queued",
                    "error": None,
                    "parsed_data": None,
                    "file_type": file_extension.lstrip('.'),
                    "processing_time": 0,
                    "file_index": i + 1,
                    "job_id": job_id,
                    "message": "Resume queued for processing. Use job_id to check status."
                })
                job_ids.append(job_id)
                successful_queued += 1
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": f"Failed to queue resume: {str(e)}",
                    "parsed_data": None,
                    "file_type": file_extension.lstrip('.'),
                    "processing_time": 0,
                    "file_index": i + 1,
                    "job_id": None
                })
                failed_files += 1
            
            # Progress update every 100 files
            if (i + 1) % 100 == 0:
                logger.info(f"Queued {i + 1}/{len(files)} files...")
        
        total_processing_time = time.time() - start_time
        
        return {
            "total_files": len(files),
            "successful_files": successful_queued,
            "failed_files": failed_files,
            "total_processing_time": total_processing_time,
            "results": results,
            "processing_mode": "async_queue_bulk",
            "queue_info": {
                "job_ids": job_ids,
                "status_endpoint": "/api/v1/bulk-processing-status/{job_id}",
                "estimated_processing_time": f"{len(job_ids) * 2} minutes",
                "max_concurrent_jobs": 1000
            },
            "success_rate": round((successful_queued / len(files) * 100) if len(files) > 0 else 0, 2)
        }
        
    except Exception as e:
        logger.error(f"Error in bulk queue processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process bulk resumes with queue: {str(e)}"
        )

@router.post("/cancel-job/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a specific resume processing job.
    
    Args:
        job_id: Job identifier to cancel
        
    Returns:
        Dict: Cancellation result
    """
    try:
        from app.services.queue_service import queue_service
        
        if not queue_service.redis_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Queue system not available"
            )
        
        success = await queue_service.cancel_job(job_id)
        
        if success:
            return {
                "message": "Job cancelled successfully",
                "job_id": job_id,
                "status": "cancelled"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found or already completed"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )

@router.post("/cancel-all-jobs")
async def cancel_all_jobs():
    """
    Cancel all active resume processing jobs.
    
    Returns:
        Dict: Cancellation result
    """
    try:
        cancelled_count = 0
        
        # Cancel all jobs in the in-memory tracking
        for job_id, job in bulk_processing_jobs.items():
            if job.get("status") in ["processing", "queued"]:
                job["status"] = "cancelled"
                job["updated_at"] = time.time()
                cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} processing jobs")
        
        return {
            "message": f"Successfully cancelled {cancelled_count} processing jobs",
            "cancelled_count": cancelled_count,
            "total_jobs": len(bulk_processing_jobs),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error cancelling all jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel all jobs: {str(e)}"
        )

@router.get("/bulk-processing-status")
async def get_all_bulk_processing_status():
    """
    Get processing status for all bulk resume jobs from all users.
    Shows how many users are uploading and their progress.
    
    Returns:
        Dict: All users' bulk processing status with user count and progress
    """
    try:
        from app.services.queue_service import queue_service
        
        # Check if Redis is available
        if not queue_service.redis_client:
            # Use global tracking for bulk processing jobs
            total_jobs = len(bulk_processing_jobs)
            active_jobs = len([job for job in bulk_processing_jobs.values() if job.get("status") in ["processing", "queued"]])
            completed_jobs = len([job for job in bulk_processing_jobs.values() if job.get("status") == "completed"])
            failed_jobs = len([job for job in bulk_processing_jobs.values() if job.get("status") == "failed"])
            duplicate_jobs = len([job for job in bulk_processing_jobs.values() if job.get("status") == "duplicate"])
            
            # Count unique users
            unique_users = set(job.get("user_id", "unknown") for job in bulk_processing_jobs.values())
            active_users = set(job.get("user_id", "unknown") for job in bulk_processing_jobs.values() if job.get("status") in ["processing", "queued"])
            
            # Calculate total file counts from all jobs
            total_files = sum(job.get("total_files", 0) for job in bulk_processing_jobs.values())
            successful_files = sum(job.get("successful_files", 0) for job in bulk_processing_jobs.values())
            failed_files = sum(job.get("failed_files", 0) for job in bulk_processing_jobs.values())
            duplicate_files = sum(job.get("duplicate_files", 0) for job in bulk_processing_jobs.values())
            
            # Calculate progress based on processed files
            total_processed_files = successful_files + failed_files + duplicate_files
            progress_percentage = round((total_processed_files / total_files * 100) if total_files > 0 else 0, 2)
            
            # Collect all file results from all jobs
            all_file_results = []
            for job in bulk_processing_jobs.values():
                logger.info(f"Job {job.get('job_id')} status: {job.get('status')}, has results: {bool(job.get('results'))}")
                if job.get("results"):
                    all_file_results.extend(job.get("results", []))
                    logger.info(f"Job {job.get('job_id')} has {len(job.get('results', []))} results")
                # Also check for partial results in active jobs
                elif job.get("status") == "processing" and job.get("processed_files", 0) > 0:
                    logger.info(f"Job {job.get('job_id')} is processing, processed files: {job.get('processed_files', 0)}")
                    # Active jobs now have incremental results
                    if job.get("results"):
                        all_file_results.extend(job.get("results", []))
                        logger.info(f"Active job {job.get('job_id')} has {len(job.get('results', []))} partial results")
            
            logger.info(f"Total file results collected: {len(all_file_results)}")
            
            return {
                "status": "operational",
                "total_jobs": total_jobs,
                "active_jobs": active_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "duplicate_jobs": duplicate_jobs,
                "total_users": len(unique_users),
                "active_users": len(active_users),
                "progress_percentage": progress_percentage,
                "jobs": list(bulk_processing_jobs.values()),
                "file_results": all_file_results,
                "redis_status": "disconnected",
                "debug_info": {
                    "jobs_count": len(bulk_processing_jobs),
                    "file_results_count": len(all_file_results),
                    "job_statuses": [job.get("status") for job in bulk_processing_jobs.values()]
                },
                "summary": {
                    "total_files": total_files,
                    "successful_files": successful_files,
                    "failed_files": failed_files,
                    "duplicate_files": duplicate_files,
                    "total_resumes_uploaded": total_jobs,
                    "users_uploading": len(unique_users),
                    "users_currently_active": len(active_users),
                    "processing_progress": f"{progress_percentage}%",
                    "estimated_completion": "All completed" if active_jobs == 0 else "Processing..."
                }
            }
        
        # Get all job statuses from Redis
        try:
            # Get all job keys
            job_keys = queue_service.redis_client.keys("job_status:*")
            all_jobs = []
            
            for job_key in job_keys:
                job_id = job_key.replace("job_status:", "")
                job_data = queue_service.redis_client.hgetall(job_key)
                
                if job_data:
                    # Parse result if available
                    result = None
                    if job_data.get("result"):
                        try:
                            result = json.loads(job_data["result"])
                        except json.JSONDecodeError:
                            result = {"error": "Invalid result format"}
                    
                    job_info = {
                        "job_id": job_id,
                        "status": job_data.get("status", "unknown"),
                        "created_at": job_data.get("created_at"),
                        "updated_at": job_data.get("updated_at"),
                        "filename": job_data.get("filename"),
                        "progress": job_data.get("progress", "0"),
                        "result": result,
                        "error": job_data.get("error")
                    }
                    all_jobs.append(job_info)
            
            # Count jobs by status
            active_jobs = len([j for j in all_jobs if j["status"] in ["queued", "processing"]])
            completed_jobs = len([j for j in all_jobs if j["status"] == "completed"])
            failed_jobs = len([j for j in all_jobs if j["status"] == "failed"])
            duplicate_jobs = len([j for j in all_jobs if j["status"] == "duplicate"])
            
            # Count unique users (based on IP or user_id if available)
            unique_users = set()
            active_users = set()
            
            for job in all_jobs:
                # Try to get user identifier from job data
                user_id = job.get("user_id") or job.get("ip_address") or "unknown"
                unique_users.add(user_id)
                
                if job["status"] in ["queued", "processing"]:
                    active_users.add(user_id)
            
            # Calculate processing progress
            total_processed = completed_jobs + failed_jobs
            progress_percentage = round((total_processed / len(all_jobs) * 100) if len(all_jobs) > 0 else 0, 2)
            
            return {
                "status": "operational",
                "total_jobs": len(all_jobs),
                "active_jobs": active_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "duplicate_jobs": duplicate_jobs,
                "total_users": len(unique_users),
                "active_users": len(active_users),
                "progress_percentage": progress_percentage,
                "jobs": all_jobs,
                "redis_status": "connected",
                "summary": {
                    "total_resumes_uploaded": len(all_jobs),
                    "users_uploading": len(unique_users),
                    "users_currently_active": len(active_users),
                    "processing_progress": f"{progress_percentage}%",
                    "estimated_completion": "2-4 hours" if active_jobs > 0 else "All completed"
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting all job statuses: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to get job statuses: {str(e)}",
                "total_jobs": 0,
                "active_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "duplicate_jobs": 0,
                "jobs": []
            }
        
    except Exception as e:
        logger.error(f"Error getting bulk processing status: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get processing status: {str(e)}",
            "total_jobs": 0,
            "active_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "duplicate_jobs": 0,
            "jobs": []
        }

@router.delete("/failed-resumes/{resume_id}")
async def delete_failed_resume(resume_id: str):
    """
    Delete a failed resume file by resume ID.
    
    Args:
        resume_id: Unique resume ID to delete
        
    Returns:
        Dict: Deletion confirmation
    """
    try:
        # Create failed folder path
        failed_folder = os.path.join(settings.UPLOAD_FOLDER, "failed")
        
        # Check if failed folder exists
        if not os.path.exists(failed_folder):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Failed resumes folder not found"
            )
        
        # Find file by resume ID (UUID pattern)
        found_file = None
        for filename in os.listdir(failed_folder):
            if filename.startswith(resume_id):
                found_file = filename
                break
        
        if not found_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Failed resume with ID '{resume_id}' not found"
            )
        
        file_path = os.path.join(failed_folder, found_file)
        
        # Delete the file
        os.remove(file_path)
        
        logger.info(f"Successfully deleted failed resume: {found_file} (ID: {resume_id})")
        
        return {
            "message": f"Failed resume with ID '{resume_id}' deleted successfully",
            "deleted_file": found_file,
            "resume_id": resume_id,
            "file_path": file_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting failed resume {resume_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete failed resume: {str(e)}"
        )

@router.get("/failed-resumes")
async def list_failed_resumes():
    """
    List all failed resume files with failure reasons.
    
    Returns:
        Dict: List of failed resume files with detailed failure information
    """
    try:
        # Create failed folder path
        failed_folder = os.path.join(settings.UPLOAD_FOLDER, "failed")
        
        # Check if failed folder exists
        if not os.path.exists(failed_folder):
            return {
                "failed_resumes": [],
                "total_count": 0,
                "message": "No failed resumes folder found"
            }
        
        # Get all files in failed folder
        failed_files = []
        for filename in os.listdir(failed_folder):
            file_path = os.path.join(failed_folder, filename)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                file_extension = os.path.splitext(filename)[1].lower()
                
                # Extract resume ID from filename (UUID part)
                resume_id = os.path.splitext(filename)[0]  # Remove extension to get UUID
                
                # Try to read failure reason from metadata file
                failure_reason = "Unknown failure reason"
                failure_type = "unknown"
                original_filename = filename  # Default to UUID filename if no metadata
                metadata_file = os.path.join(failed_folder, f"{resume_id}.metadata.json")
                
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            failure_reason = metadata.get("failure_reason", "Unknown failure reason")
                            failure_type = metadata.get("failure_type", "unknown")
                            original_filename = metadata.get("original_filename", filename)
                    except Exception as e:
                        logger.warning(f"Could not read metadata for {filename}: {e}")
                
                failed_files.append({
                    "resume_id": resume_id,
                    "filename": original_filename,  # Use original filename for display
                    "uuid_filename": filename,  # Keep UUID filename for reference
                    "file_size": file_size,
                    "file_type": file_extension.lstrip('.'),
                    "created_at": os.path.getctime(file_path),
                    "file_path": file_path,
                    "failure_reason": failure_reason,
                    "failure_type": failure_type,
                    "can_reupload": True  # All failed resumes can be re-uploaded
                })
        
        # Sort by creation time (newest first)
        failed_files.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Count by failure type
        failure_types = {}
        for file in failed_files:
            failure_type = file["failure_type"]
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
        
        return {
            "failed_resumes": failed_files,
            "total_count": len(failed_files),
            "failed_folder": failed_folder,
            "failure_summary": {
                "duplicate_resumes": failure_types.get("duplicate", 0),
                "missing_fields": failure_types.get("missing_required_fields", 0),
                "parsing_errors": failure_types.get("parsing_error", 0),
                "file_errors": failure_types.get("file_error", 0),
                "unknown_errors": failure_types.get("unknown", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing failed resumes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list failed resumes: {str(e)}"
        )


@router.post("/re-upload-failed-resumes")
async def re_upload_failed_resumes(request: Request):
    """
    Re-upload and re-process failed resume files using metadata (resume IDs).
    This API works with existing files in the failed folder, not new file uploads.
    
    Args:
        request: JSON body containing failed_resume_ids list
        
    Returns:
        Dict: Re-processing results with job tracking
    """
    start_time = time.time()
    
    try:
        # Parse JSON body to get failed resume IDs
        body = await request.json()
        failed_resume_ids = body.get("failed_resume_ids", [])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body. Expected: {'failed_resume_ids': ['id1', 'id2']}"
        )
    
    # Create a bulk processing job ID
    bulk_job_id = str(uuid.uuid4())
    user_id = "reupload_user"  # Simplified for re-upload operations
    
    # Track the bulk processing job
    bulk_processing_jobs[bulk_job_id] = {
        "job_id": bulk_job_id,
        "status": "processing",
        "user_id": user_id,
        "created_at": time.time(),
        "updated_at": time.time(),
        "total_files": len(failed_resume_ids),
        "processed_files": 0,
        "successful_files": 0,
        "failed_files": 0,
        "duplicate_files": 0,
        "progress": "0"
    }
    
    logger.info(f"Re-uploading {len(failed_resume_ids)} failed resume IDs: {failed_resume_ids}")
    
    # Validate input
    if not failed_resume_ids or len(failed_resume_ids) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No failed resume IDs provided"
        )
    
    if len(failed_resume_ids) > 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many failed resume IDs. Maximum 10,000 failed resume IDs per request."
        )
    
    # Process failed resumes
    results = []
    successful_files = [0]
    failed_files = [0]
    duplicate_files = [0]
    batch_data_to_save = []
    all_files_to_process = []
    
    try:
        failed_folder = os.path.join(settings.UPLOAD_FOLDER, "failed")
        logger.info(f"Failed folder path: {failed_folder}")
        
        if not os.path.exists(failed_folder):
            logger.error(f"Failed folder does not exist: {failed_folder}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed resumes folder not found: {failed_folder}"
            )
        
        # Process each failed resume ID
        for resume_id in failed_resume_ids:
            # Check if job was cancelled
            if bulk_job_id in bulk_processing_jobs and bulk_processing_jobs[bulk_job_id].get("status") == "cancelled":
                logger.info(f"Job {bulk_job_id} was cancelled, stopping re-upload processing")
                break
            # Find the file by resume ID
            found_file = None
            for filename in os.listdir(failed_folder):
                if filename.startswith(resume_id) and not filename.endswith('.metadata.json'):
                    found_file = filename
                    break
            
            if found_file:
                file_path = os.path.join(failed_folder, found_file)
                file_extension = os.path.splitext(found_file)[1].lower()
                
                # Read file content
                with open(file_path, "rb") as f:
                    file_content = f.read()
                
                all_files_to_process.append({
                    "filename": found_file,
                    "content": file_content,
                    "size": len(file_content),
                    "extension": file_extension,
                    "is_from_failed": True,
                    "original_resume_id": resume_id
                })
            else:
                logger.warning(f"Failed resume ID {resume_id} not found in failed folder")
                results.append({
                    "filename": resume_id,
                    "status": "failed",
                    "error": f"Failed resume ID {resume_id} not found",
                    "file_type": "unknown",
                    "processing_time": 0
                })
                failed_files[0] += 1
        
        # Process all files
        for i, file_data in enumerate(all_files_to_process):
            # Check if job was cancelled
            if bulk_job_id in bulk_processing_jobs and bulk_processing_jobs[bulk_job_id].get("status") == "cancelled":
                logger.info(f"Job {bulk_job_id} was cancelled, stopping re-upload file processing")
                break
            try:
                # Parse the resume
                parsed_data = await parse_resume_file(file_data)
                
                if parsed_data:
                    # Check for uniqueness
                    uniqueness_check = await check_resume_uniqueness(parsed_data)
                    
                    if uniqueness_check["is_unique"]:
                        # Save to database
                        resume_id = await save_resume_to_database(parsed_data, file_data)
                        
                        # Generate embedding
                        await generate_resume_embedding(resume_id, parsed_data)
                        
                        results.append({
                            "filename": file_data["filename"],
                            "status": "success",
                            "parsed_data": parsed_data,
                            "file_type": file_data["extension"].lstrip('.'),
                            "processing_time": time.time() - start_time,
                            "embedding_status": "completed",
                            "embedding_generated": True
                        })
                        successful_files[0] += 1
                    else:
                        # Mark as duplicate
                        results.append({
                            "filename": file_data["filename"],
                            "status": "duplicate",
                            "error": uniqueness_check["error"],
                            "file_type": file_data["extension"].lstrip('.'),
                            "processing_time": time.time() - start_time
                        })
                        duplicate_files[0] += 1
                else:
                    results.append({
                        "filename": file_data["filename"],
                        "status": "failed",
                        "error": "Failed to parse resume",
                        "file_type": file_data["extension"].lstrip('.'),
                        "processing_time": time.time() - start_time
                    })
                    failed_files[0] += 1
                
                # Update job progress
                if bulk_job_id in bulk_processing_jobs:
                    progress_percentage = round(((i + 1) / len(all_files_to_process)) * 100, 2)
                    bulk_processing_jobs[bulk_job_id].update({
                        "processed_files": i + 1,
                        "successful_files": successful_files[0],
                        "failed_files": failed_files[0],
                        "duplicate_files": duplicate_files[0],
                        "progress": str(progress_percentage),
                        "updated_at": time.time(),
                        "results": results.copy()
                    })
                
            except Exception as e:
                logger.error(f"Error processing file {file_data['filename']}: {str(e)}")
                results.append({
                    "filename": file_data["filename"],
                    "status": "failed",
                    "error": str(e),
                    "file_type": file_data["extension"].lstrip('.'),
                    "processing_time": time.time() - start_time
                })
                failed_files[0] += 1
        
        # Mark job as completed
        if bulk_job_id in bulk_processing_jobs:
            bulk_processing_jobs[bulk_job_id].update({
                "status": "completed",
                "updated_at": time.time()
            })
        
        total_processing_time = time.time() - start_time
        
        return {
            "job_id": bulk_job_id,
            "total_files": len(failed_resume_ids),
            "successful_files": successful_files[0],
            "failed_files": failed_files[0],
            "duplicate_files": duplicate_files[0],
            "total_processing_time": total_processing_time,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error re-uploading failed resumes: {str(e)}")
        # Mark job as failed
        if bulk_job_id in bulk_processing_jobs:
            bulk_processing_jobs[bulk_job_id].update({
                "status": "failed",
                "updated_at": time.time()
            })
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to re-upload failed resumes: {str(e)}"
        )

@router.delete("/failed-resumes")
async def delete_all_failed_resumes():
    """
    Delete all failed resume files.
    
    Returns:
        Dict: Deletion confirmation
    """
    try:
        # Create failed folder path
        failed_folder = os.path.join(settings.UPLOAD_FOLDER, "failed")
        
        # Check if failed folder exists
        if not os.path.exists(failed_folder):
            return {
                "message": "No failed resumes folder found",
                "deleted_count": 0
            }
        
        # Get all files in failed folder
        deleted_count = 0
        for filename in os.listdir(failed_folder):
            file_path = os.path.join(failed_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                deleted_count += 1
        
        logger.info(f"Successfully deleted {deleted_count} failed resumes")
        
        return {
            "message": f"Successfully deleted {deleted_count} failed resume files",
            "deleted_count": deleted_count,
            "failed_folder": failed_folder
        }
        
    except Exception as e:
        logger.error(f"Error deleting all failed resumes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete all failed resumes: {str(e)}"
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
                            parsed_data = json.loads(parsed_data)
                        except:
                            continue
                    
                    if not isinstance(parsed_data, dict):
                        continue
                    
                    # Create structured text for embedding - ONLY SKILLS AND EXPERIENCE
                    skills = parsed_data.get('Skills', [])
                    experience = parsed_data.get('Experience', [])
                    total_experience = parsed_data.get('TotalExperience', '')
                    
                    # Build text representation with ONLY skills and experience
                    text_parts = []
                    
                    # Add skills with normalization
                    if skills:
                        skills_text = ", ".join(skills) if isinstance(skills, list) else str(skills)
                        # Normalize common skill variations for better matching
                        skills_text = _normalize_skills(skills_text)
                        text_parts.append(f"Skills: {skills_text}")
                    
                    # Add experience with normalization
                    if experience:
                        if isinstance(experience, list):
                            exp_text = "; ".join([str(exp) for exp in experience])
                        else:
                            exp_text = str(experience)
                        text_parts.append(f"Experience: {exp_text}")
                    
                    # Add total experience if available
                    if total_experience:
                        text_parts.append(f"TotalExperience: {total_experience}")
                    
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
                            candidate_name = resume.get('candidate_name', 'Unknown')
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate resume embeddings: {str(e)}"
        )

# Additional endpoints for resume management
@router.get("/resumes")
async def get_all_resumes(company_id: int = Query(None, description="Company ID for data isolation")):
    """
    Get unique resumes from database with embedding status only.
    Returns only unique resumes based on candidate email.
    Excludes full embedding data for better performance.
    
    Args:
        company_id: Company ID for data isolation (optional)
    
    Returns:
        Dict: List of unique resume records with embedding status
    """
    try:
        # Get all resumes first with company filtering
        all_resumes = await database_service.get_all_resumes(limit=1000, offset=0, company_id=company_id)  # Get more to filter unique
        
        # Filter unique resumes by email
        unique_resumes = {}
        for resume in all_resumes:
            email = resume.get('candidate_email', '').lower().strip()
            if email and email not in unique_resumes:
                unique_resumes[email] = resume
        
        # Convert to list
        unique_list = list(unique_resumes.values())
        
        # Format response with embedding status only
        formatted_resumes = []
        for resume in unique_list:
            # Check if embedding exists
            embedding = resume.get('embedding')
            has_embedding = False
            embedding_status = "failed"
            
            if embedding:
                if isinstance(embedding, str):
                    try:
                        import json
                        embedding = json.loads(embedding)
                    except json.JSONDecodeError:
                        embedding = None
                
                if embedding and isinstance(embedding, list) and len(embedding) > 0:
                    has_embedding = True
                    embedding_status = "completed"
            
            formatted_resumes.append({
                "id": resume.get('id'),
                "filename": resume.get('filename'),
                "candidate_name": resume.get('candidate_name', 'Unknown'),
                "candidate_email": resume.get('candidate_email', ''),
                "candidate_phone": resume.get('candidate_phone', ''),
                "total_experience": resume.get('total_experience', ''),
                "file_type": resume.get('file_type', ''),
                "file_size": resume.get('file_size', 0),
                "created_at": resume.get('created_at'),
                "embedding_status": embedding_status,
                "embedding_generated": has_embedding,
                # Exclude full embedding data for performance
                "parsed_data": resume.get('parsed_data', {})  # Keep parsed data for now
            })
        
        return {
            "resumes": formatted_resumes,
            "total_unique": len(unique_list),
            "message": "Retrieved unique resumes with embedding status only"
        }
    except Exception as e:
        logger.error(f"Error getting unique resumes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resumes: {str(e)}"
        )

@router.get("/resumes/{resume_id}")
async def get_resume_by_id(resume_id: int):
    """
    Get a specific resume by ID.
    
    Args:
        resume_id: Resume ID
        
    Returns:
        Dict: Resume data
    """
    try:
        resume = await database_service.get_resume_by_id(resume_id)
        if not resume:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )
        return resume
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting resume {resume_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resume: {str(e)}"
        )

@router.delete("/resumes/{resume_id}")
async def delete_resume(resume_id: int):
    """
    Delete a resume by ID.
    
    Args:
        resume_id: Resume ID to delete
        
    Returns:
        Dict: Deletion confirmation
    """
    try:
        success = await database_service.delete_resume(resume_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
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

@router.delete("/resumes")
async def delete_all_resumes():
    """
    Delete all resumes.
    
    Returns:
        Dict: Deletion confirmation
    """
    try:
        deleted_count = await database_service.delete_all_resumes()
        
        return {
            "message": f"Successfully deleted {deleted_count} resumes",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Error deleting all resumes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete all resumes: {str(e)}"
        )

@router.get("/resumes/{resume_id}/download")
async def download_resume(resume_id: int):
    """
    Download a resume file by ID.
    
    Args:
        resume_id: Resume ID to download
        
    Returns:
        FileResponse: Resume file
    """
    try:
        # Get resume data from database
        resume = await database_service.get_resume_by_id(resume_id)
        if not resume:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )
        
        # Get file path
        file_path = resume.get('file_path')
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume file not found on disk"
            )
        
        # Get filename for download
        filename = resume.get('filename', 'resume.pdf')
        
        # Return file
        from fastapi.responses import FileResponse
        return FileResponse(
            path=file_path,
            filename=filename,
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


@router.post("/cancel-job/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a specific processing job.
    
    Args:
        job_id: Job ID to cancel
        
    Returns:
        Dict: Cancellation confirmation
    """
    try:
        if job_id not in bulk_processing_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        job = bulk_processing_jobs[job_id]
        
        if job.get("status") in ["completed", "failed", "cancelled"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job is already completed, failed, or cancelled"
            )
        
        # Cancel the job
        job["status"] = "cancelled"
        job["updated_at"] = time.time()
        
        logger.info(f"Cancelled job {job_id}")
        
        return {
            "message": f"Successfully cancelled job {job_id}",
            "job_id": job_id,
            "status": "cancelled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )
