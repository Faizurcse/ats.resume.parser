"""
Job Post Embeddings Controller for handling job post embedding operations.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import logging
from app.services.job_embedding_service import job_embedding_service
from app.services.database_service import DatabaseService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/job-post-embeddings", tags=["Job Post Embeddings"])

class StartEmbeddingRequest(BaseModel):
    """Request model for starting job embeddings generation."""
    force_regenerate: bool = False  # Whether to regenerate existing embeddings

class UpdateJobEmbeddingRequest(BaseModel):
    """Request model for updating a specific job's embedding."""
    job_id: int
    job_data: Dict[str, Any]  # Complete job data for embedding generation

class StartEmbeddingResponse(BaseModel):
    """Response model for starting job embeddings generation."""
    success: bool
    message: str
    total_jobs: int
    jobs_with_embeddings: int
    jobs_without_embeddings: int
    completion_percentage: float

class UpdateJobEmbeddingResponse(BaseModel):
    """Response model for updating a specific job's embedding."""
    success: bool
    message: str
    job_id: int
    embedding_size: int
    was_edited: bool

class JobEmbeddingData(BaseModel):
    """Model for individual job embedding data."""
    id: int
    title: str
    company: str
    department: str = None
    created_at: str = None
    updated_at: str = None
    embedding_size: int
    sample_embedding: List[float] = None

class GetAllEmbeddingsResponse(BaseModel):
    """Response model for getting all job embeddings."""
    success: bool
    total_jobs: int
    total_dimensions: int
    average_dimensions: float
    jobs: List[JobEmbeddingData]

@router.post("/start-embedding", response_model=StartEmbeddingResponse)
async def start_job_embeddings(request: StartEmbeddingRequest):
    """
    Start generating embeddings for all job posts.
    
    Args:
        request: StartEmbeddingRequest with optional force_regenerate flag
        
    Returns:
        StartEmbeddingResponse with processing status and summary
    """
    try:
        logger.info(f"🚀 STARTING JOB EMBEDDINGS GENERATION")
        logger.info(f"   🔄 Force Regenerate: {request.force_regenerate}")
        logger.info(f"   {'='*60}")
        
        # Get initial summary
        initial_summary = await job_embedding_service.get_job_embedding_summary()
        
        if "error" in initial_summary:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get initial summary: {initial_summary['error']}"
            )
        
        logger.info(f"📊 INITIAL STATUS:")
        logger.info(f"   📈 Total Jobs: {initial_summary['total_jobs']}")
        logger.info(f"   ✅ Jobs with Embeddings: {initial_summary['jobs_with_embeddings']}")
        logger.info(f"   ❌ Jobs without Embeddings: {initial_summary['jobs_without_embeddings']}")
        logger.info(f"   📊 Completion Rate: {initial_summary['completion_percentage']:.1f}%")
        

        
        # If no jobs without embeddings and not forcing regenerate, return early
        if initial_summary['jobs_without_embeddings'] == 0 and not request.force_regenerate:
            logger.info(f"🎉 ALL JOBS ALREADY HAVE EMBEDDINGS!")
            return StartEmbeddingResponse(
                success=True,
                message="All jobs already have embeddings",
                total_jobs=initial_summary['total_jobs'],
                jobs_with_embeddings=initial_summary['jobs_with_embeddings'],
                jobs_without_embeddings=initial_summary['jobs_without_embeddings'],
                completion_percentage=initial_summary['completion_percentage']
            )
        
        # Check if there are actually jobs to process
        if initial_summary['total_jobs'] == 0:
            logger.info(f"📭 NO JOBS IN SYSTEM")
            return StartEmbeddingResponse(
                success=True,
                message="No jobs in system",
                total_jobs=0,
                jobs_with_embeddings=0,
                jobs_without_embeddings=0,
                completion_percentage=0.0
            )
        
        # Get database service to fetch all jobs
        db_service = DatabaseService()
        
        if request.force_regenerate:
            # Get all jobs for regeneration
            all_jobs = await db_service.get_all_jobs()
            logger.info(f"🔄 FORCE REGENERATING EMBEDDINGS FOR ALL {len(all_jobs)} JOBS")
        else:
            # Get only jobs without embeddings
            jobs_without_embeddings = await db_service.get_jobs_without_embeddings()
            all_jobs = jobs_without_embeddings
            logger.info(f"🔄 GENERATING EMBEDDINGS FOR {len(all_jobs)} JOBS WITHOUT EMBEDDINGS")
        
        if not all_jobs:
            logger.info(f"📭 NO JOBS TO PROCESS")
            return StartEmbeddingResponse(
                success=True,
                message="No jobs to process",
                total_jobs=initial_summary['total_jobs'],
                jobs_with_embeddings=initial_summary['jobs_with_embeddings'],
                jobs_without_embeddings=initial_summary['jobs_without_embeddings'],
                completion_percentage=initial_summary['completion_percentage']
            )
        
        # Process all jobs in bulk
        logger.info(f"🚀 STARTING BULK EMBEDDING PROCESSING")
        logger.info(f"   📊 Total Jobs to Process: {len(all_jobs)}")
        
        bulk_result = await job_embedding_service.bulk_process_job_posts(all_jobs)
        
        if not bulk_result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Bulk processing failed: {bulk_result['error']}"
            )
        
        # Get final summary
        final_summary = await job_embedding_service.get_job_embedding_summary()
        
        if "error" in final_summary:
            logger.warning(f"⚠️ Could not get final summary: {final_summary['error']}")
            # Use bulk result for summary
            final_summary = {
                "total_jobs": initial_summary['total_jobs'],
                "jobs_with_embeddings": initial_summary['jobs_with_embeddings'] + bulk_result['summary']['successful'],
                "jobs_without_embeddings": initial_summary['jobs_without_embeddings'] - bulk_result['summary']['successful'],
                "completion_percentage": 0
            }
            if final_summary['total_jobs'] > 0:
                final_summary['completion_percentage'] = (final_summary['jobs_with_embeddings'] / final_summary['total_jobs']) * 100
        
        logger.info(f"🎉 JOB EMBEDDINGS GENERATION COMPLETED!")
        logger.info(f"   📊 FINAL STATUS:")
        logger.info(f"   📈 Total Jobs: {final_summary['total_jobs']}")
        logger.info(f"   ✅ Jobs with Embeddings: {final_summary['jobs_with_embeddings']}")
        logger.info(f"   ❌ Jobs without Embeddings: {final_summary['jobs_without_embeddings']}")
        logger.info(f"   📊 Completion Rate: {final_summary['completion_percentage']:.1f}%")
        logger.info(f"   {'='*60}")
        
        return StartEmbeddingResponse(
            success=True,
            message=f"Successfully processed {bulk_result['summary']['successful']} jobs",
            total_jobs=final_summary['total_jobs'],
            jobs_with_embeddings=final_summary['jobs_with_embeddings'],
            jobs_without_embeddings=final_summary['jobs_without_embeddings'],
            completion_percentage=final_summary['completion_percentage']
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Error starting job embeddings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start job embeddings: {str(e)}"
        )

@router.get("/all-embeddings", response_model=GetAllEmbeddingsResponse)
async def get_all_job_embeddings(limit: int = 100):
    """
    Get all job post embeddings with detailed information.
    
    Args:
        limit: Maximum number of jobs to return (default: 100, max: 1000)
        
    Returns:
        GetAllEmbeddingsResponse with all job embeddings data
    """
    try:
        # Validate limit
        if limit > 1000:
            limit = 1000
        elif limit < 1:
            limit = 100
        
        logger.info(f"🔍 RETRIEVING ALL JOB POST EMBEDDINGS")
        logger.info(f"   📊 Limit: {limit}")
        logger.info(f"   {'='*60}")
        
        # Get all jobs with embeddings
        embeddings_result = await job_embedding_service.show_all_job_embeddings(limit)
        
        if not embeddings_result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve embeddings: {embeddings_result['error']}"
            )
        
        # Transform the data to match our response model
        jobs_data = []
        for job in embeddings_result["jobs"]:
            embedding = job.get('embedding', [])
            embedding_size = len(embedding) if isinstance(embedding, list) else 0
            
            # Get sample embedding (first 5 values)
            sample_embedding = embedding[:5] if isinstance(embedding, list) and len(embedding) > 0 else None
            
            job_data = JobEmbeddingData(
                id=job.get('id'),
                title=job.get('title', 'Unknown Title'),
                company=job.get('company', 'Unknown Company'),
                department=job.get('department'),
                created_at=str(job.get('created_at')) if job.get('created_at') else None,
                updated_at=str(job.get('updated_at')) if job.get('updated_at') else None,
                embedding_size=embedding_size,
                sample_embedding=sample_embedding
            )
            jobs_data.append(job_data)
        
        logger.info(f"📊 RETRIEVED {len(jobs_data)} JOBS WITH EMBEDDINGS")
        logger.info(f"   📏 Total Dimensions: {embeddings_result['total_dimensions']}")
        logger.info(f"   📊 Average Dimensions: {embeddings_result['average_dimensions']:.1f}")
        logger.info(f"   {'='*60}")
        
        return GetAllEmbeddingsResponse(
            success=True,
            total_jobs=embeddings_result['total_jobs'],
            total_dimensions=embeddings_result['total_dimensions'],
            average_dimensions=embeddings_result['average_dimensions'],
            jobs=jobs_data
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving job embeddings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve job embeddings: {str(e)}"
        )

@router.get("/summary")
async def get_embeddings_summary():
    """
    Get a summary of job embeddings status.
    
    Returns:
        dict: Summary with counts and completion percentage
    """
    try:
        logger.info(f"📊 GETTING JOB EMBEDDINGS SUMMARY")
        
        summary = await job_embedding_service.get_job_embedding_summary()
        
        if "error" in summary:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get summary: {summary['error']}"
            )
        
        return {
            "success": True,
            "summary": summary,
            "timestamp": "2024-01-01T00:00:00Z"  # You can add actual timestamp if needed
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Error getting embeddings summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get embeddings summary: {str(e)}"
        )

@router.post("/update-job-embedding", response_model=UpdateJobEmbeddingResponse)
async def update_job_embedding(request: UpdateJobEmbeddingRequest):
    """
    Update embedding for a specific job post (useful when job is edited).
    
    Args:
        request: UpdateJobEmbeddingRequest with job ID and updated job data
        
    Returns:
        UpdateJobEmbeddingResponse with processing status
    """
    try:
        logger.info(f"🔄 UPDATING EMBEDDING FOR SPECIFIC JOB")
        logger.info(f"   🆔 Job ID: {request.job_id}")
        logger.info(f"   📋 Job Title: {request.job_data.get('title', 'Unknown')}")
        logger.info(f"   🏢 Company: {request.job_data.get('company', 'Unknown')}")
        logger.info(f"   {'='*60}")
        
        # Process the specific job
        result = await job_embedding_service.process_job_post(request.job_data)
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update job embedding: {result['error']}"
            )
        
        logger.info(f"✅ JOB EMBEDDING UPDATED SUCCESSFULLY!")
        logger.info(f"   🆔 Job ID: {request.job_id}")
        logger.info(f"   📏 Embedding Size: {result['embedding_size']}")
        logger.info(f"   🔄 Was Edited: {result.get('was_edited', False)}")
        logger.info(f"   {'='*60}")
        
        return UpdateJobEmbeddingResponse(
            success=True,
            message=result["message"],
            job_id=request.job_id,
            embedding_size=result["embedding_size"],
            was_edited=result.get("was_edited", False)
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Error updating job embedding: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update job embedding: {str(e)}"
        )

@router.post("/regenerate-all-embeddings")
async def regenerate_all_job_embeddings():
    """
    Regenerate all job embeddings using OpenAI's text-embedding-ada-002 model.
    This fixes the dimension mismatch issue where jobs had 100-dim embeddings
    while resumes have 1536-dim embeddings.
    """
    try:
        logger.info("🔄 REGENERATING ALL JOB EMBEDDINGS")
        logger.info("   🎯 Fixing dimension mismatch (100 → 1536 dimensions)")
        
        result = await job_embedding_service.regenerate_all_job_embeddings()
        
        if result["success"]:
            return result
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to regenerate embeddings: {result.get('error', 'Unknown error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in regenerate all embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint for job post embeddings service."""
    return {
        "status": "healthy", 
        "service": "job-post-embeddings",
        "endpoints": {
            "start_embedding": "POST /start-embedding",
            "update_job_embedding": "POST /update-job-embedding",
            "all_embeddings": "GET /all-embeddings",
            "summary": "GET /summary",
            "regenerate_all_embeddings": "POST /regenerate-all-embeddings"
        }
    }


