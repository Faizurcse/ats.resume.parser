"""
Get Embedding Data Controller
Provides APIs to retrieve job post and resume embedding data with status information.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from datetime import datetime

from app.services.database_service import DatabaseService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/embedding-data", tags=["Embedding Data"])

# Initialize database service
db_service = DatabaseService()

# Response Models
class JobPostEmbeddingData(BaseModel):
    """Model for job post embedding data."""
    id: int
    title: str
    company: str
    department: Optional[str] = None
    created_at: str
    embedding_size: int
    embedding_status: str  # "available", "missing", "empty"
    embedding_data: Optional[List[float]] = None  # Full embedding data
    sample_embedding: Optional[List[float]] = None  # First 5 values for preview

class ResumeEmbeddingData(BaseModel):
    """Model for resume embedding data."""
    id: int
    candidate_name: Optional[str] = None
    filename: str
    skills: Optional[List[str]] = None
    total_experience: Optional[str] = None
    created_at: str
    embedding_size: int
    embedding_status: str  # "available", "missing", "empty"
    embedding_data: Optional[List[float]] = None  # Full embedding data
    sample_embedding: Optional[List[float]] = None  # First 5 values for preview

class JobPostEmbeddingResponse(BaseModel):
    """Response model for job post embedding data."""
    success: bool
    message: str
    total_jobs: int
    jobs_with_embeddings: int
    jobs_without_embeddings: int
    data: List[JobPostEmbeddingData]

class ResumeEmbeddingResponse(BaseModel):
    """Response model for resume embedding data."""
    success: bool
    message: str
    total_resumes: int
    resumes_with_embeddings: int
    resumes_without_embeddings: int
    data: List[ResumeEmbeddingData]

@router.get("/job-posts", response_model=JobPostEmbeddingResponse)
async def get_job_post_embedding_data(
    limit: int = Query(50, description="Maximum number of job posts to return"),
    include_sample: bool = Query(False, description="Include sample embedding data (first 5 values)"),
    prioritize_with_embeddings: bool = Query(True, description="Show jobs with embeddings first")
):
    """
    Get job post embedding data with status information.
    
    Returns:
    - Job title, company, created date
    - Embedding data status (available/missing/empty)
    - Sample embedding data (optional)
    - Statistics about embedding coverage
    """
    try:
        logger.info(f"🔍 Fetching job post embedding data (limit: {limit})")
        
        # Get all jobs
        all_jobs = await db_service.get_all_jobs()
        total_jobs = len(all_jobs)
        
        # Get jobs with embeddings count
        jobs_with_embeddings_count = await db_service.get_jobs_with_embeddings_count()
        jobs_without_embeddings = total_jobs - jobs_with_embeddings_count
        
        # Sort jobs to prioritize those with embeddings if requested
        if prioritize_with_embeddings:
            # Separate jobs with and without embeddings
            jobs_with_embeddings = []
            jobs_without_embeddings_list = []
            
            for job in all_jobs:
                embedding = job.get('embedding')
                if embedding and isinstance(embedding, list) and len(embedding) > 0:
                    jobs_with_embeddings.append(job)
                else:
                    jobs_without_embeddings_list.append(job)
            
            # Combine: jobs with embeddings first, then jobs without embeddings
            sorted_jobs = jobs_with_embeddings + jobs_without_embeddings_list
        else:
            sorted_jobs = all_jobs
        
        # Limit the results
        limited_jobs = sorted_jobs[:limit]
        
        # Process job data
        job_data = []
        for job in limited_jobs:
            embedding = job.get('embedding')
            embedding_status = "missing"
            embedding_size = 0
            full_embedding_data = None
            sample_embedding = None
            
            if embedding:
                if isinstance(embedding, str):
                    try:
                        embedding = json.loads(embedding)
                    except json.JSONDecodeError:
                        embedding = None
                
                if embedding and isinstance(embedding, list) and len(embedding) > 0:
                    embedding_status = "available"
                    embedding_size = len(embedding)
                    full_embedding_data = embedding  # Full embedding data
                    
                    # Include sample embedding if requested
                    if include_sample:
                        sample_embedding = embedding[:5]  # First 5 values
                else:
                    embedding_status = "empty"
            
            job_info = JobPostEmbeddingData(
                id=job['id'],
                title=job.get('title', 'Unknown Title'),
                company=job.get('company', 'Unknown Company'),
                department=job.get('department'),
                created_at=job.get('created_at').strftime('%Y-%m-%d %H:%M:%S') if job.get('created_at') and hasattr(job.get('created_at'), 'strftime') else str(job.get('created_at', 'Unknown')),
                embedding_size=embedding_size,
                embedding_status=embedding_status,
                embedding_data=full_embedding_data,
                sample_embedding=sample_embedding
            )
            job_data.append(job_info)
        
        response = JobPostEmbeddingResponse(
            success=True,
            message=f"Successfully retrieved {len(job_data)} job posts",
            total_jobs=total_jobs,
            jobs_with_embeddings=jobs_with_embeddings_count,
            jobs_without_embeddings=jobs_without_embeddings,
            data=job_data
        )
        
        logger.info(f"✅ Job post embedding data retrieved successfully")
        logger.info(f"   📊 Total jobs: {total_jobs}")
        logger.info(f"   ✅ With embeddings: {jobs_with_embeddings_count}")
        logger.info(f"   ❌ Without embeddings: {jobs_without_embeddings}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error fetching job post embedding data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch job post embedding data: {str(e)}"
        )

@router.get("/resumes", response_model=ResumeEmbeddingResponse)
async def get_resume_embedding_data(
    limit: int = Query(50, description="Maximum number of resumes to return"),
    include_sample: bool = Query(False, description="Include sample embedding data (first 5 values)")
):
    """
    Get resume embedding data with status information.
    
    Returns:
    - Candidate name, skills, created date
    - Embedding data status (available/missing/empty)
    - Sample embedding data (optional)
    - Statistics about embedding coverage
    """
    try:
        logger.info(f"🔍 Fetching resume embedding data (limit: {limit})")
        
        # Get all resumes
        all_resumes = await db_service.get_all_resumes()
        total_resumes = len(all_resumes)
        
        # Count resumes with embeddings
        resumes_with_embeddings = 0
        for resume in all_resumes:
            embedding = resume.get('embedding')
            if embedding:
                if isinstance(embedding, str):
                    try:
                        embedding = json.loads(embedding)
                    except json.JSONDecodeError:
                        embedding = None
                
                if embedding and isinstance(embedding, list) and len(embedding) > 0:
                    resumes_with_embeddings += 1
        
        resumes_without_embeddings = total_resumes - resumes_with_embeddings
        
        # Limit the results
        limited_resumes = all_resumes[:limit]
        
        # Process resume data
        resume_data = []
        for resume in limited_resumes:
            embedding = resume.get('embedding')
            embedding_status = "missing"
            embedding_size = 0
            full_embedding_data = None
            sample_embedding = None
            
            if embedding:
                if isinstance(embedding, str):
                    try:
                        embedding = json.loads(embedding)
                    except json.JSONDecodeError:
                        embedding = None
                
                if embedding and isinstance(embedding, list) and len(embedding) > 0:
                    embedding_status = "available"
                    embedding_size = len(embedding)
                    full_embedding_data = embedding  # Full embedding data
                    
                    # Include sample embedding if requested
                    if include_sample:
                        sample_embedding = embedding[:5]  # First 5 values
                else:
                    embedding_status = "empty"
            
            # Extract skills from parsed_data
            skills = []
            parsed_data = resume.get('parsed_data', {})
            if isinstance(parsed_data, dict):
                # Look for common skill fields
                skill_fields = ['skills', 'technical_skills', 'Skills', 'Technical Skills', 'competencies']
                for field in skill_fields:
                    if field in parsed_data:
                        skill_value = parsed_data[field]
                        if isinstance(skill_value, list):
                            skills.extend(skill_value)
                        elif isinstance(skill_value, str):
                            # Split by common delimiters
                            skills.extend([s.strip() for s in skill_value.split(',') if s.strip()])
                        break
                
                # If no specific skills field, look for other relevant fields
                if not skills:
                    for key, value in parsed_data.items():
                        if 'skill' in key.lower() and isinstance(value, (str, list)):
                            if isinstance(value, list):
                                skills.extend(value)
                            else:
                                skills.extend([s.strip() for s in value.split(',') if s.strip()])
            
            resume_info = ResumeEmbeddingData(
                id=resume['id'],
                candidate_name=resume.get('candidate_name'),
                filename=resume.get('filename', 'Unknown'),
                skills=skills[:10] if skills else None,  # Limit to 10 skills
                total_experience=resume.get('total_experience'),
                created_at=resume.get('created_at').strftime('%Y-%m-%d %H:%M:%S') if resume.get('created_at') and hasattr(resume.get('created_at'), 'strftime') else str(resume.get('created_at', 'Unknown')),
                embedding_size=embedding_size,
                embedding_status=embedding_status,
                embedding_data=full_embedding_data,
                sample_embedding=sample_embedding
            )
            resume_data.append(resume_info)
        
        response = ResumeEmbeddingResponse(
            success=True,
            message=f"Successfully retrieved {len(resume_data)} resumes",
            total_resumes=total_resumes,
            resumes_with_embeddings=resumes_with_embeddings,
            resumes_without_embeddings=resumes_without_embeddings,
            data=resume_data
        )
        
        logger.info(f"✅ Resume embedding data retrieved successfully")
        logger.info(f"   📊 Total resumes: {total_resumes}")
        logger.info(f"   ✅ With embeddings: {resumes_with_embeddings}")
        logger.info(f"   ❌ Without embeddings: {resumes_without_embeddings}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error fetching resume embedding data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch resume embedding data: {str(e)}"
        )

@router.get("/status")
async def get_embedding_status():
    """
    Get overall embedding status for both job posts and resumes.
    
    Returns:
    - Summary statistics for both job posts and resumes
    - Overall embedding coverage status
    """
    try:
        logger.info("🔍 Fetching overall embedding status")
        
        # Get job post statistics
        all_jobs = await db_service.get_all_jobs()
        total_jobs = len(all_jobs)
        jobs_with_embeddings = await db_service.get_jobs_with_embeddings_count()
        jobs_without_embeddings = total_jobs - jobs_with_embeddings
        
        # Get resume statistics
        all_resumes = await db_service.get_all_resumes()
        total_resumes = len(all_resumes)
        
        resumes_with_embeddings = 0
        for resume in all_resumes:
            embedding = resume.get('embedding')
            if embedding:
                if isinstance(embedding, str):
                    try:
                        embedding = json.loads(embedding)
                    except json.JSONDecodeError:
                        embedding = None
                
                if embedding and isinstance(embedding, list) and len(embedding) > 0:
                    resumes_with_embeddings += 1
        
        resumes_without_embeddings = total_resumes - resumes_with_embeddings
        
        # Calculate percentages
        job_embedding_percentage = (jobs_with_embeddings / total_jobs * 100) if total_jobs > 0 else 0
        resume_embedding_percentage = (resumes_with_embeddings / total_resumes * 100) if total_resumes > 0 else 0
        
        status = {
            "success": True,
            "message": "Embedding status retrieved successfully",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "job_posts": {
                "total": total_jobs,
                "with_embeddings": jobs_with_embeddings,
                "without_embeddings": jobs_without_embeddings,
                "embedding_percentage": round(job_embedding_percentage, 2)
            },
            "resumes": {
                "total": total_resumes,
                "with_embeddings": resumes_with_embeddings,
                "without_embeddings": resumes_without_embeddings,
                "embedding_percentage": round(resume_embedding_percentage, 2)
            },
            "overall": {
                "total_records": total_jobs + total_resumes,
                "total_with_embeddings": jobs_with_embeddings + resumes_with_embeddings,
                "total_without_embeddings": jobs_without_embeddings + resumes_without_embeddings,
                "overall_embedding_percentage": round(
                    ((jobs_with_embeddings + resumes_with_embeddings) / (total_jobs + total_resumes) * 100) 
                    if (total_jobs + total_resumes) > 0 else 0, 2
                )
            }
        }
        
        logger.info(f"✅ Embedding status retrieved successfully")
        logger.info(f"   📊 Job posts: {jobs_with_embeddings}/{total_jobs} ({job_embedding_percentage:.1f}%)")
        logger.info(f"   📊 Resumes: {resumes_with_embeddings}/{total_resumes} ({resume_embedding_percentage:.1f}%)")
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Error fetching embedding status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch embedding status: {str(e)}"
        )
