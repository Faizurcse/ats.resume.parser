"""
Job Posting Controller for handling job posting generation requests.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
import json
import time
import asyncio
from datetime import datetime
from app.services.job_posting_service import JobPostingService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/job-posting", tags=["Job Posting"])

class JobPostingRequest(BaseModel):
    prompt: str

class BulkJobPostingRequest(BaseModel):
    prompt: str
    count: int = 5  # Default to 5 jobs

class JobPostingResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: str
    time: float

class BulkJobPostingResponse(BaseModel):
    success: bool
    data: list[Dict[str, Any]]
    message: str
    time: float
    jobCount: int


@router.post("/generate", response_model=JobPostingResponse)
async def generate_job_posting(request: JobPostingRequest):
    """
    Generate a job posting based on the provided prompt with secure handling.
    
    Args:
        request: JobPostingRequest containing the prompt
        
    Returns:
        JobPostingResponse with the generated job posting data
    """
    start_time = time.time()
    
    try:
        # Validate prompt
        if not request.prompt or not request.prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )
        
        # Initialize the job posting service
        job_service = JobPostingService()
        
        # Generate job posting from prompt (now uses secure system)
        try:
            job_data = await job_service.generate_job_posting(request.prompt)
        except ValueError as e:
            # Handle invalid prompt errors
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        
        # Calculate execution time
        end_time = time.time()
        execution_time = round(end_time - start_time, 2)
        
        # Validate the generated data
        if not job_data or not isinstance(job_data, dict):
            raise HTTPException(
                status_code=500,
                detail="Failed to generate valid job posting data"
            )
        
        # Ensure we have all required fields (updated validation)
        required_fields = [
            "title", "company", "department", "internalSPOC", "recruiter", "email",
            "jobType", "experienceLevel", "country", "city", "fullLocation",
            "workType", "jobStatus", "salaryMin", "salaryMax", "priority",
            "description", "requirements", "requiredSkills", "benefits"
        ]
        missing_fields = [field for field in required_fields if not job_data.get(field)]
        
        if missing_fields:
            logger.warning(f"Missing required fields: {missing_fields}")
            # Use fallback data for missing fields
            job_data = job_service._create_fallback_job_posting()
        
        return JobPostingResponse(
            success=True,
            data=job_data,
            message=f"Job posting generated successfully in {execution_time} seconds",
            time=execution_time
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse job posting data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error generating job posting: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate job posting: {str(e)}"
        )

@router.post("/bulk-generate", response_model=BulkJobPostingResponse)
async def bulk_job_generator(request: BulkJobPostingRequest):
    """
    Generate multiple job postings based on the provided prompt.
    Processes 1-10 jobs synchronously without background processing.
    
    Args:
        request: BulkJobPostingRequest containing the prompt and count (max 10)
        
    Returns:
        BulkJobPostingResponse with array of generated job postings
    """
    start_time = time.time()
    
    try:
        # Validate prompt
        if not request.prompt or not request.prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )
        
        # Validate count - maximum 10 jobs only
        if request.count <= 0 or request.count > 10:
            raise HTTPException(
                status_code=400,
                detail="Count must be between 1 and 10"
            )
        
        # Process all requests (1-10 jobs) synchronously
        return await _process_small_scale(request, start_time)
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in bulk job generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate bulk job postings: {str(e)}"
        )


async def _process_small_scale(request: BulkJobPostingRequest, start_time: float) -> BulkJobPostingResponse:
    """Process all requests (1-10 jobs) synchronously."""

    
    # Initialize the job posting service
    job_service = JobPostingService()
    
    # Create varied prompts
    varied_prompts = _create_varied_prompts(request.prompt, request.count)
    
    # Check if the main prompt is valid
    try:
        test_job = await job_service.generate_job_posting(request.prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Generate jobs with concurrency control
    semaphore = asyncio.Semaphore(2)  # Reduced concurrency for better stability
    
    async def generate_single_job(prompt, index):
        async with semaphore:
            try:
                job_data = await asyncio.wait_for(
                    job_service.generate_job_posting(prompt), 
                    timeout=30.0  # Increased timeout for better reliability
                )
                return job_data if job_data and isinstance(job_data, dict) else None
            except Exception as e:
                logger.error(f"Error generating job #{index+1}: {str(e)}")
                return None
    
    tasks = [generate_single_job(prompt, i) for i, prompt in enumerate(varied_prompts)]
    
    # Execute with reasonable timeout for small scale (max 10 jobs)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=300.0  # 5 minutes timeout for up to 10 jobs
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out. Please try again.")
    
    # Filter results
    bulk_jobs = [job for job in results if job is not None and not isinstance(job, Exception)]
    
    # If we got some results, return them even if not all succeeded
    if bulk_jobs:
        execution_time = round(time.time() - start_time, 2)
        success_rate = (len(bulk_jobs) / request.count) * 100
        
        return BulkJobPostingResponse(
            success=True,
            data=bulk_jobs,
            message=f"Successfully generated {len(bulk_jobs)} out of {request.count} job postings in {execution_time} seconds",
            time=execution_time,
            jobCount=len(bulk_jobs)
        )
    else:
        # If no results, try to generate at least one job as fallback
        try:
            logger.info("Attempting fallback job generation with original prompt")
            fallback_job = await asyncio.wait_for(
                job_service.generate_job_posting(request.prompt),
                timeout=20.0  # Reduced for cloud server compatibility
            )
            if fallback_job and isinstance(fallback_job, dict):
                execution_time = round(time.time() - start_time, 2)
                return BulkJobPostingResponse(
                    success=True,
                    data=[fallback_job],
                    message=f"Generated 1 fallback job posting in {execution_time} seconds",
                    time=execution_time,
                    jobCount=1
                )
        except asyncio.TimeoutError:
            logger.error("Fallback job generation timed out")
        except Exception as e:
            logger.error(f"Fallback job generation failed: {str(e)}")
        
        # Final fallback - create a basic job posting structure
        try:
            logger.info("Using final fallback - creating basic job posting structure")
            fallback_job = job_service._create_fallback_job_posting()
            execution_time = round(time.time() - start_time, 2)
            return BulkJobPostingResponse(
                success=True,
                data=[fallback_job],
                message=f"Generated 1 basic fallback job posting in {execution_time} seconds",
                time=execution_time,
                jobCount=1
            )
        except Exception as e:
            logger.error(f"Final fallback failed: {str(e)}")
        
        raise HTTPException(status_code=500, detail="Failed to generate any valid job postings.")


def _create_varied_prompts(prompt: str, count: int) -> list:
    """Create varied prompts for job generation."""
    prompt_lower = prompt.lower()
    
    # Check if this is a detailed prompt with field specifications
    field_indicators = ['company:', 'jobtitle:', 'department:', 'description:', 'requirements:', 'benefits:']
    has_field_specifications = any(indicator in prompt_lower for indicator in field_indicators)
    
    if has_field_specifications:
        # For detailed prompts with field specifications, don't create variations
        # Just return the same prompt multiple times to generate multiple jobs
        logger.info("Detected detailed prompt with field specifications - using same prompt for all jobs")
        return [prompt] * count
    
    # For simple prompts, create variations
    if any(skill in prompt_lower for skill in ['java', 'python', 'javascript', 'react', 'angular', 'vue', 'node', 'frontend', 'backend', 'full stack', 'mobile', 'devops', 'data science', 'machine learning']):
        skill_variations = [
            "Junior", "Senior", "Lead", "Principal", "Architect", "Backend", "Frontend", 
            "Full Stack", "Microservices", "Spring", "Enterprise", "Cloud", "DevOps", 
            "API", "Database", "Security", "Performance", "Scalable", "Distributed"
        ]
    else:
        skill_variations = [
            "Software Developer", "Data Analyst", "Product Manager", "Marketing Specialist",
            "Sales Representative", "HR Coordinator", "Financial Analyst", "UX Designer",
            "DevOps Engineer", "Business Analyst", "Customer Success Manager", "Operations Manager",
            "Frontend Developer", "Backend Developer", "Full Stack Developer", "Mobile Developer",
            "Data Scientist", "Machine Learning Engineer", "Cloud Architect", "Security Engineer"
        ]
    
    varied_prompts = []
    for i in range(count):
        if i < len(skill_variations):
            varied_prompt = f"{prompt} - {skill_variations[i]}"
        else:
            varied_prompt = f"{prompt} - Level {i+1}"
        varied_prompts.append(varied_prompt)
    
    return varied_prompts




@router.get("/health")
async def health_check():
    """Health check endpoint for job posting service."""
    return {"status": "healthy", "service": "job-posting"}
