"""
Job Posting Controller for handling job posting generation requests.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
import json
from datetime import datetime
from app.services.job_posting_service import JobPostingService
from app.services.background_job_service import background_job_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/job-posting", tags=["Job Posting"])

class JobPostingRequest(BaseModel):
    prompt: str

class BulkJobPostingRequest(BaseModel):
    prompt: str
    count: int = 5  # Default to 5 jobs
    batch_size: int = 10  # Optional batch size for large requests

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

class JobStatusResponse(BaseModel):
    success: bool
    job_id: str
    status: str
    completed_count: int
    total_count: int
    failed_count: int
    progress_percentage: float
    estimated_remaining_minutes: int
    results: list[Dict[str, Any]] = []

@router.post("/generate", response_model=JobPostingResponse)
async def generate_job_posting(request: JobPostingRequest):
    """
    Generate a job posting based on the provided prompt with secure handling.
    
    Args:
        request: JobPostingRequest containing the prompt
        
    Returns:
        JobPostingResponse with the generated job posting data
    """
    import time
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
    Automatically handles small (1-20), medium (21-100), and large (100+) scales.
    
    Args:
        request: BulkJobPostingRequest containing the prompt, count, and optional batch_size
        
    Returns:
        BulkJobPostingResponse with array of generated job postings or job_id for large requests
    """
    import time
    start_time = time.time()
    
    try:
        # Validate prompt
        if not request.prompt or not request.prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )
        
        # Validate count
        if request.count <= 0 or request.count > 10000:
            raise HTTPException(
                status_code=400,
                detail="Count must be between 1 and 10,000"
            )
        
        # Determine processing strategy based on count
        if request.count <= 5:
            # Very small scale: Process with minimal timeout
            return await _process_very_small_scale(request, start_time)
        elif request.count <= 20:
            # Small scale: Process synchronously with immediate response
            return await _process_small_scale(request, start_time)
        elif request.count <= 100:
            # Medium scale: Process with extended timeout
            return await _process_medium_scale(request, start_time)
        else:
            # Large scale: Start background processing and return job_id
            return await _process_large_scale(request, start_time)
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in bulk job generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate bulk job postings: {str(e)}"
        )

async def _process_very_small_scale(request: BulkJobPostingRequest, start_time: float) -> BulkJobPostingResponse:
    """Process very small scale requests (1-5 jobs) with minimal timeout."""
    import asyncio
    
    # Initialize the job posting service
    job_service = JobPostingService()
    
    # For very small scale, process sequentially to avoid timeouts
    bulk_jobs = []
    
    for i in range(request.count):
        try:
            # Create simple prompt variation
            if i == 0:
                prompt = request.prompt
            else:
                prompt = f"{request.prompt} - Job {i+1}"
            
            # Generate single job with very short timeout
            job_data = await asyncio.wait_for(
                job_service.generate_job_posting(prompt),
                timeout=3.0  # Very short timeout
            )
            
            if job_data and isinstance(job_data, dict):
                bulk_jobs.append(job_data)
                
        except asyncio.TimeoutError:
            logger.warning(f"Job {i+1} timed out, skipping")
            continue
        except Exception as e:
            logger.error(f"Error generating job {i+1}: {str(e)}")
            continue
    
    if not bulk_jobs:
        # Try one more time with the original prompt
        try:
            fallback_job = await asyncio.wait_for(
                job_service.generate_job_posting(request.prompt),
                timeout=5.0
            )
            if fallback_job:
                bulk_jobs = [fallback_job]
        except Exception as e:
            logger.error(f"Fallback job generation failed: {str(e)}")
    
    if not bulk_jobs:
        raise HTTPException(status_code=500, detail="Failed to generate any valid job postings.")
    
    execution_time = round(time.time() - start_time, 2)
    
    return BulkJobPostingResponse(
        success=True,
        data=bulk_jobs,
        message=f"Successfully generated {len(bulk_jobs)} out of {request.count} job postings in {execution_time} seconds",
        time=execution_time,
        jobCount=len(bulk_jobs)
    )

async def _process_small_scale(request: BulkJobPostingRequest, start_time: float) -> BulkJobPostingResponse:
    """Process small scale requests (1-20 jobs) synchronously."""
    import asyncio
    
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
    semaphore = asyncio.Semaphore(5)
    
    async def generate_single_job(prompt, index):
        async with semaphore:
            try:
                job_data = await asyncio.wait_for(
                    job_service.generate_job_posting(prompt), 
                    timeout=5.0  # Reduced to 5 seconds per job
                )
                return job_data if job_data and isinstance(job_data, dict) else None
            except Exception as e:
                logger.error(f"Error generating job #{index+1}: {str(e)}")
                return None
    
    tasks = [generate_single_job(prompt, i) for i, prompt in enumerate(varied_prompts)]
    
    # Execute with aggressive timeout for small scale
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=30.0  # Reduced to 30 seconds for small scale
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out. Please try with fewer jobs.")
    
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
            fallback_job = await job_service.generate_job_posting(request.prompt)
            if fallback_job:
                execution_time = round(time.time() - start_time, 2)
                return BulkJobPostingResponse(
                    success=True,
                    data=[fallback_job],
                    message=f"Generated 1 fallback job posting in {execution_time} seconds",
                    time=execution_time,
                    jobCount=1
                )
        except Exception as e:
            logger.error(f"Fallback job generation failed: {str(e)}")
        
        raise HTTPException(status_code=500, detail="Failed to generate any valid job postings.")

async def _process_medium_scale(request: BulkJobPostingRequest, start_time: float) -> BulkJobPostingResponse:
    """Process medium scale requests (21-100 jobs) with extended timeout."""
    import asyncio
    
    # Initialize the job posting service
    job_service = JobPostingService()
    
    # Create varied prompts
    varied_prompts = _create_varied_prompts(request.prompt, request.count)
    
    # Check if the main prompt is valid
    try:
        test_job = await job_service.generate_job_posting(request.prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Process in smaller batches to avoid timeouts
    batch_size = min(10, request.count)
    semaphore = asyncio.Semaphore(3)  # Reduced concurrency for medium scale
    
    async def generate_single_job(prompt, index):
        async with semaphore:
            try:
                job_data = await asyncio.wait_for(
                    job_service.generate_job_posting(prompt), 
                    timeout=10.0
                )
                return job_data if job_data and isinstance(job_data, dict) else None
            except Exception as e:
                logger.error(f"Error generating job #{index+1}: {str(e)}")
                return None
    
    # Process in batches
    all_results = []
    for i in range(0, request.count, batch_size):
        batch_prompts = varied_prompts[i:i + batch_size]
        tasks = [generate_single_job(prompt, i + j) for j, prompt in enumerate(batch_prompts)]
        
        try:
            batch_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=120.0  # Extended timeout for medium scale
            )
            all_results.extend(batch_results)
            
            # Small delay between batches
            if i + batch_size < request.count:
                await asyncio.sleep(1)
                
        except asyncio.TimeoutError:
            logger.warning(f"Batch {i//batch_size + 1} timed out, continuing with next batch")
            continue
    
    # Filter results
    bulk_jobs = [job for job in all_results if job is not None and not isinstance(job, Exception)]
    
    if not bulk_jobs:
        raise HTTPException(status_code=500, detail="Failed to generate any valid job postings.")
    
    execution_time = round(time.time() - start_time, 2)
    success_rate = (len(bulk_jobs) / request.count) * 100
    
    return BulkJobPostingResponse(
        success=True,
        data=bulk_jobs,
        message=f"Successfully generated {len(bulk_jobs)} out of {request.count} job postings in {execution_time} seconds",
        time=execution_time,
        jobCount=len(bulk_jobs)
    )

async def _process_large_scale(request: BulkJobPostingRequest, start_time: float) -> BulkJobPostingResponse:
    """Process large scale requests (100+ jobs) using background processing."""
    # Start background job
    job_id = await background_job_service.start_large_scale_generation(
        prompt=request.prompt,
        count=request.count,
        batch_size=request.batch_size
    )
    
    execution_time = round(time.time() - start_time, 2)
    
    # Return job_id for tracking instead of results
    return BulkJobPostingResponse(
        success=True,
        data=[{"job_id": job_id, "status": "started", "message": "Large-scale generation started in background"}],
        message=f"Started large-scale generation of {request.count} jobs. Use job_id '{job_id}' to track progress via /job-status/{job_id}",
        time=execution_time,
        jobCount=0  # Will be updated when job completes
    )

def _create_varied_prompts(prompt: str, count: int) -> list:
    """Create varied prompts for job generation."""
    prompt_lower = prompt.lower()
    
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



@router.get("/job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a large-scale job generation.
    
    Args:
        job_id: The job ID returned from large-scale generation
        
    Returns:
        JobStatusResponse with current progress and results
    """
    try:
        job_info = background_job_service.get_job_status(job_id)
        
        if not job_info:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found. It may have expired or never existed."
            )
        
        # Calculate progress
        total_count = job_info["total_count"]
        completed_count = job_info["completed_count"]
        failed_count = job_info["failed_count"]
        progress_percentage = (completed_count / total_count) * 100 if total_count > 0 else 0
        
        # Estimate remaining time
        if job_info["status"] == "started" and completed_count > 0:
            elapsed_time = (datetime.now() - job_info["start_time"]).total_seconds()
            rate = completed_count / elapsed_time if elapsed_time > 0 else 0
            remaining_jobs = total_count - completed_count
            estimated_remaining_minutes = int(remaining_jobs / rate / 60) if rate > 0 else 0
        else:
            estimated_remaining_minutes = 0
        
        # Get results if completed
        results = []
        if job_info["status"] == "completed":
            results = job_info.get("results", [])
        
        return JobStatusResponse(
            success=True,
            job_id=job_id,
            status=job_info["status"],
            completed_count=completed_count,
            total_count=total_count,
            failed_count=failed_count,
            progress_percentage=round(progress_percentage, 2),
            estimated_remaining_minutes=estimated_remaining_minutes,
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status for {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint for job posting service."""
    return {"status": "healthy", "service": "job-posting"}
