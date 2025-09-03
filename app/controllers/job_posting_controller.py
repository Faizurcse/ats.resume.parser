"""
Job Posting Controller for handling job posting generation requests.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
import json
from app.services.job_posting_service import JobPostingService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/job-posting", tags=["Job Posting"])

class JobPostingRequest(BaseModel):
    prompt: str

class BulkJobPostingRequest(BaseModel):
    prompt: str
    count: int = 10  # Default to 10 jobs

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
    
    Args:
        request: BulkJobPostingRequest containing the prompt and count
        
    Returns:
        BulkJobPostingResponse with array of generated job postings
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
        if request.count <= 0 or request.count > 50:
            raise HTTPException(
                status_code=400,
                detail="Count must be between 1 and 50"
            )
        
        # Initialize the job posting service
        job_service = JobPostingService()
        
        # Generate multiple job postings in parallel for faster results
        import asyncio
        
        # Check if the prompt contains specific skills to create skill-specific variations
        prompt_lower = request.prompt.lower()
        if any(skill in prompt_lower for skill in ['java', 'python', 'javascript', 'react', 'angular', 'vue', 'node', 'frontend', 'backend', 'full stack', 'mobile', 'devops', 'data science', 'machine learning']):
            # Create skill-specific variations
            skill_variations = [
                "Junior", "Senior", "Lead", "Principal", "Architect", "Backend", "Frontend", 
                "Full Stack", "Microservices", "Spring", "Enterprise", "Cloud", "DevOps", 
                "API", "Database", "Security", "Performance", "Scalable", "Distributed"
            ]
            
            # Create varied prompts for parallel processing
            varied_prompts = []
            for i in range(request.count):
                if i < len(skill_variations):
                    varied_prompt = f"{request.prompt} - {skill_variations[i]}"
                else:
                    varied_prompt = f"{request.prompt} - Level {i+1}"
                varied_prompts.append(varied_prompt)
        else:
            # Generic job variations for non-skill-specific prompts
            job_variations = [
                "Software Developer", "Data Analyst", "Product Manager", "Marketing Specialist",
                "Sales Representative", "HR Coordinator", "Financial Analyst", "UX Designer",
                "DevOps Engineer", "Business Analyst", "Customer Success Manager", "Operations Manager",
                "Frontend Developer", "Backend Developer", "Full Stack Developer", "Mobile Developer",
                "Data Scientist", "Machine Learning Engineer", "Cloud Architect", "Security Engineer"
            ]
            
            # Create varied prompts for parallel processing
            varied_prompts = []
            for i in range(request.count):
                if i < len(job_variations):
                    varied_prompt = f"{request.prompt} - {job_variations[i]}"
                else:
                    varied_prompt = f"{request.prompt} - Job #{i+1}"
                varied_prompts.append(varied_prompt)
        
        # Check if the main prompt is invalid before generating jobs
        try:
            # Test the main prompt first
            test_job = await job_service.generate_job_posting(request.prompt)
        except ValueError as e:
            # If main prompt is invalid, return error immediately
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        
        # Generate all jobs in parallel for much faster results
        async def generate_single_job(prompt, index):
            try:
                # Add timeout to prevent hanging requests
                import asyncio
                job_data = await asyncio.wait_for(
                    job_service.generate_job_posting(prompt), 
                    timeout=15.0  # 15 second timeout per job for faster results
                )
                if job_data and isinstance(job_data, dict):
                    return job_data
                else:
                    logger.warning(f"Failed to generate job #{index+1}, skipping")
                    return None
            except asyncio.TimeoutError:
                logger.error(f"Timeout generating job #{index+1}")
                return None
            except ValueError as e:
                # Handle invalid prompt errors
                logger.error(f"Invalid prompt for job #{index+1}: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Error generating job #{index+1}: {str(e)}")
                return None
        
        # Create tasks for parallel execution
        tasks = [generate_single_job(prompt, i) for i, prompt in enumerate(varied_prompts)]
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        bulk_jobs = [job for job in results if job is not None and not isinstance(job, Exception)]
        
        # Calculate execution time
        end_time = time.time()
        execution_time = round(end_time - start_time, 2)
        
        # Check if we got any valid jobs
        if not bulk_jobs:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate any valid job postings"
            )
        
        return BulkJobPostingResponse(
            success=True,
            data=bulk_jobs,
            message=f"Successfully generated {len(bulk_jobs)} job postings in {execution_time} seconds",
            time=execution_time,
            jobCount=len(bulk_jobs)
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in bulk job generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate bulk job postings: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint for job posting service."""
    return {"status": "healthy", "service": "job-posting"}
