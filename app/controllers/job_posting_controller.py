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

class JobPostingResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: str

@router.post("/generate", response_model=JobPostingResponse)
async def generate_job_posting(request: JobPostingRequest):
    """
    Generate a job posting based on the provided prompt.
    
    Args:
        request: JobPostingRequest containing the prompt
        
    Returns:
        JobPostingResponse with the generated job posting data
    """
    try:
        # Validate prompt
        if not request.prompt or not request.prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )
        
        # Initialize the job posting service
        job_service = JobPostingService()
        
        # Generate job posting from prompt
        job_data = await job_service.generate_job_posting(request.prompt)
        
        # Validate the generated data
        if not job_data or not isinstance(job_data, dict):
            raise HTTPException(
                status_code=500,
                detail="Failed to generate valid job posting data"
            )
        
        # Ensure we have at least the required fields
        required_fields = ["title", "description", "requirements", "requiredSkills"]
        missing_fields = [field for field in required_fields if not job_data.get(field)]
        
        if missing_fields:
            logger.warning(f"Missing required fields: {missing_fields}")
            # Continue anyway as we have fallback data
        
        return JobPostingResponse(
            success=True,
            data=job_data,
            message="Job posting generated successfully"
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

@router.get("/health")
async def health_check():
    """Health check endpoint for job posting service."""
    return {"status": "healthy", "service": "job-posting"}
