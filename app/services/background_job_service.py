"""
Background Job Service for handling large-scale job generation.
Processes jobs in batches to avoid timeouts and rate limits.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from app.services.job_posting_service import JobPostingService
from app.services.database_service import DatabaseService

logger = logging.getLogger(__name__)

class BackgroundJobService:
    """Service for processing large-scale job generation in the background."""
    
    def __init__(self):
        self.job_service = JobPostingService()
        self.db_service = DatabaseService()
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
    
    async def start_large_scale_generation(
        self, 
        prompt: str, 
        count: int, 
        batch_size: int = 10
    ) -> str:
        """
        Start large-scale job generation in the background.
        
        Args:
            prompt: The job generation prompt
            count: Total number of jobs to generate
            batch_size: Number of jobs to process in each batch
            
        Returns:
            job_id: Unique identifier for tracking progress
        """
        job_id = f"bulk_job_{int(time.time())}_{count}"
        
        # Initialize job tracking
        self.active_jobs[job_id] = {
            "status": "started",
            "prompt": prompt,
            "total_count": count,
            "batch_size": batch_size,
            "completed_count": 0,
            "failed_count": 0,
            "start_time": datetime.now(),
            "last_update": datetime.now(),
            "results": [],
            "errors": []
        }
        
        # Start background processing
        asyncio.create_task(self._process_large_scale_generation(job_id))
        
        logger.info(f"Started large-scale generation job {job_id} for {count} jobs")
        return job_id
    
    async def _process_large_scale_generation(self, job_id: str):
        """Process large-scale job generation in batches."""
        job_info = self.active_jobs.get(job_id)
        if not job_info:
            return
        
        try:
            prompt = job_info["prompt"]
            total_count = job_info["total_count"]
            batch_size = job_info["batch_size"]
            
            # Create varied prompts for all jobs
            varied_prompts = self._create_varied_prompts(prompt, total_count)
            
            # Process in batches
            for i in range(0, total_count, batch_size):
                batch_prompts = varied_prompts[i:i + batch_size]
                batch_results = await self._process_batch(batch_prompts, i)
                
                # Update job progress
                job_info["completed_count"] += len([r for r in batch_results if r is not None])
                job_info["failed_count"] += len([r for r in batch_results if r is None])
                job_info["results"].extend([r for r in batch_results if r is not None])
                job_info["last_update"] = datetime.now()
                
                # Save progress to database
                await self._save_job_progress(job_id, job_info)
                
                # Rate limiting - wait between batches
                await asyncio.sleep(2)
            
            # Mark job as completed
            job_info["status"] = "completed"
            job_info["end_time"] = datetime.now()
            await self._save_job_progress(job_id, job_info)
            
            logger.info(f"Completed large-scale generation job {job_id}")
            
        except Exception as e:
            logger.error(f"Error in large-scale generation job {job_id}: {str(e)}")
            job_info["status"] = "failed"
            job_info["error"] = str(e)
            job_info["end_time"] = datetime.now()
            await self._save_job_progress(job_id, job_info)
    
    async def _process_batch(self, prompts: List[str], start_index: int) -> List[Optional[Dict[str, Any]]]:
        """Process a batch of job generation requests."""
        semaphore = asyncio.Semaphore(3)  # Limit concurrent API calls
        
        async def generate_single_job(prompt: str, index: int) -> Optional[Dict[str, Any]]:
            async with semaphore:
                try:
                    job_data = await asyncio.wait_for(
                        self.job_service.generate_job_posting(prompt),
                        timeout=35.0
                    )
                    return job_data
                except Exception as e:
                    logger.error(f"Error generating job #{start_index + index + 1}: {str(e)}")
                    return None
        
        tasks = [generate_single_job(prompt, i) for i, prompt in enumerate(prompts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if r is not None and not isinstance(r, Exception)]
    
    def _create_varied_prompts(self, prompt: str, count: int) -> List[str]:
        """Create varied prompts for job generation."""
        prompt_lower = prompt.lower()
        
        if any(skill in prompt_lower for skill in ['java', 'python', 'javascript', 'react', 'angular', 'vue', 'node']):
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
    
    async def _save_job_progress(self, job_id: str, job_info: Dict[str, Any]):
        """Save job progress to database."""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_info = job_info.copy()
            for key, value in serializable_info.items():
                if isinstance(value, datetime):
                    serializable_info[key] = value.isoformat()
            
            # Save to database (you'll need to implement this table)
            # For now, we'll just log the progress
            logger.info(f"Job {job_id} progress: {job_info['completed_count']}/{job_info['total_count']} completed")
            
        except Exception as e:
            logger.error(f"Error saving job progress for {job_id}: {str(e)}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a background job."""
        return self.active_jobs.get(job_id)
    
    def get_job_results(self, job_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get the results of a completed job."""
        job_info = self.active_jobs.get(job_id)
        if job_info and job_info["status"] == "completed":
            return job_info["results"]
        return None
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs to free memory."""
        current_time = datetime.now()
        jobs_to_remove = []
        
        for job_id, job_info in self.active_jobs.items():
            if job_info["status"] in ["completed", "failed"]:
                if "end_time" in job_info:
                    age = current_time - job_info["end_time"]
                    if age.total_seconds() > max_age_hours * 3600:
                        jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]
            logger.info(f"Cleaned up old job {job_id}")

# Global instance
background_job_service = BackgroundJobService()
