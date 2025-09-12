"""
Queue Service for handling high-volume resume processing.
Uses Redis for job queuing and async processing.
"""

import json
import uuid
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import redis
from app.config.settings import settings

logger = logging.getLogger(__name__)

class QueueService:
    """Service for managing resume processing queue."""
    
    def __init__(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.Redis(
                host='localhost',  # Change to your Redis server
                port=6379,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connection established successfully")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {str(e)}")
            self.redis_client = None
    
    async def add_resume_job(self, file_data: bytes, filename: str, user_id: str = None) -> str:
        """
        Add resume processing job to queue.
        
        Args:
            file_data: Resume file content as bytes
            filename: Name of the file
            user_id: Optional user identifier
            
        Returns:
            str: Job ID for tracking
        """
        if not self.redis_client:
            raise Exception("Redis connection not available")
        
        job_id = str(uuid.uuid4())
        job_data = {
            "job_id": job_id,
            "filename": filename,
            "file_data": file_data.hex(),  # Convert bytes to hex string
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "status": "queued",
            "retry_count": 0
        }
        
        try:
            # Add to processing queue
            self.redis_client.lpush("resume_processing_queue", json.dumps(job_data))
            
            # Set job status
            self.redis_client.hset(f"job_status:{job_id}", mapping={
                "status": "queued",
                "created_at": job_data["created_at"],
                "filename": filename
            })
            
            # Set expiration (24 hours)
            self.redis_client.expire(f"job_status:{job_id}", 86400)
            
            logger.info(f"✅ Job {job_id} added to queue for file: {filename}")
            return job_id
            
        except Exception as e:
            logger.error(f"❌ Failed to add job to queue: {str(e)}")
            raise Exception(f"Failed to queue resume processing: {str(e)}")
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get job processing status.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dict: Job status information
        """
        if not self.redis_client:
            return {"error": "Redis connection not available"}
        
        try:
            status_data = self.redis_client.hgetall(f"job_status:{job_id}")
            
            if not status_data:
                return {"error": "Job not found"}
            
            # Parse result if available
            result = None
            if status_data.get("result"):
                try:
                    result = json.loads(status_data["result"])
                except json.JSONDecodeError:
                    result = {"error": "Invalid result format"}
            
            return {
                "job_id": job_id,
                "status": status_data.get("status", "unknown"),
                "created_at": status_data.get("created_at"),
                "filename": status_data.get("filename"),
                "progress": status_data.get("progress", "0"),
                "result": result,
                "error": status_data.get("error")
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get job status: {str(e)}")
            return {"error": f"Failed to get job status: {str(e)}"}
    
    async def update_job_status(self, job_id: str, status: str, progress: str = None, 
                               result: Dict[str, Any] = None, error: str = None):
        """
        Update job processing status.
        
        Args:
            job_id: Job identifier
            status: New status (queued, processing, completed, failed)
            progress: Progress percentage (0-100)
            result: Processing result
            error: Error message if failed
        """
        if not self.redis_client:
            return
        
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
            
            if progress is not None:
                update_data["progress"] = progress
            
            if result is not None:
                update_data["result"] = json.dumps(result)
            
            if error is not None:
                update_data["error"] = error
            
            self.redis_client.hset(f"job_status:{job_id}", mapping=update_data)
            
            logger.info(f"📊 Job {job_id} status updated: {status}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update job status: {str(e)}")
    
    async def get_queue_length(self) -> int:
        """Get current queue length."""
        if not self.redis_client:
            return 0
        
        try:
            return self.redis_client.llen("resume_processing_queue")
        except Exception as e:
            logger.error(f"❌ Failed to get queue length: {str(e)}")
            return 0
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a queued or processing job.
        
        Args:
            job_id: Job identifier to cancel
            
        Returns:
            bool: True if job was cancelled successfully
        """
        if not self.redis_client:
            return False
        
        try:
            # Check if job exists
            job_data = self.redis_client.hgetall(f"job_status:{job_id}")
            if not job_data:
                return False
            
            # Update job status to cancelled
            await self.update_job_status(job_id, "cancelled", error="Job cancelled by user")
            
            # Remove from queue if still queued (this is tricky with Redis lists)
            # We'll mark it as cancelled and let the processor skip it
            logger.info(f"✅ Job {job_id} cancelled successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel job {job_id}: {str(e)}")
            return False
    
    async def cancel_all_user_jobs(self, user_id: str = None) -> int:
        """
        Cancel all jobs for a specific user or all jobs.
        
        Args:
            user_id: User identifier (optional, if None cancels all jobs)
            
        Returns:
            int: Number of jobs cancelled
        """
        if not self.redis_client:
            return 0
        
        try:
            cancelled_count = 0
            job_keys = self.redis_client.keys("job_status:*")
            
            for job_key in job_keys:
                job_id = job_key.replace("job_status:", "")
                job_data = self.redis_client.hgetall(job_key)
                
                if job_data and job_data.get("status") in ["queued", "processing"]:
                    # If user_id specified, only cancel jobs for that user
                    if user_id and job_data.get("user_id") != user_id:
                        continue
                    
                    await self.update_job_status(job_id, "cancelled", error="Job cancelled by user")
                    cancelled_count += 1
            
            logger.info(f"✅ Cancelled {cancelled_count} jobs")
            return cancelled_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel jobs: {str(e)}")
            return 0

# Global queue service instance
queue_service = QueueService()
