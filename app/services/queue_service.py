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

# Global queue service instance
queue_service = QueueService()
