"""
Async Resume Processor for handling queued resume processing.
Processes resumes from Redis queue in background.
"""

import json
import logging
import asyncio
from typing import Dict, Any
from app.services.queue_service import queue_service
from app.services.file_processor import FileProcessor
from app.services.openai_service import OpenAIService
from app.services.database_service import DatabaseService

logger = logging.getLogger(__name__)

class AsyncResumeProcessor:
    """Background processor for resume queue."""
    
    def __init__(self):
        """Initialize processor services."""
        self.file_processor = FileProcessor()
        self.openai_service = OpenAIService()
        self.database_service = DatabaseService()
        self.is_processing = False
    
    async def start_processing(self):
        """Start processing resumes from queue."""
        if self.is_processing:
            logger.warning("⚠️ Processor already running")
            return
        
        self.is_processing = True
        logger.info("🚀 Starting async resume processor...")
        
        try:
            while self.is_processing:
                await self._process_next_job()
                await asyncio.sleep(1)  # Check queue every second
        except Exception as e:
            logger.error(f"❌ Processor error: {str(e)}")
        finally:
            self.is_processing = False
            logger.info("🛑 Async resume processor stopped")
    
    async def stop_processing(self):
        """Stop processing resumes."""
        self.is_processing = False
        logger.info("🛑 Stopping async resume processor...")
    
    async def _process_next_job(self):
        """Process next job from queue."""
        try:
            # Get job from queue (blocking with timeout)
            job_data_str = queue_service.redis_client.brpop(
                "resume_processing_queue", 
                timeout=1
            )
            
            if not job_data_str:
                return
            
            job_data = json.loads(job_data_str[1])
            job_id = job_data["job_id"]
            filename = job_data["filename"]
            
            logger.info(f"🔄 Processing job {job_id}: {filename}")
            
            # Update status to processing
            await queue_service.update_job_status(job_id, "processing", "10")
            
            try:
                # Convert hex string back to bytes
                file_data = bytes.fromhex(job_data["file_data"])
                
                # Process file
                await queue_service.update_job_status(job_id, "processing", "30")
                text_content = await self.file_processor.process_file(file_data, filename)
                
                # Parse with OpenAI
                await queue_service.update_job_status(job_id, "processing", "60")
                parsed_data = await self.openai_service.parse_resume_text(text_content)
                
                # Save to database
                await queue_service.update_job_status(job_id, "processing", "80")
                record_id = await self.database_service.save_resume_data(
                    filename=filename,
                    file_path=f"uploads/{filename}",
                    file_type=filename.split('.')[-1],
                    file_size=len(file_data),
                    processing_time=0.0,  # Will be calculated
                    parsed_data=parsed_data
                )
                
                # Update status to completed
                result = {
                    "success": True,
                    "record_id": record_id,
                    "filename": filename,
                    "parsed_data": parsed_data
                }
                
                await queue_service.update_job_status(
                    job_id, 
                    "completed", 
                    "100", 
                    result=result
                )
                
                logger.info(f"✅ Job {job_id} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ Job {job_id} failed: {str(e)}")
                await queue_service.update_job_status(
                    job_id, 
                    "failed", 
                    error=str(e)
                )
                
        except Exception as e:
            logger.error(f"❌ Error processing job: {str(e)}")

# Global processor instance
async_processor = AsyncResumeProcessor()
