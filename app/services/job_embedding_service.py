"""
Job Embedding Service
Simple service to generate embeddings for job posts and store them directly in the job table.
This service can be called from Node.js backend when creating/updating job posts.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime


from app.services.database_service import DatabaseService

# Configure logging
logger = logging.getLogger(__name__)

class JobEmbeddingService:
    """Service for generating and storing job post embeddings directly in job table."""
    
    def __init__(self):
        """Initialize the service."""
        self.db_service = DatabaseService()
    
    async def get_job_embedding_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all job post embeddings with counts.
        
        Returns:
            Dict[str, Any]: Summary with total jobs, jobs with embeddings, and jobs without embeddings
        """
        try:
            # Ensure database connection is initialized
            await self.db_service._get_pool()
            
            # Get total job count
            logger.info("🔍 Getting total job count...")
            total_jobs = await self.db_service.get_total_job_count()
            logger.info(f"📊 Total jobs found: {total_jobs}")
            
            # Get jobs with embeddings count
            logger.info("🔍 Getting jobs with embeddings count...")
            jobs_with_embeddings = await self.db_service.get_jobs_with_embeddings_count()
            logger.info(f"📊 Jobs with embeddings: {jobs_with_embeddings}")
            
            # Calculate jobs without embeddings
            jobs_without_embeddings = total_jobs - jobs_with_embeddings
            
            # Calculate completion percentage
            completion_percentage = (jobs_with_embeddings / total_jobs * 100) if total_jobs > 0 else 0
            
            summary = {
                "total_jobs": total_jobs,
                "jobs_with_embeddings": jobs_with_embeddings,
                "jobs_without_embeddings": jobs_without_embeddings,
                "completion_percentage": completion_percentage
            }
            
            # Log the summary with detailed formatting
            logger.info(f"📊 JOB EMBEDDING SUMMARY REPORT")
            logger.info(f"   {'='*50}")
            logger.info(f"   📈 TOTAL JOBS IN SYSTEM: {total_jobs}")
            logger.info(f"   ✅ JOBS WITH EMBEDDINGS: {jobs_with_embeddings}")
            logger.info(f"   ❌ JOBS WITHOUT EMBEDDINGS: {jobs_without_embeddings}")
            logger.info(f"   📊 COMPLETION RATE: {completion_percentage:.1f}%")
            logger.info(f"   {'='*50}")
            
            # Show progress bar
            progress_bar = self._create_progress_bar(completion_percentage)
            logger.info(f"   📊 PROGRESS: {progress_bar}")
            logger.info(f"   {'='*50}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting job embedding summary: {str(e)}")
            return {
                "error": str(e),
                "total_jobs": 0,
                "jobs_with_embeddings": 0,
                "jobs_without_embeddings": 0,
                "completion_percentage": 0
            }
    
    def _create_progress_bar(self, percentage: float, width: int = 30) -> str:
        """
        Create a visual progress bar for the completion percentage.
        
        Args:
            percentage: Completion percentage (0-100)
            width: Width of the progress bar
            
        Returns:
            String representation of the progress bar
        """
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {percentage:.1f}%"
    
    async def generate_job_embedding(self, job_data: Dict[str, Any]) -> Optional[List[float]]:
        """
        Generate embedding for a job post using OpenAI's text-embedding-3-small model.
        
        Args:
            job_data: Dictionary containing job post data
            
        Returns:
            List[float]: Embedding vector or None if failed
        """
        try:
            # Prepare text for embedding
            embedding_text = self._prepare_job_text_for_embedding(job_data)
            
            if not embedding_text:
                logger.warning("No text content found for job embedding")
                return None
            
            logger.info(f"🔄 GENERATING EMBEDDING FOR JOB:")
            logger.info(f"   📋 Job Title: {job_data.get('title', 'Unknown')}")
            logger.info(f"   🏢 Company: {job_data.get('company', 'Unknown')}")
            logger.info(f"   🆔 Job ID: {job_data.get('id', 'Unknown')}")
            
            # Generate embedding using OpenAI API (same model as resume embeddings)
            from app.services.openai_service import OpenAIService
            openai_service = OpenAIService()
            embedding = await openai_service.generate_embedding(embedding_text)
            
            if embedding:
                logger.info(f"✅ JOB EMBEDDING GENERATED SUCCESSFULLY!")
                logger.info(f"   📋 Job Title: {job_data.get('title', 'Unknown')}")
                logger.info(f"   🏢 Company: {job_data.get('company', 'Unknown')}")
                logger.info(f"   📏 Embedding Size: {len(embedding)} dimensions")
                logger.info(f"   🔢 Sample Values: {embedding[:3]}...")
                logger.info(f"   ⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Show updated summary after successful generation
                await self.get_job_embedding_summary()
                
                return embedding
            else:
                logger.error("❌ FAILED TO GENERATE JOB EMBEDDING")
                return None
                
        except Exception as e:
            logger.error(f"Error generating job embedding: {str(e)}")
            return None
    
    async def store_job_embedding(self, job_id: int, embedding: List[float]) -> bool:
        """
        Store job embedding directly in the job table.
        
        Args:
            job_id: ID of the job post
            embedding: Embedding vector
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            # Update the job table with the embedding
            success = await self.db_service.update_job_embedding(job_id, embedding)
            
            if success:
                logger.info(f"💾 JOB EMBEDDING STORED SUCCESSFULLY!")
                logger.info(f"   🆔 Job ID: {job_id}")
                logger.info(f"   📏 Embedding Size: {len(embedding)} dimensions")
                logger.info(f"   🗄️  Stored in: Ats_JobPost.embedding column")
                logger.info(f"   ⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                return True
            else:
                logger.error(f"❌ FAILED TO STORE EMBEDDING FOR JOB ID: {job_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing job embedding: {str(e)}")
            return False
    
    async def process_job_post(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a job post: generate embedding and store it in the job table.
        Automatically detects if job was edited and regenerates embedding.
        
        Args:
            job_data: Dictionary containing job post data
            
        Returns:
            Dict[str, Any]: Result with success status and details
        """
        try:
            job_id = job_data.get("id")
            if not job_id:
                return {
                    "success": False,
                    "error": "Job ID is required"
                }
            
            logger.info(f"🚀 STARTING JOB POST EMBEDDING PROCESSING")
            logger.info(f"   🆔 Job ID: {job_id}")
            logger.info(f"   📋 Job Title: {job_data.get('title', 'Unknown')}")
            logger.info(f"   🏢 Company: {job_data.get('company', 'Unknown')}")
            logger.info(f"   {'='*60}")
            
            # Check if job was edited by comparing content
            job_was_edited = await self._check_if_job_was_edited(job_id, job_data)
            
            if job_was_edited:
                logger.info(f"🔄 JOB WAS EDITED - REGENERATING EMBEDDING")
                logger.info(f"   🆔 Job ID: {job_id}")
                logger.info(f"   📋 Job Title: {job_data.get('title', 'Unknown')}")
                logger.info(f"   🏢 Company: {job_data.get('company', 'Unknown')}")
                logger.info(f"   {'='*60}")
            
            # Generate embedding
            embedding = await self.generate_job_embedding(job_data)
            
            if not embedding:
                return {
                    "success": False,
                    "error": "Failed to generate embedding"
                }
            
            # Store embedding directly in job table
            stored = await self.store_job_embedding(job_id, embedding)
            
            if stored:
                if job_was_edited:
                    logger.info(f"🎉 EDITED JOB EMBEDDING UPDATED SUCCESSFULLY!")
                else:
                    logger.info(f"🎉 NEW JOB EMBEDDING GENERATED SUCCESSFULLY!")
                
                logger.info(f"   🆔 Job ID: {job_id}")
                logger.info(f"   📋 Job Title: {job_data.get('title', 'Unknown')}")
                logger.info(f"   🏢 Company: {job_data.get('company', 'Unknown')}")
                logger.info(f"   📏 Final Embedding Size: {len(embedding)} dimensions")
                logger.info(f"   🗄️  Database: Ats_JobPost.embedding column updated")
                logger.info(f"   ⏰ Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"   {'='*60}")
                
                # Show final summary after successful processing
                await self.get_job_embedding_summary()
                
                return {
                    "success": True,
                    "job_id": job_id,
                    "embedding_size": len(embedding),
                    "message": f"Job embedding {'updated' if job_was_edited else 'generated'} successfully in job table",
                    "was_edited": job_was_edited
                }
            else:
                logger.error(f"❌ JOB POST EMBEDDING PROCESSING FAILED!")
                logger.error(f"   🆔 Job ID: {job_id}")
                logger.error(f"   📋 Job Title: {job_data.get('title', 'Unknown')}")
                logger.error(f"   🏢 Company: {job_data.get('company', 'Unknown')}")
                logger.error(f"   ❌ Error: Failed to store embedding in job table")
                logger.error(f"   {'='*60}")
                
                return {
                    "success": False,
                    "error": "Failed to store embedding in job table"
                }
                
        except Exception as e:
            logger.error(f"Error processing job post: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def bulk_process_job_posts(self, job_posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process multiple job posts in bulk.
        
        Args:
            job_posts: List of job post dictionaries
            
        Returns:
            Dict[str, Any]: Results summary
        """
        try:
            logger.info(f"🚀 STARTING BULK JOB EMBEDDING PROCESSING")
            logger.info(f"   📊 Total Jobs to Process: {len(job_posts)}")
            logger.info(f"   {'='*60}")
            
            # Show initial summary
            await self.get_job_embedding_summary()
            
            results = []
            successful = 0
            failed = 0
            
            for index, job_post in enumerate(job_posts, 1):
                logger.info(f"🔄 Processing Job {index}/{len(job_posts)}")
                result = await self.process_job_post(job_post)
                results.append({
                    "job_id": job_post.get("id"),
                    "result": result
                })
                
                if result["success"]:
                    successful += 1
                else:
                    failed += 1
                
                # Show progress after each job
                logger.info(f"📊 Progress: {index}/{len(job_posts)} ({(index/len(job_posts)*100):.1f}%)")
                logger.info(f"   ✅ Successful: {successful}")
                logger.info(f"   ❌ Failed: {failed}")
                logger.info(f"   {'='*40}")
            
            logger.info(f"🚀 BULK JOB EMBEDDING PROCESSING COMPLETED!")
            logger.info(f"   📊 Final Summary:")
            logger.info(f"      📈 Total Jobs: {len(job_posts)}")
            logger.info(f"      ✅ Successful: {successful}")
            logger.info(f"      ❌ Failed: {failed}")
            logger.info(f"      📊 Success Rate: {(successful/len(job_posts)*100):.1f}%")
            logger.info(f"   ⏰ Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   {'='*60}")
            
            # Show final summary after bulk processing
            await self.get_job_embedding_summary()
            
            return {
                "success": True,
                "summary": {
                    "total": len(job_posts),
                    "successful": successful,
                    "failed": failed
                },
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in bulk processing: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _check_if_job_was_edited(self, job_id: int, new_job_data: Dict[str, Any]) -> bool:
        """
        Check if a job was edited by comparing current content with stored content.
        
        Args:
            job_id: ID of the job to check
            new_job_data: New job data to compare
            
        Returns:
            bool: True if job was edited, False if it's new or unchanged
        """
        try:
            # Get existing job data from database
            existing_job = await self.db_service.get_job_by_id(job_id)
            
            if not existing_job:
                # Job doesn't exist in database, so it's new
                logger.info(f"🆕 Job ID {job_id} is new - no existing data found")
                return False
            
            # Prepare text for comparison
            existing_text = self._prepare_job_text_for_embedding(existing_job)
            new_text = self._prepare_job_text_for_embedding(new_job_data)
            
            # Simple text comparison (you could use more sophisticated methods)
            if existing_text.strip() == new_text.strip():
                logger.info(f"✅ Job ID {job_id} content unchanged - keeping existing embedding")
                return False
            else:
                logger.info(f"🔄 Job ID {job_id} content changed - will regenerate embedding")
                logger.info(f"   📊 Content comparison:")
                logger.info(f"      📋 Old Title: {existing_job.get('title', 'Unknown')}")
                logger.info(f"      📋 New Title: {new_job_data.get('title', 'Unknown')}")
                logger.info(f"      🏢 Old Company: {existing_job.get('company', 'Unknown')}")
                logger.info(f"      🏢 New Company: {new_job_data.get('company', 'Unknown')}")
                return True
                
        except Exception as e:
            logger.error(f"Error checking if job was edited: {str(e)}")
            # If we can't determine, assume it was edited to be safe
            return True
    
    def _prepare_job_text_for_embedding(self, job_data: Dict[str, Any]) -> str:
        """
        Prepare job post text for embedding generation using ONLY skills and experience.
        This focuses the matching on core requirements only.
        
        Args:
            job_data: Dictionary containing job post data
            
        Returns:
            Combined text string for embedding (skills + experience only)
        """
        # ONLY use skills and experience for matching
        skills = job_data.get("requiredSkills", "")
        experience = job_data.get("experienceLevel", "")
        
        # Normalize skills for better matching
        if skills:
            skills = self._normalize_skills(skills)
        
        # Combine skills and experience
        relevant_fields = [skills, experience]
        
        # Filter out empty fields and join with spaces
        return " ".join([field for field in relevant_fields if field])





    def _normalize_skills(self, skills_text: str) -> str:
        """
        Pass through skills text without normalization.
        Let GPT Text-Embedding-3-Small handle all variations naturally.
        """
        if not skills_text:
            return skills_text
        
        # No normalization - trust GPT's natural understanding
        return skills_text

# Create a global instance for easy access
job_embedding_service = JobEmbeddingService()
