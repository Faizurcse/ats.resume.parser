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
        Generate embedding for a job post using OpenAI's text-embedding-ada-002 model.
        
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
        Prepare job post text for embedding generation by combining relevant fields.
        
        Args:
            job_data: Dictionary containing job post data
            
        Returns:
            Combined text string for embedding
        """
        relevant_fields = [
            job_data.get("title", ""),
            job_data.get("company", ""),
            job_data.get("department", ""),
            job_data.get("description", ""),
            job_data.get("requirements", ""),
            job_data.get("requiredSkills", ""),
            job_data.get("benefits", ""),
            job_data.get("country", ""),
            job_data.get("city", ""),
            job_data.get("jobType", ""),
            job_data.get("experienceLevel", ""),
            job_data.get("workType", "")
        ]
        
        # Filter out empty fields and join with spaces
        return " ".join([field for field in relevant_fields if field])

    async def show_all_job_embeddings(self, limit: int = 50) -> Dict[str, Any]:
        """
        Show all job post embeddings with detailed information in the logger.
        
        Args:
            limit: Maximum number of jobs to display (default: 50)
            
        Returns:
            Dict[str, Any]: Results with job embeddings data
        """
        try:
            logger.info(f"🔍 RETRIEVING ALL JOB POST EMBEDDINGS")
            logger.info(f"   {'='*60}")
            
            # Get all jobs with embeddings
            jobs_with_embeddings = await self.db_service.get_all_jobs_with_embeddings(limit)
            
            if not jobs_with_embeddings:
                logger.info(f"   📭 NO JOBS WITH EMBEDDINGS FOUND")
                logger.info(f"   {'='*60}")
                return {
                    "success": True,
                    "message": "No jobs with embeddings found",
                    "jobs": []
                }
            
            logger.info(f"   📋 FOUND {len(jobs_with_embeddings)} JOBS WITH EMBEDDINGS")
            logger.info(f"   {'='*60}")
            
            # Display each job with embedding details
            for index, job in enumerate(jobs_with_embeddings, 1):
                logger.info(f"   🆔 JOB #{index}: {job.get('title', 'Unknown Title')}")
                logger.info(f"      🏢 Company: {job.get('company', 'Unknown Company')}")
                logger.info(f"      🆔 Job ID: {job.get('id', 'Unknown ID')}")
                logger.info(f"      📅 Created: {job.get('created_at', 'Unknown Date')}")
                logger.info(f"      📅 Updated: {job.get('updated_at', 'Unknown Date')}")
                
                # Show embedding size with proper type checking
                embedding = job.get('embedding', [])
                if isinstance(embedding, list) and len(embedding) > 0:
                    logger.info(f"      📏 Embedding Size: {len(embedding)} dimensions")
                    
                    # Show sample embedding values
                    sample_values = embedding[:5]  # Show first 5 values
                    logger.info(f"      🔢 Sample Embedding Values: {sample_values}...")
                    
                    # Show embedding statistics
                    try:
                        min_val = min(embedding)
                        max_val = max(embedding)
                        avg_val = sum(embedding) / len(embedding)
                        logger.info(f"      📊 Embedding Stats: Min={min_val:.4f}, Max={max_val:.4f}, Avg={avg_val:.4f}")
                    except (TypeError, ValueError) as e:
                        logger.info(f"      📊 Embedding Stats: Unable to calculate (data type issue)")
                else:
                    logger.info(f"      📏 Embedding Size: 0 dimensions")
                    logger.info(f"      ❌ No valid embedding data found")
                
                logger.info(f"      {'─'*50}")
            
            # Show summary statistics with proper type checking
            total_embeddings = len(jobs_with_embeddings)
            total_dimensions = 0
            for job in jobs_with_embeddings:
                embedding = job.get('embedding', [])
                if isinstance(embedding, list) and len(embedding) > 0:
                    total_dimensions += len(embedding)
            
            avg_dimensions = total_dimensions / total_embeddings if total_embeddings > 0 else 0
            
            logger.info(f"   📊 EMBEDDING STATISTICS SUMMARY")
            logger.info(f"      {'='*50}")
            logger.info(f"      📈 Total Jobs with Embeddings: {total_embeddings}")
            logger.info(f"      📏 Total Dimensions: {total_dimensions}")
            logger.info(f"      📊 Average Dimensions per Job: {avg_dimensions:.1f}")
            logger.info(f"      {'='*50}")
            
            return {
                "success": True,
                "total_jobs": total_embeddings,
                "total_dimensions": total_dimensions,
                "average_dimensions": avg_dimensions,
                "jobs": jobs_with_embeddings
            }
            
        except Exception as e:
            logger.error(f"Error retrieving job embeddings: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "jobs": []
            }

    async def show_job_embedding_details(self, job_id: int) -> Dict[str, Any]:
        """
        Show detailed information for a specific job embedding.
        
        Args:
            job_id: ID of the job to show details for
            
        Returns:
            Dict[str, Any]: Job embedding details
        """
        try:
            logger.info(f"🔍 SHOWING DETAILS FOR JOB ID: {job_id}")
            logger.info(f"   {'='*60}")
            
            # Get specific job with embedding
            job = await self.db_service.get_job_with_embedding(job_id)
            
            if not job:
                logger.info(f"   ❌ JOB WITH ID {job_id} NOT FOUND OR NO EMBEDDING")
                logger.info(f"   {'='*60}")
                return {
                    "success": False,
                    "error": f"Job with ID {job_id} not found or has no embedding"
                }
            
            # Display detailed job information
            logger.info(f"   📋 JOB DETAILS:")
            logger.info(f"      🆔 Job ID: {job.get('id', 'Unknown ID')}")
            logger.info(f"      📋 Title: {job.get('title', 'Unknown Title')}")
            logger.info(f"      🏢 Company: {job.get('company', 'Unknown Company')}")
            logger.info(f"      🏢 Department: {job.get('department', 'Unknown Department')}")
            logger.info(f"      📅 Created: {job.get('created_at', 'Unknown Date')}")
            logger.info(f"      📅 Updated: {job.get('updated_at', 'Unknown Date')}")
            
            # Show embedding details with proper type checking
            embedding = job.get('embedding', [])
            if isinstance(embedding, list) and len(embedding) > 0:
                logger.info(f"      📏 Embedding Size: {len(embedding)} dimensions")
                logger.info(f"   🔢 EMBEDDING DETAILS:")
                logger.info(f"      📏 Vector Length: {len(embedding)}")
                
                # Show first 10 values
                first_values = embedding[:10]
                logger.info(f"      🔢 First 10 Values: {first_values}")
                
                # Show last 10 values
                last_values = embedding[-10:] if len(embedding) > 10 else embedding
                logger.info(f"      🔢 Last 10 Values: {last_values}")
                
                # Show statistics with error handling
                try:
                    min_val = min(embedding)
                    max_val = max(embedding)
                    avg_val = sum(embedding) / len(embedding)
                    logger.info(f"      📊 Statistics:")
                    logger.info(f"         📉 Minimum Value: {min_val:.6f}")
                    logger.info(f"         📈 Maximum Value: {max_val:.6f}")
                    logger.info(f"         📊 Average Value: {avg_val:.6f}")
                    
                    # Show value distribution
                    positive_count = sum(1 for val in embedding if isinstance(val, (int, float)) and val > 0)
                    negative_count = sum(1 for val in embedding if isinstance(val, (int, float)) and val < 0)
                    zero_count = sum(1 for val in embedding if isinstance(val, (int, float)) and val == 0)
                    
                    logger.info(f"      📊 Value Distribution:")
                    logger.info(f"         ➕ Positive Values: {positive_count} ({(positive_count/len(embedding)*100):.1f}%)")
                    logger.info(f"         ➖ Negative Values: {negative_count} ({(negative_count/len(embedding)*100):.1f}%)")
                    logger.info(f"         ➖ Zero Values: {zero_count} ({(zero_count/len(embedding)*100):.1f}%)")
                except (TypeError, ValueError) as e:
                    logger.info(f"      📊 Statistics: Unable to calculate (data type issue)")
            else:
                logger.info(f"      📏 Embedding Size: 0 dimensions")
                logger.info(f"   ❌ NO VALID EMBEDDING DATA FOUND")
            
            logger.info(f"   {'='*60}")
            
            return {
                "success": True,
                "job": job,
                "embedding_stats": {
                    "size": len(embedding) if isinstance(embedding, list) else 0,
                    "min": min(embedding) if isinstance(embedding, list) and len(embedding) > 0 and all(isinstance(x, (int, float)) for x in embedding) else 0,
                    "max": max(embedding) if isinstance(embedding, list) and len(embedding) > 0 and all(isinstance(x, (int, float)) for x in embedding) else 0,
                    "avg": sum(embedding) / len(embedding) if isinstance(embedding, list) and len(embedding) > 0 and all(isinstance(x, (int, float)) for x in embedding) else 0
                } if isinstance(embedding, list) and len(embedding) > 0 else None
            }
            
        except Exception as e:
            logger.error(f"Error showing job embedding details: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _create_simple_fallback_embedding(self, text: str) -> List[float]:
        """
        Create a simple fallback embedding when AI_SEARCH is not available.
        This creates a basic vector representation based on word frequencies.
        """
        try:
            # Simple word-frequency based embedding
            text = text.lower()
            words = text.split()
            
            # Create a 100-dimensional vector
            embedding = [0.0] * 100
            
            # Fill the first 50 dimensions with word frequency features
            word_freq = {}
            for word in words:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
            
            # Normalize word frequencies
            max_freq = max(word_freq.values()) if word_freq else 1
            for i, (word, freq) in enumerate(list(word_freq.items())[:50]):
                embedding[i] = (freq / max_freq) * 0.5
            
            # Fill remaining dimensions with simple features
            embedding[90] = min(len(text) / 1000.0, 1.0)  # Text length
            embedding[91] = min(len(words) / 100.0, 1.0)   # Word count
            embedding[92] = 1.0 if any(word in text for word in ['developer', 'engineer', 'programmer']) else 0.0
            embedding[93] = 1.0 if any(word in text for word in ['java', 'python', 'javascript']) else 0.0
            embedding[94] = 1.0 if any(word in text for word in ['react', 'angular', 'vue']) else 0.0
            embedding[95] = 1.0 if any(word in text for word in ['experience', 'skill', 'project']) else 0.0
            embedding[96] = 1.0 if any(word in text for word in ['education', 'degree', 'university']) else 0.0
            embedding[97] = 1.0 if any(word in text for word in ['database', 'sql', 'nosql']) else 0.0
            embedding[98] = 1.0 if any(word in text for word in ['web', 'mobile', 'api']) else 0.0
            embedding[99] = 1.0 if any(word in text for word in ['aws', 'azure', 'cloud']) else 0.0
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating fallback embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * 100

    async def regenerate_all_job_embeddings(self) -> Dict[str, Any]:
        """
        Regenerate all job embeddings using OpenAI's text-embedding-ada-002 model.
        This fixes the dimension mismatch issue where jobs had 100-dim embeddings
        while resumes have 1536-dim embeddings.
        
        Returns:
            Dict[str, Any]: Result with success status and details
        """
        try:
            logger.info(f"🔄 REGENERATING ALL JOB EMBEDDINGS")
            logger.info(f"   🎯 Fixing dimension mismatch (100 → 1536 dimensions)")
            logger.info(f"   {'='*60}")
            
            # Get all jobs
            all_jobs = await self.db_service.get_all_jobs_with_embeddings(limit=1000)
            if not all_jobs:
                return {
                    "success": False,
                    "error": "No jobs found in the system"
                }
            
            logger.info(f"📊 Found {len(all_jobs)} jobs to process")
            
            # Process each job
            success_count = 0
            error_count = 0
            
            for job in all_jobs:
                try:
                    job_id = job['id']
                    logger.info(f"🔄 Regenerating embedding for job {job_id}: {job.get('title', 'Unknown')}")
                    
                    # Generate new embedding
                    new_embedding = await self.generate_job_embedding(job)
                    
                    if new_embedding:
                        # Store the new embedding
                        success = await self.store_job_embedding(job_id, new_embedding)
                        if success:
                            success_count += 1
                            logger.info(f"✅ Successfully regenerated embedding for job {job_id}")
                        else:
                            error_count += 1
                            logger.error(f"❌ Failed to store new embedding for job {job_id}")
                    else:
                        error_count += 1
                        logger.error(f"❌ Failed to generate new embedding for job {job_id}")
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"❌ Error processing job {job.get('id', 'Unknown')}: {str(e)}")
                    continue
            
            # Show final summary
            logger.info(f"🎉 JOB EMBEDDING REGENERATION COMPLETED!")
            logger.info(f"   {'='*60}")
            logger.info(f"   ✅ Successfully regenerated: {success_count} jobs")
            logger.info(f"   ❌ Errors: {error_count} jobs")
            logger.info(f"   📊 Total processed: {len(all_jobs)} jobs")
            logger.info(f"   {'='*60}")
            
            return {
                "success": True,
                "total_jobs": len(all_jobs),
                "success_count": success_count,
                "error_count": error_count,
                "message": f"Successfully regenerated {success_count} out of {len(all_jobs)} job embeddings"
            }
            
        except Exception as e:
            logger.error(f"Error regenerating all job embeddings: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# Create a global instance for easy access
job_embedding_service = JobEmbeddingService()
