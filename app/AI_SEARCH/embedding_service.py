"""
Embedding service for creating and managing vector embeddings for resumes.
"""

import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional

# Try to import sentence_transformers, fallback to simple implementation if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence_transformers not available. Using fallback embedding method.")

from app.services.database_service import DatabaseService

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for creating and managing resume embeddings."""
    
    def __init__(self):
        """Initialize embedding service with sentence transformer model or fallback."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use the full sentence transformer model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dimension = 384
            logger.info("Using sentence transformer model for embeddings")
        else:
            # Fallback to simple TF-IDF like approach
            self.model = None
            self.embedding_dimension = 100
            logger.info("Using fallback embedding method")
        
        self.db_service = DatabaseService()
        
    async def create_resume_embedding(self, resume_id: int) -> Dict[str, Any]:
        """
        Create embedding for a specific resume.
        
        Args:
            resume_id (int): Resume ID to create embedding for
            
        Returns:
            Dict[str, Any]: Result of embedding creation
        """
        try:
            # Get resume data from database
            resume_data = await self.db_service.get_resume_by_id(resume_id)
            if not resume_data:
                return {
                    "success": False,
                    "resume_id": resume_id,
                    "embedding_status": "failed",
                    "message": f"Resume with ID {resume_id} not found"
                }
            
            # Debug logging
            logger.info(f"Resume {resume_id} data type: {type(resume_data['parsed_data'])}")
            logger.info(f"Resume {resume_id} parsed_data sample: {str(resume_data['parsed_data'])[:200]}...")
            
            # Extract text content for embedding
            # Handle case where parsed_data might be a string or dict
            parsed_data = resume_data['parsed_data']
            if isinstance(parsed_data, str):
                try:
                    import json
                    parsed_data = json.loads(parsed_data)
                    logger.info(f"Successfully parsed JSON string for resume {resume_id}")
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse parsed_data as JSON for resume {resume_id}")
                    parsed_data = {}
            
            text_content = self._extract_text_for_embedding(parsed_data)
            
            # Create embedding
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                embedding = self.model.encode(text_content)
                embedding = embedding.tolist()
            else:
                embedding = self._create_fallback_embedding(text_content)
            
            # Save embedding to database
            await self._save_embedding_to_db(resume_id, embedding)
            
            return {
                "success": True,
                "resume_id": resume_id,
                "embedding_status": "created",
                "message": "Embedding created successfully"
            }
            
        except Exception as e:
            logger.error(f"Error creating embedding for resume {resume_id}: {str(e)}")
            return {
                "success": False,
                "resume_id": resume_id,
                "embedding_status": "failed",
                "message": f"Error creating embedding: {str(e)}"
            }
    
    def _create_fallback_embedding(self, text: str) -> List[float]:
        """
        Create a simple fallback embedding when sentence_transformers is not available.
        This creates a more meaningful vector representation based on word frequencies and key terms.
        """
        try:
            # Simple but more effective word-frequency based embedding
            text = text.lower()
            words = text.split()
            
            # Create a 100-dimensional vector
            embedding = [0.0] * 100
            
            # Key technology and skill words with higher weights
            tech_keywords = {
                'java': 0.8, 'python': 0.8, 'javascript': 0.8, 'react': 0.8, 'node': 0.7,
                'developer': 0.9, 'engineer': 0.9, 'programmer': 0.8, 'coder': 0.7,
                'frontend': 0.8, 'backend': 0.8, 'fullstack': 0.8, 'full-stack': 0.8,
                'web': 0.7, 'mobile': 0.7, 'desktop': 0.6, 'api': 0.7, 'database': 0.7,
                'sql': 0.7, 'nosql': 0.7, 'mongodb': 0.7, 'postgresql': 0.7,
                'aws': 0.7, 'azure': 0.7, 'cloud': 0.7, 'docker': 0.7, 'kubernetes': 0.7,
                'git': 0.6, 'agile': 0.6, 'scrum': 0.6, 'testing': 0.6, 'debug': 0.6
            }
            
            # Experience and education keywords
            exp_keywords = {
                'experience': 0.6, 'years': 0.6, 'senior': 0.7, 'junior': 0.5, 'lead': 0.7,
                'team': 0.6, 'project': 0.6, 'management': 0.6, 'architect': 0.7,
                'degree': 0.5, 'university': 0.5, 'college': 0.5, 'bachelor': 0.5, 'master': 0.6,
                'certification': 0.6, 'certified': 0.6
            }
            
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
            
            # Fill dimensions 50-79 with technology keyword matches
            for i, (keyword, weight) in enumerate(tech_keywords.items()):
                if keyword in text:
                    embedding[50 + i] = weight
            
            # Fill dimensions 80-89 with experience/education features
            for i, (keyword, weight) in enumerate(exp_keywords.items()):
                if keyword in text:
                    embedding[80 + i] = weight
            
            # Fill dimensions 90-99 with general text features
            embedding[90] = min(len(text) / 1000.0, 1.0)  # Text length
            embedding[91] = min(len(words) / 100.0, 1.0)   # Word count
            embedding[92] = 1.0 if any(word in text for word in ['java', 'python', 'javascript']) else 0.0
            embedding[93] = 1.0 if any(word in text for word in ['react', 'angular', 'vue']) else 0.0
            embedding[94] = 1.0 if any(word in text for word in ['developer', 'engineer', 'programmer']) else 0.0
            embedding[95] = 1.0 if any(word in text for word in ['experience', 'skill', 'project']) else 0.0
            embedding[96] = 1.0 if any(word in text for word in ['education', 'degree', 'university']) else 0.0
            embedding[97] = 1.0 if any(word in text for word in ['database', 'sql', 'nosql']) else 0.0
            embedding[98] = 1.0 if any(word in text for word in ['web', 'mobile', 'api']) else 0.0
            embedding[99] = 1.0 if any(word in text for word in ['aws', 'azure', 'cloud']) else 0.0
            
            logger.debug(f"Created fallback embedding for text: '{text[:50]}...' - Vector: {embedding[:10]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating fallback embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * 100
    
    async def create_embeddings_for_all_resumes(self) -> Dict[str, Any]:
        """
        Create embeddings for all resumes in the database.
        
        Returns:
            Dict[str, Any]: Result of batch embedding creation
        """
        try:
            # Get all resumes without embeddings
            resumes = await self.db_service.get_all_resumes(limit=1000, offset=0)
            
            success_count = 0
            failed_count = 0
            
            for resume in resumes:
                try:
                    result = await self.create_resume_embedding(resume['id'])
                    if result['success']:
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Error processing resume {resume['id']}: {str(e)}")
                    failed_count += 1
            
            return {
                "success": True,
                "total_processed": len(resumes),
                "successful": success_count,
                "failed": failed_count,
                "message": f"Processed {len(resumes)} resumes. {success_count} successful, {failed_count} failed."
            }
            
        except Exception as e:
            logger.error(f"Error in batch embedding creation: {str(e)}")
            return {
                "success": False,
                "message": f"Error in batch embedding creation: {str(e)}"
            }
    
    def _extract_text_for_embedding(self, parsed_data: Dict[str, Any]) -> str:
        """
        Extract and combine relevant text from parsed resume data for embedding.
        
        Args:
            parsed_data (Dict[str, Any]): Parsed resume data
            
        Returns:
            str: Combined text for embedding
        """
        try:
            # Debug logging
            logger.info(f"Extracting text from parsed_data type: {type(parsed_data)}")
            logger.info(f"Parsed_data keys: {list(parsed_data.keys()) if isinstance(parsed_data, dict) else 'Not a dict'}")
            
            text_parts = []
            
            # Extract key fields that are most relevant for search
            key_fields = [
                'Name', 'Skills', 'Experience', 'Education', 'Summary', 
                'TechnicalSkills', 'ProgrammingLanguages', 'Technologies',
                'WorkExperience', 'Projects', 'Certifications'
            ]
            
            for field in key_fields:
                if field in parsed_data and parsed_data[field]:
                    value = parsed_data[field]
                    if isinstance(value, list):
                        text_parts.extend([str(item) for item in value])
                    else:
                        text_parts.append(str(value))
            
            # Combine all text parts
            combined_text = " ".join(text_parts)
            
            # Clean and normalize text
            cleaned_text = " ".join(combined_text.split())
            
            logger.info(f"Extracted text length: {len(cleaned_text)}")
            logger.info(f"Text sample: {cleaned_text[:100]}...")
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error in _extract_text_for_embedding: {str(e)}")
            logger.error(f"Parsed_data: {parsed_data}")
            raise
    
    async def _save_embedding_to_db(self, resume_id: int, embedding: List[float]) -> bool:
        """
        Save embedding to database.
        
        Args:
            resume_id (int): Resume ID
            embedding (List[float]): Vector embedding
            
        Returns:
            bool: True if saved successfully
        """
        try:
            pool = await self.db_service._get_pool()
            
            async with pool.acquire() as conn:
                # Check if embedding already exists
                existing = await conn.fetchrow('''
                    SELECT id FROM resume_embeddings WHERE resume_id = $1
                ''', resume_id)
                
                if existing:
                    # Update existing embedding
                    await conn.execute('''
                        UPDATE resume_embeddings 
                        SET embedding = $1, updated_at = NOW()
                        WHERE resume_id = $2
                    ''', json.dumps(embedding), resume_id)
                else:
                    # Insert new embedding
                    await conn.execute('''
                        INSERT INTO resume_embeddings (resume_id, embedding, created_at, updated_at)
                        VALUES ($1, $2, NOW(), NOW())
                    ''', resume_id, json.dumps(embedding))
                
                logger.info(f"Embedding saved for resume {resume_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving embedding for resume {resume_id}: {str(e)}")
            return False
    
    async def get_embedding_status(self) -> Dict[str, Any]:
        """
        Get status of embeddings in the database.
        
        Returns:
            Dict[str, Any]: Embedding status information
        """
        try:
            pool = await self.db_service._get_pool()
            
            async with pool.acquire() as conn:
                # Get total resumes count
                total_resumes = await conn.fetchval('SELECT COUNT(*) FROM resume_data')
                
                # Get embedded resumes count
                embedded_resumes = await conn.fetchval('SELECT COUNT(*) FROM resume_embeddings')
                
                # Calculate percentage
                embedding_percentage = (embedded_resumes / total_resumes * 100) if total_resumes > 0 else 0
                
                # Get last updated time
                last_updated = await conn.fetchval('''
                    SELECT MAX(updated_at) FROM resume_embeddings
                ''')
                
                return {
                    "total_resumes": total_resumes,
                    "embedded_resumes": embedded_resumes,
                    "pending_resumes": total_resumes - embedded_resumes,
                    "embedding_percentage": round(embedding_percentage, 2),
                    "last_updated": last_updated.isoformat() if last_updated else None
                }
                
        except Exception as e:
            logger.error(f"Error getting embedding status: {str(e)}")
            return {
                "total_resumes": 0,
                "embedded_resumes": 0,
                "pending_resumes": 0,
                "embedding_percentage": 0.0,
                "last_updated": None
            }
