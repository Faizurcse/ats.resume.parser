"""
Search service for AI-powered resume search using vector embeddings.
"""

import logging
import json
import time
import numpy as np
from typing import List, Dict, Any

# Try to import sentence_transformers, fallback to simple implementation if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence_transformers not available. Using fallback search method.")

from app.services.database_service import DatabaseService

logger = logging.getLogger(__name__)

class SearchService:
    """Service for AI-powered resume search using vector embeddings."""
    
    def __init__(self):
        """Initialize search service with sentence transformer model or fallback."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Using sentence transformer model for search")
        else:
            self.model = None
            logger.info("Using fallback search method")
        
        self.db_service = DatabaseService()
    
    async def search_resumes(self, query: str, limit: int = 10, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Search resumes using vector similarity.
        
        Args:
            query (str): Search query (e.g., 'Java developer')
            limit (int): Maximum number of results
            similarity_threshold (float): Minimum similarity score (0.0 to 1.0)
            
        Returns:
            Dict[str, Any]: Search results with similarity scores
        """
        logger.info(f"🚀 SEARCH SERVICE CALLED with query: '{query}', limit: {limit}, threshold: {similarity_threshold}")
        start_time = time.time()
        
        try:
            # Create embedding for search query
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                query_embedding = self.model.encode(query)
            else:
                query_embedding = self._create_fallback_embedding(query)
            
            logger.info(f"✅ Query embedding created: length={len(query_embedding)}")
            
            # Ensure query_embedding is a 1D numpy array
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            # Ensure it's 1D
            if query_embedding.ndim == 0:
                query_embedding = query_embedding.reshape(1)
            elif query_embedding.ndim > 1:
                query_embedding = query_embedding.flatten()
            
            logger.info(f"🔧 Query embedding processed: shape={query_embedding.shape}, sample={query_embedding[:5]}")
            
            # Search for similar resumes
            logger.info(f"🔍 Calling _search_similar_resumes...")
            results = await self._search_similar_resumes(query_embedding, limit, similarity_threshold)
            logger.info(f"🔍 _search_similar_resumes returned {len(results)} results")
            
            search_time = time.time() - start_time
            
            logger.info(f"🎯 Search completed in {search_time:.3f}s - Found {len(results)} results")
            
            return {
                "query": query,
                "total_results": len(results),
                "results": results,
                "search_time": round(search_time, 3)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in resume search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "query": query,
                "total_results": 0,
                "results": [],
                "search_time": 0.0,
                "error": str(e)
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
    
    async def _search_similar_resumes(self, query_embedding: np.ndarray, limit: int, similarity_threshold: float) -> List[Dict[str, Any]]:
        """
        Search for resumes similar to the query using vector similarity.
        
        Args:
            query_embedding (np.ndarray): Query vector embedding
            limit (int): Maximum number of results
            similarity_threshold (float): Minimum similarity score
            
        Returns:
            List[Dict[str, Any]]: List of similar resumes with scores
        """
        logger.info(f"🔍 _search_similar_resumes called with limit={limit}, threshold={similarity_threshold}")
        try:
            pool = await self.db_service._get_pool()
            logger.info(f"✅ Database pool acquired")
            
            async with pool.acquire() as conn:
                # Get all resume embeddings
                logger.info(f"🔍 Executing database query...")
                embeddings_data = await conn.fetch('''
                    SELECT re.resume_id, re.embedding, rd.candidate_name, rd.candidate_email, 
                           rd.filename, rd.parsed_data
                    FROM resume_embeddings re
                    JOIN resume_data rd ON re.resume_id = rd.id
                ''')
                
                logger.info(f"✅ Found {len(embeddings_data)} resume embeddings in database")
                
                if not embeddings_data:
                    logger.warning("⚠️ No resume embeddings found in database")
                    return []
                
                # Calculate similarities
                similarities = []
                logger.info(f"🔍 Processing {len(embeddings_data)} embeddings...")
                
                for i, record in enumerate(embeddings_data):
                    logger.info(f"🔍 Processing record {i+1}/{len(embeddings_data)}: resume_id={record['resume_id']}")
                    try:
                        # Parse embedding from JSON string
                        embedding = record['embedding']
                        if isinstance(embedding, str):
                            try:
                                embedding = json.loads(embedding)
                                logger.info(f"✅ Parsed JSON embedding for resume {record['resume_id']}")
                            except json.JSONDecodeError:
                                logger.warning(f"⚠️ Could not parse embedding JSON for resume {record['resume_id']}")
                                continue
                        
                        # Ensure embedding is a list
                        if not isinstance(embedding, list):
                            logger.warning(f"⚠️ Embedding for resume {record['resume_id']} is not a list: {type(embedding)}")
                            continue
                        
                        logger.info(f"✅ Resume {record['resume_id']} embedding length: {len(embedding)}")
                        
                        # Calculate cosine similarity
                        similarity = self._calculate_cosine_similarity(query_embedding, embedding)
                        logger.info(f"✅ Resume {record['resume_id']} similarity: {similarity:.4f} (threshold: {similarity_threshold})")
                        
                        if similarity >= similarity_threshold:
                            logger.info(f"🎯 Resume {record['resume_id']} meets threshold: {similarity:.4f} >= {similarity_threshold}")
                            # Parse parsed_data if it's a string
                            parsed_data = record['parsed_data']
                            if isinstance(parsed_data, str):
                                try:
                                    parsed_data = json.loads(parsed_data)
                                except json.JSONDecodeError:
                                    parsed_data = {}
                            
                            similarities.append({
                                'resume_id': record['resume_id'],
                                'similarity_score': similarity,
                                'similarity_percentage': round(similarity * 100, 2),
                                'candidate_name': record['candidate_name'] or 'Unknown',
                                'candidate_email': record['candidate_email'] or 'No email',
                                'filename': record['filename'],
                                'parsed_data': parsed_data
                            })
                        else:
                            logger.debug(f"📉 Resume {record['resume_id']} below threshold: {similarity:.4f} < {similarity_threshold}")
                    except Exception as e:
                        logger.warning(f"❌ Error processing embedding for resume {record['resume_id']}: {str(e)}")
                        continue
                
                logger.info(f"🎯 Total similarities calculated: {len(similarities)}")
                
                # Sort by similarity score (highest first)
                similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
                
                # Return top results
                return similarities[:limit]
                
        except Exception as e:
            logger.error(f"❌ Error searching similar resumes: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1 (np.ndarray): First vector
            vec2 (List[float]): Second vector
            
        Returns:
            float: Cosine similarity score (0.0 to 1.0)
        """
        try:
            # Convert to numpy arrays
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            logger.debug(f"Vector 1 shape: {v1.shape}, Vector 2 shape: {v2.shape}")
            logger.debug(f"Vector 1 sample: {v1[:5]}, Vector 2 sample: {v2[:5]}")
            
            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            logger.debug(f"Dot product: {dot_product}, Norm v1: {norm_v1}, Norm v2: {norm_v2}")
            
            if norm_v1 == 0 or norm_v2 == 0:
                logger.warning(f"Zero norm detected: v1={norm_v1}, v2={norm_v2}")
                return 0.0
            
            similarity = dot_product / (norm_v1 * norm_v2)
            logger.debug(f"Raw similarity: {similarity}")
            
            # Ensure result is between 0 and 1
            final_similarity = max(0.0, min(1.0, similarity))
            logger.debug(f"Final similarity: {final_similarity}")
            
            return final_similarity
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    async def get_search_suggestions(self, partial_query: str) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        Args:
            partial_query (str): Partial search query
            
        Returns:
            List[str]: List of search suggestions
        """
        try:
            # Common job titles and skills for suggestions
            suggestions = [
                "Java Developer", "Python Developer", "Frontend Developer", "Backend Developer",
                "Full Stack Developer", "Data Scientist", "DevOps Engineer", "UI/UX Designer",
                "Product Manager", "Software Engineer", "React Developer", "Node.js Developer",
                "Machine Learning Engineer", "Cloud Engineer", "Database Administrator"
            ]
            
            # Filter suggestions based on partial query
            if partial_query:
                filtered_suggestions = [
                    suggestion for suggestion in suggestions 
                    if partial_query.lower() in suggestion.lower()
                ]
                return filtered_suggestions[:5]  # Return top 5 suggestions
            
            return suggestions[:5]
            
        except Exception as e:
            logger.error(f"Error getting search suggestions: {str(e)}")
            return []
