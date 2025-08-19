#!/usr/bin/env python3
"""
Debug script to test the search functionality step by step.
"""

import asyncio
import logging
from app.AI_SEARCH.search_service import SearchService
from app.services.database_service import DatabaseService
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_search():
    """Debug the search functionality step by step."""
    try:
        search_service = SearchService()
        db_service = DatabaseService()
        
        # Test 1: Check if we can get database connection
        logger.info("=== Test 1: Database Connection ===")
        pool = await db_service._get_pool()
        logger.info("✅ Database connection successful")
        
        # Test 2: Check if embeddings exist
        logger.info("=== Test 2: Check Embeddings ===")
        async with pool.acquire() as conn:
            embeddings_count = await conn.fetchval('SELECT COUNT(*) FROM resume_embeddings')
            logger.info(f"Found {embeddings_count} embeddings in database")
            
            # Get a sample embedding
            sample_embedding = await conn.fetchrow('SELECT * FROM resume_embeddings LIMIT 1')
            if sample_embedding:
                logger.info(f"Sample embedding ID: {sample_embedding['resume_id']}")
                logger.info(f"Sample embedding data type: {type(sample_embedding['embedding'])}")
                logger.info(f"Sample embedding length: {len(sample_embedding['embedding'])}")
            else:
                logger.error("No embeddings found!")
                return
        
        # Test 3: Test fallback embedding creation
        logger.info("=== Test 3: Test Fallback Embedding ===")
        test_query = "React developer"
        query_embedding = search_service._create_fallback_embedding(test_query)
        logger.info(f"Query: '{test_query}'")
        logger.info(f"Query embedding length: {len(query_embedding)}")
        logger.info(f"Query embedding sample: {query_embedding[:10]}")
        
        # Test 4: Test similarity calculation
        logger.info("=== Test 4: Test Similarity Calculation ===")
        async with pool.acquire() as conn:
            # Get first few embeddings
            embeddings_data = await conn.fetch('''
                SELECT re.resume_id, re.embedding, rd.candidate_name, rd.filename
                FROM resume_embeddings re
                JOIN resume_data rd ON re.resume_id = rd.id
                LIMIT 3
            ''')
            
            for record in embeddings_data:
                resume_id = record['resume_id']
                embedding = record['embedding']
                candidate_name = record['candidate_name']
                filename = record['filename']
                
                logger.info(f"\n--- Resume {resume_id}: {candidate_name} ({filename}) ---")
                logger.info(f"Raw embedding type: {type(embedding)}")
                logger.info(f"Raw embedding length: {len(embedding)}")
                logger.info(f"Raw embedding sample: {embedding[:100]}...")
                
                # Parse the embedding using the same logic as the search service
                try:
                    if isinstance(embedding, str):
                        import json
                        parsed_embedding = json.loads(embedding)
                        logger.info(f"Parsed embedding type: {type(parsed_embedding)}")
                        logger.info(f"Parsed embedding length: {len(parsed_embedding)}")
                        logger.info(f"Parsed embedding sample: {parsed_embedding[:10]}")
                    else:
                        parsed_embedding = embedding
                        logger.info(f"Embedding was already parsed: {type(parsed_embedding)}")
                    
                    # Calculate similarity using the search service method
                    similarity = search_service._calculate_cosine_similarity(query_embedding, parsed_embedding)
                    logger.info(f"Similarity with query: {similarity:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error processing embedding: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Test 5: Test actual search
        logger.info("=== Test 5: Test Actual Search ===")
        results = await search_service.search_resumes("React developer", limit=5, similarity_threshold=0.01)
        logger.info(f"Search results: {results}")
        
    except Exception as e:
        logger.error(f"Error in debug search: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_search())
