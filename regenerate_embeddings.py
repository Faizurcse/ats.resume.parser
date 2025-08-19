#!/usr/bin/env python3
"""
Script to regenerate all resume embeddings using the improved fallback method.
This is needed because we updated the embedding algorithm.
"""

import asyncio
import logging
from app.AI_SEARCH.embedding_service import EmbeddingService
from app.services.database_service import DatabaseService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def regenerate_all_embeddings():
    """Regenerate embeddings for all resumes in the database."""
    try:
        db_service = DatabaseService()
        embedding_service = EmbeddingService()
        
        # Get all resume IDs
        pool = await db_service._get_pool()
        async with pool.acquire() as conn:
            resume_ids = await conn.fetch('SELECT id FROM resume_data ORDER BY id')
        
        logger.info(f"Found {len(resume_ids)} resumes to regenerate embeddings for")
        
        # Delete all existing embeddings
        async with pool.acquire() as conn:
            await conn.execute('DELETE FROM resume_embeddings')
        logger.info("Deleted all existing embeddings")
        
        # Regenerate embeddings
        success_count = 0
        failed_count = 0
        
        for record in resume_ids:
            resume_id = record['id']
            try:
                result = await embedding_service.create_resume_embedding(resume_id)
                if result['success']:
                    success_count += 1
                    logger.info(f"✅ Regenerated embedding for resume {resume_id}")
                else:
                    failed_count += 1
                    logger.error(f"❌ Failed to regenerate embedding for resume {resume_id}: {result['message']}")
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Exception regenerating embedding for resume {resume_id}: {str(e)}")
        
        logger.info(f"🎯 Embedding regeneration complete: {success_count} successful, {failed_count} failed")
        
        # Check final status
        status = await embedding_service.get_embedding_status()
        logger.info(f"Final embedding status: {status}")
        
    except Exception as e:
        logger.error(f"Fatal error in embedding regeneration: {str(e)}")

if __name__ == "__main__":
    asyncio.run(regenerate_all_embeddings())
