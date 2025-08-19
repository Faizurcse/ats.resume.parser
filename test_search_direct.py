#!/usr/bin/env python3
"""
Test script to test the search service directly.
"""

import asyncio
import logging
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from AI_SEARCH.search_service import SearchService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_search():
    """Test the search service directly."""
    try:
        logger.info("🚀 Starting direct search service test")
        
        # Create search service
        search_service = SearchService()
        logger.info("✅ Search service created")
        
        # Test fallback embedding creation
        test_query = "developer"
        logger.info(f"🔍 Testing query: '{test_query}'")
        
        embedding = search_service._create_fallback_embedding(test_query)
        logger.info(f"✅ Fallback embedding created: length={len(embedding)}, sample={embedding[:10]}")
        
        # Test search
        logger.info("🔍 Testing search...")
        results = await search_service.search_resumes(test_query, limit=5, similarity_threshold=0.001)
        
        logger.info(f"🎯 Search results: {results}")
        
    except Exception as e:
        logger.error(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_search())
