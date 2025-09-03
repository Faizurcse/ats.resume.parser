#!/usr/bin/env python3
"""
Simple run script for the Resume Parser application.
"""

import asyncio
import uvicorn
from app.config.settings import settings
from app.services.job_embedding_service import job_embedding_service

async def show_startup_status():
    """Show comprehensive startup status."""
    try:
        # Import and use the comprehensive startup status service
        from app.services.startup_status_service import startup_status_service
        
        # Display comprehensive startup status
        await startup_status_service.display_comprehensive_startup_status()
        
    except Exception as e:
        print(f"⚠️  Could not show comprehensive startup status: {str(e)}")
        # Fallback to basic startup message
        print("🚀 Starting Resume Parser Backend...")
        print("=" * 60)
        print(f"🎯 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
        print(f"🌐 Server will be available at: http://localhost:{settings.PORT}")
        print(f"📚 API Documentation: http://localhost:{settings.PORT}/docs")
        print(f"📖 ReDoc Documentation: http://localhost:{settings.PORT}/redoc")
        print("=" * 60)

if __name__ == "__main__":
    # Show startup status
    asyncio.run(show_startup_status())
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
