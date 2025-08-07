#!/usr/bin/env python3
"""
Simple run script for the Resume Parser application.
"""

import uvicorn
from app.config.settings import settings

if __name__ == "__main__":
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Server will be available at: http://localhost:8000")
    print(f"API Documentation: http://localhost:8000/docs")
    print(f"ReDoc Documentation: http://localhost:8000/redoc")
    print("-" * 50)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
