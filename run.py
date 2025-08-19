#!/usr/bin/env python3
"""
Simple run script for the Resume Parser application.
"""

import uvicorn
from app.config.settings import settings

if __name__ == "__main__":
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Server will be available at: http://localhost:{settings.PORT}")
    print(f"API Documentation: http://localhost:{settings.PORT}/docs")
    print(f"ReDoc Documentation: http://localhost:{settings.PORT}/redoc")
    print("-" * 50)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
