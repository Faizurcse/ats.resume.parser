"""
Main FastAPI application entry point.
Simplified version with only essential endpoints.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from app.config.settings import settings
from app.controllers.resume_controller import router as resume_router
from app.controllers.job_posting_controller import router as job_posting_router
from app.controllers.download_resume_controller import router as download_resume_router
from app.AI_SEARCH.controller import router as ai_search_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware - must be added before other middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aiats.workisy.in",
        "https://pyats.workisy.in",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],  # Allowed origins
    allow_credentials=True,  # Allow credentials
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Specific methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"]  # Expose all headers
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Add processing time header to all responses.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming requests and their processing time.
    """
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response

# Test CORS endpoint
@app.get("/test-cors")
async def test_cors():
    """
    Test endpoint to verify CORS is working.
    """
    from fastapi.responses import JSONResponse
    response = JSONResponse(
        content={"message": "CORS test successful", "timestamp": time.time()}
    )
    return response

# Test OpenAI API key endpoint
@app.get("/test-openai")
async def test_openai_api():
    """
    Test endpoint to verify OpenAI API key is working.
    """
    try:
        from app.services.openai_service import OpenAIService
        from app.config.settings import settings
        
        # Check if API key is set
        if not settings.OPENAI_API_KEY:
            return {
                "status": "❌ FAILED",
                "message": "OpenAI API key is not set in .env file",
                "error": "OPENAI_API_KEY environment variable is missing",
                "timestamp": time.time()
            }
        
        # Test if API key is valid
        openai_service = OpenAIService()
        
        # Try a simple API call
        test_response = openai_service.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test message"}],
            max_tokens=10
        )
        
        return {
            "status": "✅ WORKING",
            "message": "OpenAI API key is valid and working!",
            "api_key_set": True,
            "api_key_valid": True,
            "model": "gpt-3.5-turbo",
            "test_response": test_response.choices[0].message.content,
            "timestamp": time.time()
        }
        
    except Exception as e:
        error_msg = str(e)
        
        # Check for specific error types
        if "Invalid API key" in error_msg or "authentication" in error_msg.lower():
            return {
                "status": "❌ FAILED",
                "message": "OpenAI API key is invalid or expired",
                "error": error_msg,
                "api_key_set": True,
                "api_key_valid": False,
                "timestamp": time.time()
            }
        elif "insufficient" in error_msg.lower() or "quota" in error_msg.lower():
            return {
                "status": "❌ FAILED",
                "message": "OpenAI account has insufficient credits/quota",
                "error": error_msg,
                "api_key_set": True,
                "api_key_valid": True,
                "quota_issue": True,
                "timestamp": time.time()
            }
        elif "rate limit" in error_msg.lower():
            return {
                "status": "⚠️ WARNING",
                "message": "OpenAI API rate limit exceeded",
                "error": error_msg,
                "api_key_set": True,
                "api_key_valid": True,
                "rate_limit_issue": True,
                "timestamp": time.time()
            }
        else:
            return {
                "status": "❌ FAILED",
                "message": "OpenAI API test failed with unknown error",
                "error": error_msg,
                "api_key_set": True,
                "api_key_valid": False,
                "timestamp": time.time()
            }

# Include routers
app.include_router(resume_router)
app.include_router(job_posting_router)
app.include_router(download_resume_router)
app.include_router(ai_search_router)



# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with application information.
    
    Returns:
        dict: Application information and available endpoints
    """
    return {
        "message": "Resume Parser API",
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "endpoints": {
            "health": "/api/v1/health",
            "parse_resume": "/api/v1/parse-resume",
            "generate_job_posting": "/api/v1/job-posting/generate",
            "all_resumes": "/api/v1/resumes",
            "download_unique_resumes": "/api/v1/download/resumes",
            "download_unique_resumes_with_files": "/api/v1/download/resumes/with-files",
            "download_all_resumes_admin": "/api/v1/download/resumes/all",
            "download_resume_file": "/api/v1/download/resume/{resume_id}",
            "ai_search": "/ai-search/search",
            "embedding_status": "/ai-search/embedding-status",
            "test_cors": "/test-cors",
            "test_openai": "/test-openai",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "supported_formats": settings.ALLOWED_EXTENSIONS,
        "max_file_size_mb": settings.MAX_FILE_SIZE / (1024 * 1024)
    }

# Health check endpoint (root level)
@app.get("/health")
async def health_check():
    """
    Health check endpoint at root level.
    
    Returns:
        dict: Application health status
    """
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "timestamp": time.time()
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled exceptions.
    
    Args:
        request (Request): The request that caused the exception
        exc (Exception): The unhandled exception
        
    Returns:
        JSONResponse: Error response
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "error_code": "INTERNAL_ERROR"
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.
    """
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Max file size: {settings.MAX_FILE_SIZE} bytes")
    logger.info(f"Supported formats: {settings.ALLOWED_EXTENSIONS}")
    
    # Validate settings
    try:
        settings.validate_settings()
        logger.info("Settings validation passed")
    except Exception as e:
        logger.error(f"Settings validation failed: {str(e)}")
        raise
    
    # Initialize database and ensure schema is up to date
    try:
        from app.services.database_service import DatabaseService
        db_service = DatabaseService()
        await db_service._initialize()
        logger.info("Database schema validation completed")
    except Exception as e:
        logger.error(f"Database schema validation failed: {str(e)}")
        # Don't fail startup for database issues, but log them

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event handler.
    """
    logger.info(f"Shutting down {settings.APP_NAME}")

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
