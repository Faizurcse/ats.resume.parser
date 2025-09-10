"""
Main FastAPI application entry point.
Simplified version with only essential endpoints.
"""

import logging
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from app.config.settings import settings
from app.controllers.resume_controller import router as resume_router
from app.controllers.job_posting_controller import router as job_posting_router
from app.controllers.download_resume_controller import router as download_resume_router
from app.controllers.job_post_embeddings_controller import router as job_post_embeddings_router
from app.controllers.candidates_matching_external_controller import router as candidates_matching_router
from app.controllers.get_embedding_data_controller import router as embedding_data_router



# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
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

# Configure response size limits
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses > 1KB

# Add CORS middleware - must be added before other middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using wildcard origins
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],  # Specific methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
    max_age=86400  # Cache preflight response for 24 hours
)

# Add preflight handler for CORS
@app.options("/{full_path:path}")
async def preflight_handler(full_path: str):
    """
    Handle preflight OPTIONS requests for CORS.
    """
    return JSONResponse(
        content={"message": "Preflight request handled"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400"
        }
    )

# Add CORS headers middleware
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    """
    Add CORS headers to all responses.
    """
    response = await call_next(request)
    
    # Add CORS headers if not already present
    if "Access-Control-Allow-Origin" not in response.headers:
        response.headers["Access-Control-Allow-Origin"] = "*"
    if "Access-Control-Allow-Methods" not in response.headers:
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, HEAD"
    if "Access-Control-Allow-Headers" not in response.headers:
        response.headers["Access-Control-Allow-Headers"] = "*"
    
    return response

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
        content={
            "message": "CORS test successful", 
            "timestamp": time.time(),
            "cors_enabled": True,
            "allowed_origins": ["*"]  # Allow all origins
        }
    )
    # Add CORS headers explicitly
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, HEAD"
    response.headers["Access-Control-Allow-Headers"] = "*"
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
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": "Hello, this is a test message"}],
            max_tokens=10
        )
        
        return {
            "status": "✅ WORKING",
            "message": "OpenAI API key is valid and working!",
            "api_key_set": True,
            "api_key_valid": True,
            "model": settings.OPENAI_MODEL,
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
app.include_router(job_post_embeddings_router)
app.include_router(candidates_matching_router)
app.include_router(embedding_data_router)









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
            "health": "/health",
            "startup_status": "/startup-status",
            "parse_resume": "/api/v1/parse-resume",
            "resume_embeddings_status": "/api/v1/resume-embeddings-status",
            "generate_resume_embeddings": "/api/v1/generate-resume-embeddings",
            "generate_job_posting": "/api/v1/job-posting/generate",
            "all_resumes": "/api/v1/resumes",
            "download_unique_resumes": "/api/v1/download/resumes",
            "download_unique_resumes_with_files": "/api/v1/download/resumes/with-files",
            "download_all_resumes_admin": "/api/v1/download/resumes/all",
            "download_resume_file": "/api/v1/download/resume/{resume_id}",

            "job_post_embeddings": {
                "start_embedding": "/job-post-embeddings/start-embedding",
                "all_embeddings": "/job-post-embeddings/all-embeddings",
                "summary": "/job-post-embeddings/summary",
                "health": "/job-post-embeddings/health"
            },
            "candidates_matching": {
                "fast_matching": "/api/v1/candidates-matching/job/{job_id}/candidates-fast",
                "all_matches": "/api/v1/candidates-matching/all-matches",
                "populate_locations": "/api/v1/candidates-matching/populate-job-locations",
                "health": "/api/v1/candidates-matching/health"
            },
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

# Startup status endpoint
@app.get("/startup-status")
async def get_startup_status():
    """
    Get comprehensive startup status information.
    
    Returns:
        dict: Complete startup status report
    """
    try:
        from app.services.startup_status_service import startup_status_service
        return await startup_status_service.display_comprehensive_startup_status()
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
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
    Application startup event handler with comprehensive status display.
    """
    try:
        # Show comprehensive startup status
        from app.services.startup_status_service import startup_status_service
        
        # Display comprehensive startup status with timeout
        try:
            await asyncio.wait_for(
                startup_status_service.display_comprehensive_startup_status(),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            logger.warning("⚠️  Startup status check timed out - database may be slow to respond")
            logger.info("🚀 Starting Resume Parser Backend...")
            logger.info("=" * 60)
            logger.info(f"🎯 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
            logger.info(f"🌐 Server will be available at: http://localhost:{settings.PORT}")
            logger.info(f"📚 API Documentation: http://localhost:{settings.PORT}/docs")
            logger.info(f"📖 ReDoc Documentation: http://localhost:{settings.PORT}/redoc")
            logger.info("=" * 60)
            logger.info("ℹ️  For detailed system status, visit: /startup-status")
        except Exception as e:
            logger.error(f"⚠️  Could not show comprehensive startup status: {str(e)}")
            # Fallback to basic startup message
            logger.info("🚀 Starting Resume Parser Backend...")
            logger.info("=" * 60)
            logger.info(f"🎯 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
            logger.info(f"🌐 Server will be available at: http://localhost:{settings.PORT}")
            logger.info(f"📚 API Documentation: http://localhost:{settings.PORT}/docs")
            logger.info(f"📖 ReDoc Documentation: http://localhost:{settings.PORT}/redoc")
            logger.info("=" * 60)
            logger.info("ℹ️  For detailed system status, visit: /startup-status")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        # Don't re-raise the exception to prevent startup failure

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
