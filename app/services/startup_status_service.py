"""
Startup Status Service
Comprehensive service to display project status on startup including:
- Server status
- Database connection
- OpenAI API connection
- Job embeddings status
- Resume embeddings status
- Matching system status
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.services.database_service import DatabaseService
from app.services.openai_service import OpenAIService
from app.services.job_embedding_service import job_embedding_service
from app.config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

class StartupStatusService:
    """Service for displaying comprehensive startup status."""
    
    def __init__(self):
        """Initialize the startup status service."""
        self.db_service = DatabaseService()
        self.openai_service = None
        self.startup_time = datetime.now()
    
    async def display_comprehensive_startup_status(self) -> Dict[str, Any]:
        """
        Display comprehensive startup status with all system components.
        
        Returns:
            Dict[str, Any]: Complete startup status report
        """
        try:
            logger.info("🚀" + "="*80)
            logger.info("🚀 ATS RESUME PARSER - COMPREHENSIVE STARTUP STATUS")
            logger.info("🚀" + "="*80)
            logger.info(f"⏰ Startup Time: {self.startup_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"📱 Application: {settings.APP_NAME} v{settings.APP_VERSION}")
            logger.info(f"🌐 Server Port: {settings.PORT}")
            logger.info(f"🔧 Debug Mode: {settings.DEBUG}")
            logger.info("="*80)
            
            # Initialize status tracking
            status_report = {
                "startup_time": self.startup_time.isoformat(),
                "app_name": settings.APP_NAME,
                "app_version": settings.APP_VERSION,
                "server_port": settings.PORT,
                "debug_mode": settings.DEBUG,
                "components": {}
            }
            
            # 1. Server Status
            await self._display_server_status(status_report)
            
            # 2. Database Connection Status
            await self._display_database_status(status_report)
            
            # 3. OpenAI API Status
            await self._display_openai_status(status_report)
            
            # 4. Job Embeddings Status (with proper database initialization)
            try:
                # Ensure database is fully initialized before making queries
                await asyncio.wait_for(self.db_service._get_pool(), timeout=5.0)
                await asyncio.sleep(1.0)  # Wait for database to be ready
                await self._display_job_embeddings_status(status_report)
            except asyncio.TimeoutError:
                logger.warning("Job embeddings status check timed out - database may be slow")
                status_report["components"]["job_embeddings"] = {
                    "status": "⚠️ TIMEOUT",
                    "message": "Database connection timeout - job embeddings status unavailable"
                }
            except Exception as e:
                logger.warning(f"Job embeddings status check failed: {str(e)}")
                status_report["components"]["job_embeddings"] = {
                    "status": "⚠️ WARNING",
                    "message": "Could not check job embeddings status"
                }
            
            # 5. Resume Embeddings Status (with proper database initialization)
            try:
                # Ensure database is fully initialized before making queries
                await asyncio.wait_for(self.db_service._get_pool(), timeout=5.0)
                await asyncio.sleep(1.0)  # Wait for database to be ready
                await self._display_resume_embeddings_status(status_report)
            except asyncio.TimeoutError:
                logger.warning("Resume embeddings status check timed out - database may be slow")
                status_report["components"]["resume_embeddings"] = {
                    "status": "⚠️ TIMEOUT",
                    "message": "Database connection timeout - resume embeddings status unavailable"
                }
            except Exception as e:
                logger.warning(f"Resume embeddings status check failed: {str(e)}")
                status_report["components"]["resume_embeddings"] = {
                    "status": "⚠️ WARNING", 
                    "message": "Could not check resume embeddings status"
                }
            
            # 6. Matching System Status (with proper database initialization)
            try:
                # Ensure database is fully initialized before making queries
                await asyncio.wait_for(self.db_service._get_pool(), timeout=5.0)
                await asyncio.sleep(1.0)  # Wait for database to be ready
                await self._display_matching_system_status(status_report)
            except asyncio.TimeoutError:
                logger.warning("Matching system status check timed out - database may be slow")
                status_report["components"]["matching_system"] = {
                    "status": "⚠️ TIMEOUT",
                    "message": "Database connection timeout - matching system status unavailable"
                }
            except Exception as e:
                logger.warning(f"Matching system status check failed: {str(e)}")
                status_report["components"]["matching_system"] = {
                    "status": "⚠️ WARNING",
                    "message": "Could not check matching system status"
                }
            
            # 7. Final Summary
            await self._display_final_summary(status_report)
            
            return status_report
            
        except Exception as e:
            logger.error(f"Error displaying startup status: {str(e)}")
            return {
                "error": str(e),
                "startup_time": self.startup_time.isoformat()
            }
    
    async def _display_server_status(self, status_report: Dict[str, Any]):
        """Display server startup status."""
        try:
            logger.info("🌐 SERVER STATUS")
            logger.info("   " + "─"*50)
            logger.info("   ✅ FastAPI Server: STARTED SUCCESSFULLY")
            logger.info(f"   🌐 Server URL: http://localhost:{settings.PORT}")
            logger.info(f"   📚 API Documentation: http://localhost:{settings.PORT}/docs")
            logger.info(f"   📖 ReDoc Documentation: http://localhost:{settings.PORT}/redoc")
            logger.info("   🔧 Middleware: CORS, GZip, Request Logging")
            logger.info("   📊 Response Headers: X-Process-Time")
            logger.info("   " + "─"*50)
            
            status_report["components"]["server"] = {
                "status": "✅ SUCCESS",
                "url": f"http://localhost:{settings.PORT}",
                "docs_url": f"http://localhost:{settings.PORT}/docs",
                "redoc_url": f"http://localhost:{settings.PORT}/redoc",
                "middleware": ["CORS", "GZip", "Request Logging"],
                "response_headers": ["X-Process-Time"]
            }
            
        except Exception as e:
            logger.error(f"   ❌ Server Status Error: {str(e)}")
            status_report["components"]["server"] = {
                "status": "❌ ERROR",
                "error": str(e)
            }
    
    async def _display_database_status(self, status_report: Dict[str, Any]):
        """Display database connection status."""
        try:
            logger.info("🗄️  DATABASE CONNECTION STATUS")
            logger.info("   " + "─"*50)
            
            # Test database connection with timeout
            await asyncio.wait_for(self.db_service._get_pool(), timeout=10.0)
            
            logger.info("   ✅ Database: CONNECTED SUCCESSFULLY")
            logger.info("   🏢 Host: 147.93.155.233:5432")
            logger.info("   🗄️  Database: ai_ats")
            logger.info("   👤 User: root")
            logger.info("   🔌 Connection Pool: Active")
            logger.info("   📊 Tables: resume_data, Ats_JobPost")
            logger.info("   " + "─"*50)
            
            status_report["components"]["database"] = {
                "status": "✅ CONNECTED",
                "host": "147.93.155.233",
                "port": 5432,
                "database": "ai_ats",
                "user": "root",
                "connection_pool": "Active",
                "tables": ["resume_data", "Ats_JobPost"]
            }
            
        except asyncio.TimeoutError:
            logger.warning("   ⚠️  Database Connection: TIMEOUT (10s)")
            logger.info("   🏢 Host: 147.93.155.233:5432")
            logger.info("   🗄️  Database: ai_ats")
            logger.info("   👤 User: root")
            logger.info("   ⚠️  Connection may be slow - server will start anyway")
            logger.info("   " + "─"*50)
            
            status_report["components"]["database"] = {
                "status": "⚠️ TIMEOUT",
                "host": "147.93.155.233",
                "port": 5432,
                "database": "ai_ats",
                "user": "root",
                "warning": "Connection timeout - server will start anyway"
            }
            
        except Exception as e:
            logger.error(f"   ❌ Database Connection Failed: {str(e)}")
            logger.info("   🏢 Host: 147.93.155.233:5432")
            logger.info("   🗄️  Database: ai_ats")
            logger.info("   👤 User: root")
            logger.info("   ⚠️  Server will start but database features may not work")
            logger.info("   " + "─"*50)
            
            status_report["components"]["database"] = {
                "status": "❌ CONNECTION FAILED",
                "host": "147.93.155.233",
                "port": 5432,
                "database": "ai_ats",
                "user": "root",
                "error": str(e),
                "warning": "Database features may not work"
            }
    
    async def _display_openai_status(self, status_report: Dict[str, Any]):
        """Display OpenAI API connection status."""
        try:
            logger.info("🤖 OPENAI API STATUS")
            logger.info("   " + "─"*50)
            
            # Check if API key is set
            if not settings.OPENAI_API_KEY:
                logger.info("   ❌ OpenAI API Key: NOT SET")
                logger.info("   ⚠️  AI features will not work")
                status_report["components"]["openai"] = {
                    "status": "❌ API KEY NOT SET",
                    "model": settings.OPENAI_MODEL,
                    "warning": "AI features will not work"
                }
                return
            
            # Initialize OpenAI service
            self.openai_service = OpenAIService()
            
            # Test API connection
            test_response = self.openai_service.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": "Hello, this is a test message"}],
                max_tokens=10
            )
            
            logger.info("   ✅ OpenAI API: CONNECTED SUCCESSFULLY")
            logger.info(f"   🤖 Model: {settings.OPENAI_MODEL}")
            logger.info(f"   🔑 API Key: Set and Valid")
            logger.info(f"   📊 Max Tokens: {settings.OPENAI_MAX_TOKENS}")
            logger.info(f"   🌡️  Temperature: {settings.OPENAI_TEMPERATURE}")
            logger.info(f"   ✅ Test Response: {test_response.choices[0].message.content}")
            logger.info("   " + "─"*50)
            
            status_report["components"]["openai"] = {
                "status": "✅ CONNECTED",
                "model": settings.OPENAI_MODEL,
                "api_key_set": True,
                "api_key_valid": True,
                "max_tokens": settings.OPENAI_MAX_TOKENS,
                "temperature": settings.OPENAI_TEMPERATURE,
                "test_response": test_response.choices[0].message.content
            }
            
        except Exception as e:
            error_msg = str(e)
            if "Invalid API key" in error_msg or "authentication" in error_msg.lower():
                logger.info("   ❌ OpenAI API Key: INVALID OR EXPIRED")
                logger.info("   ⚠️  AI features will not work")
                status_report["components"]["openai"] = {
                    "status": "❌ INVALID API KEY",
                    "model": settings.OPENAI_MODEL,
                    "error": "API key is invalid or expired",
                    "warning": "AI features will not work"
                }
            elif "insufficient" in error_msg.lower() or "quota" in error_msg.lower():
                logger.info("   ❌ OpenAI API: INSUFFICIENT CREDITS/QUOTA")
                logger.info("   ⚠️  AI features may not work")
                status_report["components"]["openai"] = {
                    "status": "❌ INSUFFICIENT QUOTA",
                    "model": settings.OPENAI_MODEL,
                    "error": "Account has insufficient credits/quota",
                    "warning": "AI features may not work"
                }
            else:
                logger.error(f"   ❌ OpenAI API Error: {error_msg}")
                status_report["components"]["openai"] = {
                    "status": "❌ CONNECTION ERROR",
                    "model": settings.OPENAI_MODEL,
                    "error": error_msg
                }
    
    async def _display_job_embeddings_status(self, status_report: Dict[str, Any]):
        """Display job embeddings status."""
        try:
            logger.info("💼 JOB EMBEDDINGS STATUS")
            logger.info("   " + "─"*50)
            
            # Get job embedding summary
            summary = await job_embedding_service.get_job_embedding_summary()
            
            if "error" not in summary:
                total_jobs = summary['total_jobs']
                jobs_with_embeddings = summary['jobs_with_embeddings']
                jobs_without_embeddings = summary['jobs_without_embeddings']
                completion_percentage = summary['completion_percentage']
                
                logger.info(f"   📊 Total Jobs in System: {total_jobs}")
                logger.info(f"   ✅ Jobs with Embeddings: {jobs_with_embeddings}")
                logger.info(f"   ❌ Jobs without Embeddings: {jobs_without_embeddings}")
                logger.info(f"   📈 Completion Rate: {completion_percentage:.1f}%")
                
                # Show progress bar
                progress_bar = self._create_progress_bar(completion_percentage)
                logger.info(f"   📊 Progress: {progress_bar}")
                
                # Show embedding details if available
                if jobs_with_embeddings > 0:
                    logger.info(f"   📏 Embedding Model: text-embedding-3-small")
                    logger.info(f"   📊 Embedding Dimensions: 1536 per job")
                
                status_report["components"]["job_embeddings"] = {
                    "status": "✅ ACTIVE",
                    "total_jobs": total_jobs,
                    "jobs_with_embeddings": jobs_with_embeddings,
                    "jobs_without_embeddings": jobs_without_embeddings,
                    "completion_percentage": completion_percentage,
                    "progress_bar": progress_bar
                }
            else:
                logger.warning(f"   ⚠️  Could not get job embedding status: {summary['error']}")
                status_report["components"]["job_embeddings"] = {
                    "status": "⚠️  ERROR",
                    "error": summary['error']
                }
            
            logger.info("   " + "─"*50)
            
        except Exception as e:
            logger.error(f"   ❌ Job Embeddings Status Error: {str(e)}")
            status_report["components"]["job_embeddings"] = {
                "status": "❌ ERROR",
                "error": str(e)
            }
    
    async def _display_resume_embeddings_status(self, status_report: Dict[str, Any]):
        """Display resume embeddings status."""
        try:
            logger.info("📄 RESUME EMBEDDINGS STATUS")
            logger.info("   " + "─"*50)
            
            # Get resume statistics
            all_resumes = await self.db_service.get_all_resumes(limit=1000)
            resumes_with_embeddings = await self.db_service.get_all_resumes_with_embeddings(limit=1000)
            
            total_resumes = len(all_resumes)
            resumes_with_emb = len(resumes_with_embeddings)
            resumes_without_emb = total_resumes - resumes_with_emb
            
            completion_percentage = (resumes_with_emb / total_resumes * 100) if total_resumes > 0 else 0
            
            logger.info(f"   📊 Total Resumes in System: {total_resumes}")
            logger.info(f"   ✅ Resumes with Embeddings: {resumes_with_emb}")
            logger.info(f"   ❌ Resumes without Embeddings: {resumes_without_emb}")
            logger.info(f"   📈 Completion Rate: {completion_percentage:.1f}%")
            
            # Show progress bar
            progress_bar = self._create_progress_bar(completion_percentage)
            logger.info(f"   📊 Progress: {progress_bar}")
            
            # Show embedding details if available
            if resumes_with_emb > 0:
                total_dimensions = 0
                for resume in resumes_with_embeddings:
                    embedding = resume.get('embedding', [])
                    if isinstance(embedding, list) and len(embedding) > 0:
                        total_dimensions += len(embedding)
                
                avg_dimensions = total_dimensions / resumes_with_emb if resumes_with_emb > 0 else 0
                logger.info(f"   📏 Embedding Dimensions: {total_dimensions}")
                logger.info(f"   📊 Avg Dimensions per Resume: {avg_dimensions:.1f}")
            
            status_report["components"]["resume_embeddings"] = {
                "status": "✅ ACTIVE",
                "total_resumes": total_resumes,
                "resumes_with_embeddings": resumes_with_emb,
                "resumes_without_embeddings": resumes_without_emb,
                "completion_percentage": completion_percentage,
                "progress_bar": progress_bar
            }
            
            logger.info("   " + "─"*50)
            
        except Exception as e:
            logger.error(f"   ❌ Resume Embeddings Status Error: {str(e)}")
            status_report["components"]["resume_embeddings"] = {
                "status": "❌ ERROR",
                "error": str(e)
            }
    
    async def _display_matching_system_status(self, status_report: Dict[str, Any]):
        """Display matching system status."""
        try:
            logger.info("🎯 CANDIDATE MATCHING SYSTEM STATUS")
            logger.info("   " + "─"*50)
            
            # Check if we have both jobs and resumes with embeddings
            job_summary = await job_embedding_service.get_job_embedding_summary()
            all_resumes = await self.db_service.get_all_resumes_with_embeddings(limit=1000)
            
            jobs_with_emb = job_summary.get('jobs_with_embeddings', 0) if "error" not in job_summary else 0
            resumes_with_emb = len(all_resumes)
            
            if jobs_with_emb > 0 and resumes_with_emb > 0:
                logger.info("   ✅ Matching System: READY")
                logger.info(f"   💼 Jobs with Embeddings: {jobs_with_emb}")
                logger.info(f"   📄 Resumes with Embeddings: {resumes_with_emb}")
                logger.info("   🔍 Hybrid Matching: Available")
                logger.info("   📊 Semantic Search: Available")
                logger.info("   🎯 Candidate Ranking: Available")
                
                status_report["components"]["matching_system"] = {
                    "status": "✅ READY",
                    "jobs_with_embeddings": jobs_with_emb,
                    "resumes_with_embeddings": resumes_with_emb,
                    "hybrid_matching": True,
                    "semantic_search": True,
                    "candidate_ranking": True
                }
            elif jobs_with_emb > 0:
                logger.info("   ⚠️  Matching System: PARTIALLY READY")
                logger.info(f"   💼 Jobs with Embeddings: {jobs_with_emb}")
                logger.info(f"   📄 Resumes with Embeddings: {resumes_with_emb}")
                logger.info("   ⚠️  Need more resumes with embeddings for matching")
                
                status_report["components"]["matching_system"] = {
                    "status": "⚠️  PARTIALLY READY",
                    "jobs_with_embeddings": jobs_with_emb,
                    "resumes_with_embeddings": resumes_with_emb,
                    "warning": "Need more resumes with embeddings"
                }
            elif resumes_with_emb > 0:
                logger.info("   ⚠️  Matching System: PARTIALLY READY")
                logger.info(f"   💼 Jobs with Embeddings: {jobs_with_emb}")
                logger.info(f"   📄 Resumes with Embeddings: {resumes_with_emb}")
                logger.info("   ⚠️  Need more jobs with embeddings for matching")
                
                status_report["components"]["matching_system"] = {
                    "status": "⚠️  PARTIALLY READY",
                    "jobs_with_embeddings": jobs_with_emb,
                    "resumes_with_embeddings": resumes_with_emb,
                    "warning": "Need more jobs with embeddings"
                }
            else:
                logger.info("   ❌ Matching System: NOT READY")
                logger.info(f"   💼 Jobs with Embeddings: {jobs_with_emb}")
                logger.info(f"   📄 Resumes with Embeddings: {resumes_with_emb}")
                logger.info("   ⚠️  Need both jobs and resumes with embeddings")
                
                status_report["components"]["matching_system"] = {
                    "status": "❌ NOT READY",
                    "jobs_with_embeddings": jobs_with_emb,
                    "resumes_with_embeddings": resumes_with_emb,
                    "warning": "Need both jobs and resumes with embeddings"
                }
            
            logger.info("   " + "─"*50)
            
        except Exception as e:
            logger.error(f"   ❌ Matching System Status Error: {str(e)}")
            status_report["components"]["matching_system"] = {
                "status": "❌ ERROR",
                "error": str(e)
            }
    
    async def _display_final_summary(self, status_report: Dict[str, Any]):
        """Display final startup summary."""
        try:
            logger.info("🎉 STARTUP SUMMARY")
            logger.info("="*80)
            
            # Count successful components
            components = status_report.get("components", {})
            successful_components = 0
            total_components = len(components)
            
            for component_name, component_data in components.items():
                status = component_data.get("status", "❌ UNKNOWN")
                if "✅" in status:
                    successful_components += 1
            
            logger.info(f"📊 System Health: {successful_components}/{total_components} components ready")
            logger.info(f"⏰ Startup Duration: {(datetime.now() - self.startup_time).total_seconds():.2f} seconds")
            logger.info("")
            
            # Show component status
            logger.info("🔧 COMPONENT STATUS:")
            for component_name, component_data in components.items():
                status = component_data.get("status", "❌ UNKNOWN")
                component_display_name = component_name.replace("_", " ").title()
                logger.info(f"   {status} {component_display_name}")
            
            logger.info("")
            logger.info("🌐 SERVER IS READY TO ACCEPT REQUESTS!")
            logger.info(f"📚 API Documentation: http://localhost:{settings.PORT}/docs")
            logger.info(f"📖 ReDoc Documentation: http://localhost:{settings.PORT}/redoc")
            logger.info("="*80)
            
            # Add final summary to status report
            status_report["final_summary"] = {
                "system_health": f"{successful_components}/{total_components}",
                "successful_components": successful_components,
                "total_components": total_components,
                "startup_duration_seconds": (datetime.now() - self.startup_time).total_seconds(),
                "server_ready": True,
                "api_docs_url": f"http://localhost:{settings.PORT}/docs",
                "redoc_url": f"http://localhost:{settings.PORT}/redoc"
            }
            
        except Exception as e:
            logger.error(f"Error displaying final summary: {str(e)}")
    
    def _create_progress_bar(self, percentage: float, width: int = 30) -> str:
        """
        Create a visual progress bar for the completion percentage.
        
        Args:
            percentage: Completion percentage (0-100)
            width: Width of the progress bar
            
        Returns:
            String representation of the progress bar
        """
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {percentage:.1f}%"

# Create a global instance for easy access
startup_status_service = StartupStatusService()
