"""
Pydantic schemas for the Resume Parser application.
Simplified version with only essential models.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class BatchResumeParseResponse(BaseModel):
    """Response model for resume parsing endpoint (supports both single and multiple files)."""
    
    total_files: int = Field(description="Total number of files processed")
    successful_files: int = Field(description="Number of files successfully processed")
    failed_files: int = Field(description="Number of files that failed to process")
    total_processing_time: float = Field(description="Total time taken to process all files")
    results: List[Dict[str, Any]] = Field(description="Results for each processed file")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "total_files": 3,
                "successful_files": 2,
                "failed_files": 1,
                "total_processing_time": 5.2,
                "results": [
                    {
                        "filename": "resume1.pdf",
                        "status": "success",
                        "parsed_data": {
                            "Name": "John Doe",
                            "Email": "john.doe@email.com"
                        },
                        "file_type": "pdf",
                        "processing_time": 2.1
                    },
                    {
                        "filename": "resume2.png",
                        "status": "success",
                        "parsed_data": {
                            "Name": "Jane Smith",
                            "Email": "jane.smith@email.com"
                        },
                        "file_type": "png",
                        "processing_time": 1.8
                    },
                    {
                        "filename": "invalid.txt",
                        "status": "failed",
                        "error": "No text could be extracted from the file",
                        "file_type": "txt",
                        "processing_time": 0.1
                    }
                ]
            }
        }

class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    error_code: Optional[str] = Field(None, description="Error code for client handling")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "error": "Invalid file format",
                "detail": "Only PDF, DOCX, DOC, TXT, RTF, PNG, JPG, JPEG, WEBP files are supported",
                "error_code": "INVALID_FILE_FORMAT"
            }
        }

class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(description="Application status")
    version: str = Field(description="Application version")
    timestamp: str = Field(description="Current timestamp")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }
