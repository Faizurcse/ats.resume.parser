"""
Pydantic schemas for the Resume Parser application.
Simplified version with only essential models.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ResumeParseResponse(BaseModel):
    """Response model for resume parsing endpoint."""
    
    # Dynamic fields based on resume content
    # Using Dict[str, Any] to allow flexible field names
    parsed_data: Dict[str, Any] = Field(
        description="Parsed resume data with dynamic fields"
    )
    file_type: str = Field(description="Type of uploaded file")
    processing_time: float = Field(description="Time taken to process the file in seconds")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "parsed_data": {
                    "Name": "John Doe",
                    "Email": "john.doe@email.com",
                    "Phone": "+1-555-123-4567",
                    "TotalExperience": "5 years",
                    "Experience": [
                        {
                            "Company": "Tech Corp",
                            "Position": "Software Engineer",
                            "Duration": "2020-2023",
                            "Description": "Developed web applications"
                        }
                    ],
                    "Education": [
                        {
                            "Institution": "University of Technology",
                            "Degree": "Bachelor's",
                            "Field": "Computer Science",
                            "Year": "2020"
                        }
                    ],
                    "Skills": ["Python", "FastAPI", "React", "Docker"]
                },
                "file_type": "pdf",
                "processing_time": 2.5
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
