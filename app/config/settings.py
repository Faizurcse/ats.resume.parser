"""
Configuration settings for the Resume Parser application.
Simplified version with only essential settings.
"""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings and configuration management."""
    
    # Database Configuration
    PORT: str = os.getenv("PORT", "8000")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    # OpenAI API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    
    # File Processing Configuration
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB default
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "1"))  # its only for candaite job applications Maximum files per batch (single resume only)
    MAX_TOTAL_BATCH_SIZE: int = int(os.getenv("MAX_TOTAL_BATCH_SIZE", "104857600"))  # 100MB total batch size (increased from 50MB)
    ALLOWED_EXTENSIONS: List[str] = [
        ".pdf", ".docx", ".doc", ".txt", ".rtf",
        ".png", ".jpg", ".jpeg", ".webp"
    ]
    
    # Application Configuration
    APP_NAME: str = "Resume Parser API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "AI-powered resume parsing API with multi-format support"
    
    # OCR Configuration (EasyOCR only - pip-installable)
    OCR_LANGUAGE: str = os.getenv("OCR_LANGUAGE", "en")
    OCR_CONFIDENCE_THRESHOLD: float = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.5"))
    USE_GPU: bool = os.getenv("USE_GPU", "False").lower() == "true"
    
    # API Configuration
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    PORT: int = int(PORT)
    
    # File Upload Configuration
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "uploads")
    
    @classmethod
    def validate_settings(cls) -> bool:
        """
        Validate that all required settings are properly configured.
        
        Returns:
            bool: True if all required settings are valid
        """
        if not cls.OPENAI_API_KEY:
            print("⚠️  Warning: OPENAI_API_KEY not set. Some features may not work.")
            # Don't raise error, just warn
        
        # DATABASE_URL is optional since database service has hardcoded values
        if not cls.DATABASE_URL:
            print("ℹ️  Info: DATABASE_URL not set. Using hardcoded database connection.")
        
        return True

# Create settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate_settings()
except ValueError as e:
    print(f"Configuration Error: {e}")
    print("Please check your .env file and ensure all required variables are set.")
