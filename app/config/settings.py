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
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    # OpenAI API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    
    # File Processing Configuration
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB default
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "10"))  # Maximum files per batch
    MAX_TOTAL_BATCH_SIZE: int = int(os.getenv("MAX_TOTAL_BATCH_SIZE", "52428800"))  # 50MB total batch size
    ALLOWED_EXTENSIONS: List[str] = [
        ".pdf", ".docx", ".doc", ".txt", ".rtf",
        ".png", ".jpg", ".jpeg", ".webp"
    ]
    
    # Application Configuration
    APP_NAME: str = "Resume Parser API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "AI-powered resume parsing API with multi-format support"
    
    # OCR Configuration
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "tesseract")
    OCR_LANGUAGE: str = os.getenv("OCR_LANGUAGE", "eng")
    
    # API Configuration
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    @classmethod
    def validate_settings(cls) -> bool:
        """
        Validate that all required settings are properly configured.
        
        Returns:
            bool: True if all required settings are valid
        """
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required in environment variables")
        
        if not cls.DATABASE_URL:
            raise ValueError("DATABASE_URL is required in environment variables")
        
        return True

# Create settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate_settings()
except ValueError as e:
    print(f"Configuration Error: {e}")
    print("Please check your .env file and ensure all required variables are set.")
