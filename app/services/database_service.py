"""
Database service for handling database operations.
Uses asyncpg for direct PostgreSQL connection.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncpg
from urllib.parse import urlparse

from app.config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseService:
    """Service for database operations using asyncpg."""
    
    def __init__(self):
        """Initialize database connection."""
        self.pool = None
        self._init_done = False
    
    async def _get_pool(self):
        """Get database connection pool."""
        if not self._init_done:
            await self._initialize()
        return self.pool
    
    async def _initialize(self):
        """Initialize database connection pool and create tables."""
        try:
            # Parse database URL
            parsed_url = urlparse(settings.DATABASE_URL)
            
            # Extract connection parameters
            host = parsed_url.hostname
            port = parsed_url.port or 5432
            user = parsed_url.username
            password = parsed_url.password
            database = parsed_url.path.lstrip('/')
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                ssl='require'
            )
            
            # Create tables
            await self._create_tables()
            
            self._init_done = True
            logger.info("Database service initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    async def _create_tables(self):
        """Create database tables."""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS resume_data (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(50) NOT NULL,
                    file_size INTEGER NOT NULL,
                    processing_time FLOAT NOT NULL,
                    parsed_data JSONB NOT NULL,
                    candidate_name VARCHAR(255),
                    candidate_email VARCHAR(255),
                    candidate_phone VARCHAR(100),
                    total_experience VARCHAR(100),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            ''')
            
            # Create indexes
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_resume_data_filename ON resume_data(filename)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_resume_data_candidate_name ON resume_data(candidate_name)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_resume_data_candidate_email ON resume_data(candidate_email)')
    
    async def save_resume_data(self, 
                              filename: str, 
                              file_type: str, 
                              file_size: int,
                              processing_time: float,
                              parsed_data: Dict[str, Any]) -> int:
        """
        Save parsed resume data to database.
        
        Args:
            filename (str): Original filename
            file_type (str): File type (pdf, docx, png, etc.)
            file_size (int): File size in bytes
            processing_time (float): Processing time in seconds
            parsed_data (Dict[str, Any]): Parsed resume data
            
        Returns:
            int: ID of the saved record
        """
        try:
            pool = await self._get_pool()
            
            # Extract key information from parsed data
            candidate_name = parsed_data.get("Name", "")
            candidate_email = parsed_data.get("Email", "")
            candidate_phone = parsed_data.get("Phone", "")
            total_experience = parsed_data.get("TotalExperience", "")
            
            async with pool.acquire() as conn:
                # Insert resume data
                record = await conn.fetchrow('''
                    INSERT INTO resume_data 
                    (filename, file_type, file_size, processing_time, parsed_data, 
                     candidate_name, candidate_email, candidate_phone, total_experience)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                ''', filename, file_type, file_size, processing_time, 
                     json.dumps(parsed_data), candidate_name, candidate_email, 
                     candidate_phone, total_experience)
                
                record_id = record['id']
                logger.info(f"Resume data saved to database with ID: {record_id}")
                return record_id
                
        except Exception as e:
            logger.error(f"Error saving resume data: {str(e)}")
            raise Exception(f"Failed to save resume data: {str(e)}")
    
    async def save_batch_resume_data(self, resume_data_list: List[Dict[str, Any]]) -> List[int]:
        """
        Save multiple parsed resume data to database in batch.
        
        Args:
            resume_data_list (List[Dict[str, Any]]): List of resume data dictionaries
                Each dict should contain: filename, file_type, file_size, processing_time, parsed_data
            
        Returns:
            List[int]: List of IDs of the saved records
        """
        try:
            pool = await self._get_pool()
            
            async with pool.acquire() as conn:
                # Start transaction
                async with conn.transaction():
                    record_ids = []
                    
                    for resume_data in resume_data_list:
                        # Extract key information from parsed data
                        candidate_name = resume_data['parsed_data'].get("Name", "")
                        candidate_email = resume_data['parsed_data'].get("Email", "")
                        candidate_phone = resume_data['parsed_data'].get("Phone", "")
                        total_experience = resume_data['parsed_data'].get("TotalExperience", "")
                        
                        # Insert resume data
                        record = await conn.fetchrow('''
                            INSERT INTO resume_data 
                            (filename, file_type, file_size, processing_time, parsed_data, 
                             candidate_name, candidate_email, candidate_phone, total_experience)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                            RETURNING id
                        ''', resume_data['filename'], resume_data['file_type'], 
                             resume_data['file_size'], resume_data['processing_time'],
                             json.dumps(resume_data['parsed_data']), candidate_name, 
                             candidate_email, candidate_phone, total_experience)
                        
                        record_ids.append(record['id'])
                    
                    logger.info(f"Batch saved {len(record_ids)} resume records to database")
                    return record_ids
                
        except Exception as e:
            logger.error(f"Error saving batch resume data: {str(e)}")
            raise Exception(f"Failed to save batch resume data: {str(e)}")
    
    async def get_resume_by_id(self, resume_id: int) -> Optional[Dict[str, Any]]:
        """
        Get resume data by ID.
        
        Args:
            resume_id (int): Resume record ID
            
        Returns:
            Optional[Dict[str, Any]]: Resume data or None if not found
        """
        try:
            pool = await self._get_pool()
            
            async with pool.acquire() as conn:
                record = await conn.fetchrow('''
                    SELECT * FROM resume_data WHERE id = $1
                ''', resume_id)
                
                if record:
                    return {
                        "id": record['id'],
                        "filename": record['filename'],
                        "file_type": record['file_type'],
                        "file_size": record['file_size'],
                        "processing_time": record['processing_time'],
                        "parsed_data": record['parsed_data'],
                        "candidate_name": record['candidate_name'],
                        "candidate_email": record['candidate_email'],
                        "candidate_phone": record['candidate_phone'],
                        "total_experience": record['total_experience'],
                        "created_at": record['created_at'].isoformat() if record['created_at'] else None,
                        "updated_at": record['updated_at'].isoformat() if record['updated_at'] else None
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error getting resume {resume_id}: {str(e)}")
            raise Exception(f"Failed to get resume data: {str(e)}")
    
    async def get_all_resumes(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get all resume records with pagination.
        
        Args:
            limit (int): Number of records to return
            offset (int): Number of records to skip
            
        Returns:
            List[Dict[str, Any]]: List of resume records
        """
        try:
            pool = await self._get_pool()
            
            async with pool.acquire() as conn:
                records = await conn.fetch('''
                    SELECT id, filename, file_type, candidate_name, candidate_email, 
                           total_experience, created_at
                    FROM resume_data 
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                ''', limit, offset)
                
                return [
                    {
                        "id": record['id'],
                        "filename": record['filename'],
                        "file_type": record['file_type'],
                        "candidate_name": record['candidate_name'],
                        "candidate_email": record['candidate_email'],
                        "total_experience": record['total_experience'],
                        "created_at": record['created_at'].isoformat() if record['created_at'] else None
                    }
                    for record in records
                ]
                
        except Exception as e:
            logger.error(f"Error getting resumes: {str(e)}")
            raise Exception(f"Failed to get resume data: {str(e)}")
    
    async def search_resumes(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search resumes by candidate name or email.
        
        Args:
            search_term (str): Search term
            
        Returns:
            List[Dict[str, Any]]: List of matching resume records
        """
        try:
            pool = await self._get_pool()
            
            async with pool.acquire() as conn:
                records = await conn.fetch('''
                    SELECT id, filename, candidate_name, candidate_email, 
                           total_experience, created_at
                    FROM resume_data 
                    WHERE candidate_name ILIKE $1 OR candidate_email ILIKE $1
                    ORDER BY created_at DESC
                ''', f'%{search_term}%')
                
                return [
                    {
                        "id": record['id'],
                        "filename": record['filename'],
                        "candidate_name": record['candidate_name'],
                        "candidate_email": record['candidate_email'],
                        "total_experience": record['total_experience'],
                        "created_at": record['created_at'].isoformat() if record['created_at'] else None
                    }
                    for record in records
                ]
                
        except Exception as e:
            logger.error(f"Error searching resumes: {str(e)}")
            raise Exception(f"Failed to search resume data: {str(e)}")
    
    async def delete_resume(self, resume_id: int) -> bool:
        """
        Delete resume record by ID.
        
        Args:
            resume_id (int): Resume record ID
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            pool = await self._get_pool()
            
            async with pool.acquire() as conn:
                result = await conn.execute('''
                    DELETE FROM resume_data WHERE id = $1
                ''', resume_id)
                
                if result == "DELETE 1":
                    logger.info(f"Resume record deleted: {resume_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error deleting resume {resume_id}: {str(e)}")
            raise Exception(f"Failed to delete resume data: {str(e)}")
