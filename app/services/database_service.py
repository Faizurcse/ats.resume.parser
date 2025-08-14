"""
Database service for handling database operations.
Uses asyncpg for direct PostgreSQL connection.
"""

import logging
import json
import os
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
            # Create table if it doesn't exist
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS resume_data (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_path VARCHAR(500) NOT NULL,
                    file_type VARCHAR(50) NOT NULL,
                    file_size INTEGER NOT NULL,
                    processing_time FLOAT NOT NULL,
                    parsed_data JSONB NOT NULL,
                    candidate_name VARCHAR(255),
                    candidate_email VARCHAR(255),
                    candidate_phone VARCHAR(100),
                    total_experience VARCHAR(100),
                    is_unique BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            ''')
            
            # Add missing columns if they don't exist (for existing tables)
            await self._add_missing_columns(conn)
            
            # No duplicate handling needed - keep all resumes
            # await self._handle_duplicate_emails(conn)  # Commented out
            
            # Create indexes
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_resume_data_filename ON resume_data(filename)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_resume_data_candidate_name ON resume_data(candidate_name)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_resume_data_candidate_email ON resume_data(candidate_email)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_resume_data_is_unique ON resume_data(is_unique)')
            
            # No unique constraint - allow multiple resumes per email
            # Create regular index for performance
            try:
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_candidate_email 
                    ON resume_data(candidate_email) 
                    WHERE candidate_email IS NOT NULL AND candidate_email != ''
                ''')
                logger.info("Regular index on candidate_email created successfully")
            except Exception as e:
                logger.warning(f"Could not create index on candidate_email: {str(e)}")
                # Continue execution even if index creation fails
    
    async def _add_missing_columns(self, conn):
        """Add missing columns to existing tables for backward compatibility."""
        try:
            # Check if file_path column exists
            columns = await conn.fetch('''
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'resume_data' AND column_name = 'file_path'
            ''')
            
            if not columns:
                # Add file_path column without default to avoid conflicts
                await conn.execute('ALTER TABLE resume_data ADD COLUMN file_path VARCHAR(500)')
                # Update existing records to have empty string
                await conn.execute('UPDATE resume_data SET file_path = "" WHERE file_path IS NULL')
                # Make it NOT NULL after updating
                await conn.execute('ALTER TABLE resume_data ALTER COLUMN file_path SET NOT NULL')
                logger.info("Added missing file_path column")
            
            # Check if is_unique column exists
            columns = await conn.fetch('''
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'resume_data' AND column_name = 'is_unique'
            ''')
            
            if not columns:
                # Add is_unique column without default to avoid conflicts
                await conn.execute('ALTER TABLE resume_data ADD COLUMN is_unique BOOLEAN')
                # Update existing records to have TRUE
                await conn.execute('UPDATE resume_data SET is_unique = TRUE WHERE is_unique IS NULL')
                # Make it NOT NULL after updating
                await conn.execute('ALTER TABLE resume_data ALTER COLUMN is_unique SET NOT NULL')
                # Set default for future records
                await conn.execute('ALTER TABLE resume_data ALTER COLUMN is_unique SET DEFAULT TRUE')
                logger.info("Added missing is_unique column")
                
        except Exception as e:
            logger.warning(f"Could not add missing columns: {str(e)}")
            # Continue execution even if column addition fails
        
        # Try to populate file_path for existing records that have empty file_path
        await self._populate_missing_file_paths(conn)
    
    async def _populate_missing_file_paths(self, conn):
        """Populate file_path for existing records that have empty file_path."""
        try:
            # Find records with empty file_path
            records_with_empty_path = await conn.fetch('''
                SELECT id, filename, file_type 
                FROM resume_data 
                WHERE file_path = '' OR file_path IS NULL
            ''')
            
            if records_with_empty_path:
                logger.info(f"Found {len(records_with_empty_path)} records with empty file_path")
                
                for record in records_with_empty_path:
                    # Try to find the actual file in the uploads folder
                    upload_folder = settings.UPLOAD_FOLDER
                    if os.path.exists(upload_folder):
                        # Look for files that might match this record
                        # Since we can't know the exact UUID, we'll try to find by extension
                        file_extension = os.path.splitext(record['filename'])[1]
                        files_in_folder = [f for f in os.listdir(upload_folder) if f.endswith(file_extension)]
                        
                        if files_in_folder:
                            # Use the first matching file (this is a best-effort approach)
                            actual_filename = files_in_folder[0]
                            actual_file_path = os.path.join(upload_folder, actual_filename)
                            
                            # Update the database with the found file path
                            await conn.execute('''
                                UPDATE resume_data 
                                SET file_path = $1 
                                WHERE id = $2
                            ''', actual_file_path, record['id'])
                            
                            logger.info(f"Updated record {record['id']} with file_path: {actual_file_path}")
                        else:
                            logger.warning(f"No matching file found for record {record['id']} with filename: {record['filename']}")
                    else:
                        logger.warning(f"Upload folder does not exist: {upload_folder}")
                        
        except Exception as e:
            logger.warning(f"Could not populate missing file paths: {str(e)}")
            # Continue execution even if file path population fails
    
    async def update_file_path(self, resume_id: int, file_path: str) -> bool:
        """
        Update the file_path for a specific resume record.
        
        Args:
            resume_id (int): Resume record ID
            file_path (str): New file path
            
        Returns:
            bool: True if updated successfully
        """
        try:
            pool = await self._get_pool()
            
            async with pool.acquire() as conn:
                result = await conn.execute('''
                    UPDATE resume_data 
                    SET file_path = $1, updated_at = NOW()
                    WHERE id = $2
                ''', file_path, resume_id)
                
                if result == "UPDATE 1":
                    logger.info(f"Updated file_path for resume {resume_id}: {file_path}")
                    return True
                else:
                    logger.warning(f"No rows updated for resume {resume_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating file_path for resume {resume_id}: {str(e)}")
            return False
    
    async def _handle_duplicate_emails(self, conn):
        """Handle duplicate emails by keeping only the most recent resume for each email."""
        try:
            # Find duplicate emails
            duplicate_emails = await conn.fetch('''
                SELECT candidate_email, COUNT(*) as count
                FROM resume_data 
                WHERE candidate_email IS NOT NULL AND candidate_email != ''
                GROUP BY candidate_email 
                HAVING COUNT(*) > 1
            ''')
            
            if duplicate_emails:
                logger.info(f"Found {len(duplicate_emails)} duplicate email addresses")
                
                for duplicate in duplicate_emails:
                    email = duplicate['candidate_email']
                    count = duplicate['count']
                    
                    if count > 1:
                        # Keep the most recent resume for this email, delete others
                        await conn.execute('''
                            DELETE FROM resume_data 
                            WHERE candidate_email = $1 
                            AND id NOT IN (
                                SELECT id FROM resume_data 
                                WHERE candidate_email = $1 
                                ORDER BY created_at DESC 
                                LIMIT 1
                            )
                        ''', email)
                        
                        logger.info(f"Removed {count - 1} duplicate resumes for email: {email}")
                
                logger.info("Duplicate email cleanup completed")
            else:
                logger.info("No duplicate emails found")
                
        except Exception as e:
            logger.warning(f"Could not handle duplicate emails: {str(e)}")
            # Continue execution even if duplicate handling fails
    
    async def save_resume_data(self, 
                              filename: str, 
                              file_path: str,
                              file_type: str, 
                              file_size: int,
                              processing_time: float,
                              parsed_data: Dict[str, Any]) -> int:
        """
        Save parsed resume data to database.
        ALL resumes are saved without checking for duplicates.
        
        Args:
            filename (str): Original filename
            file_path (str): Path where file is stored
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
                # ALWAYS INSERT new resume data (no duplicate checking)
                record = await conn.fetchrow('''
                    INSERT INTO resume_data 
                    (filename, file_path, file_type, file_size, processing_time, parsed_data, 
                     candidate_name, candidate_email, candidate_phone, total_experience)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id
                ''', filename, file_path, file_type, file_size, processing_time, 
                     json.dumps(parsed_data), candidate_name, candidate_email, 
                     candidate_phone, total_experience)
                
                record_id = record['id']
                logger.info(f"New resume data saved to database with ID: {record_id}")
                return record_id
                
        except Exception as e:
            logger.error(f"Error saving resume data: {str(e)}")
            raise Exception(f"Failed to save resume data: {str(e)}")
    
    async def save_batch_resume_data(self, resume_data_list: List[Dict[str, Any]]) -> List[int]:
        """
        Save multiple parsed resume data to database in batch.
        ALL resumes are saved without checking for duplicates.
        
        Args:
            resume_data_list (List[Dict[str, Any]]): List of resume data dictionaries
                Each dict should contain: filename, file_path, file_type, file_size, processing_time, parsed_data
            
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
                        
                        # ALWAYS INSERT new resume data (no duplicate checking)
                        record = await conn.fetchrow('''
                            INSERT INTO resume_data 
                            (filename, file_path, file_type, file_size, processing_time, parsed_data, 
                             candidate_name, candidate_email, candidate_phone, total_experience)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                            RETURNING id
                        ''', resume_data['filename'], resume_data['file_path'], 
                             resume_data['file_type'], resume_data['file_size'], 
                             resume_data['processing_time'], json.dumps(resume_data['parsed_data']), 
                             candidate_name, candidate_email, candidate_phone, total_experience)
                        
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
                        "file_path": record.get('file_path', ''),  # Add file_path field
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
                # First, check what columns exist in the table
                columns_info = await conn.fetch('''
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'resume_data'
                ''')
                existing_columns = {col['column_name'] for col in columns_info}
                
                # Build query based on available columns
                if 'file_path' in existing_columns and 'is_unique' in existing_columns:
                    # Full schema - use all columns
                    query = '''
                        SELECT id, filename, file_path, file_type, candidate_name, candidate_email, 
                               total_experience, parsed_data, created_at
                        FROM resume_data 
                        ORDER BY created_at DESC
                        LIMIT $1 OFFSET $2
                    '''
                else:
                    # Legacy schema - use only existing columns
                    query = '''
                        SELECT id, filename, file_type, candidate_name, candidate_email, 
                               total_experience, parsed_data, created_at
                        FROM resume_data 
                        ORDER BY created_at DESC
                        LIMIT $1 OFFSET $2
                    '''
                
                records = await conn.fetch(query, limit, offset)
                
                return [
                    {
                        "id": record['id'],
                        "filename": record['filename'],
                        "file_path": record.get('file_path', ''),
                        "file_type": record['file_type'],
                        "candidate_name": record['candidate_name'],
                        "candidate_email": record['candidate_email'],
                        "total_experience": record['total_experience'],
                        "parsed_data": record['parsed_data'],
                        "created_at": record['created_at'].isoformat() if record['created_at'] else None
                    }
                    for record in records
                ]
                
        except Exception as e:
            logger.error(f"Error getting resumes: {str(e)}")
            logger.error(f"Full error details: {e.__class__.__name__}: {str(e)}")
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
                    SELECT id, filename, file_path, candidate_name, candidate_email, 
                           total_experience, parsed_data, created_at
                    FROM resume_data 
                    WHERE candidate_name ILIKE $1 OR candidate_email ILIKE $1
                    ORDER BY created_at DESC
                ''', f'%{search_term}%')
                
                return [
                    {
                        "id": record['id'],
                        "filename": record['filename'],
                        "file_path": record.get('file_path', ''),
                        "candidate_name": record['candidate_name'],
                        "candidate_email": record['candidate_email'],
                        "total_experience": record['total_experience'],
                        "parsed_data": record['parsed_data'],
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
    
    async def get_unique_resumes_for_download(self) -> List[Dict[str, Any]]:
        """
        Get all unique resumes for download with basic information.
        This method returns only unique resumes based on candidate email.
        
        Returns:
            List[Dict[str, Any]]: List of unique resume records
        """
        try:
            pool = await self._get_pool()
            
            async with pool.acquire() as conn:
                records = await conn.fetch('''
                    SELECT DISTINCT ON (candidate_email) 
                           id, filename, file_path, file_type, candidate_name, candidate_email, 
                           total_experience, created_at
                    FROM resume_data 
                    WHERE candidate_email IS NOT NULL AND candidate_email != ''
                    ORDER BY candidate_email, created_at DESC
                ''')
                
                return [
                    {
                        "id": record['id'],
                        "filename": record['filename'],
                        "file_path": record.get('file_path', ''),
                        "file_type": record['file_type'],
                        "candidate_name": record['candidate_name'],
                        "candidate_email": record['candidate_email'],
                        "total_experience": record['total_experience'],
                        "created_at": record['created_at'].isoformat() if record['created_at'] else None
                    }
                    for record in records
                ]
                
        except Exception as e:
            logger.error(f"Error getting unique resumes for download: {str(e)}")
            raise Exception(f"Failed to get unique resume data: {str(e)}")
    
    async def get_unique_resumes_with_files(self) -> List[Dict[str, Any]]:
        """
        Get all unique resumes with file information for download.
        This method returns only unique resumes based on candidate email.
        
        Returns:
            List[Dict[str, Any]]: List of unique resume records with file info
        """
        try:
            pool = await self._get_pool()
            
            async with pool.acquire() as conn:
                records = await conn.fetch('''
                    SELECT DISTINCT ON (candidate_email) 
                           id, filename, file_path, file_type, candidate_name, candidate_email, 
                           total_experience, created_at
                    FROM resume_data 
                    WHERE candidate_email IS NOT NULL AND candidate_email != ''
                    ORDER BY candidate_email, created_at DESC
                ''')
                
                return [
                    {
                        "id": record['id'],
                        "filename": record['filename'],
                        "file_path": record.get('file_path', ''),
                        "file_type": record['file_type'],
                        "candidate_name": record['candidate_name'],
                        "candidate_email": record['candidate_email'],
                        "total_experience": record['total_experience'],
                        "created_at": record['created_at'].isoformat() if record['created_at'] else None
                    }
                    for record in records
                ]
                
        except Exception as e:
            logger.error(f"Error getting unique resumes with files: {str(e)}")
            raise Exception(f"Failed to get unique resume data with files: {str(e)}")
    
    async def get_all_resumes_including_duplicates(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get ALL resumes including duplicates for admin purposes.
        This method returns all resumes without filtering for uniqueness.
        
        Args:
            limit (int): Number of records to return
            offset (int): Number of records to skip
            
        Returns:
            List[Dict[str, Any]]: List of all resume records
        """
        try:
            pool = await self._get_pool()
            
            async with pool.acquire() as conn:
                records = await conn.fetch('''
                    SELECT id, filename, file_path, file_type, candidate_name, candidate_email, 
                           total_experience, parsed_data, created_at
                    FROM resume_data 
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                ''', limit, offset)
                
                return [
                    {
                        "id": record['id'],
                        "filename": record['filename'],
                        "file_path": record.get('file_path', ''),
                        "file_type": record['file_type'],
                        "candidate_name": record['candidate_name'],
                        "candidate_email": record['candidate_email'],
                        "total_experience": record['total_experience'],
                        "parsed_data": record['parsed_data'],
                        "created_at": record['created_at'].isoformat() if record['created_at'] else None
                    }
                    for record in records
                ]
                
        except Exception as e:
            logger.error(f"Error getting all resumes including duplicates: {str(e)}")
            raise Exception(f"Failed to get all resume data: {str(e)}")
