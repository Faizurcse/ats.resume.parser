"""
File processing service for handling different file formats.
Extracts text content from various file types including PDF, DOCX, images, etc.
"""

import io
import logging
from typing import Optional

# Import file processing libraries
import fitz  # PyMuPDF
import docx2txt
from PIL import Image
import pytesseract
from app.config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileProcessor:
    """Service for processing different file formats and extracting text content."""
    
    def __init__(self):
        """Initialize the file processor with OCR configuration."""
        # Configure Tesseract path if specified
        if hasattr(settings, 'TESSERACT_CMD') and settings.TESSERACT_CMD != "tesseract":
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
    
    async def process_file(self, file_content: bytes, filename: str) -> str:
        """
        Process a file and extract text content.
        
        Args:
            file_content (bytes): File content as bytes
            filename (str): Name of the file
            
        Returns:
            str: Extracted text content
            
        Raises:
            ValueError: If file format is not supported
            Exception: If file processing fails
        """
        try:
            # Get file extension
            import os
            file_extension = os.path.splitext(filename)[1].lower()
            
            # Process based on file type
            if file_extension == '.pdf':
                return await self._process_pdf(file_content)
            elif file_extension in ['.docx', '.doc']:
                return await self._process_docx(file_content)
            elif file_extension in ['.txt', '.rtf']:
                return await self._process_text(file_content)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.webp']:
                return await self._process_image(file_content)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            raise
    
    async def _process_pdf(self, file_content: bytes) -> str:
        """
        Extract text from PDF file using PyMuPDF.
        
        Args:
            file_content (bytes): PDF file content
            
        Returns:
            str: Extracted text content
        """
        try:
            text_content = []
            
            # Open PDF with PyMuPDF from bytes
            with fitz.open(stream=file_content, filetype="pdf") as pdf_document:
                # Iterate through all pages
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    
                    # Extract text from the page
                    page_text = page.get_text()
                    if page_text.strip():
                        text_content.append(page_text)
            
            # Join all page content
            full_text = "\n".join(text_content)
            logger.info(f"Successfully extracted text from PDF: {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise Exception(f"Failed to process PDF file: {str(e)}")
    
    async def _process_docx(self, file_content: bytes) -> str:
        """
        Extract text from DOCX/DOC file using docx2txt.
        
        Args:
            file_content (bytes): DOCX/DOC file content
            
        Returns:
            str: Extracted text content
        """
        try:
            # Create a temporary file-like object
            file_stream = io.BytesIO(file_content)
            
            # Extract text using docx2txt
            text_content = docx2txt.process(file_stream)
            
            if not text_content or not text_content.strip():
                raise Exception("No text content found in the document")
            
            logger.info(f"Successfully extracted text from DOCX: {len(text_content)} characters")
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"Error processing DOCX: {str(e)}")
            raise Exception(f"Failed to process DOCX file: {str(e)}")
    
    async def _process_text(self, file_content: bytes) -> str:
        """
        Extract text from plain text files.
        
        Args:
            file_content (bytes): Text file content
            
        Returns:
            str: Extracted text content
        """
        try:
            # Decode text content
            text_content = file_content.decode('utf-8')
            
            if not text_content or not text_content.strip():
                raise Exception("No text content found in the file")
            
            logger.info(f"Successfully extracted text from text file: {len(text_content)} characters")
            return text_content.strip()
            
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                text_content = file_content.decode('latin-1')
                logger.info(f"Successfully extracted text from text file: {len(text_content)} characters")
                return text_content.strip()
            except Exception as e:
                logger.error(f"Error processing text file: {str(e)}")
                raise Exception(f"Failed to process text file: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing text file: {str(e)}")
            raise Exception(f"Failed to process text file: {str(e)}")
    
    async def _process_image(self, file_content: bytes) -> str:
        """
        Extract text from image files using OCR.
        
        Args:
            file_content (bytes): Image file content
            
        Returns:
            str: Extracted text content
        """
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(file_content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text using OCR
            text_content = pytesseract.image_to_string(image)
            
            if not text_content or not text_content.strip():
                raise Exception("No text content found in the image")
            
            logger.info(f"Successfully extracted text from image: {len(text_content)} characters")
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise Exception(f"Failed to process image file: {str(e)}")
