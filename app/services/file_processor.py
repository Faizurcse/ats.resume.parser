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
# import pytesseract  # Removed - requires external Tesseract installation
import easyocr
import numpy as np
from app.config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileProcessor:
    """Service for processing different file formats and extracting text content."""
    
    def __init__(self):
        """Initialize the file processor with OCR configuration."""
        # Initialize EasyOCR reader (will download models on first use)
        self.easyocr_reader = None
        try:
            logger.info("Initializing EasyOCR...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have GPU
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            logger.error("Please ensure EasyOCR is properly installed: pip install easyocr")
            raise Exception(f"EasyOCR initialization failed: {e}. Please check installation.")
    
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
        Extract text from image files using EasyOCR.
        
        Args:
            file_content (bytes): Image file content
            
        Returns:
            str: Extracted text content
        """
        try:
            # Check if EasyOCR is available
            if not self.easyocr_reader:
                raise Exception("EasyOCR is not available. Please check installation.")
            
            # Open image from bytes
            image = Image.open(io.BytesIO(file_content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert PIL image to numpy array
            image_array = np.array(image)
            
            # Extract text using EasyOCR
            results = self.easyocr_reader.readtext(image_array)
            
            # Extract text from results
            text_content = []
            for (bbox, text, prob) in results:
                if prob > 0.5:  # Only include text with confidence > 50%
                    text_content.append(text)
            
            extracted_text = " ".join(text_content)
            
            if not extracted_text.strip():
                raise Exception("No text content could be extracted from the image")
            
            logger.info(f"Successfully extracted text from image using EasyOCR: {len(extracted_text)} characters")
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise Exception(f"Failed to process image file: {str(e)}")
