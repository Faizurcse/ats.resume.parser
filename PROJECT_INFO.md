# Resume Parser API - Project Information

## Overview

The Resume Parser API is a professional FastAPI-based application that uses OpenAI API for extracting structured information from resumes. The application supports multiple file formats and returns JSON output with column-wise structured data.

## 🚀 Features

### Core Functionality
- **Multi-format Support**: PDF, DOCX, DOC, TXT, RTF, PNG, JPG, JPEG, WEBP
- **OCR Processing**: Automatic text extraction from image formats using Tesseract
- **AI-Powered Parsing**: Uses OpenAI API for intelligent resume parsing
- **Dynamic Response**: Generates JSON fields based on actual resume content
- **Professional MVC Structure**: Clean, maintainable codebase

### Technical Features
- **File Validation**: Comprehensive file type and size validation
- **Error Handling**: Robust error handling with detailed error messages
- **Security**: File upload security and content validation
- **Logging**: Comprehensive logging for monitoring and debugging
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation

## 📁 Project Structure

```
ResumeParser/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── main.py                   # FastAPI application entry point
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py           # Application settings and environment variables
│   ├── models/                   # Data models and schemas
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic models for request/response
│   ├── services/                 # Business logic services
│   │   ├── __init__.py
│   │   ├── file_processor.py     # File processing and text extraction
│   │   ├── ocr_service.py        # OCR functionality for images
│   │   └── openai_service.py     # OpenAI API integration
│   ├── controllers/              # API endpoints and request handling
│   │   ├── __init__.py
│   │   └── resume_controller.py  # Resume parsing endpoints
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       └── helpers.py            # Common utility functions
├── tests/                        # Test suite
│   ├── __init__.py
│   └── test_resume_parser.py     # Unit and integration tests
├── uploads/                      # Temporary file storage
├── requirements.txt              # Python dependencies
├── env.example                   # Environment variables template
├── run.py                       # Application startup script
├── README.md                    # Project documentation
└── PROJECT_INFO.md              # This file
```

## 🏗️ Architecture

### MVC Pattern Implementation

#### Models (app/models/)
- **schemas.py**: Pydantic models for data validation
  - `ResumeParseRequest`: Request model for file uploads
  - `ResumeParseResponse`: Response model with parsed data
  - `ErrorResponse`: Error response model
  - `HealthResponse`: Health check response model
  - `FileType`: Enum for supported file types

#### Views (FastAPI Response Models)
- Dynamic JSON responses based on resume content
- Consistent error response format
- Health check and API information endpoints

#### Controllers (app/controllers/)
- **resume_controller.py**: API endpoints and request handling
  - `POST /api/v1/parse-resume`: Main resume parsing endpoint
  - `POST /api/v1/upload-file`: File upload endpoint
  - `GET /api/v1/supported-formats`: Supported formats endpoint
  - `GET /api/v1/health`: Health check endpoint

#### Services (app/services/)
- **file_processor.py**: File processing and text extraction
  - PDF processing with PyMuPDF
  - DOCX/DOC processing with docx2txt
  - Text file processing
  - Image OCR processing
- **ocr_service.py**: OCR functionality
  - Tesseract integration
  - Image preprocessing
  - Text cleaning and validation
- **openai_service.py**: OpenAI API integration
  - Resume parsing with AI
  - Prompt engineering
  - Response parsing and validation

## 🔧 Configuration

### Environment Variables (env.example)
```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=2000
OPENAI_TEMPERATURE=0.1

# File Upload Configuration
UPLOAD_DIR=uploads
MAX_FILE_SIZE=10485760

# OCR Configuration
TESSERACT_CMD=tesseract
OCR_LANGUAGE=eng

# Application Configuration
DEBUG=False
```

### Settings Management (app/config/settings.py)
- Centralized configuration management
- Environment variable validation
- Default value handling
- Settings validation on startup

## 📋 API Endpoints

### Core Endpoints

#### POST /api/v1/parse-resume
**Purpose**: Parse resume from uploaded file
**Request**: Multipart form data with file upload
**Response**: JSON with parsed resume data
**Example Response**:
```json
{
  "parsed_data": {
    "Name": "John Doe",
    "Email": "john.doe@email.com",
    "Phone": "+1-555-123-4567",
    "Experience": [
      {
        "Company": "Tech Company Inc.",
        "Position": "Senior Software Engineer",
        "Duration": "2020-2023"
      }
    ],
    "Skills": ["Python", "JavaScript", "React"]
  },
  "file_type": "pdf",
  "processing_time": 2.5
}
```

#### GET /api/v1/health
**Purpose**: Health check endpoint
**Response**: Application status and version

#### GET /api/v1/supported-formats
**Purpose**: Get supported file formats
**Response**: List of supported formats and descriptions

### Root Endpoints

#### GET /
**Purpose**: Application information
**Response**: API overview and available endpoints

#### GET /health
**Purpose**: Root level health check
**Response**: Basic health status

#### GET /api/info
**Purpose**: Detailed API information
**Response**: Comprehensive API details and configuration

## 🔍 File Processing

### Supported Formats

#### Documents
- **PDF**: Using PyMuPDF for text extraction
- **DOCX**: Using python-docx and docx2txt
- **DOC**: Using docx2txt for legacy Word documents
- **TXT**: Direct text processing
- **RTF**: Rich text format processing

#### Images
- **PNG**: OCR processing with Tesseract
- **JPG/JPEG**: OCR processing with Tesseract
- **WEBP**: OCR processing with Tesseract

### Processing Pipeline

1. **File Upload**: Multipart form data handling
2. **Validation**: File type, size, and content validation
3. **Text Extraction**: Format-specific text extraction
4. **OCR Processing**: For image formats
5. **AI Parsing**: OpenAI API integration
6. **Response Generation**: Structured JSON output

## 🤖 AI Integration

### OpenAI Service Features
- **Intelligent Parsing**: Context-aware resume parsing
- **Dynamic Fields**: Generates fields based on actual content
- **Structured Output**: Consistent JSON format
- **Error Handling**: Robust API error handling

### Prompt Engineering
The application uses carefully crafted prompts to ensure:
- Consistent JSON output format
- Accurate field extraction
- Handling of missing information
- Proper array handling for lists

## 🛡️ Security Features

### File Upload Security
- File type validation
- File size limits
- Content validation
- Temporary file handling
- Automatic cleanup

### Data Protection
- No persistent storage of uploaded files
- Temporary file processing
- Secure file handling
- Input sanitization

## 📊 Error Handling

### Comprehensive Error Management
- **File Validation Errors**: Invalid formats, sizes
- **Processing Errors**: OCR failures, parsing errors
- **API Errors**: OpenAI API failures
- **System Errors**: Unexpected exceptions

### Error Response Format
```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "error_code": "ERROR_CODE"
}
```

## 🧪 Testing

### Test Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **API Tests**: Endpoint functionality testing
- **Error Tests**: Error handling validation

### Test Categories
- File processing validation
- OCR functionality
- OpenAI integration
- API endpoint testing
- Error handling scenarios

## 🚀 Deployment

### Prerequisites
1. **Python 3.8+**
2. **Tesseract OCR**: Required for image processing
3. **OpenAI API Key**: Required for AI parsing

### Installation Steps
1. Clone the repository
2. Create virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Configure environment variables
5. Install Tesseract OCR
6. Run the application

### Running the Application
```bash
# Using the run script
python run.py

# Using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📈 Performance

### Optimization Features
- **Async Processing**: Non-blocking file processing
- **Efficient OCR**: Optimized image preprocessing
- **Caching**: Temporary file management
- **Memory Management**: Proper cleanup and resource handling

### Monitoring
- Request/response logging
- Processing time tracking
- Error rate monitoring
- Performance metrics

## 🔧 Development

### Code Quality
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Detailed docstrings
- **Comments**: Inline code comments
- **Error Handling**: Robust exception handling

### Best Practices
- **MVC Architecture**: Clean separation of concerns
- **Dependency Injection**: Service-based architecture
- **Configuration Management**: Environment-based settings
- **Logging**: Comprehensive logging system

## 📚 Dependencies

### Core Dependencies
- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **Python-multipart**: File upload handling

### File Processing
- **PyMuPDF**: PDF processing
- **python-docx**: DOCX processing
- **docx2txt**: DOC/DOCX text extraction
- **Pillow**: Image processing
- **pytesseract**: OCR functionality

### AI Integration
- **OpenAI**: AI API integration

### Utilities
- **python-dotenv**: Environment variable management
- **aiofiles**: Async file operations

## 🎯 Use Cases

### Primary Use Cases
1. **HR Departments**: Automated resume screening
2. **Recruitment Agencies**: Bulk resume processing
3. **Job Boards**: Resume parsing for job matching
4. **ATS Systems**: Integration with applicant tracking systems

### Integration Scenarios
- **Web Applications**: Frontend integration
- **Mobile Apps**: API consumption
- **Enterprise Systems**: Large-scale processing
- **Automation Workflows**: Automated resume processing

## 🔮 Future Enhancements

### Planned Features
- **Batch Processing**: Multiple file processing
- **Custom Fields**: User-defined field extraction
- **Template Matching**: Resume template recognition
- **Language Support**: Multi-language OCR and parsing
- **Advanced Analytics**: Resume quality scoring

### Technical Improvements
- **Caching**: Redis integration for performance
- **Queue System**: Background job processing
- **Database Integration**: Persistent storage options
- **Microservices**: Service decomposition
- **Containerization**: Docker support

## 📞 Support

### Documentation
- **API Documentation**: Auto-generated at `/docs`
- **ReDoc Documentation**: Alternative docs at `/redoc`
- **Code Comments**: Comprehensive inline documentation
- **README**: Setup and usage instructions

### Troubleshooting
- **Logs**: Detailed application logging
- **Error Messages**: Descriptive error responses
- **Health Checks**: System status monitoring
- **Validation**: Input validation and feedback

This project represents a professional, production-ready resume parsing solution with comprehensive features, robust error handling, and a clean, maintainable architecture.
