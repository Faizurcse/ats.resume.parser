# Resume Parser API

A powerful AI-powered resume parsing API that supports multiple file formats and batch processing. Built with FastAPI, PostgreSQL, and OpenAI.

## 🚀 Features

- **Unified Endpoint**: One endpoint handles both single and multiple file uploads
- **Flexible Upload**: Upload 1 file or N number of files (up to 10)
- **Multiple File Formats**: Support for PDF, DOCX, DOC, TXT, RTF, PNG, JPG, JPEG, WEBP
- **AI-Powered Parsing**: Uses OpenAI GPT to extract structured data from resumes
- **Database Storage**: All parsed data automatically saved to PostgreSQL
- **Error Handling**: Individual file error tracking with detailed error messages
- **Performance Metrics**: Processing time tracking for each file and total batch
- **OCR Support**: Text extraction from images using EasyOCR (pip-installable)
- **RESTful API**: Complete CRUD operations for resume management

## 📋 API Endpoints

### Main Upload Endpoint
```
POST /api/v1/parse-resume
```

**Request:**
- Content-Type: `multipart/form-data`
- Body: Files with field name `files` (can be 1 or multiple files)

**Response:**
```json
{
  "total_files": 3,
  "successful_files": 2,
  "failed_files": 1,
  "total_processing_time": 5.2,
  "results": [
    {
      "filename": "faiz.pdf",
      "status": "success",
      "parsed_data": {
        "Name": "Faiz Ahmed",
        "Email": "faiz@email.com",
        "Phone": "+1-555-123-4567",
        "TotalExperience": "5 years",
        "Experience": [...],
        "Education": [...],
        "Skills": [...]
      },
      "file_type": "pdf",
      "processing_time": 2.1
    }
  ]
}
```

### Other Endpoints

- `GET /api/v1/health` - Health check
- `GET /api/v1/resumes` - Get all resumes with pagination
- `GET /api/v1/resumes/{resume_id}` - Get specific resume
- `GET /api/v1/resumes/search/{search_term}` - Search resumes
- `DELETE /api/v1/resumes/{resume_id}` - Delete resume

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- PostgreSQL database
- OpenAI API key
- No external OCR installation required (uses pip-installable alternatives)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ResumeParser
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the example environment file and configure it:

```bash
cp env.example .env
```

Edit `.env` with your configuration:

```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/resume_parser

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=2000
OPENAI_TEMPERATURE=0.1

# File Processing Configuration
MAX_FILE_SIZE=10485760
MAX_BATCH_SIZE=10
MAX_TOTAL_BATCH_SIZE=52428800

# OCR Configuration (New pip-installable alternatives)
OCR_METHOD=easyocr
OCR_LANGUAGE=en
OCR_CONFIDENCE_THRESHOLD=0.5
USE_GPU=False

# Application Configuration
DEBUG=False
```

### 4. Database Setup

The application will automatically create the required database tables on first run.

### 5. Run the Application

```bash
python run.py
```

The API will be available at `http://localhost:8000`

## 📝 Usage Examples

### Single File Upload (Python)

```python
import requests

# Upload single file
with open('faiz.pdf', 'rb') as f:
    files = [('files', ('faiz.pdf', f.read(), 'application/pdf'))]

response = requests.post(
    'http://localhost:8000/api/v1/parse-resume',
    files=files
)

if response.status_code == 200:
    result = response.json()
    print(f"Processed {result['total_files']} file")
    print(f"Successful: {result['successful_files']}")
    
    if result['results']:
        file_result = result['results'][0]
        if file_result['status'] == 'success':
            print(f"✅ {file_result['filename']}: {file_result['parsed_data']['Name']}")
```

### Multiple Files Upload (Python)

```python
import requests

# Upload multiple files
files = [
    ('files', ('faiz.pdf', open('faiz.pdf', 'rb'), 'application/pdf')),
    ('files', ('faiz.png', open('faiz.png', 'rb'), 'image/png')),
    ('files', ('resume.txt', open('resume.txt', 'rb'), 'text/plain'))
]

response = requests.post(
    'http://localhost:8000/api/v1/parse-resume',
    files=files
)

if response.status_code == 200:
    result = response.json()
    print(f"Processed {result['total_files']} files")
    print(f"Successful: {result['successful_files']}")
    print(f"Failed: {result['failed_files']}")
    
    for file_result in result['results']:
        if file_result['status'] == 'success':
            print(f"✅ {file_result['filename']}: {file_result['parsed_data']['Name']}")
        else:
            print(f"❌ {file_result['filename']}: {file_result['error']}")
```

### cURL Examples

**Single file:**
```bash
curl -X POST "http://localhost:8000/api/v1/parse-resume" \
  -F "files=@faiz.pdf"
```

**Multiple files:**
```bash
curl -X POST "http://localhost:8000/api/v1/parse-resume" \
  -F "files=@faiz.pdf" \
  -F "files=@faiz.png" \
  -F "files=@resume.txt"
```

### JavaScript Example

```javascript
const formData = new FormData();

// Single file
formData.append('files', file1);

// OR Multiple files
formData.append('files', file1);
formData.append('files', file2);
formData.append('files', file3);

fetch('http://localhost:8000/api/v1/parse-resume', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log(`Processed ${data.total_files} files`);
  console.log(`Successful: ${data.successful_files}`);
  console.log(`Failed: ${data.failed_files}`);
});
```

## 🧪 Testing

You can test the API using:

- **cURL commands** (examples provided above)
- **Postman** or similar API testing tools
- **Your own client applications** using the provided code examples
- **FastAPI's automatic documentation** at `http://localhost:8000/docs`

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-3.5-turbo` |
| `OPENAI_MAX_TOKENS` | Maximum tokens for AI response | `2000` |
| `OPENAI_TEMPERATURE` | AI response randomness | `0.1` |
| `MAX_FILE_SIZE` | Maximum file size in bytes | `10485760` (10MB) |
| `MAX_BATCH_SIZE` | Maximum files per request | `10` |
| `MAX_TOTAL_BATCH_SIZE` | Maximum total batch size | `52428800` (50MB) |
| `OCR_LANGUAGE` | OCR language | `en` |
| `OCR_CONFIDENCE_THRESHOLD` | Minimum confidence for OCR text | `0.5` |
| `USE_GPU` | Use GPU acceleration if available | `False` |
| `DEBUG` | Debug mode | `False` |

### Supported File Formats

- **Documents**: PDF, DOCX, DOC, TXT, RTF
- **Images**: PNG, JPG, JPEG, WEBP

### Limits

- **Maximum files per request**: 10
- **Maximum file size**: 10MB per file
- **Maximum total batch size**: 50MB
- **Supported formats**: PDF, DOCX, DOC, TXT, RTF, PNG, JPG, JPEG, WEBP

## 🔍 Error Handling

The API provides detailed error information for each file:

- **File validation errors**: Invalid format, size limits, missing filename
- **Processing errors**: Text extraction failures, AI parsing errors
- **Database errors**: Connection issues, save failures

Each file result includes:
- `status`: "success" or "failed"
- `error`: Detailed error message (if failed)
- `processing_time`: Time taken to process the file

## 💾 Database Schema

The application uses PostgreSQL with the following schema:

```sql
CREATE TABLE resume_data (
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
);
```

## 🚀 Performance

- **Unified processing**: Same endpoint handles 1 or N files
- **Batch database saves**: Successful files are saved in a single transaction
- **Memory efficient**: Files are processed one at a time to manage memory usage
- **Timeout handling**: Individual file timeouts don't affect other files
- **Async processing**: Non-blocking file processing and database operations

## 🔧 Troubleshooting

### Common Issues

1. **"Maximum files allowed" error**
   - Reduce the number of files in your request
   - Check `MAX_BATCH_SIZE` setting

2. **"File size exceeds limit" error**
   - Compress or reduce file sizes
   - Check `MAX_FILE_SIZE` setting

3. **"Total batch size exceeds limit" error**
   - Reduce the number or size of files
   - Check `MAX_TOTAL_BATCH_SIZE` setting

4. **"No text could be extracted" error**
   - Ensure the file contains readable text
   - Try a different file format
   - Check if the file is corrupted

5. **Database connection errors**
   - Verify `DATABASE_URL` is correct
   - Ensure PostgreSQL is running
   - Check network connectivity

6. **OpenAI API errors**
   - Verify `OPENAI_API_KEY` is valid
   - Check API quota and billing
   - Ensure internet connectivity

### Logs

Check the application logs for detailed error information:

```bash
# View logs
tail -f app.log
```

## 📊 Monitoring

The API provides processing metrics:

- Total processing time for the batch
- Individual file processing times
- Success/failure counts
- Detailed error messages for failed files

Use these metrics to monitor performance and identify issues.

## 🎯 Key Benefits

- **One Endpoint**: No need to choose between single or batch endpoints
- **Flexible**: Upload 1 file or N files with the same API call
- **Consistent Response**: Same response format regardless of file count
- **Easy Integration**: Simple to integrate into any application
- **Scalable**: Handles from 1 to 10 files efficiently
- **AI-Powered**: Advanced resume parsing using OpenAI
- **Multi-format**: Supports various file formats including images
- **Database Storage**: Automatic data persistence and retrieval

## 📁 Project Structure

```
ResumeParser/
├── app/
│   ├── config/
│   │   └── settings.py          # Configuration management
│   ├── controllers/
│   │   └── resume_controller.py # API endpoints
│   ├── models/
│   │   └── schemas.py           # Pydantic models
│   ├── services/
│   │   ├── database_service.py  # Database operations
│   │   ├── file_processor.py    # File processing & OCR
│   │   └── openai_service.py    # AI parsing
│   └── main.py                  # FastAPI application
├── requirements.txt              # Python dependencies
├── run.py                       # Application entry point
├── env.example                  # Environment variables template
└── README.md                    # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section above
- Review the logs for detailed error information
