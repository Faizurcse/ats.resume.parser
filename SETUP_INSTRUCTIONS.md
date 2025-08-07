# Resume Parser Setup Instructions

## Prerequisites

1. **Python 3.13** (or compatible version)
2. **Tesseract OCR** - Required for OCR functionality
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install and add to PATH

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd ResumeParser
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Copy the environment example file:
```bash
copy env.example .env
```

Edit `.env` file with your configuration:
```
OPENAI_API_KEY=your_openai_api_key_here
UPLOAD_DIR=uploads
MAX_FILE_SIZE=10485760
```

## Troubleshooting

### PyMuPDF Installation Issues
If you encounter issues with PyMuPDF installation on Windows:

1. **Try pre-compiled wheels:**
   ```bash
   pip install PyMuPDF==1.24.11 --only-binary=PyMuPDF
   ```

2. **Alternative: Use pdfplumber**
   If PyMuPDF continues to fail, you can use the alternative requirements:
   ```bash
   pip install -r requirements_alternative.txt
   ```

### Pillow Installation Issues
If Pillow fails to install:

1. **Use a compatible version:**
   ```bash
   pip install Pillow==10.4.0 --only-binary=Pillow
   ```

### Pydantic Installation Issues
If Pydantic fails due to Rust compilation:

1. **Use pre-compiled version:**
   ```bash
   pip install pydantic==2.11.7 --only-binary=pydantic
   ```

### Tesseract OCR Issues
If pytesseract fails to find Tesseract:

1. **Install Tesseract OCR:**
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install to default location (usually `C:\Program Files\Tesseract-OCR`)
   - Add to PATH environment variable

2. **Verify installation:**
   ```bash
   python -c "import pytesseract; print('Tesseract is available')"
   ```

## Running the Application

1. **Start the server:**
   ```bash
   python run.py
   ```

2. **Access the API:**
   - Swagger UI: http://localhost:8000/docs
   - API Base: http://localhost:8000

## File Upload Directory

Make sure the `uploads` directory exists:
```bash
mkdir uploads
```

## Supported File Formats

- PDF files (.pdf)
- Word documents (.docx)
- Text files (.txt)
- Image files (.png, .jpg, .jpeg) - requires OCR

## Notes

- The application uses OpenAI API for text processing
- OCR functionality requires Tesseract to be installed
- File uploads are limited to 10MB by default
- All uploaded files are stored in the `uploads` directory
