# Resume Parser API

A FastAPI-based resume parser application that combines **Computer Vision (OCR)** and **Natural Language Processing (NLP)** to extract structured information from resumes. The application supports multiple file formats and returns JSON output with intelligent data parsing.

## 🎯 **Technology Stack**

### **Computer Vision (CV) Components:**
- ✅ **OCR (Optical Character Recognition)** - Tesseract for image-to-text conversion
- ✅ **Image Processing** - PIL (Python Imaging Library) for image manipulation
- ✅ **Multi-format Image Support** - PNG, JPG, JPEG, WEBP processing

### **Natural Language Processing (NLP) Components:**
- ✅ **GPT-3.5-turbo** - OpenAI API for intelligent text understanding
- ✅ **Text Analysis** - Extracts structured data from unstructured text
- ✅ **Information Extraction** - Identifies names, emails, experience, skills, etc.

### **Hybrid Approach (CV + NLP):**
- ✅ **Computer Vision** → **OCR** → **Text Extraction**
- ✅ **NLP** → **GPT Analysis** → **Structured Data**
- ✅ **End-to-End Processing** - From image/text to structured JSON

## 🚀 **Features**

- **Multi-format Support**: PDF, DOCX, DOC, TXT, RTF, PNG, JPG, JPEG, WEBP
- **Computer Vision**: OCR processing for image-based resumes
- **NLP Processing**: AI-powered text understanding and information extraction
- **Database Storage**: PostgreSQL database for storing parsed resume data
- **RESTful API**: Complete CRUD operations for resume data management
- **Dynamic Response**: Generates JSON fields based on actual resume content
- **Experience Calculation**: Automatic calculation of total experience in months/years
- **Professional Architecture**: Clean, maintainable MVC structure

## 🏗️ **Project Architecture**

```
ResumeParser/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config/
│   │   └── settings.py         # Configuration management
│   ├── models/
│   │   ├── schemas.py          # Pydantic models and schemas
│   │   └── database.py         # SQLAlchemy database models
│   ├── services/
│   │   ├── file_processor.py   # CV + File processing logic
│   │   ├── openai_service.py   # NLP + OpenAI API integration
│   │   └── database_service.py # Database operations and CRUD
│   └── controllers/
│       └── resume_controller.py # API endpoints
├── requirements.txt            # Python dependencies
├── setup_database.py          # Database setup script
└── README.md                  # Project documentation
```

## 🔧 **Code Structure & Working**

### **1. Computer Vision Processing (`app/services/file_processor.py`)**

```python
# Computer Vision Component: OCR for Image Processing
async def _process_image(self, file_content: bytes) -> str:
    """
    Computer Vision: Extract text from image files using OCR.
    This is a Computer Vision task - converting visual text to digital text.
    """
    try:
        # CV Step 1: Load image from bytes
        image = Image.open(io.BytesIO(file_content))
        
        # CV Step 2: Image preprocessing (convert to RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # CV Step 3: OCR - Computer Vision text extraction
        text_content = pytesseract.image_to_string(image)
        
        return text_content.strip()
    except Exception as e:
        raise Exception(f"Computer Vision OCR failed: {str(e)}")
```

### **2. NLP Processing (`app/services/openai_service.py`)**

```python
# NLP Component: AI-powered text understanding
async def parse_resume_text(self, resume_text: str) -> Dict[str, Any]:
    """
    NLP: Parse resume text using OpenAI GPT for intelligent understanding.
    This is an NLP task - understanding and structuring human language.
    """
    try:
        # NLP Step 1: Create prompt for GPT
        prompt = self._create_resume_parsing_prompt(resume_text)
        
        # NLP Step 2: Call OpenAI API (GPT-3.5-turbo)
        response = self._call_openai_api(prompt)
        
        # NLP Step 3: Parse and structure the response
        parsed_data = self._parse_openai_response(response)
        
        return parsed_data
    except Exception as e:
        raise Exception(f"NLP processing failed: {str(e)}")
```

### **3. Hybrid Processing Pipeline (`app/controllers/resume_controller.py`)**

```python
# Hybrid Approach: CV + NLP Pipeline
async def parse_resume(file: UploadFile = File(...)):
    """
    Complete pipeline: Computer Vision → NLP → Structured Output
    """
    try:
        # Step 1: File validation and reading
        file_content = await file.read()
        
        # Step 2: Computer Vision (if image) or Direct Text Extraction
        extracted_text = await file_processor.process_file(file_content, file.filename)
        
        # Step 3: NLP Processing with GPT
        parsed_data = await openai_service.parse_resume_text(extracted_text)
        
        # Step 4: Return structured JSON
        return ResumeParseResponse(
            parsed_data=parsed_data,
            file_type=file_type,
            processing_time=processing_time
        )
    except Exception as e:
        raise HTTPException(detail=f"Processing failed: {str(e)}")
```

## 📊 **Technology Breakdown**

### **Computer Vision (CV) Usage:**
- **OCR Processing**: Tesseract for image-to-text conversion
- **Image Formats**: PNG, JPG, JPEG, WEBP
- **Image Preprocessing**: RGB conversion, format handling
- **Text Extraction**: Visual text → Digital text

### **Natural Language Processing (NLP) Usage:**
- **Text Understanding**: GPT-3.5-turbo for intelligent parsing
- **Information Extraction**: Names, emails, experience, skills
- **Structured Output**: Unstructured text → JSON data
- **Experience Calculation**: Automatic duration calculation

### **Hybrid Processing Flow:**
```
Input File → CV (OCR) → Text → NLP (GPT) → Structured JSON
     ↓           ↓        ↓        ↓           ↓
   Image    Computer   Raw    Natural    Structured
   File     Vision    Text   Language   Data
```

## 🛠️ **Setup Instructions**

### **Prerequisites**

1. **Python 3.8+** installed
2. **Tesseract OCR** (for Computer Vision):
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

### **Installation**

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ResumeParser
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**:
   Create a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   MAX_FILE_SIZE=10485760
   TESSERACT_CMD=tesseract
   OCR_LANGUAGE=eng
   ```

5. **Run the application**:
   ```bash
   python run.py
   ```

## 📡 **API Usage**

### **Resume Parsing Endpoint**

**Endpoint**: `POST /api/v1/parse-resume`

**Supports**: PDF, DOCX, DOC, TXT, RTF, PNG, JPG, JPEG, WEBP

**Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/parse-resume" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@resume.pdf"
```

**Response Example**:
```json
{
  "parsed_data": {
    "Name": "John Doe",
    "Email": "john.doe@email.com",
    "Phone": "+1-555-123-4567",
    "TotalExperience": "5 years 3 months",
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
```

### **Health Check**

**Endpoint**: `GET /api/v1/health`

**Response**: Application status and version

## 🔍 **Technology Details**

### **Computer Vision Components:**

1. **Tesseract OCR**:
   - Converts image text to digital text
   - Supports multiple languages
   - Handles various image formats

2. **PIL (Python Imaging Library)**:
   - Image preprocessing
   - Format conversion (RGB)
   - Image manipulation

3. **Image Processing Pipeline**:
   ```
   Image File → PIL Load → RGB Convert → Tesseract OCR → Text Output
   ```

### **NLP Components:**

1. **OpenAI GPT-3.5-turbo**:
   - Intelligent text understanding
   - Context-aware parsing
   - Structured information extraction

2. **Text Analysis Pipeline**:
   ```
   Raw Text → GPT Analysis → Structured JSON → Experience Calculation
   ```

3. **Information Extraction**:
   - Personal details (name, email, phone)
   - Professional experience
   - Education history
   - Skills and certifications
   - Projects and achievements

## 🎯 **Key Features**

### **Computer Vision Features:**
- ✅ **Multi-format Image Support**: PNG, JPG, JPEG, WEBP
- ✅ **OCR Processing**: Automatic text extraction from images
- ✅ **Image Preprocessing**: RGB conversion, format handling
- ✅ **Error Handling**: Robust CV error management

### **NLP Features:**
- ✅ **Intelligent Parsing**: GPT-powered text understanding
- ✅ **Dynamic Field Detection**: Adapts to resume content
- ✅ **Experience Calculation**: Automatic duration computation
- ✅ **Structured Output**: Clean JSON responses

### **Hybrid Features:**
- ✅ **Seamless Integration**: CV → NLP pipeline
- ✅ **Multi-format Support**: Images and documents
- ✅ **Intelligent Processing**: Best of both worlds
- ✅ **Scalable Architecture**: Easy to extend

## 🗄️ **Database Setup**

### **1. Environment Configuration**
Create a `.env` file in the project root with your database URL:
```bash
DATABASE_URL=postgresql://neondb_owner:npg_h6IxCm7NduUE@ep-broad-bonus-a1yhtql1-pooler.ap-southeast-1.aws.neon.tech/Ats?sslmode=require&channel_binding=require
OPENAI_API_KEY=your_openai_api_key_here
DEBUG=True
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Database Features**
- **PostgreSQL Integration**: Direct database connection using asyncpg
- **Automatic Storage**: All parsed resumes are automatically saved to PostgreSQL
- **CRUD Operations**: Complete API for managing resume data
- **Search Functionality**: Search resumes by candidate name or email using SQL ILIKE
- **Pagination**: Efficient data retrieval with pagination support
- **Connection Pooling**: Optimized database connections
- **Indexes**: Database indexes for faster queries

### **4. API Endpoints**
- `POST /api/v1/parse-resume` - Parse and save resume
- `GET /api/v1/resumes/{id}` - Get resume by ID
- `GET /api/v1/resumes` - Get all resumes with pagination
- `GET /api/v1/resumes/search/{term}` - Search resumes
- `DELETE /api/v1/resumes/{id}` - Delete resume

### **5. Database Schema**
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

## 🚀 **Performance**

- **Computer Vision**: Fast OCR processing with Tesseract
- **NLP**: Efficient GPT API calls with caching
- **Database**: Fast PostgreSQL queries with indexing
- **Hybrid**: Optimized pipeline for maximum accuracy
- **Response Time**: 2-10 seconds depending on file size and complexity

## 🔧 **Development**

### **Adding New CV Features:**
```python
# Add new image format support
async def _process_new_format(self, file_content: bytes) -> str:
    # Computer Vision processing for new format
    pass
```

### **Adding New NLP Features:**
```python
# Add new information extraction
def _extract_new_field(self, text: str) -> str:
    # NLP processing for new field
    pass
```

## 📈 **Future Enhancements**

### **Computer Vision Improvements:**
- Advanced image preprocessing
- Layout analysis
- Table extraction
- Multi-language OCR

### **NLP Improvements:**
- Custom GPT fine-tuning
- Enhanced information extraction
- Sentiment analysis
- Skill matching

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure CV and NLP components work together
5. Submit a pull request

## 📄 **License**

MIT License - see LICENSE file for details

---

**This project demonstrates the power of combining Computer Vision and Natural Language Processing for intelligent document processing!** 🚀
