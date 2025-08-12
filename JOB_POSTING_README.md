# Job Posting Generation API

This API intelligently generates job postings using OpenAI's AI capabilities. The service automatically detects prompt complexity and generates fields accordingly:

- **Detailed Prompts** → Generate ALL fields with specific information
- **Simple Prompts** → Generate only relevant fields with reasonable defaults

## 🧠 Smart Field Generation

### Detailed Prompt Example
**Input:** *"Generate a job posting for an AI Specialist position at Appit Software Solutions. The role is located in Hyderabad, India. The job requires experience in machine learning, data science, and Python. The salary range is 50,000 - 80,000 INR. The recruiter is Hemanth Avvaru. The job is full-time and onsite. The position requires proficiency in TensorFlow and other machine learning tools. Benefits include health insurance, paid time off, and 401K. The priority is high."*

**Result:** All 20 fields populated with specific details from your prompt.

### Simple Prompt Example  
**Input:** *"Create a job posting for a React Developer"*

**Result:** Only relevant fields populated, others use reasonable defaults.

## Features

- **AI-Powered Generation**: Uses OpenAI to create professional job postings
- **Smart Field Detection**: Automatically determines prompt complexity
- **Adaptive Output**: Generates fields based on available information
- **Structured Output**: Returns data in a consistent JSON format
- **Easy Integration**: Simple REST API endpoint

## API Endpoint

### Generate Job Posting

**POST** `/api/v1/job-posting/generate`

**Request Body:**
```json
{
  "prompt": "Create a job posting for a React Developer"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "title": "React Developer",
    "company": "TechCorp Solutions",
    "department": "Engineering",
    "internalSPOC": "John Smith",
    "recruiter": "Sarah Johnson",
    "email": "careers@techcorp.com",
    "jobType": "Full-time",
    "experienceLevel": "Intermediate",
    "country": "United States",
    "city": "New York",
    "fullLocation": "New York, NY, United States",
    "workType": "HYBRID",
    "jobStatus": "ACTIVE",
    "salaryMin": 80000,
    "salaryMax": 120000,
    "priority": "Medium",
    "description": "We are seeking a React Developer...",
    "requirements": "Bachelor's degree in Computer Science...",
    "requiredSkills": "React, JavaScript, HTML, CSS, Git",
    "benefits": "Health insurance, 401k, flexible work hours..."
  },
  "message": "Job posting generated successfully"
}
```

## Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | String | Yes | Job title/position |
| `company` | String | Yes | Company name |
| `department` | String | No | Department name |
| `internalSPOC` | String | No | Internal point of contact |
| `recruiter` | String | No | Recruiter name |
| `email` | String | Yes | Contact email for applications |
| `jobType` | String | Yes | Full-time/Part-time/Contract/Internship |
| `experienceLevel` | String | No | Entry/Intermediate/Senior/Executive |
| `country` | String | No | Country location |
| `city` | String | No | City location |
| `fullLocation` | String | No | Complete location description |
| `workType` | String | No | ONSITE/REMOTE/HYBRID |
| `jobStatus` | String | No | ACTIVE/INACTIVE/CLOSED |
| `salaryMin` | Integer | Yes | Minimum salary |
| `salaryMax` | Integer | Yes | Maximum salary |
| `priority` | String | No | High/Medium/Low |
| `description` | String | Yes | Detailed job description |
| `requirements` | String | Yes | Job requirements and qualifications |
| `requiredSkills` | String | Yes | Required technical and soft skills |
| `benefits` | String | Yes | Company benefits and perks |

## Prompt Examples

### Detailed Prompts (Generate All Fields)
- *"Generate a job posting for an AI Specialist position at Appit Software Solutions. The role is located in Hyderabad, India. The job requires experience in machine learning, data science, and Python. The salary range is 50,000 - 80,000 INR. The recruiter is Hemanth Avvaru. The job is full-time and onsite. The position requires proficiency in TensorFlow and other machine learning tools. Benefits include health insurance, paid time off, and 401K. The priority is high."*

### Simple Prompts (Generate Relevant Fields Only)
- *"Create a job posting for a React Developer"*
- *"Generate a job posting for a Marketing Manager"*
- *"Create a job posting for a Data Scientist"*

### Medium Prompts (Mix of Both)
- *"Generate a job posting for a Data Scientist in New York with Python skills"*
- *"Create a job posting for a Frontend Developer in London"*

## Usage Examples

### Python
```python
import requests
import json

url = "http://localhost:8000/api/v1/job-posting/generate"

# Simple prompt
data = {"prompt": "Create a job posting for a React Developer"}
response = requests.post(url, json=data)
job_posting = response.json()
print(json.dumps(job_posting, indent=2))

# Detailed prompt
detailed_data = {
    "prompt": "Generate a job posting for an AI Specialist position at Appit Software Solutions. The role is located in Hyderabad, India. The job requires experience in machine learning, data science, and Python. The salary range is 50,000 - 80,000 INR. The recruiter is Hemanth Avvaru. The job is full-time and onsite. The position requires proficiency in TensorFlow and other machine learning tools. Benefits include health insurance, paid time off, and 401K. The priority is high."
}
response = requests.post(url, json=detailed_data)
detailed_job = response.json()
print(json.dumps(detailed_job, indent=2))
```

### JavaScript/Node.js
```javascript
// Simple prompt
const simpleResponse = await fetch('http://localhost:8000/api/v1/job-posting/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        prompt: 'Create a job posting for a React Developer'
    })
});

// Detailed prompt
const detailedResponse = await fetch('http://localhost:8000/api/v1/job-posting/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        prompt: 'Generate a job posting for an AI Specialist position at Appit Software Solutions. The role is located in Hyderabad, India. The job requires experience in machine learning, data science, and Python. The salary range is 50,000 - 80,000 INR. The recruiter is Hemanth Avvaru. The job is full-time and onsite. The position requires proficiency in TensorFlow and other machine learning tools. Benefits include health insurance, paid time off, and 401K. The priority is high.'
    })
});
```

### cURL
```bash
# Simple prompt
curl -X POST "http://localhost:8000/api/v1/job-posting/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Create a job posting for a React Developer"}'

# Detailed prompt
curl -X POST "http://localhost:8000/api/v1/job-posting/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Generate a job posting for an AI Specialist position at Appit Software Solutions. The role is located in Hyderabad, India. The job requires experience in machine learning, data science, and Python. The salary range is 50,000 - 80,000 INR. The recruiter is Hemanth Avvaru. The job is full-time and onsite. The position requires proficiency in TensorFlow and other machine learning tools. Benefits include health insurance, paid time off, and 401K. The priority is high."}'
```

## Testing

You can test the API using the provided test script:

```bash
python test_job_posting.py
```

This will test both detailed and simple prompt scenarios to show the difference in field generation.

Or use the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Health Check

**GET** `/api/v1/job-posting/health`

Returns the health status of the job posting service.

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- **400**: Bad Request (invalid input)
- **500**: Internal Server Error (service error)

Error responses include:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Requirements

- OpenAI API key configured in environment variables
- FastAPI application running
- Internet connection for OpenAI API calls

## Environment Variables

Make sure these are set in your `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=2000
OPENAI_TEMPERATURE=0.1
```
