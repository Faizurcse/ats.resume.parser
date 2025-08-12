# 🧠 Smart Field Generation - Job Posting API

## What It Does

The API now **intelligently detects prompt complexity** and generates fields accordingly:

### 🔍 **Simple Prompt** → **Minimal Fields**
**Input:** *"Create a job posting for a React Developer"*

**Result:** Only essential fields populated, others use reasonable defaults
- ✅ Generates: title, company, description, requirements, skills
- 🔧 Uses defaults: location, salary, benefits, etc.

### 📝 **Detailed Prompt** → **All Fields**
**Input:** *"Generate a job posting for an AI Specialist position at Appit Software Solutions. The role is located in Hyderabad, India. The job requires experience in machine learning, data science, and Python. The salary range is 50,000 - 80,000 INR. The recruiter is Hemanth Avvaru. The job is full-time and onsite. The position requires proficiency in TensorFlow and other machine learning tools. Benefits include health insurance, paid time off, and 401K. The priority is high."*

**Result:** ALL 20 fields populated with specific details from your prompt
- ✅ Generates: Every field with exact information provided
- 🎯 Uses: Specific company, location, salary, recruiter, benefits, etc.

## How It Works

1. **Analyzes your prompt** for keywords like:
   - `salary`, `benefits`, `recruiter`, `department`, `priority`
   - `experience level`, `work type`, `job status`, `requirements`
   - `skills`, `location`, `country`, `city`

2. **Counts detailed elements** - if 3+ found → Detailed mode
3. **Chooses generation strategy**:
   - **Detailed mode**: Fill ALL fields with specific info
   - **Simple mode**: Fill relevant fields, use defaults for others

## Examples

### Simple Prompts (Minimal Fields)
```
"Create a job posting for a React Developer"
"Generate a job posting for a Marketing Manager"
"Create a job posting for a Data Scientist"
```

### Detailed Prompts (All Fields)
```
"Generate a job posting for an AI Specialist position at Appit Software Solutions. The role is located in Hyderabad, India. The job requires experience in machine learning, data science, and Python. The salary range is 50,000 - 80,000 INR. The recruiter is Hemanth Avvaru. The job is full-time and onsite. The position requires proficiency in TensorFlow and other machine learning tools. Benefits include health insurance, paid time off, and 401K. The priority is high."
```

## Benefits

- 🎯 **Smart**: Automatically adapts to your input
- ⚡ **Fast**: Simple prompts generate quickly
- 📋 **Complete**: Detailed prompts get full information
- 🔧 **Flexible**: Works with any level of detail
- 💡 **Intelligent**: No need to specify format

## Test It

```bash
# Run the demo
python demo_smart_generation.py

# Run the test suite
python test_job_posting.py

# Use the API
curl -X POST "http://localhost:8000/api/v1/job-posting/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Create a job posting for a React Developer"}'
```

## API Endpoint

**POST** `/api/v1/job-posting/generate`

The API automatically detects your prompt complexity and generates the appropriate number of fields!
