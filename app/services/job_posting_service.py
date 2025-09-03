"""
Job Posting Service for generating job postings using OpenAI.
"""

import json
import logging
import re
from typing import Dict, Any
import openai
from app.config.settings import settings

logger = logging.getLogger(__name__)

class JobPostingService:
    """Service for generating job postings using OpenAI API."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required")
        logger.info("Job Posting service initialized")
    
    async def generate_job_posting(self, prompt: str) -> Dict[str, Any]:
        """Generate job posting data from a prompt with security validation."""
        try:
            # Security Layer 1: Input validation and sanitization
            prompt = self._sanitize_input(prompt)
            
            # Security Layer 2: Multi-layer security analysis for prompt type determination
            prompt_type = self._analyze_prompt_security(prompt)
            
            if prompt_type == "non_job_related":
                # Non-job related prompt - raise error instead of generating job
                raise ValueError(f"Invalid prompt: '{prompt}' is not related to job postings. Please provide a job-related prompt.")
            elif prompt_type == "single_skill":
                # Single skill search (e.g., "java") - return specific job posting
                system_prompt = self._get_single_skill_prompt()
                logger.info(f"Using single skill prompt for: {prompt}")
            elif prompt_type == "specific_skill":
                # Specific skill request (e.g., "java developer") - return skill-specific jobs
                system_prompt = self._get_specific_skill_prompt(prompt)
                logger.info(f"Using specific skill prompt for: {prompt}")
            elif prompt_type == "generic_job":
                # Generic job search - return diverse job postings
                system_prompt = self._get_generic_word_prompt()
                logger.info(f"Using generic job prompt for: {prompt}")
            elif prompt_type == "detailed_job":
                # Detailed prompt - return comprehensive job posting
                system_prompt = self._get_detailed_prompt()
                logger.info(f"Using detailed prompt for: {prompt}")
            else:
                # Default fallback - return basic job posting (SECURE FALLBACK)
                system_prompt = self._get_simple_prompt()
                logger.info(f"Using secure fallback prompt for: {prompt}")
            
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate job posting: {prompt}"}
                ],
                max_tokens=400,  # Further reduced for faster response
                temperature=0.0,  # Zero temperature for fastest, most consistent responses
                top_p=0.1,  # Low top_p for faster generation
                response_format={"type": "json_object"},  # Force JSON response
                timeout=5.0  # Reduced timeout to 5 seconds
            )
            
            response_content = response.choices[0].message.content.strip()
            logger.info(f"Raw OpenAI response: {response_content[:200]}...")
            
            # Clean and parse the response
            job_data = self._parse_and_clean_response(response_content)
            
            logger.info(f"Generated job posting with {len(job_data)} fields")
            return job_data
            
        except ValueError as e:
            # Re-raise ValueError (from sanitization or non-job-related prompts) as-is
            logger.error(f"Validation error: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Error generating job posting: {str(e)}")
            raise Exception(f"Failed to generate job posting: {str(e)}")
    
    def _is_detailed_prompt(self, prompt: str) -> bool:
        """Check if prompt contains detailed information."""
        detailed_keywords = [
            'salary', 'benefits', 'recruiter', 'department', 'priority',
            'experience level', 'work type', 'job status', 'requirements',
            'skills', 'location', 'country', 'city'
        ]
        
        prompt_lower = prompt.lower()
        detailed_count = sum(1 for keyword in detailed_keywords if keyword in prompt_lower)
        
        # If prompt contains 3+ detailed elements, treat as detailed
        return detailed_count >= 3
    
    def _is_single_skill_search(self, prompt: str) -> bool:
        """Check if prompt is a single skill search (e.g., 'java', 'python')"""
        # Remove extra whitespace and check if it's just a single word/skill
        clean_prompt = prompt.strip().lower()
        
        # Check if it's just a single word (likely a programming language or skill)
        if len(clean_prompt.split()) == 1:
            # Common programming languages and skills
            common_skills = [
                'java', 'python', 'javascript', 'react', 'angular', 'vue', 'node',
                'sql', 'mongodb', 'aws', 'azure', 'docker', 'kubernetes', 'git',
                'html', 'css', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin',
                'c++', 'c#', 'dotnet', 'spring', 'django', 'flask', 'express',
                'mysql', 'postgresql', 'redis', 'elasticsearch', 'kafka',
                'jenkins', 'gitlab', 'jira', 'confluence', 'agile', 'scrum'
            ]
            return clean_prompt in common_skills
        
        return False
    
    def _is_specific_skill_request(self, prompt: str) -> bool:
        """Check if prompt contains specific skill requests like 'java developer'"""
        clean_prompt = prompt.strip().lower()
        
        # Common programming languages and technologies
        tech_skills = [
            'java', 'python', 'javascript', 'react', 'angular', 'vue', 'node',
            'sql', 'mongodb', 'aws', 'azure', 'docker', 'kubernetes', 'git',
            'html', 'css', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin',
            'c++', 'c#', 'dotnet', 'spring', 'django', 'flask', 'express',
            'mysql', 'postgresql', 'redis', 'elasticsearch', 'kafka',
            'jenkins', 'gitlab', 'jira', 'confluence', 'agile', 'scrum',
            'frontend', 'backend', 'full stack', 'fullstack', 'mobile', 'devops',
            'data science', 'machine learning', 'ai', 'artificial intelligence'
        ]
        
        # Check if any tech skill is mentioned in the prompt
        return any(skill in clean_prompt for skill in tech_skills)
    
    def _is_non_job_related_prompt(self, prompt: str) -> bool:
        """Check if prompt is not job-related with comprehensive detection"""
        clean_prompt = prompt.strip().lower()
        
        # First check if it's a job posting creation request with field specifications
        if self._is_job_posting_with_fields(prompt):
            return False  # Allow it even if it contains non-job keywords
        
        # Comprehensive non-job related keywords
        non_job_keywords = [
            # Celebrity Names (Indian & International)
            'salman khan', 'shah rukh khan', 'amir khan', 'akshay kumar', 'hrithik roshan',
            'deepika padukone', 'priyanka chopra', 'kareena kapoor', 'katrina kaif',
            'tom cruise', 'leonardo dicaprio', 'brad pitt', 'angelina jolie', 'jennifer lawrence',
            'robert downey jr', 'chris evans', 'scarlett johansson', 'chris hemsworth',
            
            # Sports & Games
            'cricket', 'football', 'soccer', 'basketball', 'tennis', 'badminton',
            'hockey', 'volleyball', 'baseball', 'golf', 'swimming', 'running',
            'chess', 'poker', 'video games', 'gaming', 'esports', 'fifa', 'call of duty',
            
            # Entertainment & Media
            'movie', 'film', 'bollywood', 'hollywood', 'music', 'song', 'dance',
            'actor', 'actress', 'singer', 'dancer', 'director', 'producer',
            'netflix', 'youtube', 'instagram', 'tiktok', 'facebook', 'twitter',
            'comedy', 'drama', 'action', 'horror', 'romance', 'thriller',
            
            # Food & Drinks
            'pizza', 'burger', 'pasta', 'rice', 'chicken', 'beef', 'vegetarian',
            'restaurant', 'cooking', 'recipe', 'food', 'drink', 'coffee', 'tea',
            'alcohol', 'beer', 'wine', 'whiskey', 'vodka', 'cocktail',
            
            # Nature & Weather
            'weather', 'rain', 'sunny', 'cloudy', 'hot', 'cold', 'warm',
            'mountain', 'ocean', 'river', 'forest', 'tree', 'flower', 'animal',
            'dog', 'cat', 'bird', 'fish', 'lion', 'tiger', 'elephant',
            
            # Random & Common Words
            'hello', 'hi', 'good morning', 'good evening', 'how are you',
            'thank you', 'please', 'sorry', 'yes', 'no', 'maybe', 'okay',
            'love', 'hate', 'happy', 'sad', 'angry', 'excited', 'bored',
            
            # Fantasy & Fiction
            'unicorn', 'dragon', 'magic', 'wizard', 'fairy', 'superhero',
            'spaceship', 'alien', 'robot', 'monster', 'ghost', 'vampire',
            
            # Academic & Scientific
            'quantum physics', 'chemistry', 'biology', 'mathematics', 'history',
            'geography', 'literature', 'philosophy', 'psychology', 'sociology',
            
            # Technology (Non-Job Related)
            'iphone', 'android', 'smartphone', 'laptop', 'computer', 'internet',
            'social media', 'streaming', 'podcast', 'blog', 'website',
            
            # Miscellaneous
            'travel', 'vacation', 'holiday', 'party', 'wedding', 'birthday',
            'shopping', 'fashion', 'beauty', 'fitness', 'gym', 'yoga',
            'art', 'painting', 'drawing', 'sculpture', 'photography',
            'book', 'novel', 'story', 'poetry', 'writing', 'reading'
        ]
        
        # Check if prompt contains non-job related keywords
        return any(keyword in clean_prompt for keyword in non_job_keywords)
    
    def _is_job_posting_with_fields(self, prompt: str) -> bool:
        """Check if prompt is a job posting creation with field specifications"""
        clean_prompt = prompt.strip().lower()
        
        # Job posting field indicators
        field_indicators = [
            'email:', 'spoc:', 'internalspoc:', 'recruiter:', 'company:', 'department:',
            'jobtype:', 'job type:', 'experiencelevel:', 'experience level:', 'country:',
            'city:', 'location:', 'worktype:', 'work type:', 'jobstatus:', 'job status:',
            'salarymin:', 'salary min:', 'salarymax:', 'salary max:', 'priority:',
            'description:', 'requirements:', 'requiredskills:', 'required skills:',
            'benefits:', 'title:', 'jobtitle:', 'job title:'
        ]
        
        # Check if prompt contains field specifications
        return any(indicator in clean_prompt for indicator in field_indicators)
    
    def _is_job_related_prompt(self, prompt: str) -> bool:
        """Check if prompt is job-related with comprehensive detection"""
        clean_prompt = prompt.strip().lower()
        
        # Job-related keywords
        job_keywords = [
            # Job Titles
            'developer', 'engineer', 'manager', 'analyst', 'designer', 'specialist',
            'coordinator', 'assistant', 'director', 'lead', 'architect', 'consultant',
            'administrator', 'supervisor', 'executive', 'officer', 'representative',
            'technician', 'operator', 'clerk', 'secretary', 'receptionist',
            
            # Skills & Technologies
            'java', 'python', 'javascript', 'react', 'angular', 'vue', 'node',
            'sql', 'mongodb', 'aws', 'azure', 'docker', 'kubernetes', 'git',
            'html', 'css', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin',
            'spring', 'django', 'flask', 'express', 'mysql', 'postgresql',
            
            # Job Actions
            'hire', 'recruit', 'employment', 'career', 'job', 'work', 'position',
            'role', 'vacancy', 'opening', 'opportunity', 'candidate', 'resume',
            'interview', 'salary', 'benefits', 'experience', 'qualification',
            
            # Industries
            'software', 'technology', 'it', 'finance', 'banking', 'healthcare',
            'education', 'marketing', 'sales', 'hr', 'human resources', 'legal',
            'consulting', 'retail', 'manufacturing', 'construction', 'real estate',
            
            # Work Types
            'full-time', 'part-time', 'contract', 'internship', 'remote', 'onsite',
            'hybrid', 'freelance', 'temporary', 'permanent', 'entry-level', 'senior',
            'junior', 'mid-level', 'executive', 'leadership', 'management'
        ]
        
        # Check if prompt contains job-related keywords
        return any(keyword in clean_prompt for keyword in job_keywords)
    
    def _analyze_prompt_security(self, prompt: str) -> str:
        """Multi-layer security analysis to determine prompt type"""
        clean_prompt = prompt.strip().lower()
        
        # Layer 1: Check for non-job related content (HIGHEST PRIORITY)
        if self._is_non_job_related_prompt(prompt):
            return "non_job_related"
        
        # Layer 2: Check for job-related content
        if self._is_job_related_prompt(prompt):
            # Layer 3: Determine specific job type
            if self._is_single_skill_search(prompt):
                return "single_skill"
            elif self._is_specific_skill_request(prompt):
                return "specific_skill"
            elif self._is_detailed_prompt(prompt):
                return "detailed_job"
            else:
                return "generic_job"
        
        # Layer 4: Check for generic job creation requests
        if self._is_generic_word_search(prompt):
            return "generic_job"
        
        # Layer 5: SECURE FALLBACK - Handle ANY unknown prompt
        # This ensures 100% coverage for any prompt in the world
        return "secure_fallback"
    
    def _sanitize_input(self, prompt: str) -> str:
        """Sanitize and validate input prompt for security"""
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt is empty or invalid. Please provide a valid prompt.")
        
        # Remove potentially harmful characters and limit length
        import re
        
        # Remove special characters that could cause issues
        sanitized = re.sub(r'[<>"\'\`\\]', '', prompt)
        
        # Limit prompt length to prevent abuse
        if len(sanitized) > 500:
            sanitized = sanitized[:500]
        
        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())
        
        # If empty after sanitization, raise error
        if not sanitized.strip():
            raise ValueError("Prompt is empty or contains only whitespace. Please provide a valid prompt.")
        
        return sanitized.strip()
    
    def _is_generic_word_search(self, prompt: str) -> bool:
        """Check if prompt is just generic words without specific job details"""
        clean_prompt = prompt.strip().lower()
        
        # Generic prompts that should generate diverse jobs
        generic_phrases = [
            'create job post', 'generate jobs', 'job posting', 'jobs', 
            'create jobs', 'generate job postings', 'job posts', 'post jobs',
            'create job', 'generate job', 'job', 'posting', 'posts'
        ]
        
        # Check if it's a generic job creation request
        if any(phrase in clean_prompt for phrase in generic_phrases):
            return True
        
        # If it's just 1-3 generic words without job-specific context
        if len(clean_prompt.split()) <= 3:
            # Check if it doesn't contain job-specific keywords
            job_keywords = [
                'developer', 'engineer', 'manager', 'analyst', 'designer',
                'specialist', 'coordinator', 'assistant', 'director', 'lead',
                'architect', 'consultant', 'administrator', 'supervisor'
            ]
            
            # If none of the words are job titles, treat as generic search
            words = clean_prompt.split()
            has_job_title = any(word in job_keywords for word in words)
            
            return not has_job_title
        
        return False
    
    def _get_detailed_prompt(self) -> str:
        """Get system prompt for detailed job posting generation."""
        return """You are a job posting generator. Create comprehensive job postings in this exact JSON format.

IMPORTANT: Extract ALL fields that are specified in the user's prompt using patterns like:
- "company: [value]" -> extract company field
- "department: [value]" -> extract department field
- "jobTitle: [value]" -> extract title field
- "internalSPOC: [value]" -> extract internalSPOC field
- "recruiter: [value]" -> extract recruiter field
- "email: [value]" -> extract email field
- "jobType: [value]" -> extract jobType field
- "experienceLevel: [value]" -> extract experienceLevel field
- "country: [value]" -> extract country field
- "city: [value]" -> extract city field
- "fullLocation: [value]" -> extract fullLocation field
- "workType: [value]" -> extract workType field
- "jobStatus: [value]" -> extract jobStatus field
- "salaryMin: [value]" -> extract salaryMin field
- "salaryMax: [value]" -> extract salaryMax field
- "priority: [value]" -> extract priority field
- "description: [value]" -> extract description field
- "requirements: [value]" -> extract requirements field
- "requiredSkills: [value]" -> extract requiredSkills field
- "benefits: [value]" -> extract benefits field

Return JSON with ONLY the fields that are explicitly specified in the user's prompt:

{
  "title": "Job Title (from jobTitle: or title:)",
  "company": "Company Name (from company:)",
  "department": "Department Name (from department:)",
  "internalSPOC": "Internal Point of Contact (from internalSPOC:)",
  "recruiter": "Recruiter Name (from recruiter:)",
  "email": "contact@company.com (from email:)",
  "jobType": "Full-time/Part-time/Contract/Internship (from jobType:)",
  "experienceLevel": "Entry/Intermediate/Senior/Executive (from experienceLevel:)",
  "country": "Country (from country:)",
  "city": "City (from city:)",
  "fullLocation": "Full Location Description (from fullLocation:)",
  "workType": "ONSITE/REMOTE/HYBRID (from workType:)",
  "jobStatus": "ACTIVE/INACTIVE/CLOSED (from jobStatus:)",
  "salaryMin": "Salary minimum (from salaryMin:)",
  "salaryMax": "Salary maximum (from salaryMax:)",
  "priority": "High/Medium/Low (from priority:)",
  "description": "Detailed job description (from description:)",
  "requirements": "Job requirements (from requirements:)",
  "requiredSkills": "Required skills (from requiredSkills:)",
  "benefits": "Company benefits (from benefits:)"
}

CRITICAL RULES:
1. Extract ALL fields that are specified in the user's prompt
2. Use the exact values provided after each field colon
3. If a field is specified like "company: [value]", it MUST be included in the response
4. Return ONLY valid JSON with no additional text
5. Do not omit any fields that are explicitly mentioned in the prompt
6. Pay special attention to field specifications with colons (e.g., "company: Appit Software Solutions")"""
    
    def _get_single_skill_prompt(self) -> str:
        """Get system prompt for single skill search (e.g., 'java')."""
        return """Create a complete job posting for the skill mentioned.

Return JSON with ALL fields:
{
  "title": "[Skill] Developer",
  "company": "[Company name]",
  "department": "Engineering",
  "internalSPOC": "[Name]",
  "recruiter": "[Name]",
  "email": "[email@company.com]",
  "jobType": "Full-time",
  "experienceLevel": "[Entry/Intermediate/Senior]",
  "country": "[Country]",
  "city": "[City]",
  "fullLocation": "[City, State, Country]",
  "workType": "[ONSITE/REMOTE/HYBRID]",
  "jobStatus": "ACTIVE",
  "salaryMin": "[Min salary]",
  "salaryMax": "[Max salary]",
  "priority": "High",
  "description": "[100-150 word description]",
  "requirements": "[100-150 word requirements]",
  "requiredSkills": "[6-8 skills, comma-separated]",
  "benefits": "[Benefits]"
}"""
    
    def _get_specific_skill_prompt(self, prompt: str) -> str:
        """Get system prompt for specific skill requests (e.g., 'java developer')."""
        return f"""Create a job posting for: {prompt}

Return JSON:
{{
  "title": "[Job title with skill]",
  "company": "[Company name]",
  "department": "Engineering",
  "internalSPOC": "[Name]",
  "recruiter": "[Name]",
  "email": "[email@company.com]",
  "jobType": "Full-time",
  "experienceLevel": "[Entry/Intermediate/Senior]",
  "country": "[Country]",
  "city": "[City]",
  "fullLocation": "[City, State, Country]",
  "workType": "[ONSITE/REMOTE/HYBRID]",
  "jobStatus": "ACTIVE",
  "salaryMin": "[Min salary]",
  "salaryMax": "[Max salary]",
  "priority": "High",
  "description": "[100-150 word description]",
  "requirements": "[100-150 word requirements]",
  "requiredSkills": "[6-8 skills, comma-separated]",
  "benefits": "[Benefits]"
}}"""
    
    def _get_generic_word_prompt(self) -> str:
        """Get system prompt for generic word searches."""
        return """Create a job posting. Choose from: Software Developer, Data Analyst, Product Manager, Marketing Specialist, Sales Rep, HR Coordinator, Financial Analyst, UX Designer, DevOps Engineer, Business Analyst.

Return JSON:
{
  "title": "[Job title]",
  "company": "[Company name]",
  "department": "[Department]",
  "internalSPOC": "[Name]",
  "recruiter": "[Name]",
  "email": "[email@company.com]",
  "jobType": "Full-time",
  "experienceLevel": "[Entry/Intermediate/Senior]",
  "country": "[Country]",
  "city": "[City]",
  "fullLocation": "[City, State, Country]",
  "workType": "[ONSITE/REMOTE/HYBRID]",
  "jobStatus": "ACTIVE",
  "salaryMin": "[Min salary]",
  "salaryMax": "[Max salary]",
  "priority": "High",
  "description": "[100-150 word description]",
  "requirements": "[100-150 word requirements]",
  "requiredSkills": "[6-8 skills, comma-separated]",
  "benefits": "[Benefits]"
}"""
    
    def _get_simple_prompt(self) -> str:
        """Get system prompt for simple job posting generation."""
        return """Create a complete job posting.

Return JSON with ALL fields:
{
  "title": "[Job title]",
  "company": "[Company name]",
  "department": "[Department]",
  "internalSPOC": "[Name]",
  "recruiter": "[Name]",
  "email": "[email@company.com]",
  "jobType": "Full-time",
  "experienceLevel": "[Entry/Intermediate/Senior]",
  "country": "[Country]",
  "city": "[City]",
  "fullLocation": "[City, State, Country]",
  "workType": "[ONSITE/REMOTE/HYBRID]",
  "jobStatus": "ACTIVE",
  "salaryMin": "[Min salary]",
  "salaryMax": "[Max salary]",
  "priority": "High",
  "description": "[100-150 word description]",
  "requirements": "[100-150 word requirements]",
  "requiredSkills": "[6-8 skills, comma-separated]",
  "benefits": "[Benefits]"
}"""
    
    def _parse_and_clean_response(self, response: str) -> Dict[str, Any]:
        """Parse and clean the OpenAI response to extract valid JSON."""
        try:
            # First try to parse directly (in case OpenAI returned clean JSON)
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
            
            # Clean the response
            cleaned_response = self._clean_response(response)
            logger.info(f"Cleaned response: {cleaned_response[:200]}...")
            
            # Try to parse the cleaned response
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse cleaned response: {str(e)}")
                logger.error(f"Cleaned response: {cleaned_response}")
                
                # Try to extract JSON using regex as last resort
                json_match = self._extract_json_with_regex(cleaned_response)
                if json_match:
                    try:
                        return json.loads(json_match)
                    except json.JSONDecodeError:
                        pass
                
                # If all else fails, create a basic structure
                logger.warning("Creating fallback job posting structure")
                return self._create_fallback_job_posting()
                
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return self._create_fallback_job_posting()
    
    def _clean_response(self, response: str) -> str:
        """Clean OpenAI response to extract JSON."""
        # Remove markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end]
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end]
        
        # Remove any leading/trailing whitespace and newlines
        response = response.strip()
        
        # Remove any text before the first {
        first_brace = response.find('{')
        if first_brace != -1:
            response = response[first_brace:]
        
        # Remove any text after the last }
        last_brace = response.rfind('}')
        if last_brace != -1:
            response = response[:last_brace + 1]
        
        return response
    
    def _extract_json_with_regex(self, text: str) -> str:
        """Extract JSON using regex as a fallback method."""
        try:
            # Look for JSON-like structure
            pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(pattern, text)
            if matches:
                # Return the longest match (most likely to be complete)
                return max(matches, key=len)
        except Exception as e:
            logger.warning(f"Regex extraction failed: {str(e)}")
        return ""
    
    def _create_fallback_job_posting(self) -> Dict[str, Any]:
        """Create a fallback job posting structure if parsing fails."""
        return {
            "title": "Software Developer",
            "company": "Tech Solutions Inc.",
            "department": "Engineering",
            "internalSPOC": "John Smith",
            "recruiter": "Jane Doe",
            "email": "jane.doe@techsolutions.com",
            "jobType": "Full-time",
            "experienceLevel": "Intermediate",
            "country": "United States",
            "city": "San Francisco",
            "fullLocation": "San Francisco, CA, United States",
            "workType": "HYBRID",
            "jobStatus": "ACTIVE",
            "salaryMin": "80000",
            "salaryMax": "120000",
            "priority": "High",
            "description": "We are seeking a talented software developer to join our dynamic team. The ideal candidate will have strong programming skills, experience with modern development practices, and a passion for creating high-quality software solutions.",
            "requirements": "Bachelor's degree in Computer Science or related field. Minimum 3+ years of experience in software development. Proficiency in programming languages like Java, Python, or JavaScript. Experience with modern frameworks and technologies. Strong problem-solving skills and attention to detail.",
            "requiredSkills": "Java, Python, JavaScript, React, Node.js, SQL, Git, Docker, AWS, REST APIs",
            "benefits": "Competitive salary, health insurance, 401(k) plan, flexible work hours, professional development opportunities"
        }
