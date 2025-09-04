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
            
            if prompt_type == "invalid_prompt":
                # Invalid prompt pattern (random characters, etc.) - raise error
                raise ValueError(f"Invalid prompt: '{prompt}' contains random characters or is not a meaningful job description. Please provide a proper job-related prompt.")
            elif prompt_type == "non_job_related":
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
                
                # Pre-process the prompt to extract fields explicitly
                extracted_fields = self._extract_fields_from_prompt(prompt)
                if extracted_fields and len(extracted_fields) >= 5:  # If we have enough fields, use direct approach
                    logger.info(f"Extracted {len(extracted_fields)} fields from prompt - using direct field mapping")
                    # Use direct field mapping instead of AI generation
                    return self._create_job_from_extracted_fields(extracted_fields)
                elif extracted_fields:
                    logger.info(f"Extracted {len(extracted_fields)} fields from prompt - using AI with extracted fields")
                    # Modify the prompt to include extracted fields
                    prompt = self._format_extracted_fields(extracted_fields)
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
                max_tokens=2000,  # Increased for comprehensive job descriptions
                temperature=0.1,  # Slight randomness for variety
                top_p=0.9,  # Higher top_p for better quality
                response_format={"type": "json_object"},  # Force JSON response
                timeout=30.0  # Increased timeout to 30 seconds for reliability
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
            # Try to provide more specific error messages
            if "timeout" in str(e).lower():
                raise Exception(f"Request timed out while generating job posting. Please try again.")
            elif "rate limit" in str(e).lower():
                raise Exception(f"API rate limit exceeded. Please try again in a few moments.")
            elif "authentication" in str(e).lower() or "api key" in str(e).lower():
                raise Exception(f"API authentication failed. Please check your OpenAI API key.")
            else:
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
    
    def _is_invalid_prompt_pattern(self, prompt: str) -> bool:
        """Check if prompt contains invalid patterns like random characters"""
        clean_prompt = prompt.strip()
        
        # Skip validation for detailed job posting prompts (they contain field specifications)
        if self._is_job_posting_with_fields(prompt):
            return False
        
        import re
        
        # Check for obvious random character sequences (more strict patterns)
        # Pattern: 15+ consecutive letters without spaces (keyboard mashing)
        random_letter_pattern = re.compile(r'[a-z]{15,}|[A-Z]{15,}')
        if random_letter_pattern.search(clean_prompt):
            return True
        
        # Check for random number sequences (10+ consecutive digits)
        random_number_pattern = re.compile(r'[0-9]{10,}')
        if random_number_pattern.search(clean_prompt):
            return True
        
        # Check for random special character sequences (8+ consecutive special chars)
        random_special_pattern = re.compile(r'[^a-zA-Z0-9\s]{8,}')
        if random_special_pattern.search(clean_prompt):
            return True
        
        # Check for repeated characters (10+ same character in a row)
        repeated_char_pattern = re.compile(r'(.)\1{10,}')
        if repeated_char_pattern.search(clean_prompt):
            return True
        
        # Check for minimum meaningful words (at least 2 words with length > 1)
        words = [word for word in clean_prompt.split() if len(word) > 1]
        if len(words) < 2:
            return True
        
        # Check for job-related keywords (must contain at least one)
        job_keywords = [
            'job', 'position', 'role', 'developer', 'engineer', 'manager', 'analyst', 
            'designer', 'programmer', 'sales', 'marketing', 'hr', 'finance', 'admin', 
            'support', 'consultant', 'specialist', 'coordinator', 'assistant', 'director', 
            'lead', 'senior', 'junior', 'intern', 'freelance', 'remote', 'full-time', 
            'part-time', 'contract', 'employee', 'staff', 'worker', 'professional',
            'career', 'employment', 'hiring', 'recruitment', 'vacancy', 'opening',
            'company', 'department', 'salary', 'experience', 'skills', 'requirements',
            'benefits', 'location', 'work', 'team', 'project', 'technology', 'software'
        ]
        
        clean_lower = clean_prompt.lower()
        has_job_keywords = any(keyword in clean_lower for keyword in job_keywords)
        if not has_job_keywords:
            return True
        
        return False
    
    def _extract_fields_from_prompt(self, prompt: str) -> Dict[str, str]:
        """Extract field specifications from detailed prompts"""
        extracted_fields = {}
        
        # Field patterns to look for - improved regex patterns
        field_patterns = {
            'company': r'company:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'department': r'department:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'jobTitle': r'jobTitle:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'internalSPOC': r'internalSPOC:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'recruiter': r'recruiter:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'email': r'email:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'jobType': r'jobType:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'experienceLevel': r'experienceLevel:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'country': r'country:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'city': r'city:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'fullLocation': r'fullLocation:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'workType': r'workType:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'jobStatus': r'jobStatus:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'salaryMin': r'salaryMin:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'salaryMax': r'salaryMax:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'priority': r'priority:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'description': r'description:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'requirements': r'requirements:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'requiredSkills': r'requiredSkills:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)',
            'benefits': r'benefits:\s*([^a-z][^:]*?)(?=\s+[a-z]+:|$)'
        }
        
        import re
        for field_name, pattern in field_patterns.items():
            match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                if value:
                    extracted_fields[field_name] = value
                    logger.info(f"Extracted {field_name}: {value[:50]}...")
        
        return extracted_fields
    
    def _format_extracted_fields(self, extracted_fields: Dict[str, str]) -> str:
        """Format extracted fields into a clear prompt for the AI"""
        formatted_prompt = "Generate a job posting with the following specifications:\n\n"
        
        for field_name, value in extracted_fields.items():
            formatted_prompt += f"{field_name}: {value}\n"
        
        formatted_prompt += "\nPlease create a complete job posting using ALL the above specifications."
        
        return formatted_prompt
    
    def _create_job_from_extracted_fields(self, extracted_fields: Dict[str, str]) -> Dict[str, Any]:
        """Create job posting directly from extracted fields"""
        job_data = {}
        
        # Map extracted fields to job data structure
        field_mapping = {
            'company': 'company',
            'department': 'department', 
            'jobTitle': 'title',
            'internalSPOC': 'internalSPOC',
            'recruiter': 'recruiter',
            'email': 'email',
            'jobType': 'jobType',
            'experienceLevel': 'experienceLevel',
            'country': 'country',
            'city': 'city',
            'fullLocation': 'fullLocation',
            'workType': 'workType',
            'jobStatus': 'jobStatus',
            'salaryMin': 'salaryMin',
            'salaryMax': 'salaryMax',
            'priority': 'priority',
            'description': 'description',
            'requirements': 'requirements',
            'requiredSkills': 'requiredSkills',
            'benefits': 'benefits'
        }
        
        # Map all extracted fields
        for field_name, value in extracted_fields.items():
            if field_name in field_mapping:
                job_data[field_mapping[field_name]] = value
                logger.info(f"Mapped {field_name} -> {field_mapping[field_name]}: {value[:50]}...")
        
        # Ensure all required fields are present (fill with empty string if missing)
        required_fields = ['title', 'company', 'department', 'internalSPOC', 'recruiter', 'email', 
                          'jobType', 'experienceLevel', 'country', 'city', 'fullLocation', 'workType', 
                          'jobStatus', 'salaryMin', 'salaryMax', 'priority', 'description', 
                          'requirements', 'requiredSkills', 'benefits']
        
        for field in required_fields:
            if field not in job_data:
                job_data[field] = ""
        
        logger.info(f"Created job posting with {len(job_data)} fields from extracted data")
        return job_data
    
    def _is_job_posting_with_fields(self, prompt: str) -> bool:
        """Check if prompt is a job posting creation with field specifications"""
        clean_prompt = prompt.strip().lower()
        
        # Job posting field indicators (all lowercase for case-insensitive matching)
        field_indicators = [
            'email:', 'spoc:', 'internalspoc:', 'recruiter:', 'company:', 'department:',
            'jobtype:', 'job type:', 'experiencelevel:', 'experience level:', 'country:',
            'city:', 'location:', 'worktype:', 'work type:', 'jobstatus:', 'job status:',
            'salarymin:', 'salary min:', 'salarymax:', 'salary max:', 'priority:',
            'description:', 'requirements:', 'requiredskills:', 'required skills:',
            'benefits:', 'title:', 'jobtitle:', 'job title:',
            # Additional field indicators for comprehensive job descriptions (lowercase)
            'fulllocation:', 'worktype:', 'jobstatus:', 'salarymin:', 'salarymax:',
            'requiredskills:', 'internalspoc:', 'experiencelevel:', 'jobtitle:'
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
            'principal', 'senior', 'junior', 'staff', 'associate', 'vice',
            
            # Skills & Technologies
            'java', 'python', 'javascript', 'react', 'angular', 'vue', 'node',
            'sql', 'mongodb', 'aws', 'azure', 'docker', 'kubernetes', 'git',
            'html', 'css', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin',
            'spring', 'django', 'flask', 'express', 'mysql', 'postgresql',
            
            # Job Actions
            'hire', 'recruit', 'employment', 'career', 'job', 'work', 'position',
            'role', 'vacancy', 'opening', 'opportunity', 'candidate', 'resume',
            'interview', 'salary', 'benefits', 'experience', 'qualification',
            'seeking', 'join', 'team', 'developing', 'implementing', 'collaborating',
            'deliver', 'products', 'transform', 'industries', 'growth', 'opportunities',
            'groundbreaking', 'projects', 'initiatives', 'mentor', 'contribute',
            'strategy', 'roadmap', 'bachelor', 'degree', 'computer', 'science',
            'machine', 'learning', 'artificial', 'intelligence', 'proficiency',
            'deep', 'learning', 'computer', 'vision', 'natural', 'language',
            'processing', 'cloud', 'platforms', 'mlops', 'problem', 'solving',
            'analytical', 'skills', 'communication', 'teamwork', 'abilities',
            'pipelines', 'databases', 'engineering', 'understanding', 'software',
            'development', 'practices', 'agile', 'methodologies', 'health', 'insurance',
            'coverage', 'dental', 'vision', 'retirement', 'plan', 'company', 'match',
            'paid', 'time', 'off', 'flexible', 'working', 'hours', 'professional',
            'development', 'budget', 'gym', 'membership', 'reimbursement', 'free',
            'lunch', 'snacks', 'remote', 'work', 'options', 'annual', 'performance',
            'bonus', 'stock', 'options', 'wellness', 'programs', 'team', 'building',
            'events', 'conference', 'attendance', 'support', 'certification',
            'reimbursement', 'mentorship', 'programs', 'advancement', 'opportunities',
            
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
        
        # Layer 0: Check for invalid/random character patterns (HIGHEST PRIORITY)
        if self._is_invalid_prompt_pattern(prompt):
            return "invalid_prompt"
        
        # Layer 1: Check for job posting creation with field specifications
        if self._is_job_posting_with_fields(prompt):
            return "detailed_job"
        
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
        
        # Layer 5: Check for non-job related content (LOWER PRIORITY)
        if self._is_non_job_related_prompt(prompt):
            return "non_job_related"
        
        # Layer 6: SECURE FALLBACK - Handle ANY unknown prompt
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

CRITICAL INSTRUCTIONS:
1. The user's prompt contains detailed field specifications with colons (e.g., "company: Appit Software Solutions")
2. You MUST extract and include ALL fields that are explicitly mentioned with colons
3. Do not omit any fields that are provided in the user's input
4. Extract the exact values after each colon, including long descriptions

FIELD EXTRACTION PATTERNS:
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
- "description: [value]" -> extract description field (can be very long)
- "requirements: [value]" -> extract requirements field (can be very long)
- "requiredSkills: [value]" -> extract requiredSkills field (can be very long)
- "benefits: [value]" -> extract benefits field (can be very long)

IMPORTANT: For long field values (like description, requirements, requiredSkills, benefits), extract the ENTIRE content after the colon until the next field or end of prompt.

Return JSON with ALL fields that are explicitly specified in the user's prompt:

{
  "title": "Extract from jobTitle: field",
  "company": "Extract from company: field",
  "department": "Extract from department: field",
  "internalSPOC": "Extract from internalSPOC: field",
  "recruiter": "Extract from recruiter: field",
  "email": "Extract from email: field",
  "jobType": "Extract from jobType: field",
  "experienceLevel": "Extract from experienceLevel: field",
  "country": "Extract from country: field",
  "city": "Extract from city: field",
  "fullLocation": "Extract from fullLocation: field",
  "workType": "Extract from workType: field",
  "jobStatus": "Extract from jobStatus: field",
  "salaryMin": "Extract from salaryMin: field",
  "salaryMax": "Extract from salaryMax: field",
  "priority": "Extract from priority: field",
  "description": "Extract from description: field (include full text)",
  "requirements": "Extract from requirements: field (include full text)",
  "requiredSkills": "Extract from requiredSkills: field (include full text)",
  "benefits": "Extract from benefits: field (include full text)"
}

CRITICAL RULES:
1. Extract ALL fields that are specified in the user's prompt
2. Use the exact values provided after each field colon
3. If a field is specified like "company: [value]", it MUST be included in the response
4. For long fields (description, requirements, requiredSkills, benefits), extract ALL text after the colon
5. Return ONLY valid JSON with no additional text
6. Do not omit any fields that are explicitly mentioned in the prompt
7. Pay special attention to field specifications with colons (e.g., "company: Appit Software Solutions")
8. If a field value is very long, include the ENTIRE value, not just a summary

EXAMPLE EXTRACTION:
If user provides: "description: We are seeking a talented Senior AI Specialist to join our cutting-edge AI team..."
Extract: "We are seeking a talented Senior AI Specialist to join our cutting-edge AI team..." (full text)

If user provides: "requiredSkills: Python, TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy..."
Extract: "Python, TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy..." (full list)"""
    
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
