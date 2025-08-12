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
        """Generate job posting data from a prompt."""
        try:
            # Analyze prompt type to determine generation strategy
            if self._is_single_skill_search(prompt):
                # Single skill search (e.g., "java") - return specific job posting
                system_prompt = self._get_single_skill_prompt()
                logger.info(f"Using single skill prompt for: {prompt}")
            elif self._is_generic_word_search(prompt):
                # Generic word search - return n skills and 1000 char description
                system_prompt = self._get_generic_word_prompt()
                logger.info(f"Using generic word prompt for: {prompt}")
            elif self._is_detailed_prompt(prompt):
                # Detailed prompt - return comprehensive job posting
                system_prompt = self._get_detailed_prompt()
                logger.info(f"Using detailed prompt for: {prompt}")
            else:
                # Simple prompt - return basic job posting
                system_prompt = self._get_simple_prompt()
                logger.info(f"Using simple prompt for: {prompt}")
            
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate job posting: {prompt}"}
                ],
                max_tokens=settings.OPENAI_MAX_TOKENS,
                temperature=0.3,  # Lower temperature for more consistent JSON
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            response_content = response.choices[0].message.content.strip()
            logger.info(f"Raw OpenAI response: {response_content[:200]}...")
            
            # Clean and parse the response
            job_data = self._parse_and_clean_response(response_content)
            
            logger.info(f"Generated job posting with {len(job_data)} fields")
            return job_data
            
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
    
    def _is_generic_word_search(self, prompt: str) -> bool:
        """Check if prompt is just generic words without specific job details"""
        clean_prompt = prompt.strip().lower()
        
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
        return """You are a job posting generator for single skill searches. 
When given a single skill like "java", "python", etc., create a specific job posting for that role.

Return JSON in this exact format:

{
  "title": "[Skill] Developer",
  "description": "Job description based on role",
  "requirements": "Job requirements based on role", 
  "requiredSkills": "Required skills based on role"
}

CRITICAL RULES:
1. For the title, use the skill name + "Developer" (e.g., "Java Developer", "Python Developer")
2. Generate a realistic job description for that specific role
3. Generate realistic requirements for that specific role
4. Generate realistic required skills for that specific role
5. Return ONLY valid JSON with no additional text
6. Make the content specific to the skill mentioned (e.g., if "java", mention Java-specific technologies)"""
    
    def _get_generic_word_prompt(self) -> str:
        """Get system prompt for generic word searches."""
        return """You are a job posting generator for generic word searches.
When given generic words, generate a comprehensive job posting with multiple skills and detailed descriptions.

Return JSON in this exact format:

{
  "title": "Software Developer",
  "description": "[Generate a detailed job description with approximately 1000 characters]",
  "requirements": "[Generate detailed job requirements with approximately 1000 characters]",
  "requiredSkills": "[Generate 8-12 specific technical skills, separated by commas]"
}

CRITICAL RULES:
1. Generate a detailed description with approximately 1000 characters
2. Generate detailed requirements with approximately 1000 characters  
3. Generate 8-12 specific technical skills, separated by commas
4. Make the content comprehensive and professional
5. Return ONLY valid JSON with no additional text
6. Ensure the description and requirements are detailed enough to fill 1000 characters"""
    
    def _get_simple_prompt(self) -> str:
        """Get system prompt for simple job posting generation."""
        return """You are a job posting generator. Create job postings in this exact JSON format.

IMPORTANT: Extract ALL fields that are specified in the user's prompt using patterns like:
- "company: [value]" -> extract company field
- "department: [value]" -> extract department field
- "jobTitle: [value]" -> extract title field
- etc.

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
  "description": "Job description based on role",
  "requirements": "Job requirements based on role",
  "requiredSkills": "Required skills based on role",
  "benefits": "Company benefits (from benefits:)"
}

CRITICAL RULES:
1. Extract ALL fields that are specified in the user's prompt
2. Use the exact values provided after each field colon
3. If a field is specified like "company: [value]", it MUST be included in the response
4. Focus on the core job information: title, description, requirements, skills
5. Return ONLY valid JSON with no additional text
6. Pay special attention to field specifications with colons (e.g., "company: [value]")"""
    
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
            "description": "We are seeking a talented software developer to join our dynamic team. The ideal candidate will have strong programming skills, experience with modern development practices, and a passion for creating high-quality software solutions. You will work on exciting projects, collaborate with cross-functional teams, and contribute to the development of innovative applications that drive business value.",
            "requirements": "Bachelor's degree in Computer Science, Software Engineering, or related field. Minimum 3+ years of experience in software development. Proficiency in multiple programming languages (Java, Python, JavaScript, etc.). Experience with modern frameworks and technologies. Strong problem-solving skills and attention to detail. Excellent communication and teamwork abilities. Experience with version control systems (Git) and agile development methodologies. Knowledge of database design and SQL. Understanding of software testing principles and practices.",
            "requiredSkills": "Java, Python, JavaScript, React, Node.js, SQL, Git, Docker, AWS, REST APIs, Agile, Testing"
        }
