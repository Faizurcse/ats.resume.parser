"""
Explanation Utils
Hardcoded explanations for fast matching without GPT calls.
Pre-written explanations based on similarity scores for skills, experience, and overall matching.
"""

from typing import Dict, Any

def get_skills_explanation(score: float, job_skills: str, candidate_skills: list) -> str:
    """
    Get hardcoded skills explanation based on similarity score.
    
    Args:
        score: Similarity score (0.0 to 1.0)
        job_skills: Job required skills
        candidate_skills: Candidate skills list
        
    Returns:
        str: Explanation for skills match
    """
    if score >= 0.9:
        return f"🎯 EXCELLENT SKILLS MATCH: {score:.1%} - Perfect alignment with required skills. Candidate has all essential technologies and frameworks needed for this role."
    elif score >= 0.8:
        return f"🌟 STRONG SKILLS MATCH: {score:.1%} - Excellent technical skills alignment. Candidate possesses most required skills with strong foundation in core technologies."
    elif score >= 0.7:
        return f"👍 GOOD SKILLS MATCH: {score:.1%} - Strong technical compatibility. Candidate has relevant skills and can quickly adapt to role requirements."
    elif score >= 0.6:
        return f"⚠️ MODERATE SKILLS MATCH: {score:.1%} - Reasonable skills alignment. Some training may be needed but candidate shows good potential."
    elif score >= 0.4:
        return f"🔶 FAIR SKILLS MATCH: {score:.1%} - Basic skills compatibility. Significant training required but candidate has transferable abilities."
    else:
        return f"❌ POOR SKILLS MATCH: {score:.1%} - Limited skills alignment. Extensive training needed for role success."

def get_experience_explanation(score: float, job_experience_level: str, candidate_experience: str) -> str:
    """
    Get hardcoded experience explanation based on similarity score.
    
    Args:
        score: Similarity score (0.0 to 1.0)
        job_experience_level: Job experience requirement
        candidate_experience: Candidate experience
        
    Returns:
        str: Explanation for experience match
    """
    if score >= 0.9:
        return f"🎯 EXCELLENT EXPERIENCE FIT: {score:.1%} - Perfect experience alignment. Candidate's background perfectly matches {job_experience_level} requirements."
    elif score >= 0.8:
        return f"🌟 STRONG EXPERIENCE FIT: {score:.1%} - Excellent experience compatibility. Candidate has solid foundation for {job_experience_level} role."
    elif score >= 0.7:
        return f"👍 GOOD EXPERIENCE FIT: {score:.1%} - Strong experience alignment. Candidate's background is well-suited for this position."
    elif score >= 0.6:
        return f"⚠️ MODERATE EXPERIENCE FIT: {score:.1%} - Reasonable experience match. Some mentoring may be beneficial for optimal performance."
    elif score >= 0.4:
        return f"🔶 FAIR EXPERIENCE FIT: {score:.1%} - Basic experience compatibility. Training and support will ensure success in this role."
    else:
        return f"❌ POOR EXPERIENCE FIT: {score:.1%} - Limited experience alignment. Significant development needed for role requirements."

def get_overall_explanation(score: float, job_title: str, candidate_name: str) -> str:
    """
    Get hardcoded overall explanation based on similarity score.
    
    Args:
        score: Similarity score (0.0 to 1.0)
        job_title: Job title
        candidate_name: Candidate name
        
    Returns:
        str: Explanation for overall match
    """
    if score >= 0.9:
        return f"🎯 EXCELLENT OVERALL MATCH: {score:.1%} - {candidate_name} is an ideal candidate for {job_title}. Perfect alignment across all criteria with immediate contribution potential."
    elif score >= 0.8:
        return f"🌟 STRONG OVERALL MATCH: {score:.1%} - {candidate_name} shows excellent compatibility with {job_title} requirements. Strong potential for role success."
    elif score >= 0.7:
        return f"👍 GOOD OVERALL MATCH: {score:.1%} - {candidate_name} is a solid fit for {job_title}. Good alignment with minor development areas."
    elif score >= 0.6:
        return f"⚠️ MODERATE OVERALL MATCH: {score:.1%} - {candidate_name} has reasonable compatibility with {job_title}. Some training and support recommended."
    elif score >= 0.4:
        return f"🔶 FAIR OVERALL MATCH: {score:.1%} - {candidate_name} shows basic alignment with {job_title}. Structured onboarding and mentoring needed."
    else:
        return f"❌ POOR OVERALL MATCH: {score:.1%} - {candidate_name} has limited compatibility with {job_title}. Extensive development program required."

def get_fit_status(score: float) -> str:
    """
    Get fit status based on similarity score.
    
    Args:
        score: Similarity score (0.0 to 1.0)
        
    Returns:
        str: FIT or NOT FIT
    """
    return "FIT" if score >= 0.7 else "NOT FIT"

def get_rating(score: float) -> str:
    """
    Get rating based on similarity score.
    
    Args:
        score: Similarity score (0.0 to 1.0)
        
    Returns:
        str: Rating (Excellent, Good, Fair, Poor)
    """
    if score >= 0.9:
        return "Excellent"
    elif score >= 0.8:
        return "Strong"
    elif score >= 0.7:
        return "Good"
    elif score >= 0.6:
        return "Moderate"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Poor"

# Pre-defined explanation templates for different score ranges
EXPLANATION_TEMPLATES = {
    "skills": {
        "excellent": [
            "🎯 PERFECT SKILLS ALIGNMENT: All required technologies present with advanced expertise",
            "🌟 OUTSTANDING TECHNICAL MATCH: Complete skill set with proven experience",
            "✅ IDEAL CANDIDATE: Perfect technology stack alignment for immediate productivity"
        ],
        "good": [
            "👍 STRONG SKILLS MATCH: Core technologies present with good foundation",
            "🚀 SOLID TECHNICAL FIT: Essential skills covered with growth potential",
            "💪 COMPETENT CANDIDATE: Strong technical background for role success"
        ],
        "moderate": [
            "⚠️ REASONABLE SKILLS FIT: Some required skills present, training needed",
            "🔧 DEVELOPING CANDIDATE: Basic skills with learning potential",
            "📚 GROWTH OPPORTUNITY: Foundation skills with development areas"
        ],
        "poor": [
            "❌ LIMITED SKILLS MATCH: Few required skills present, extensive training needed",
            "🔴 SKILL GAP: Significant development required for role success",
            "⚠️ HIGH TRAINING NEED: Major skill development program required"
        ]
    },
    "experience": {
        "excellent": [
            "🎯 PERFECT EXPERIENCE FIT: Ideal background for this role level",
            "🌟 OUTSTANDING EXPERIENCE: Proven track record in similar positions",
            "✅ IDEAL CANDIDATE: Perfect experience alignment for immediate impact"
        ],
        "good": [
            "👍 STRONG EXPERIENCE FIT: Solid background with relevant experience",
            "🚀 COMPETENT CANDIDATE: Good experience foundation for role success",
            "💪 EXPERIENCED PROFESSIONAL: Strong background with growth potential"
        ],
        "moderate": [
            "⚠️ REASONABLE EXPERIENCE FIT: Some relevant experience, mentoring beneficial",
            "🔧 DEVELOPING CANDIDATE: Basic experience with learning potential",
            "📚 GROWTH OPPORTUNITY: Foundation experience with development areas"
        ],
        "poor": [
            "❌ LIMITED EXPERIENCE FIT: Minimal relevant experience, extensive development needed",
            "🔴 EXPERIENCE GAP: Significant background development required",
            "⚠️ HIGH DEVELOPMENT NEED: Major experience building program required"
        ]
    },
    "overall": {
        "excellent": [
            "🎯 PERFECT MATCH: Ideal candidate with complete alignment across all criteria",
            "🌟 OUTSTANDING CANDIDATE: Excellent fit with immediate contribution potential",
            "✅ TOP CHOICE: Perfect match for role requirements and company culture"
        ],
        "good": [
            "👍 STRONG MATCH: Solid candidate with good alignment across key areas",
            "🚀 COMPETENT CANDIDATE: Strong fit with minor development areas",
            "💪 GOOD CHOICE: Well-suited candidate with growth potential"
        ],
        "moderate": [
            "⚠️ REASONABLE MATCH: Decent candidate with some alignment, training needed",
            "🔧 DEVELOPING CANDIDATE: Basic fit with development potential",
            "📚 GROWTH OPPORTUNITY: Foundation candidate with structured support needed"
        ],
        "poor": [
            "❌ LIMITED MATCH: Poor alignment across most criteria, extensive development needed",
            "🔴 MAJOR GAPS: Significant misalignment requiring comprehensive development",
            "⚠️ HIGH RISK: Limited compatibility with role requirements"
        ]
    }
}

def get_detailed_explanation(score: float, category: str, job_title: str = "", candidate_name: str = "") -> str:
    """
    Get detailed explanation using pre-defined templates.
    
    Args:
        score: Similarity score (0.0 to 1.0)
        category: Type of explanation (skills, experience, overall)
        job_title: Job title for context
        candidate_name: Candidate name for context
        
    Returns:
        str: Detailed explanation
    """
    if category not in EXPLANATION_TEMPLATES:
        return f"{category.title()} match: {score:.1%}"
    
    if score >= 0.9:
        level = "excellent"
    elif score >= 0.7:
        level = "good"
    elif score >= 0.5:
        level = "moderate"
    else:
        level = "poor"
    
    templates = EXPLANATION_TEMPLATES[category][level]
    import random
    base_explanation = random.choice(templates)
    
    # Add score and context
    if category == "overall" and job_title and candidate_name:
        return f"{base_explanation} ({score:.1%}) - {candidate_name} for {job_title}"
    else:
        return f"{base_explanation} ({score:.1%})"
