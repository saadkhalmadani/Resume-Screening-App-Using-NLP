import streamlit as st
from sentence_transformers import SentenceTransformer, util
import re
import string
from collections import Counter
from typing import List, Dict, Tuple

# Cache the SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load model
model = load_model()

# Extract named entities and skills from text using pattern matching
def extract_entities(text):
    entities = []
    
    # Simple tokenization without external dependencies
    # Remove punctuation and split by whitespace
    words = text.translate(str.maketrans('', '', string.punctuation)).split()
    
    # Pattern-based skill extraction
    skill_patterns = [
        r'\b(?:Python|Java|JavaScript|C\+\+|C#|SQL|HTML|CSS|React|Angular|Vue|Node\.js|Django|Flask|FastAPI|Spring|\.NET)\b',
        r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Git|Jenkins|CI/CD|DevOps|Agile|Scrum)\b',
        r'\b(?:Machine Learning|Deep Learning|AI|NLP|Data Science|Analytics|Statistics)\b',
        r'\b(?:TensorFlow|PyTorch|Scikit-learn|Pandas|NumPy|Matplotlib|Tableau|Power BI)\b',
        r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|Spark|Hadoop|Kafka)\b'
    ]
    
    for pattern in skill_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append((match.group(), 'SKILL'))
    
    # Extract organizations (simple pattern matching)
    org_patterns = [
        r'\b(?:Google|Microsoft|Amazon|Apple|Meta|Netflix|Tesla|IBM|Oracle|Salesforce)\b',
        r'\b(?:University|College|Institute|Corp|Corporation|Inc|Ltd|LLC)\b'
    ]
    
    for pattern in org_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append((match.group(), 'ORG'))
    
    # Extract person names (simple heuristic - capitalized words)
    person_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
    matches = re.finditer(person_pattern, text)
    for match in matches:
        name = match.group()
        # Filter out common false positives
        if not any(word in name.lower() for word in ['data', 'machine', 'computer', 'software', 'senior']):
            entities.append((name, 'PERSON'))
    
    return entities

# Compute similarity score between resume and job description
def match_resume_to_job(resume_text, job_description):
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_job = model.encode(job_description, convert_to_tensor=True)
    similarity = util.cos_sim(emb_resume, emb_job)
    return float(similarity[0][0])

# Enhanced keyword density analysis
def analyze_keyword_density(text: str, job_description: str) -> Dict:
    """Analyze keyword density and frequency in resume vs job description"""
    
    # Extract keywords from job description
    job_words = re.findall(r'\b[a-zA-Z]{3,}\b', job_description.lower())
    job_counter = Counter(job_words)
    
    # Get top keywords from job description (excluding common words)
    common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'does', 'let', 'man', 'put', 'say', 'she', 'too', 'use', 'will', 'with', 'work', 'experience', 'position', 'candidate', 'team', 'company'}
    
    important_keywords = {word: count for word, count in job_counter.most_common(20) 
                         if word not in common_words and len(word) > 3}
    
    # Analyze keyword presence in resume
    resume_lower = text.lower()
    keyword_analysis = {}
    
    for keyword, job_freq in important_keywords.items():
        resume_freq = len(re.findall(r'\b' + re.escape(keyword) + r'\b', resume_lower))
        keyword_analysis[keyword] = {
            'job_frequency': job_freq,
            'resume_frequency': resume_freq,
            'density_score': min(resume_freq / max(job_freq, 1), 1.0)
        }
    
    return keyword_analysis

# Missing skills recommendation
def recommend_missing_skills(resume_text: str, job_description: str) -> List[str]:
    """Recommend skills that appear in job description but not in resume"""
    
    # Define comprehensive skill categories
    all_skills = [
        # Programming Languages
        'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust', 'Ruby', 'PHP', 'Swift', 'Kotlin',
        'Scala', 'R', 'MATLAB', 'Perl', 'Shell', 'Bash', 'PowerShell',
        
        # Web Technologies
        'HTML', 'CSS', 'React', 'Angular', 'Vue', 'Node.js', 'Express', 'Django', 'Flask', 'FastAPI', 'Spring',
        'ASP.NET', 'Laravel', 'Bootstrap', 'jQuery', 'Webpack', 'Babel',
        
        # Databases
        'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Cassandra', 'DynamoDB', 'Elasticsearch', 'Neo4j',
        'Oracle', 'SQLite', 'MariaDB',
        
        # Cloud & DevOps
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Jenkins', 'GitLab CI', 'GitHub Actions', 'Terraform',
        'Ansible', 'Chef', 'Puppet', 'Helm', 'Istio',
        
        # Data Science & ML
        'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy',
        'Matplotlib', 'Seaborn', 'Jupyter', 'Apache Spark', 'Hadoop', 'Kafka', 'Airflow',
        
        # Methodologies
        'Agile', 'Scrum', 'Kanban', 'DevOps', 'CI/CD', 'TDD', 'BDD', 'Microservices', 'RESTful', 'GraphQL'
    ]
    
    resume_lower = resume_text.lower()
    job_lower = job_description.lower()
    
    missing_skills = []
    
    for skill in all_skills:
        skill_pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        
        # Check if skill is mentioned in job description but not in resume
        if (re.search(skill_pattern, job_lower) and 
            not re.search(skill_pattern, resume_lower)):
            missing_skills.append(skill)
    
    return missing_skills[:10]  # Return top 10 missing skills

# Experience level detection
def detect_experience_level(resume_text: str) -> Tuple[str, float]:
    """Detect experience level based on resume content"""
    
    # Experience indicators
    senior_indicators = [
        'senior', 'lead', 'principal', 'architect', 'director', 'manager', 'head of',
        'years of experience', 'expert', 'advanced', 'mentor', 'team lead'
    ]
    
    mid_indicators = [
        'experienced', 'proficient', 'skilled', 'specialist', 'developer',
        'engineer', 'analyst', 'consultant'
    ]
    
    junior_indicators = [
        'junior', 'entry', 'graduate', 'intern', 'trainee', 'associate',
        'recent graduate', 'new grad', 'fresher'
    ]
    
    text_lower = resume_text.lower()
    
    # Count indicators
    senior_count = sum(1 for indicator in senior_indicators if indicator in text_lower)
    mid_count = sum(1 for indicator in mid_indicators if indicator in text_lower)
    junior_count = sum(1 for indicator in junior_indicators if indicator in text_lower)
    
    # Extract years of experience
    years_pattern = r'(\d+)[\s-]*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)'
    years_matches = re.findall(years_pattern, text_lower)
    
    max_years = 0
    if years_matches:
        max_years = max(int(year) for year in years_matches)
    
    # Determine level based on years and indicators
    if max_years >= 7 or senior_count >= 3:
        return "Senior (7+ years)", 0.9
    elif max_years >= 3 or (mid_count >= 2 and senior_count >= 1):
        return "Mid-Level (3-6 years)", 0.7
    elif max_years >= 1 or junior_count >= 1:
        return "Junior (1-2 years)", 0.4
    else:
        return "Entry Level (0-1 year)", 0.2

# Detailed scoring breakdown
def get_detailed_score_breakdown(resume_text: str, job_description: str) -> Dict:
    """Provide detailed scoring breakdown with multiple factors"""
    
    # Get basic similarity score
    similarity_score = match_resume_to_job(resume_text, job_description)
    
    # Get keyword analysis
    keyword_analysis = analyze_keyword_density(resume_text, job_description)
    
    # Calculate keyword match score
    if keyword_analysis:
        keyword_scores = [data['density_score'] for data in keyword_analysis.values()]
        keyword_match_score = sum(keyword_scores) / len(keyword_scores)
    else:
        keyword_match_score = 0.0
    
    # Get experience level
    experience_level, experience_score = detect_experience_level(resume_text)
    
    # Calculate skills coverage
    missing_skills = recommend_missing_skills(resume_text, job_description)
    total_skills_in_job = len(re.findall(r'\b(?:Python|Java|JavaScript|SQL|AWS|React|Docker|Kubernetes|Machine Learning|Data Science|Agile|Scrum)\b', job_description, re.IGNORECASE))
    skills_coverage = max(0, 1 - (len(missing_skills) / max(total_skills_in_job, 1)))
    
    # Calculate overall score with weights
    overall_score = (
        similarity_score * 0.4 +           # 40% semantic similarity
        keyword_match_score * 0.3 +        # 30% keyword matching
        skills_coverage * 0.2 +             # 20% skills coverage
        experience_score * 0.1              # 10% experience level
    )
    
    return {
        'overall_score': overall_score,
        'similarity_score': similarity_score,
        'keyword_match_score': keyword_match_score,
        'skills_coverage': skills_coverage,
        'experience_score': experience_score,
        'experience_level': experience_level,
        'missing_skills_count': len(missing_skills)
    }

# Generate comprehensive improvement recommendations
def generate_improvement_recommendations(resume_text: str, job_description: str, score_breakdown: Dict) -> Dict:
    """Generate specific, actionable recommendations to improve resume score"""
    
    recommendations = {
        'high_priority': [],
        'medium_priority': [],
        'low_priority': [],
        'specific_examples': [],
        'keyword_suggestions': [],
        'format_improvements': []
    }
    
    missing_skills = recommend_missing_skills(resume_text, job_description)
    keyword_analysis = analyze_keyword_density(resume_text, job_description)
    
    # High Priority Recommendations (biggest impact on score)
    if score_breakdown['skills_coverage'] < 0.6:
        recommendations['high_priority'].append({
            'title': '🎯 Add Missing Critical Skills',
            'description': f'Your resume is missing {len(missing_skills)} key skills mentioned in the job description.',
            'action': f'Add these skills to your resume: {", ".join(missing_skills[:5])}',
            'impact': 'High (+15-25 points)',
            'example': 'Instead of "worked with databases" → "Developed SQL queries and managed PostgreSQL databases"'
        })
    
    if score_breakdown['keyword_match_score'] < 0.5:
        low_keywords = [k for k, v in keyword_analysis.items() if v['density_score'] < 0.3][:3]
        recommendations['high_priority'].append({
            'title': '🔑 Increase Keyword Density',
            'description': 'Your resume doesn\'t mention key terms frequently enough.',
            'action': f'Use these important keywords more often: {", ".join(low_keywords)}',
            'impact': 'High (+10-20 points)',
            'example': 'Mention "Python" 3-4 times instead of once, in different contexts'
        })
    
    if score_breakdown['similarity_score'] < 0.4:
        recommendations['high_priority'].append({
            'title': '📝 Improve Content Relevance',
            'description': 'Your resume content doesn\'t closely match the job requirements.',
            'action': 'Rewrite job descriptions to mirror the language used in the job posting',
            'impact': 'Very High (+20-30 points)',
            'example': 'Use exact phrases from job description in your experience bullets'
        })
    
    # Medium Priority Recommendations
    if score_breakdown['experience_score'] < 0.6:
        recommendations['medium_priority'].append({
            'title': '👨‍💼 Highlight Seniority Level',
            'description': 'Your experience level isn\'t clearly communicated.',
            'action': 'Add specific years of experience and leadership examples',
            'impact': 'Medium (+8-15 points)',
            'example': '"5+ years Python development" or "Led team of 3 developers"'
        })
    
    if len(missing_skills) > 0:
        recommendations['medium_priority'].append({
            'title': '🛠️ Expand Technical Skills Section',
            'description': 'Consider adding a dedicated skills section with missing technologies.',
            'action': f'Create a "Technical Skills" section including: {", ".join(missing_skills[:3])}',
            'impact': 'Medium (+10-15 points)',
            'example': 'Technical Skills: Python, SQL, AWS, Docker, React, Git'
        })
    
    # Low Priority Recommendations
    recommendations['low_priority'].append({
        'title': '📄 Optimize Resume Format',
        'description': 'Ensure your resume follows ATS-friendly formatting.',
        'action': 'Use standard headings, bullet points, and avoid complex formatting',
        'impact': 'Low (+3-8 points)',
        'example': 'Use headings like "Experience", "Education", "Skills" instead of creative titles'
    })
    
    if 'university' not in resume_text.lower() and 'college' not in resume_text.lower():
        recommendations['low_priority'].append({
            'title': '🎓 Add Education Details',
            'description': 'Include relevant education information.',
            'action': 'Add degree, university, and graduation year',
            'impact': 'Low (+2-5 points)',
            'example': 'Bachelor of Science in Computer Science, XYZ University (2020)'
        })
    
    # Specific Examples and Templates
    recommendations['specific_examples'] = [
        {
            'category': 'Achievement-Focused Bullets',
            'before': 'Worked on web development projects',
            'after': 'Developed 5+ responsive web applications using React and Node.js, improving user engagement by 30%',
            'why': 'Specific numbers and technologies show measurable impact'
        },
        {
            'category': 'Technical Implementation',
            'before': 'Used Python for data analysis',
            'after': 'Implemented Python-based ETL pipelines using Pandas and SQL, processing 10M+ records daily',
            'why': 'Shows scale, specific tools, and business impact'
        },
        {
            'category': 'Leadership and Collaboration',
            'before': 'Worked with team members',
            'after': 'Collaborated with cross-functional team of 8 developers and designers to deliver features 25% faster',
            'why': 'Quantifies team size and improvement metrics'
        }
    ]
    
    # Keyword Usage Suggestions
    if keyword_analysis:
        recommendations['keyword_suggestions'] = [
            {
                'keyword': keyword,
                'current_usage': data['resume_frequency'],
                'suggested_usage': max(3, data['job_frequency']),
                'context_ideas': get_keyword_context_suggestions(keyword)
            }
            for keyword, data in list(keyword_analysis.items())[:5]
            if data['density_score'] < 0.5
        ]
    
    return recommendations

def get_keyword_context_suggestions(keyword: str) -> List[str]:
    """Get context suggestions for using specific keywords"""
    
    contexts = {
        'python': [
            'Developed Python applications using Django/Flask',
            'Automated workflows with Python scripts',
            'Implemented data analysis pipelines in Python'
        ],
        'sql': [
            'Designed and optimized SQL queries',
            'Managed SQL Server/PostgreSQL databases',
            'Created SQL reports and dashboards'
        ],
        'aws': [
            'Deployed applications on AWS cloud platform',
            'Utilized AWS services (EC2, S3, Lambda)',
            'Implemented AWS infrastructure automation'
        ],
        'machine learning': [
            'Developed machine learning models using scikit-learn',
            'Implemented machine learning algorithms for prediction',
            'Applied machine learning techniques to business problems'
        ],
        'react': [
            'Built responsive UIs with React.js',
            'Developed React components and hooks',
            'Created React-based single-page applications'
        ]
    }
    
    return contexts.get(keyword.lower(), [
        f'Utilized {keyword} in project development',
        f'Implemented solutions using {keyword}',
        f'Applied {keyword} best practices'
    ])

# Generate industry-specific advice
def get_industry_advice(job_description: str) -> Dict:
    """Provide industry-specific resume advice based on job description"""
    
    job_lower = job_description.lower()
    advice = {
        'industry': 'General',
        'key_trends': [],
        'must_have_skills': [],
        'nice_to_have_skills': [],
        'format_tips': []
    }
    
    # Detect industry
    if any(word in job_lower for word in ['data scientist', 'data analyst', 'analytics', 'machine learning']):
        advice['industry'] = 'Data Science'
        advice['key_trends'] = ['AI/ML expertise', 'Big Data processing', 'Cloud platforms', 'Statistical analysis']
        advice['must_have_skills'] = ['Python/R', 'SQL', 'Statistics', 'Data visualization']
        advice['nice_to_have_skills'] = ['Deep Learning', 'MLOps', 'Spark', 'Tableau']
        advice['format_tips'] = ['Include portfolio links', 'Showcase data projects', 'Quantify model performance']
        
    elif any(word in job_lower for word in ['software engineer', 'developer', 'programming']):
        advice['industry'] = 'Software Development'
        advice['key_trends'] = ['Cloud-native development', 'DevOps integration', 'Microservices', 'API development']
        advice['must_have_skills'] = ['Programming languages', 'Version control (Git)', 'Testing', 'Debugging']
        advice['nice_to_have_skills'] = ['Docker', 'Kubernetes', 'CI/CD', 'Cloud platforms']
        advice['format_tips'] = ['Include GitHub profile', 'Highlight code quality', 'Show project architecture']
        
    elif any(word in job_lower for word in ['devops', 'cloud', 'infrastructure', 'sre']):
        advice['industry'] = 'DevOps/Cloud'
        advice['key_trends'] = ['Infrastructure as Code', 'Container orchestration', 'Monitoring', 'Security']
        advice['must_have_skills'] = ['Cloud platforms', 'Automation tools', 'Scripting', 'Monitoring']
        advice['nice_to_have_skills'] = ['Terraform', 'Kubernetes', 'Prometheus', 'Security tools']
        advice['format_tips'] = ['Emphasize automation', 'Show infrastructure scale', 'Include certifications']
        
    elif any(word in job_lower for word in ['frontend', 'ui', 'ux', 'web developer']):
        advice['industry'] = 'Frontend Development'
        advice['key_trends'] = ['Modern frameworks', 'Mobile-first design', 'Performance optimization', 'Accessibility']
        advice['must_have_skills'] = ['HTML/CSS/JavaScript', 'React/Vue/Angular', 'Responsive design', 'Browser tools']
        advice['nice_to_have_skills'] = ['TypeScript', 'Testing frameworks', 'Build tools', 'Design systems']
        advice['format_tips'] = ['Include portfolio/demos', 'Show design skills', 'Highlight user experience']
    
    return advice

