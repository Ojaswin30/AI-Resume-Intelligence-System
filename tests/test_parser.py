import pytest
from app.services.pdf_parser import ResumePDFParser

SAMPLE_RESUME_TEXT = """
Jane Doe
jane.doe@example.com

SUMMARY
Experienced software engineer focused on Python backend services and data pipelines.

SKILLS
Languages: Python, SQL, JavaScript
Frameworks: FastAPI, PyTorch, HuggingFace
Tools: Docker, Git

EXPERIENCE
Backend Software Engineer | TechCorp (2022 - Present)
• Built high-performance microservices using FastAPI and PostgreSQL.
• Optimized database queries, reducing average API response latency by 35%.

PROJECTS
NLP Sentiment Analyzer
• Created an NLP pipeline using Python and HuggingFace Transformers for document classification.

EDUCATION
Bachelor of Science in Computer Science, University of California (2018 - 2022)
"""

def test_resume_parser_structured_extraction():
    parser = ResumePDFParser()
    parsed = parser.parse_text(SAMPLE_RESUME_TEXT)

    assert parsed.name == "Jane Doe"
    assert parsed.email == "jane.doe@example.com"
    assert "Python" in parsed.skills or "python" in [s.lower() for s in parsed.skills]
    assert len(parsed.experience) > 0
    assert len(parsed.projects) > 0
    assert len(parsed.education) > 0
    assert len(parsed.all_bullets) >= 2
