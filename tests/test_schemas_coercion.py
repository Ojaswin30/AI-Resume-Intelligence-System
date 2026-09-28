import pytest
from app.models.schemas import GroundedAnalysis

def test_grounded_analysis_string_coercion():
    # Simulate SLM returning strings instead of arrays
    raw_data = {
        "summary_analysis": "Solid candidate.",
        "key_strengths": "1. Strong Python skills\n2. Experience with FastAPI",
        "critical_gaps": "Missing explicit FAISS experience",
        "honest_recommendations": "Given candidate experience, recommend highlighting AI projects.",
        "section_improvements": {
            "skills": "Ensure skill section matches JD.",
            "experience": "1. Detail backend scale\n2. Add metrics",
            "education": "Highlight degree"
        }
    }
    
    analysis = GroundedAnalysis(**raw_data)
    
    assert isinstance(analysis.key_strengths, list)
    assert len(analysis.key_strengths) >= 2
    assert isinstance(analysis.honest_recommendations, list)
    assert len(analysis.honest_recommendations) >= 1
    assert isinstance(analysis.section_improvements["skills"], list)
    assert isinstance(analysis.section_improvements["experience"], list)
    assert len(analysis.section_improvements["experience"]) >= 2
