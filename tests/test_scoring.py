import pytest
from app.models.schemas import EvidenceMatch, MatchStatus, RequirementCategory, ImportanceLevel
from app.services.scoring_engine import ScoringEngine

def test_deterministic_scoring_calculation():
    engine = ScoringEngine()
    
    matches = [
        EvidenceMatch(
            requirement="Python",
            category=RequirementCategory.TECHNICAL_SKILLS,
            importance=ImportanceLevel.HIGH,
            status=MatchStatus.STRONG,
            score=0.95
        ),
        EvidenceMatch(
            requirement="FastAPI",
            category=RequirementCategory.TECHNICAL_SKILLS,
            importance=ImportanceLevel.MEDIUM,
            status=MatchStatus.STRONG,
            score=0.90
        ),
        EvidenceMatch(
            requirement="RAG / Vector DB",
            category=RequirementCategory.TECHNICAL_SKILLS,
            importance=ImportanceLevel.HIGH,
            status=MatchStatus.MISSING,
            score=0.10
        ),
        EvidenceMatch(
            requirement="3+ years backend experience",
            category=RequirementCategory.EXPERIENCE,
            importance=ImportanceLevel.HIGH,
            status=MatchStatus.STRONG,
            score=0.85
        )
    ]
    
    score_report = engine.calculate_score(matches)
    
    assert 0.0 <= score_report.overall_score <= 100.0
    assert "technical_skills" in score_report.category_scores
    assert "experience" in score_report.category_scores
    assert score_report.category_scores["technical_skills"].score > 0
