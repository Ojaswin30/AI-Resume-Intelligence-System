import pytest
from app.models.schemas import ParsedResume, EducationEntry, JDRequirement, RequirementCategory, ImportanceLevel
from app.services.embedding_engine import EmbeddingEngine
from app.services.matching_engine import MatchingEngine
from app.services.scoring_engine import ScoringEngine

def test_matching_ojaswin_profile():
    embedding_engine = EmbeddingEngine()
    matcher = MatchingEngine(embedding_engine)
    scorer = ScoringEngine()

    resume = ParsedResume(
        name="Ojaswin Aggarwal",
        email="ojaswin@example.com",
        skills=["Python", "NumPy", "Pandas", "NLP", "HuggingFace", "TensorFlow", "Keras", "SQL", "SQL Server", "SQLite", "Git", "Streamlit", "OpenCV", "AI Agent Tooling"],
        education=[EducationEntry(degree="B.Tech Computer Science Engineering (AI & ML Honours) Amity University 2022-2026 [CGPA 8.58]")],
        projects=[],
        experience=[],
        all_bullets=[
            "[Skills] Python, NumPy, Pandas, SQL",
            "[Skills] NLP, HuggingFace, TensorFlow, AI Agent Tooling",
            "[Skills] Git, Azure DevOps, Streamlit",
            "[Education] B.Tech Computer Science Engineering (AI & ML Honours) Amity University"
        ]
    )

    jd_requirements = [
        JDRequirement(
            requirement="Strong proficiency in Python (including standard libraries like NumPy and Pandas)",
            category=RequirementCategory.TECHNICAL_SKILLS,
            importance=ImportanceLevel.HIGH,
            core_entities=["Python", "NumPy", "Pandas"]
        ),
        JDRequirement(
            requirement="Solid understanding of core NLP (Natural Language Processing) concepts, tokenization, embeddings, and basic Transformer architecture",
            category=RequirementCategory.TECHNICAL_SKILLS,
            importance=ImportanceLevel.HIGH,
            core_entities=["NLP", "Transformers", "Embeddings"]
        ),
        JDRequirement(
            requirement="Comfortable using Git and GitHub for team collaboration",
            category=RequirementCategory.TECHNICAL_SKILLS,
            importance=ImportanceLevel.MEDIUM,
            core_entities=["Git", "GitHub"]
        ),
        JDRequirement(
            requirement="Basic understanding of vector databases (e.g., Pinecone, ChromaDB, or FAISS) and standard SQL",
            category=RequirementCategory.TECHNICAL_SKILLS,
            importance=ImportanceLevel.MEDIUM,
            core_entities=["FAISS", "SQL"]
        ),
        JDRequirement(
            requirement="Education: B.Tech / B.E. / M.Tech in Computer Science, AI or related discipline",
            category=RequirementCategory.EDUCATION,
            importance=ImportanceLevel.HIGH,
            core_entities=["B.Tech", "Computer Science"]
        ),
        JDRequirement(
            requirement="Experience: 0 years (Graduating students or recent graduates with hands-on project portfolios)",
            category=RequirementCategory.EXPERIENCE,
            importance=ImportanceLevel.HIGH,
            core_entities=["Fresher"]
        )
    ]

    matches = matcher.match_requirements(resume, jd_requirements)
    score_report = scorer.calculate_score(matches)

    # Assert accurate classification
    python_match = next(m for m in matches if "Python" in m.requirement)
    assert python_match.status.value == "STRONG"
    assert python_match.score >= 0.85

    edu_match = next(m for m in matches if "Education" in m.requirement)
    assert edu_match.status.value == "STRONG"
    assert edu_match.score >= 0.90

    git_match = next(m for m in matches if "Git" in m.requirement)
    assert git_match.status.value == "STRONG"

    sql_match = next(m for m in matches if "SQL" in m.requirement)
    assert sql_match.status.value in ["STRONG", "PARTIAL"]

    # Overall score should be in 75-95 range, NOT 33!
    assert score_report.overall_score >= 70.0
