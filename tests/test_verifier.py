import pytest
from app.models.schemas import ParsedResume, ExperienceEntry, ProjectEntry
from app.services.claim_verifier import ClaimVerifier

def test_verifier_catches_hallucinated_technologies():
    verifier = ClaimVerifier()
    
    resume = ParsedResume(
        name="Alex Dev",
        skills=["Python", "FastAPI", "SQL"],
        experience=[
            ExperienceEntry(
                role="Backend Developer",
                bullets=["Built REST APIs using Python and FastAPI."]
            )
        ],
        projects=[
            ProjectEntry(
                title="Chatbot",
                bullets=["Developed a basic rule-based NLP bot using Python."]
            )
        ]
    )
    
    original_bullet = "Built REST APIs using Python and FastAPI."
    
    # 1. Honest rewrite with existing facts
    honest_rewrite = "Engineered high-throughput REST APIs utilizing Python and FastAPI."
    passed_honest, warnings_honest = verifier.verify_bullet_rewrite(original_bullet, honest_rewrite, resume)
    assert passed_honest is True
    assert len(warnings_honest) == 0

    # 2. Dishonest rewrite inventing Kubernetes and FAISS
    hallucinated_rewrite = "Architected distributed Kubernetes microservices with FAISS vector indexing."
    passed_fake, warnings_fake = verifier.verify_bullet_rewrite(original_bullet, hallucinated_rewrite, resume)
    assert passed_fake is False
    assert len(warnings_fake) > 0
    assert any("Kubernetes" in w or "FAISS" in w for w in warnings_fake)
