import pytest
from app.main import app
from app.api.routes import router, orchestrator
from app.services.orchestrator import ResumeIntelligenceOrchestrator

def test_full_app_and_orchestrator_initialization():
    assert app is not None
    assert router is not None
    assert orchestrator is not None
    assert isinstance(orchestrator, ResumeIntelligenceOrchestrator)
