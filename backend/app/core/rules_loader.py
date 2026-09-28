import json
import yaml
from pathlib import Path
from typing import Dict, Any

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

def load_scoring_rules() -> Dict[str, Any]:
    file_path = CONFIG_DIR / "scoring_rules.yaml"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "scoring_weights": {
            "technical_skills": 0.40,
            "experience": 0.25,
            "projects": 0.20,
            "education": 0.10,
            "certifications": 0.05
        },
        "thresholds": {"strong_match": 0.72, "partial_match": 0.45, "weak_match": 0.25},
        "importance_multipliers": {"high": 1.2, "medium": 1.0, "low": 0.7, "bonus": 0.5}
    }

def load_resume_rules() -> Dict[str, Any]:
    file_path = CONFIG_DIR / "resume_rules.yaml"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "resume_rules": {
            "prohibit_hallucination": True,
            "allow_reordering": True,
            "allow_rephrasing": True,
            "allow_new_claims": False,
            "require_evidence_for_metrics": True
        }
    }

def load_skill_taxonomy() -> Dict[str, list]:
    file_path = CONFIG_DIR / "skill_taxonomy.json"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_system_prompts() -> Dict[str, str]:
    file_path = CONFIG_DIR / "system_prompts.yaml"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}
