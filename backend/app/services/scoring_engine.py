from typing import List, Dict
from app.models.schemas import (
    EvidenceMatch, DeterministicScore, CategoryScoreBreakdown, RequirementCategory
)
from app.core.rules_loader import load_scoring_rules

class ScoringEngine:
    def __init__(self):
        self.rules = load_scoring_rules()
        self.weights = self.rules.get("scoring_weights", {
            "technical_skills": 0.40,
            "experience": 0.25,
            "projects": 0.20,
            "education": 0.10,
            "certifications": 0.05
        })
        self.importance_multipliers = self.rules.get("importance_multipliers", {
            "high": 1.2,
            "medium": 1.0,
            "low": 0.7,
            "bonus": 0.5
        })

    def calculate_score(self, evidence_matches: List[EvidenceMatch]) -> DeterministicScore:
        if not evidence_matches:
            return DeterministicScore(
                overall_score=0.0,
                category_scores={},
                evidence_matrix=[]
            )

        # Group matches by category
        cat_matches: Dict[str, List[EvidenceMatch]] = {}
        for match in evidence_matches:
            cat_key = match.category.value if hasattr(match.category, "value") else str(match.category)
            cat_matches.setdefault(cat_key, []).append(match)

        category_breakdowns: Dict[str, CategoryScoreBreakdown] = {}
        total_weighted_score = 0.0
        active_weights_sum = 0.0

        for cat_name, cat_weight in self.weights.items():
            matches = cat_matches.get(cat_name, [])
            if not matches:
                # If no requirements were defined for this category in the JD, skip weighting penalty
                continue

            # Calculate weighted average score for this category
            weighted_sum = 0.0
            multiplier_sum = 0.0

            for m in matches:
                imp = m.importance.value if hasattr(m.importance, "value") else str(m.importance)
                mult = self.importance_multipliers.get(imp, 1.0)
                weighted_sum += (m.score * 100.0) * mult
                multiplier_sum += mult

            cat_score = round(weighted_sum / max(multiplier_sum, 1.0), 1)
            
            category_breakdowns[cat_name] = CategoryScoreBreakdown(
                category=RequirementCategory(cat_name) if cat_name in RequirementCategory._value2member_map_ else RequirementCategory.TECHNICAL_SKILLS,
                score=cat_score,
                weight=cat_weight,
                requirements_count=len(matches)
            )

            total_weighted_score += cat_score * cat_weight
            active_weights_sum += cat_weight

        # Normalize overall score based on present categories
        final_overall_score = round(total_weighted_score / max(active_weights_sum, 0.01), 1)

        return DeterministicScore(
            overall_score=min(100.0, max(0.0, final_overall_score)),
            category_scores=category_breakdowns,
            evidence_matrix=evidence_matches
        )
