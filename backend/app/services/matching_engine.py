import re
from typing import List, Dict, Set
from app.models.schemas import (
    ParsedResume, JDRequirement, EvidenceMatch, MatchStatus, RequirementCategory
)
from app.services.embedding_engine import EmbeddingEngine
from app.core.rules_loader import load_scoring_rules, load_skill_taxonomy

class MatchingEngine:
    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.scoring_rules = load_scoring_rules()
        self.taxonomy = load_skill_taxonomy()
        self.thresholds = self.scoring_rules.get("thresholds", {
            "strong_match": 0.72,
            "partial_match": 0.45,
            "weak_match": 0.25
        })

    def match_requirements(self, resume: ParsedResume, requirements: List[JDRequirement]) -> List[EvidenceMatch]:
        evidence_matches: List[EvidenceMatch] = []
        all_bullets = resume.all_bullets or self._synthesize_bullets(resume)

        for req in requirements:
            match = self._evaluate_single_requirement(req, resume, all_bullets)
            evidence_matches.append(match)

        return evidence_matches

    def _evaluate_single_requirement(
        self, req: JDRequirement, resume: ParsedResume, all_bullets: List[str]
    ) -> EvidenceMatch:
        req_text = req.requirement
        target_entities = set([e.lower() for e in req.core_entities])
        if not target_entities:
            target_entities = set([w.lower() for w in req_text.split() if len(w) > 3])

        # 1. Check exact / taxonomy keyword presence across resume
        resume_entities = self._extract_resume_entities(resume)
        matched_entities = []
        for ent in target_entities:
            aliases = self.taxonomy.get(ent, [ent])
            for alias in aliases:
                if any(alias in res_ent for res_ent in resume_entities):
                    matched_entities.append(ent)
                    break

        entity_match_ratio = len(matched_entities) / max(len(target_entities), 1)

        # 2. Semantic vector matching against all resume bullets
        semantic_scores = self.embedding_engine.compute_similarity(req_text, all_bullets)
        
        best_bullet = None
        best_semantic_score = 0.0
        best_section = None

        if semantic_scores:
            max_idx = int(max(range(len(semantic_scores)), key=lambda i: semantic_scores[i]))
            best_semantic_score = float(semantic_scores[max_idx])
            best_bullet = all_bullets[max_idx]
            
            # Determine section from bullet tag like [Role] or [Skills]
            if best_bullet.startswith("["):
                end_tag = best_bullet.find("]")
                if end_tag != -1:
                    best_section = best_bullet[1:end_tag]

        # 3. Blended deterministic score
        # 60% semantic similarity + 40% explicit entity match
        if target_entities:
            combined_score = (0.60 * best_semantic_score) + (0.40 * entity_match_ratio)
        else:
            combined_score = best_semantic_score

        # If high semantic score with verified entities, boost slightly
        if entity_match_ratio > 0.8 and best_semantic_score > 0.6:
            combined_score = min(1.0, combined_score + 0.1)

        # 4. Status classification based on configured thresholds
        if combined_score >= self.thresholds["strong_match"]:
            status = MatchStatus.STRONG
            notes = f"Strong direct evidence found with verified terms: {', '.join(matched_entities) if matched_entities else 'Semantics aligned'}."
        elif combined_score >= self.thresholds["partial_match"]:
            status = MatchStatus.PARTIAL
            notes = "Partial evidence or related domain experience detected; direct explicit keyword/metric is limited."
        else:
            status = MatchStatus.MISSING
            best_bullet = None
            notes = "No verifiable evidence or matching projects found for this requirement in the resume."

        return EvidenceMatch(
            requirement=req.requirement,
            category=req.category,
            importance=req.importance,
            status=status,
            score=round(combined_score, 4),
            best_matching_bullet=best_bullet,
            matched_section=best_section,
            matched_entities=list(set(matched_entities)),
            evidence_notes=notes
        )

    def _extract_resume_entities(self, resume: ParsedResume) -> Set[str]:
        entities = set()
        for s in resume.skills:
            entities.add(s.lower())
        for exp in resume.experience:
            entities.add(exp.role.lower())
            for b in exp.bullets:
                for word in re.findall(r"\b[A-Za-z0-9\.\+#]+\b", b):
                    entities.add(word.lower())
        for prj in resume.projects:
            entities.add(prj.title.lower())
            for t in prj.technologies:
                entities.add(t.lower())
            for b in prj.bullets:
                for word in re.findall(r"\b[A-Za-z0-9\.\+#]+\b", b):
                    entities.add(word.lower())
        return entities

    def _synthesize_bullets(self, resume: ParsedResume) -> List[str]:
        bullets = []
        if resume.skills:
            bullets.append(f"[Skills] {', '.join(resume.skills)}")
        for exp in resume.experience:
            for b in exp.bullets:
                bullets.append(f"[{exp.role}] {b}")
        for prj in resume.projects:
            for b in prj.bullets:
                bullets.append(f"[{prj.title}] {b}")
        return bullets
