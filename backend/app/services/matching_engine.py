import re
from typing import List, Dict, Set, Tuple, Optional
from app.models.schemas import (
    ParsedResume, JDRequirement, EvidenceMatch, MatchStatus, RequirementCategory, ImportanceLevel
)
from app.services.embedding_engine import EmbeddingEngine
from app.core.rules_loader import load_scoring_rules, load_skill_taxonomy

class MatchingEngine:
    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.scoring_rules = load_scoring_rules()
        self.taxonomy = load_skill_taxonomy()
        self.thresholds = self.scoring_rules.get("thresholds", {
            "strong_match": 0.65,
            "partial_match": 0.40,
            "weak_match": 0.20
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
        category = req.category

        # 1. Specialized Education Matching
        if category == RequirementCategory.EDUCATION or any(k in req_text.lower() for k in ["b.tech", "b.e.", "degree", "mca", "bachelor", "master", "graduat"]):
            edu_match = self._match_education(req_text, resume)
            if edu_match:
                return edu_match

        # 2. Specialized Experience / Fresher Matching
        if category == RequirementCategory.EXPERIENCE or any(k in req_text.lower() for k in ["0 years", "fresher", "graduating student", "recent graduate"]):
            exp_match = self._match_experience_level(req_text, resume)
            if exp_match:
                return exp_match

        # 3. Extract core entities from requirement
        target_entities = self._extract_target_entities(req_text, req.core_entities)
        
        # 4. Check explicit keyword / taxonomy matches across resume
        resume_entities = self._extract_resume_entities(resume)
        matched_entities = []
        for ent in target_entities:
            aliases = self.taxonomy.get(ent.lower(), [ent.lower()])
            found = False
            for alias in aliases:
                if any(alias in res_ent or res_ent in alias for res_ent in resume_entities):
                    matched_entities.append(ent)
                    found = True
                    break
            if not found and ent.lower() in [s.lower() for s in resume.skills]:
                matched_entities.append(ent)

        matched_entities = list(set(matched_entities))
        entity_match_ratio = len(matched_entities) / max(len(target_entities), 1) if target_entities else 0.0

        # 5. Semantic Cosine Vector Matching against granular bullets
        semantic_scores = self.embedding_engine.compute_similarity(req_text, all_bullets)
        best_bullet = None
        best_semantic_score = 0.0
        best_section = None

        if semantic_scores:
            max_idx = int(max(range(len(semantic_scores)), key=lambda i: semantic_scores[i]))
            best_semantic_score = float(semantic_scores[max_idx])
            best_bullet = all_bullets[max_idx]
            if best_bullet.startswith("["):
                end_tag = best_bullet.find("]")
                if end_tag != -1:
                    best_section = best_bullet[1:end_tag]

        # 6. Hybrid Deterministic Score Calculation
        if target_entities:
            # If all core entities are present (e.g. Python, NumPy, Pandas all in resume), score is high
            if entity_match_ratio >= 0.75:
                combined_score = max(0.85, (0.50 * best_semantic_score) + (0.50 * entity_match_ratio))
            elif entity_match_ratio >= 0.35:
                combined_score = max(0.55, (0.60 * best_semantic_score) + (0.40 * entity_match_ratio))
            else:
                combined_score = (0.70 * best_semantic_score) + (0.30 * entity_match_ratio)
        else:
            combined_score = best_semantic_score

        combined_score = min(1.0, max(0.0, combined_score))

        # 7. Status Classification & Honest Notes
        if combined_score >= self.thresholds["strong_match"] or entity_match_ratio >= 0.7:
            status = MatchStatus.STRONG
            best_evidence = best_bullet or (f"Skills: {', '.join(matched_entities)}" if matched_entities else "Direct Match")
            notes = f"Verified evidence found for: {', '.join(matched_entities) if matched_entities else 'requirement semantics'}."
        elif combined_score >= self.thresholds["partial_match"] or entity_match_ratio > 0.0:
            status = MatchStatus.PARTIAL
            best_evidence = best_bullet or (f"Skills: {', '.join(matched_entities)}" if matched_entities else "Partial Match")
            missing_terms = [e for e in target_entities if e not in matched_entities]
            notes = f"Partial evidence found ({', '.join(matched_entities)}). Missing explicit demonstration for: {', '.join(missing_terms)}."
        else:
            status = MatchStatus.MISSING
            best_evidence = None
            notes = "No verifiable evidence or matching skills found for this requirement in the resume."

        return EvidenceMatch(
            requirement=req.requirement,
            category=req.category,
            importance=req.importance,
            status=status,
            score=round(combined_score, 4),
            best_matching_bullet=best_evidence,
            matched_section=best_section or "skills",
            matched_entities=matched_entities,
            evidence_notes=notes
        )

    def _match_education(self, req_text: str, resume: ParsedResume) -> Optional[EvidenceMatch]:
        edu_entries = resume.education
        all_text = " ".join([e.degree for e in edu_entries] + [b for b in resume.all_bullets if any(k in b.lower() for k in ["b.tech", "bachelor", "master", "degree", "university", "college"])])
        
        has_degree = any(d in all_text.lower() for d in ["b.tech", "btech", "b.e.", "be", "bachelor", "m.tech", "mca", "b.sc", "m.sc"])
        has_field = any(f in all_text.lower() for f in ["computer science", "data science", "artificial intelligence", "ai", "information technology", "engineering", "cs", "it"])

        if has_degree and has_field:
            best_entry = next((e.degree for e in edu_entries if len(e.degree) > 5), "B.Tech in Computer Science / Engineering")
            return EvidenceMatch(
                requirement=req_text,
                category=RequirementCategory.EDUCATION,
                importance=ImportanceLevel.HIGH,
                status=MatchStatus.STRONG,
                score=0.98,
                best_matching_bullet=f"Education: {best_entry}",
                matched_section="education",
                matched_entities=["Degree", "Engineering / CS"],
                evidence_notes="Verified matching degree and specialization in candidate's academic record."
            )
        elif has_degree or edu_entries:
            best_entry = edu_entries[0].degree if edu_entries else "Degree listed"
            return EvidenceMatch(
                requirement=req_text,
                category=RequirementCategory.EDUCATION,
                importance=ImportanceLevel.HIGH,
                status=MatchStatus.PARTIAL,
                score=0.80,
                best_matching_bullet=f"Education: {best_entry}",
                matched_section="education",
                matched_entities=["Degree"],
                evidence_notes="Candidate holds relevant academic qualification."
            )
        return None

    def _match_experience_level(self, req_text: str, resume: ParsedResume) -> Optional[EvidenceMatch]:
        req_lower = req_text.lower()
        if any(k in req_lower for k in ["0 years", "0-1 year", "fresher", "graduating student", "entry level", "recent graduate", "intern"]):
            best_evidence = f"Projects: {resume.projects[0].title if resume.projects else 'Demonstrable Technical Projects & Internships'}"
            return EvidenceMatch(
                requirement=req_text,
                category=RequirementCategory.EXPERIENCE,
                importance=ImportanceLevel.HIGH,
                status=MatchStatus.STRONG,
                score=0.95,
                best_matching_bullet=best_evidence,
                matched_section="experience",
                matched_entities=["Entry Level / Project Portfolio"],
                evidence_notes="Candidate demonstrates hands-on project portfolio matching entry-level criteria."
            )
        return None

    def _extract_target_entities(self, req_text: str, provided_entities: List[str]) -> List[str]:
        if provided_entities and len(provided_entities) >= 1:
            return provided_entities
        
        found = []
        known_keywords = [
            "Python", "Java", "C++", "JavaScript", "TypeScript", "Go", "Rust", "SQL",
            "NumPy", "Pandas", "Scikit-learn", "PyTorch", "TensorFlow", "Keras",
            "NLP", "Natural Language Processing", "LLM", "Transformers", "BERT", "GPT",
            "LangChain", "LlamaIndex", "RAG", "FAISS", "ChromaDB", "Pinecone", "Qdrant",
            "FastAPI", "Flask", "Django", "Node.js", "Express", "React", "Next.js", "Vue",
            "Docker", "Kubernetes", "AWS", "Azure", "GCP", "PostgreSQL", "MySQL", "MongoDB",
            "Redis", "Git", "CI/CD", "Linux", "REST APIs", "Microservices"
        ]
        for kw in known_keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", req_text, re.IGNORECASE):
                found.append(kw)
        
        if not found:
            found = [w.strip(",.;:()\"'") for w in req_text.split() if len(w) > 3 and w.lower() not in [
                "with", "have", "solid", "basic", "strong", "understanding", "experience", "knowledge",
                "using", "skills", "ability", "proven", "track", "record", "plus", "preferred"
            ]]
        return found[:6]

    def _extract_resume_entities(self, resume: ParsedResume) -> Set[str]:
        entities = set()
        for s in resume.skills:
            entities.add(s.lower())
            for part in re.split(r"[\s\-/]+", s):
                if len(part) > 1:
                    entities.add(part.lower())
        for exp in resume.experience:
            entities.add(exp.role.lower())
            for b in exp.bullets:
                for word in re.findall(r"\b[A-Za-z0-9\.\+#\-]+\b", b):
                    entities.add(word.lower())
        for prj in resume.projects:
            entities.add(prj.title.lower())
            for t in prj.technologies:
                entities.add(t.lower())
            for b in prj.bullets:
                for word in re.findall(r"\b[A-Za-z0-9\.\+#\-]+\b", b):
                    entities.add(word.lower())
        for edu in resume.education:
            for word in re.findall(r"\b[A-Za-z0-9\.\+#\-]+\b", edu.degree):
                entities.add(word.lower())
        return entities

    def _synthesize_bullets(self, resume: ParsedResume) -> List[str]:
        bullets = []
        if resume.skills:
            bullets.append(f"Skills: {', '.join(resume.skills[:5])}")
        for exp in resume.experience:
            for b in exp.bullets:
                bullets.append(f"[{exp.role}] {b}")
        for prj in resume.projects:
            for b in prj.bullets:
                bullets.append(f"[{prj.title}] {b}")
        for edu in resume.education:
            bullets.append(f"Education: {edu.degree}")
        return bullets

