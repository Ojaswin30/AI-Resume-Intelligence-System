import re
from typing import Tuple, List, Set
from app.models.schemas import ParsedResume
from app.core.rules_loader import load_resume_rules, load_skill_taxonomy

COMMON_TECH_PATTERNS = [
    r"\b[A-Z][a-zA-Z0-9\+\#\.]+\b",  # Capitalized tech terms (Docker, Kubernetes, LangChain)
    r"\b[a-z0-9\+\#\.]+(?:db|sql|api|js|py|ai|ml)\b"  # Tech suffixes (mongodb, postgresql, restapi, nodejs)
]

COMMON_ENGLISH_WORDS = {
    "The", "This", "Developed", "Built", "Implemented", "Designed", "Created", "Optimized",
    "Engineered", "Managed", "Led", "Integrated", "Collaborated", "Reduced", "Increased",
    "Enhanced", "Utilized", "Leveraged", "Achieved", "Supported", "Maintained", "Applied",
    "High", "Low", "System", "Service", "Architecture", "Performance", "Efficiency",
    "Production", "Backend", "Frontend", "API", "Pipeline", "Model", "Data", "Code"
}

class ClaimVerifier:
    def __init__(self):
        self.rules = load_resume_rules()
        self.taxonomy = load_skill_taxonomy()

    def verify_bullet_rewrite(
        self, original_bullet: str, rewritten_bullet: str, resume: ParsedResume
    ) -> Tuple[bool, List[str]]:
        warnings: List[str] = []
        
        # 1. Gather all authorized ground-truth entities from resume and original bullet
        authorized_tokens = self._get_authorized_tokens(original_bullet, resume)

        # 2. Extract potential technical entities from rewritten bullet
        rewritten_entities = self._extract_entities(rewritten_bullet)

        # 3. Check for unauthorized/invented technical entities
        for entity in rewritten_entities:
            normalized = entity.lower()
            if entity in COMMON_ENGLISH_WORDS or len(entity) <= 2:
                continue

            # Check if this entity or any taxonomy alias was present in authorized tokens
            is_authorized = False
            if normalized in authorized_tokens:
                is_authorized = True
            else:
                # Check taxonomy
                aliases = self.taxonomy.get(normalized, [normalized])
                if any(a in authorized_tokens for a in aliases):
                    is_authorized = True

            if not is_authorized:
                warnings.append(f"Detected ungrounded technology/claim '{entity}' not present in original resume.")

        # 4. Check for invented metrics if rule is enabled
        require_metrics_evidence = self.rules.get("resume_rules", {}).get("require_evidence_for_metrics", True)
        if require_metrics_evidence:
            orig_numbers = set(re.findall(r"\b\d+[%kKmM]?\b", original_bullet))
            new_numbers = set(re.findall(r"\b\d+[%kKmM]?\b", rewritten_bullet))
            invented_numbers = new_numbers - orig_numbers
            if invented_numbers:
                warnings.append(f"Detected unverified metric(s) {list(invented_numbers)} not found in original bullet.")

        passed = len(warnings) == 0
        return passed, warnings

    def _get_authorized_tokens(self, original_bullet: str, resume: ParsedResume) -> Set[str]:
        tokens = set()
        # Words from original bullet
        for w in re.findall(r"[A-Za-z0-9\+\#\.]+", original_bullet):
            tokens.add(w.lower())

        # Skills from resume
        for s in resume.skills:
            tokens.add(s.lower())
            for part in re.split(r"[\s\-/]+", s):
                tokens.add(part.lower())

        # Entities in projects and experience
        for prj in resume.projects:
            tokens.add(prj.title.lower())
            for t in prj.technologies:
                tokens.add(t.lower())
        for exp in resume.experience:
            tokens.add(exp.role.lower())
            for b in exp.bullets:
                for w in re.findall(r"[A-Za-z0-9\+\#\.]+", b):
                    tokens.add(w.lower())

        return tokens

    def _extract_entities(self, text: str) -> List[str]:
        found = set()
        for pattern in COMMON_TECH_PATTERNS:
            matches = re.findall(pattern, text)
            for m in matches:
                found.add(m)
        return list(found)
