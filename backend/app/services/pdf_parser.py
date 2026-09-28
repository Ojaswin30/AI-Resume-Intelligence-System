import re
import io
from typing import Dict, List, Optional
from app.models.schemas import (
    ParsedResume, ExperienceEntry, ProjectEntry, EducationEntry
)

SECTION_HEADERS = {
    "summary": ["summary", "professional summary", "about me", "profile", "objective"],
    "skills": ["skills", "technical skills", "skills & technologies", "core competencies", "technologies", "tools"],
    "experience": ["experience", "work experience", "professional experience", "employment history", "work history"],
    "projects": ["projects", "personal projects", "academic projects", "key projects"],
    "education": ["education", "academic background", "qualifications"],
    "certifications": ["certifications", "certificates", "licenses & certifications", "achievements", "awards"]
}

class ResumePDFParser:
    def parse_pdf_bytes(self, pdf_bytes: bytes) -> ParsedResume:
        raw_text = self._extract_text(pdf_bytes)
        return self.parse_text(raw_text)

    def _extract_text(self, pdf_bytes: bytes) -> str:
        text = ""
        # Try pdfplumber first
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception:
            pass

        # Fallback to pypdf
        if not text.strip():
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            except Exception:
                pass

        if not text.strip():
            # If plain text was passed directly
            try:
                text = pdf_bytes.decode("utf-8", errors="ignore")
            except Exception:
                text = ""

        return text.strip()

    def parse_text(self, raw_text: str) -> ParsedResume:
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        
        # 1. Contact info heuristics
        name = lines[0] if lines else "Candidate"
        email = None
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", raw_text)
        if email_match:
            email = email_match.group(0)

        # 2. Section segmentation
        sections = self._segment_sections(lines)

        # 3. Parse Skills
        skills = self._parse_skills(sections.get("skills", []))

        # 4. Parse Education
        education = self._parse_education(sections.get("education", []))

        # 5. Parse Experience
        experience = self._parse_experience(sections.get("experience", []))

        # 6. Parse Projects
        projects = self._parse_projects(sections.get("projects", []))

        # 7. Parse Summary & Certifications
        summary = " ".join(sections.get("summary", [])) if sections.get("summary") else None
        certifications = [line.lstrip("•-* ") for line in sections.get("certifications", []) if len(line) > 3]

        # 8. Collect all discrete bullets across all sections
        all_bullets = self._extract_all_bullets(raw_text, experience, projects, skills)

        return ParsedResume(
            name=name,
            email=email,
            summary=summary,
            skills=skills,
            experience=experience,
            projects=projects,
            education=education,
            certifications=certifications,
            all_bullets=all_bullets
        )

    def _segment_sections(self, lines: List[str]) -> Dict[str, List[str]]:
        sections: Dict[str, List[str]] = {}
        current_section = "summary"
        sections[current_section] = []

        for line in lines:
            normalized = re.sub(r"[^a-zA-Z\s]", "", line).strip().lower()
            detected = None
            for sec_name, header_aliases in SECTION_HEADERS.items():
                if normalized in header_aliases or any(normalized.startswith(h) and len(normalized) <= len(h) + 4 for h in header_aliases):
                    detected = sec_name
                    break
            
            if detected:
                current_section = detected
                if current_section not in sections:
                    sections[current_section] = []
            else:
                sections.setdefault(current_section, []).append(line)

        return sections

    def _parse_skills(self, lines: List[str]) -> List[str]:
        skills = set()
        for line in lines:
            cleaned = re.sub(r"^(Languages|Frameworks|Tools|Databases|Libraries|Skills):", "", line, flags=re.IGNORECASE)
            items = re.split(r"[,|•·;/\t]+", cleaned)
            for item in items:
                skill = item.strip().strip("•-* ")
                if skill and len(skill) < 40 and not skill.lower().startswith("proficient in"):
                    skills.add(skill)
        return sorted(list(skills))

    def _parse_education(self, lines: List[str]) -> List[EducationEntry]:
        entries = []
        for line in lines:
            if any(term in line.lower() for term in ["bachelor", "master", "b.tech", "m.tech", "b.s.", "m.s.", "phd", "degree", "university", "institute", "college"]):
                entries.append(EducationEntry(degree=line.strip("•-* ")))
        if not entries and lines:
            entries.append(EducationEntry(degree=" ".join(lines[:2])))
        return entries

    def _parse_experience(self, lines: List[str]) -> List[ExperienceEntry]:
        entries = []
        current_entry = None
        
        for line in lines:
            is_bullet = bool(re.match(r"^([•\-\*]|\d+\.)", line))
            if not is_bullet and len(line) < 80 and any(kw in line.lower() for kw in ["engineer", "developer", "intern", "lead", "architect", "analyst", "manager", "associate"]):
                if current_entry:
                    entries.append(current_entry)
                current_entry = ExperienceEntry(role=line.strip())
            elif current_entry:
                current_entry.bullets.append(line.lstrip("•-* 1234567890.").strip())
            else:
                if len(line) > 15:
                    if not current_entry:
                        current_entry = ExperienceEntry(role="Professional Experience")
                    current_entry.bullets.append(line.lstrip("•-* ").strip())

        if current_entry:
            entries.append(current_entry)
        return entries

    def _parse_projects(self, lines: List[str]) -> List[ProjectEntry]:
        entries = []
        current_entry = None

        for line in lines:
            is_bullet = bool(re.match(r"^([•\-\*]|\d+\.)", line))
            if not is_bullet and len(line) < 70 and not line.endswith("."):
                if current_entry:
                    entries.append(current_entry)
                current_entry = ProjectEntry(title=line.strip())
            elif current_entry:
                current_entry.bullets.append(line.lstrip("•-* 1234567890.").strip())
            else:
                if len(line) > 15:
                    if not current_entry:
                        current_entry = ProjectEntry(title="Relevant Project")
                    current_entry.bullets.append(line.lstrip("•-* ").strip())

        if current_entry:
            entries.append(current_entry)
        return entries

    def _extract_all_bullets(self, raw_text: str, experience: List[ExperienceEntry], projects: List[ProjectEntry], skills: List[str]) -> List[str]:
        bullets = []
        for exp in experience:
            for b in exp.bullets:
                if len(b) > 10:
                    bullets.append(f"[{exp.role}] {b}")
        for prj in projects:
            for b in prj.bullets:
                if len(b) > 10:
                    bullets.append(f"[{prj.title}] {b}")
        if skills:
            bullets.append(f"[Skills] {', '.join(skills)}")

        if not bullets:
            # Fallback to regex splitting on bullet markers or sentence endings
            for line in raw_text.split("\n"):
                cleaned = line.strip().lstrip("•-* ")
                if len(cleaned) > 20:
                    bullets.append(cleaned)
        return bullets
