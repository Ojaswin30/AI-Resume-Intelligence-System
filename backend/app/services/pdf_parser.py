import re
import io
from typing import Dict, List, Optional
from app.models.schemas import (
    ParsedResume, ExperienceEntry, ProjectEntry, EducationEntry
)

SECTION_HEADERS = {
    "education": ["education", "education & publications", "academic background", "qualifications", "academics"],
    "skills": ["skills", "technical skills", "skills & technologies", "core competencies", "technologies", "tools", "technical proficiencies", "technical expertise"],
    "experience": ["experience", "work experience", "professional experience", "employment history", "work history", "internships", "internship experience"],
    "projects": ["projects", "personal projects", "academic projects", "key projects", "notable projects"],
    "certifications": ["certifications", "certificates", "licenses & certifications", "achievements", "awards", "publications", "extracurricular"]
}

class ResumePDFParser:
    def parse_pdf_bytes(self, pdf_bytes: bytes) -> ParsedResume:
        raw_text = self._extract_text(pdf_bytes)
        return self.parse_text(raw_text)

    def _extract_text(self, pdf_bytes: bytes) -> str:
        text = ""
        # Try pdfplumber with layout preservation
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text(layout=False)
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
        education = self._parse_education(sections.get("education", []) + self._find_edu_lines(lines))

        # 5. Parse Experience
        experience = self._parse_experience(sections.get("experience", []))

        # 6. Parse Projects
        projects = self._parse_projects(sections.get("projects", []))

        # 7. Summary & Certifications
        summary = " ".join(sections.get("summary", [])) if sections.get("summary") else None
        certifications = [line.lstrip("•-* ") for line in sections.get("certifications", []) if len(line) > 3]

        # 8. Extract discrete, highly-granular bullets
        all_bullets = self._extract_all_bullets(raw_text, experience, projects, skills, education)

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
            clean_header = re.sub(r"[^a-zA-Z\s&]", "", line).strip().lower()
            detected = None
            for sec_name, header_aliases in SECTION_HEADERS.items():
                if clean_header in header_aliases or any(clean_header == h for h in header_aliases):
                    detected = sec_name
                    break
                elif len(clean_header) <= 35 and any(clean_header.startswith(h) for h in header_aliases):
                    detected = sec_name
                    break

            if detected:
                current_section = detected
                if current_section not in sections:
                    sections[current_section] = []
            else:
                sections.setdefault(current_section, []).append(line)

        return sections

    def _find_edu_lines(self, lines: List[str]) -> List[str]:
        edu_lines = []
        for line in lines:
            lower = line.lower()
            if any(term in lower for term in ["b.tech", "btech", "b.e.", "m.tech", "bachelor", "master", "cgpa", "university", "institute of technology"]):
                edu_lines.append(line)
        return edu_lines

    def _parse_skills(self, lines: List[str]) -> List[str]:
        skills = set()
        for line in lines:
            # Skip education or GPA artifacts in skill lines
            if any(skip in line.lower() for skip in ["cgpa", "university", "2022-", "2023-", "2024-", "2025-", "2026-", "education"]):
                continue

            cleaned = re.sub(r"^(Languages|Frameworks|Tools|Databases|Libraries|Skills|Technical Skills|Core Competencies):", "", line, flags=re.IGNORECASE)
            items = re.split(r"[,|•·;/\t\(\)]+", cleaned)
            for item in items:
                skill = item.strip().strip("•-*[]{} ")
                # Remove artifacts
                if skill and len(skill) < 40 and not re.search(r"\b\d{4}\b", skill) and not skill.lower().startswith("cgpa"):
                    skills.add(skill)
        return sorted(list(skills))

    def _parse_education(self, lines: List[str]) -> List[EducationEntry]:
        entries = []
        seen = set()
        for line in lines:
            lower = line.lower()
            if any(term in lower for term in ["b.tech", "btech", "b.e.", "m.tech", "mca", "bachelor", "master", "degree", "computer science", "amity", "university"]):
                cleaned = line.strip("•-* ")
                if cleaned not in seen and len(cleaned) > 5:
                    seen.add(cleaned)
                    entries.append(EducationEntry(degree=cleaned))
        if not entries and lines:
            entries.append(EducationEntry(degree=" ".join(lines[:2])))
        return entries

    def _parse_experience(self, lines: List[str]) -> List[ExperienceEntry]:
        entries = []
        current_entry = None
        for line in lines:
            is_bullet = bool(re.match(r"^([•\-\*·]|\d+\.)", line))
            if not is_bullet and len(line) < 80 and any(kw in line.lower() for kw in ["engineer", "developer", "intern", "lead", "architect", "analyst", "trainee", "associate"]):
                if current_entry:
                    entries.append(current_entry)
                current_entry = ExperienceEntry(role=line.strip())
            elif current_entry:
                current_entry.bullets.append(line.lstrip("•-*· 1234567890.").strip())
            else:
                if len(line) > 15:
                    if not current_entry:
                        current_entry = ExperienceEntry(role="Experience")
                    current_entry.bullets.append(line.lstrip("•-*· ").strip())

        if current_entry:
            entries.append(current_entry)
        return entries

    def _parse_projects(self, lines: List[str]) -> List[ProjectEntry]:
        entries = []
        current_entry = None
        for line in lines:
            is_bullet = bool(re.match(r"^([•\-\*·]|\d+\.)", line))
            if not is_bullet and len(line) < 70 and not line.endswith("."):
                if current_entry:
                    entries.append(current_entry)
                current_entry = ProjectEntry(title=line.strip())
            elif current_entry:
                current_entry.bullets.append(line.lstrip("•-*· 1234567890.").strip())
            else:
                if len(line) > 15:
                    if not current_entry:
                        current_entry = ProjectEntry(title="Project")
                    current_entry.bullets.append(line.lstrip("•-*· ").strip())

        if current_entry:
            entries.append(current_entry)
        return entries

    def _extract_all_bullets(
        self, raw_text: str, experience: List[ExperienceEntry], projects: List[ProjectEntry], skills: List[str], education: List[EducationEntry]
    ) -> List[str]:
        bullets = []

        # 1. Project bullets
        for prj in projects:
            for b in prj.bullets:
                if len(b) > 8:
                    bullets.append(f"[{prj.title}] {b}")

        # 2. Experience bullets
        for exp in experience:
            for b in exp.bullets:
                if len(b) > 8:
                    bullets.append(f"[{exp.role}] {b}")

        # 3. Education entries
        for edu in education:
            bullets.append(f"[Education] {edu.degree}")

        # 4. Small, focused skill groupings (3-5 skills each so embeddings are NOT diluted)
        if skills:
            chunk_size = 4
            for i in range(0, len(skills), chunk_size):
                sub_skills = skills[i:i + chunk_size]
                bullets.append(f"[Skills] {', '.join(sub_skills)}")

        # 5. Raw bullet split fallback
        if len(bullets) < 3:
            for line in raw_text.split("\n"):
                cleaned = line.strip().lstrip("•-*· ")
                if len(cleaned) > 20:
                    bullets.append(cleaned)

        return bullets
