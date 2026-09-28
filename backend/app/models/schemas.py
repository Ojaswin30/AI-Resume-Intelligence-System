import re
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator

class MatchStatus(str, Enum):
    STRONG = "STRONG"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"

class RequirementCategory(str, Enum):
    TECHNICAL_SKILLS = "technical_skills"
    EXPERIENCE = "experience"
    PROJECTS = "projects"
    EDUCATION = "education"
    CERTIFICATIONS = "certifications"

class ImportanceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BONUS = "bonus"

class EducationEntry(BaseModel):
    degree: str
    institution: Optional[str] = None
    year: Optional[str] = None
    gpa_or_details: Optional[str] = None

class ExperienceEntry(BaseModel):
    role: str
    company: Optional[str] = None
    duration: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)

class ProjectEntry(BaseModel):
    title: str
    technologies: List[str] = Field(default_factory=list)
    bullets: List[str] = Field(default_factory=list)

class ParsedResume(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[ExperienceEntry] = Field(default_factory=list)
    projects: List[ProjectEntry] = Field(default_factory=list)
    education: List[EducationEntry] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    all_bullets: List[str] = Field(default_factory=list)

class JDRequirement(BaseModel):
    requirement: str
    category: RequirementCategory = RequirementCategory.TECHNICAL_SKILLS
    importance: ImportanceLevel = ImportanceLevel.HIGH
    core_entities: List[str] = Field(default_factory=list)

class JDDecomposition(BaseModel):
    role_title: Optional[str] = "Target Role"
    requirements: List[JDRequirement] = Field(default_factory=list)

class EvidenceMatch(BaseModel):
    requirement: str
    category: RequirementCategory
    importance: ImportanceLevel
    status: MatchStatus
    score: float = Field(..., description="Cosine similarity / combined score [0.0 - 1.0]")
    best_matching_bullet: Optional[str] = None
    matched_section: Optional[str] = None
    matched_entities: List[str] = Field(default_factory=list)
    evidence_notes: Optional[str] = None

class CategoryScoreBreakdown(BaseModel):
    category: RequirementCategory
    score: float = Field(..., description="Category score scaled 0-100")
    weight: float
    requirements_count: int

class DeterministicScore(BaseModel):
    overall_score: float = Field(..., description="Overall score 0-100")
    category_scores: Dict[str, CategoryScoreBreakdown]
    evidence_matrix: List[EvidenceMatch]

class BulletRewrite(BaseModel):
    original_bullet: str
    rewritten_bullet: str
    target_requirement: str
    rationale: str
    verification_passed: bool
    hallucination_warnings: List[str] = Field(default_factory=list)

class InterviewQuestion(BaseModel):
    category: str
    question: str
    context_or_reason: str
    sample_focus_points: List[str] = Field(default_factory=list)

class InterviewPrep(BaseModel):
    technical_questions: List[InterviewQuestion] = Field(default_factory=list)
    resume_deep_dives: List[InterviewQuestion] = Field(default_factory=list)
    gap_questions: List[InterviewQuestion] = Field(default_factory=list)

class GroundedAnalysis(BaseModel):
    summary_analysis: str = "Candidate analysis complete."
    key_strengths: List[str] = Field(default_factory=list)
    critical_gaps: List[str] = Field(default_factory=list)
    honest_recommendations: List[str] = Field(default_factory=list)
    section_improvements: Dict[str, List[str]] = Field(default_factory=dict)

    @field_validator("key_strengths", "critical_gaps", "honest_recommendations", mode="before")
    @classmethod
    def coerce_list_fields(cls, v):
        if isinstance(v, str):
            # Split by numbered items (1., 2.) or newlines/bullets
            items = re.split(r"(?:\r?\n|•|\d+\.)\s*", v)
            return [it.strip().lstrip("-*• ") for it in items if it.strip()]
        if isinstance(v, list):
            return [str(it).strip() for it in v if str(it).strip()]
        return []

    @field_validator("section_improvements", mode="before")
    @classmethod
    def coerce_section_improvements(cls, v):
        if not isinstance(v, dict):
            return {}
        cleaned = {}
        for key, val in v.items():
            if isinstance(val, str):
                items = re.split(r"(?:\r?\n|•|\d+\.)\s*", val)
                cleaned[key] = [it.strip().lstrip("-*• ") for it in items if it.strip()]
            elif isinstance(val, list):
                cleaned[key] = [str(it).strip() for it in val if str(it).strip()]
            else:
                cleaned[key] = [str(val)]
        return cleaned

class FullAnalysisReport(BaseModel):
    parsed_resume: ParsedResume
    jd_decomposition: JDDecomposition
    deterministic_score: DeterministicScore
    grounded_analysis: GroundedAnalysis
    bullet_rewrites: List[BulletRewrite] = Field(default_factory=list)
    interview_prep: InterviewPrep
