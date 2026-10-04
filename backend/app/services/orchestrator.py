import json
from typing import Optional, List
from app.models.schemas import (
    ParsedResume, JDDecomposition, JDRequirement, DeterministicScore,
    GroundedAnalysis, BulletRewrite, InterviewPrep, InterviewQuestion,
    FullAnalysisReport, MatchStatus, EvidenceMatch
)
from app.services.pdf_parser import ResumePDFParser
from app.services.llm_engine import LocalSLMEngine
from app.services.embedding_engine import EmbeddingEngine
from app.services.matching_engine import MatchingEngine
from app.services.scoring_engine import ScoringEngine
from app.services.claim_verifier import ClaimVerifier
from app.core.rules_loader import load_system_prompts

class ResumeIntelligenceOrchestrator:
    def __init__(self):
        self.parser = ResumePDFParser()
        self.llm = LocalSLMEngine()
        self.embedding_engine = EmbeddingEngine()
        self.matcher = MatchingEngine(self.embedding_engine)
        self.scorer = ScoringEngine()
        self.verifier = ClaimVerifier()
        self.prompts = load_system_prompts()

    async def run_full_analysis(self, pdf_bytes: bytes, jd_text: str) -> FullAnalysisReport:
        # Validate JD text length
        clean_jd = jd_text.strip()
        if len(clean_jd.split()) < 6:
            raise ValueError("The Job Description is too brief. Please paste a full job description or check 'Use Sample Job Description'.")

        # 1. Parse Resume PDF into structured JSON
        parsed_resume = self.parser.parse_pdf_bytes(pdf_bytes)

        # 2. Decompose Job Description using Local SLM
        jd_decomp = await self.decompose_jd(clean_jd)

        # 3. Deterministic Requirement/Evidence Matching
        evidence_matches = self.matcher.match_requirements(parsed_resume, jd_decomp.requirements)

        # 4. Deterministic Scoring Calculation
        score_report = self.scorer.calculate_score(evidence_matches)

        # 5. Local SLM Grounded Analysis & Gap Analysis
        grounded_analysis = await self.generate_grounded_analysis(parsed_resume, jd_decomp, score_report)

        # 6. Resume Bullet Rewrites + Verification Guardrail
        bullet_rewrites = await self.generate_verified_bullet_rewrites(parsed_resume, evidence_matches)

        # 7. Targeted Interview Preparation
        interview_prep = await self.generate_interview_prep(parsed_resume, jd_decomp, score_report)

        return FullAnalysisReport(
            parsed_resume=parsed_resume,
            jd_decomposition=jd_decomp,
            deterministic_score=score_report,
            grounded_analysis=grounded_analysis,
            bullet_rewrites=bullet_rewrites,
            interview_prep=interview_prep
        )

    async def decompose_jd(self, jd_text: str) -> JDDecomposition:
        system_prompt = self.prompts.get("jd_decomposition_prompt", "Decompose the JD into requirements.")
        prompt = f"Analyze and decompose this Job Description:\n\n{jd_text}"
        
        result_json = await self.llm.generate_json(prompt, system_prompt)
        
        reqs = []
        for r in result_json.get("requirements", []):
            try:
                reqs.append(JDRequirement(**r))
            except Exception:
                reqs.append(JDRequirement(
                    requirement=r.get("requirement", str(r)),
                    category=r.get("category", "technical_skills"),
                    importance=r.get("importance", "high"),
                    core_entities=r.get("core_entities", [])
                ))

        if not reqs:
            reqs = [
                JDRequirement(requirement="Core Technical Proficiency & Programming", category="technical_skills", importance="high", core_entities=["Python", "Engineering"]),
                JDRequirement(requirement="Software Engineering & System Design", category="experience", importance="high", core_entities=["Architecture", "APIs"]),
                JDRequirement(requirement="Hands-on Project Development & Delivery", category="projects", importance="high", core_entities=["Projects"]),
                JDRequirement(requirement="Relevant Technical Degree or Background", category="education", importance="medium", core_entities=["Computer Science"])
            ]

        return JDDecomposition(
            role_title=result_json.get("role_title", "Target Role"),
            requirements=reqs
        )

    async def generate_grounded_analysis(
        self, resume: ParsedResume, jd_decomp: JDDecomposition, score_report: DeterministicScore
    ) -> GroundedAnalysis:
        system_prompt = self.prompts.get("grounded_analysis_prompt", "Provide grounded analysis.")
        
        strong_evidence = [f"- {m.requirement}: {ResumePDFParser.clean_evidence_text(m.best_matching_bullet)}" for m in score_report.evidence_matrix if m.status == MatchStatus.STRONG]
        missing_evidence = [f"- {m.requirement} ({m.category.value if hasattr(m.category, 'value') else m.category})" for m in score_report.evidence_matrix if m.status == MatchStatus.MISSING]
        partial_evidence = [f"- {m.requirement}" for m in score_report.evidence_matrix if m.status == MatchStatus.PARTIAL]

        prompt = f"""
CANDIDATE SCORE: {score_report.overall_score}/100
STRONG MATCHES FOUND:
{chr(10).join(strong_evidence) if strong_evidence else 'None'}

PARTIAL MATCHES:
{chr(10).join(partial_evidence) if partial_evidence else 'None'}

MISSING REQUIREMENTS:
{chr(10).join(missing_evidence) if missing_evidence else 'None'}

CANDIDATE SKILLS: {', '.join(resume.skills)}
ROLE TARGET: {jd_decomp.role_title}
"""
        result_json = await self.llm.generate_json(prompt, system_prompt)

        # Ensure non-empty strengths and gaps
        strengths = result_json.get("key_strengths") or [m.requirement for m in score_report.evidence_matrix if m.status == MatchStatus.STRONG][:4]
        if not strengths:
            strengths = ["Strong foundational computer science background", "Demonstrated hands-on programming experience"]

        gaps = result_json.get("critical_gaps") or [m.requirement for m in score_report.evidence_matrix if m.status == MatchStatus.MISSING][:4]
        if not gaps:
            gaps = ["Explicit demonstration of cloud deployment (AWS/Azure/GCP)", "Quantified business metrics in project bullets"]

        recommendations = result_json.get("honest_recommendations") or [
            "Highlight hands-on project implementations that directly touch on missing JD keywords.",
            "Add measurable outcomes (e.g. 'reduced latency by 20%', 'supported 10k+ requests') to existing bullets.",
            "Organize technical skills into clear categories at the top of the resume."
        ]

        section_tips = result_json.get("section_improvements") or {
            "experience": ["Emphasize engineering scale, system architecture, and measurable impact."],
            "projects": ["Structure project descriptions with Context, Action, Tech Stack, and Impact."],
            "skills": ["Group technical skills clearly by category (Languages, Frameworks, Cloud, Databases)."]
        }

        return GroundedAnalysis(
            summary_analysis=result_json.get("summary_analysis", "Candidate demonstrates a solid engineering foundation with actionable opportunities for JD alignment."),
            key_strengths=strengths,
            critical_gaps=gaps,
            honest_recommendations=recommendations,
            section_improvements=section_tips
        )

    async def generate_verified_bullet_rewrites(
        self, resume: ParsedResume, evidence_matches: List[EvidenceMatch]
    ) -> List[BulletRewrite]:
        rewrites: List[BulletRewrite] = []
        system_prompt = self.prompts.get("bullet_rewriting_prompt", "Rewrite bullet honestly.")

        # Extract authentic descriptive candidate bullets (experience & project bullets)
        actionable_bullets = []
        for exp in resume.experience:
            for b in exp.bullets:
                clean_b = ResumePDFParser.clean_evidence_text(b).strip()
                if len(clean_b) > 20 and not any(skip in clean_b.lower() for skip in ["full-time", "internship", "associate software engineer"]):
                    actionable_bullets.append((clean_b, "Experience"))
        for prj in resume.projects:
            for b in prj.bullets:
                clean_b = ResumePDFParser.clean_evidence_text(b).strip()
                if len(clean_b) > 20:
                    actionable_bullets.append((clean_b, "Projects"))

        if not actionable_bullets:
            for b in resume.all_bullets:
                clean_b = ResumePDFParser.clean_evidence_text(b).strip()
                if len(clean_b) > 25 and not clean_b.lower().startswith("skills:") and not clean_b.lower().startswith("education:"):
                    actionable_bullets.append((clean_b, "Work"))

        # Select top 3 distinct candidate bullets
        selected_candidates = actionable_bullets[:3]

        for idx, (orig_bullet, section) in enumerate(selected_candidates):
            # Find matching target requirement
            target_req = evidence_matches[idx % len(evidence_matches)].requirement if evidence_matches else "Software Engineering"
            target_entities = evidence_matches[idx % len(evidence_matches)].matched_entities if evidence_matches else []

            prompt = f"Original Bullet: {orig_bullet}\nTarget Requirement: {target_req}\nAllowed Entities: {target_entities}"
            
            res_json = await self.llm.generate_json(prompt, system_prompt)
            rewritten = res_json.get("rewritten_bullet", orig_bullet)
            rationale = res_json.get("rationale", f"Strengthened action verbs and aligned phrasing with '{target_req}'.")

            passed, warnings = self.verifier.verify_bullet_rewrite(orig_bullet, rewritten, resume)

            rewrites.append(BulletRewrite(
                original_bullet=orig_bullet,
                rewritten_bullet=rewritten,
                target_requirement=target_req,
                rationale=rationale,
                verification_passed=passed,
                hallucination_warnings=warnings
            ))

        return rewrites

    async def generate_interview_prep(
        self, resume: ParsedResume, jd_decomp: JDDecomposition, score_report: DeterministicScore
    ) -> InterviewPrep:
        system_prompt = self.prompts.get("interview_prep_prompt", "Generate targeted interview questions.")
        
        strong_reqs = [m.requirement for m in score_report.evidence_matrix if m.status == MatchStatus.STRONG][:3]
        missing_reqs = [m.requirement for m in score_report.evidence_matrix if m.status == MatchStatus.MISSING][:3]

        prompt = f"""
ROLE TITLE: {jd_decomp.role_title}
STRONG MATCHES: {', '.join(strong_reqs) if strong_reqs else 'Core Engineering'}
GAPS / MISSING SKILLS: {', '.join(missing_reqs) if missing_reqs else 'Cloud / Vector Systems'}
CANDIDATE BULLETS SAMPLE:
{chr(10).join(resume.all_bullets[:5])}
"""
        res_json = await self.llm.generate_json(prompt, system_prompt)
        
        tech_q = res_json.get("technical_questions") or [
            {
                "category": "Core Architecture",
                "question": f"How do you ensure high performance, fault tolerance, and concurrency when building applications with {strong_reqs[0] if strong_reqs else 'your tech stack'}?",
                "context_or_reason": "Evaluates architectural depth and production reliability best practices.",
                "sample_focus_points": ["Asynchronous tasks", "Caching strategies", "Connection pooling"]
            },
            {
                "category": "API & System Design",
                "question": "How do you structure RESTful APIs or microservices for scalability and maintainability?",
                "context_or_reason": "Tests system design principles and clean code architecture.",
                "sample_focus_points": ["Data serialization", "Authentication/Rate limiting", "Error boundaries"]
            }
        ]

        deep_dive_q = res_json.get("resume_deep_dives") or [
            {
                "category": "Project Deep-Dive",
                "question": "Walk me through the technical architecture of your most impactful project listed on your resume.",
                "context_or_reason": "Verifies technical ownership, design decisions, and hands-on contribution.",
                "sample_focus_points": ["Architecture diagram", "Trade-offs considered", "Key metrics achieved"]
            },
            {
                "category": "Debugging & Performance",
                "question": "Describe a difficult performance bottleneck or bug you diagnosed in your work. What tools and steps did you use?",
                "context_or_reason": "Evaluates debugging methodology and root-cause analysis skills.",
                "sample_focus_points": ["Profiling tools", "Root cause diagnosis", "Preventative testing"]
            }
        ]

        gap_q = res_json.get("gap_questions") or [
            {
                "category": "Skill Gap Strategy",
                "question": f"This position requires {missing_reqs[0] if missing_reqs else 'modern containerization and cloud tooling'}. How would you approach quickly ramping up on this stack?",
                "context_or_reason": "Assesses fast-learning agility and conceptual foundation in areas not heavily evidenced on the resume.",
                "sample_focus_points": ["Core concepts", "Hands-on side projects", "Ramp-up timeline"]
            }
        ]

        return InterviewPrep(
            technical_questions=[InterviewQuestion(**q) if isinstance(q, dict) else q for q in tech_q],
            resume_deep_dives=[InterviewQuestion(**q) if isinstance(q, dict) else q for q in deep_dive_q],
            gap_questions=[InterviewQuestion(**q) if isinstance(q, dict) else q for q in gap_q]
        )

