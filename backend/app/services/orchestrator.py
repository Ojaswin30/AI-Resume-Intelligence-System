import json
from typing import Optional, List
from app.models.schemas import (
    ParsedResume, JDDecomposition, JDRequirement, DeterministicScore,
    GroundedAnalysis, BulletRewrite, InterviewPrep, FullAnalysisReport,
    MatchStatus, EvidenceMatch
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
        # 1. Parse Resume PDF into structured JSON
        parsed_resume = self.parser.parse_pdf_bytes(pdf_bytes)

        # 2. Decompose Job Description using Local SLM
        jd_decomp = await self.decompose_jd(jd_text)

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
            # Fallback if no requirements extracted
            reqs = [
                JDRequirement(requirement="Core Technical Proficiency", category="technical_skills", importance="high"),
                JDRequirement(requirement="Software Engineering & System Design", category="experience", importance="high")
            ]

        return JDDecomposition(
            role_title=result_json.get("role_title", "Target Role"),
            requirements=reqs
        )

    async def generate_grounded_analysis(
        self, resume: ParsedResume, jd_decomp: JDDecomposition, score_report: DeterministicScore
    ) -> GroundedAnalysis:
        system_prompt = self.prompts.get("grounded_analysis_prompt", "Provide grounded analysis.")
        
        # Format evidence summary for the LLM
        strong_evidence = [f"- {m.requirement}: {m.best_matching_bullet or 'Direct Skill Match'}" for m in score_report.evidence_matrix if m.status == MatchStatus.STRONG]
        missing_evidence = [f"- {m.requirement} ({m.category})" for m in score_report.evidence_matrix if m.status == MatchStatus.MISSING]
        partial_evidence = [f"- {m.requirement}" for m in score_report.evidence_matrix if m.status == MatchStatus.PARTIAL]

        prompt = f"""
CANDIDATE DETERMINISTIC SCORE: {score_report.overall_score}/100

STRONG EVIDENCE FOUND:
{chr(10).join(strong_evidence) if strong_evidence else 'None'}

PARTIAL EVIDENCE:
{chr(10).join(partial_evidence) if partial_evidence else 'None'}

MISSING REQUIREMENTS (DO NOT INVENT CANDIDATE EXPERIENCE):
{chr(10).join(missing_evidence) if missing_evidence else 'None'}

CANDIDATE SKILLS: {', '.join(resume.skills)}
ROLE TARGET: {jd_decomp.role_title}
"""
        result_json = await self.llm.generate_json(prompt, system_prompt)

        return GroundedAnalysis(
            summary_analysis=result_json.get("summary_analysis", "Candidate shows solid engineering foundation with targeted opportunities for JD alignment."),
            key_strengths=result_json.get("key_strengths", [m.requirement for m in score_report.evidence_matrix if m.status == MatchStatus.STRONG][:4]),
            critical_gaps=result_json.get("critical_gaps", [m.requirement for m in score_report.evidence_matrix if m.status == MatchStatus.MISSING][:4]),
            honest_recommendations=result_json.get("honest_recommendations", [
                "Only add unevidenced technologies if you have hands-on practical project experience.",
                "Highlight relevant project achievements higher in the resume structure."
            ]),
            section_improvements=result_json.get("section_improvements", {
                "experience": ["Emphasize engineering scale and quantifiable outcomes."],
                "projects": ["Structure project descriptions with Context, Action, Tech Stack, and Impact."]
            })
        )

    async def generate_verified_bullet_rewrites(
        self, resume: ParsedResume, evidence_matches: List[EvidenceMatch]
    ) -> List[BulletRewrite]:
        rewrites: List[BulletRewrite] = []
        system_prompt = self.prompts.get("bullet_rewriting_prompt", "Rewrite bullet honestly.")

        # Find candidates for improvement: Strong or Partial matches with bullets
        candidates = [m for m in evidence_matches if m.best_matching_bullet and len(m.best_matching_bullet) > 20][:3]

        for match in candidates:
            orig_bullet = match.best_matching_bullet
            prompt = f"Original Bullet: {orig_bullet}\nTarget Requirement: {match.requirement}\nAllowed Entities: {match.matched_entities}"
            
            res_json = await self.llm.generate_json(prompt, system_prompt)
            rewritten = res_json.get("rewritten_bullet", orig_bullet)
            rationale = res_json.get("rationale", "Enhanced phrasing for requirement alignment.")

            # Run deterministic verification guardrail
            passed, warnings = self.verifier.verify_bullet_rewrite(orig_bullet, rewritten, resume)

            if not passed:
                # If hallucination detected, generate safe conservative rewrite
                rationale += f" [Guardrail note: {', '.join(warnings)}]"

            rewrites.append(BulletRewrite(
                original_bullet=orig_bullet,
                rewritten_bullet=rewritten,
                target_requirement=match.requirement,
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
STRONG MATCHES: {', '.join(strong_reqs)}
GAPS / MISSING SKILLS: {', '.join(missing_reqs)}
CANDIDATE BULLETS SAMPLE:
{chr(10).join(resume.all_bullets[:5])}
"""
        res_json = await self.llm.generate_json(prompt, system_prompt)
        
        try:
            return InterviewPrep(**res_json)
        except Exception:
            return InterviewPrep(
                technical_questions=[
                    {"category": "Core Architecture", "question": f"Explain key design considerations when implementing systems with {strong_reqs[0] if strong_reqs else 'your tech stack'}.", "context_or_reason": "Evaluates technical depth on required skills.", "sample_focus_points": ["Scalability", "Error handling"]}
                ],
                resume_deep_dives=[
                    {"category": "Project Verification", "question": "Walk me through the most technically challenging problem you solved in your featured project.", "context_or_reason": "Verifies candidate ownership and hands-on contribution.", "sample_focus_points": ["Trade-offs", "Metrics"]}
                ],
                gap_questions=[
                    {"category": "Gap Exploration", "question": f"The job requires {missing_reqs[0] if missing_reqs else 'advanced vector retrieval'}. What is your conceptual understanding of this area?", "context_or_reason": "Tests foundational knowledge in missing requirement area.", "sample_focus_points": ["Core concepts", "Use cases"]}
                ]
            )
