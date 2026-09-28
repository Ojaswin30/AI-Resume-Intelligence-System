from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from app.models.schemas import (
    FullAnalysisReport, ParsedResume, JDDecomposition, BulletRewrite
)
from app.services.orchestrator import ResumeIntelligenceOrchestrator
from app.services.pdf_parser import ResumePDFParser
from app.services.claim_verifier import ClaimVerifier

router = APIRouter(prefix="/api", tags=["Resume Intelligence"])
orchestrator = ResumeIntelligenceOrchestrator()
parser = ResumePDFParser()
verifier = ClaimVerifier()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine": "Local SLM Resume Intelligence Engine",
        "version": "2.0.0"
    }

@router.post("/analyze-full", response_model=FullAnalysisReport)
async def analyze_full_resume_and_jd(
    resume_file: UploadFile = File(..., description="Resume in PDF format"),
    jd_text: str = Form(..., description="Raw text of the target Job Description")
):
    try:
        pdf_bytes = await resume_file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        report = await orchestrator.run_full_analysis(pdf_bytes, jd_text)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")

@router.post("/parse-resume", response_model=ParsedResume)
async def parse_resume_only(
    resume_file: UploadFile = File(...)
):
    try:
        pdf_bytes = await resume_file.read()
        return parser.parse_pdf_bytes(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse resume PDF: {str(e)}")

@router.post("/decompose-jd", response_model=JDDecomposition)
async def decompose_job_description(
    jd_text: str = Form(...)
):
    try:
        return await orchestrator.decompose_jd(jd_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to decompose JD: {str(e)}")
