import asyncio
import sys
from pathlib import Path
import re

# Add backend directory to sys.path so imports work seamlessly
backend_dir = Path(__file__).resolve().parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import streamlit as st
import pandas as pd
from backend.app.services.orchestrator import ResumeIntelligenceOrchestrator
from backend.app.services.pdf_parser import ResumePDFParser
from backend.app.core.rules_loader import load_scoring_rules

# Page Configuration
st.set_page_config(
    page_title="ResumeAI — Intelligent ATS Resume Checker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Header styling */
    .header-badge {
        display: inline-block;
        padding: 4px 12px;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }
    
    /* Cards */
    .feature-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    /* Score Banner */
    .score-card {
        background: linear-gradient(145deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }
    
    /* Pills & Badges */
    .pill-green {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    .pill-amber {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    .pill-red {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    
    /* Rewrite Box */
    .rewrite-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .bullet-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .bullet-content {
        font-size: 0.95rem;
        line-height: 1.5;
        padding: 10px 12px;
        border-radius: 8px;
    }
    .bullet-orig {
        background: #0d1117;
        border-left: 3px solid #64748b;
        color: #cbd5e1;
    }
    .bullet-optimized {
        background: rgba(16, 185, 129, 0.08);
        border-left: 3px solid #10b981;
        color: #e2e8f0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Sample JD Templates
SAMPLE_JDS = {
    "Senior AI & GenAI Engineer": """Senior AI / GenAI Software Engineer

Requirements:
• Strong proficiency in Python, FastAPI, and asynchronous backend engineering.
• Hands-on experience developing LLM applications and RAG systems using HuggingFace Transformers, LangChain, or LlamaIndex.
• Experience with vector databases and semantic indexing (FAISS, ChromaDB, or Pinecone).
• Solid understanding of relational and NoSQL databases (PostgreSQL, Redis) and Docker containerization.
• Bachelor's or Master's degree in Computer Science, Data Science, or related engineering discipline.""",

    "Full-Stack Software Engineer": """Full-Stack Software Engineer (Python & React)

Requirements:
• 2+ years experience building web applications with Python (FastAPI/Django) and modern React/TypeScript.
• Strong foundation in RESTful API design, database modeling (PostgreSQL, SQL), and state management.
• Experience with Docker containerization, CI/CD pipelines, and cloud deployment (AWS/GCP).
• Bachelor's degree in Computer Science or demonstrable equivalent technical experience.""",

    "Machine Learning Engineer": """Machine Learning Engineer

Requirements:
• Proficiency in Python and ML libraries: NumPy, Pandas, Scikit-learn, PyTorch, or TensorFlow.
• Experience with end-to-end ML model development, training, feature engineering, and evaluation.
• Experience deploying ML models as REST APIs using FastAPI or Flask with Docker.
• Bachelor's or Master's in Computer Science, Mathematics, or related quantitative field."""
}

# Initialize Orchestrator in session state
@st.cache_resource
def get_orchestrator():
    return ResumeIntelligenceOrchestrator()

orchestrator = get_orchestrator()

# Header
st.markdown('<div class="header-badge">🛡️ 100% Local & Private Processing</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">ResumeAI — ATS Matcher & Resume Coach</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Instantly analyze your resume against any job description, discover keyword gaps, and get ATS-optimized bullet rewrites.</div>', unsafe_allow_html=True)

# 2-Column Input Area
col_resume, col_jd = st.columns([1, 1], gap="large")

with col_resume:
    st.markdown("### 📄 1. Upload Your Resume")
    uploaded_file = st.file_uploader("Upload resume PDF", type=["pdf"], help="Your file is processed locally and never leaves your computer.")
    
    if uploaded_file:
        file_size_kb = uploaded_file.size / 1024
        st.success(f"✓ **{uploaded_file.name}** ({file_size_kb:.1f} KB) loaded successfully.")
    else:
        st.info("💡 Upload your current PDF resume to evaluate keyword match & bullet quality.")

with col_jd:
    st.markdown("### 🎯 2. Target Job Description (JD)")
    
    preset_choice = st.selectbox(
        "Load a sample job description or paste your own:",
        ["✍️ Paste Custom Job Description"] + [f"📋 Sample: {k}" for k in SAMPLE_JDS.keys()],
        index=0
    )
    
    if preset_choice.startswith("📋 Sample: "):
        preset_key = preset_choice.replace("📋 Sample: ", "")
        default_jd_text = SAMPLE_JDS.get(preset_key, "")
    else:
        default_jd_text = ""
        
    jd_input = st.text_area(
        "Job Description Requirements",
        value=default_jd_text,
        height=180,
        placeholder="Paste the target job description or requirements here..."
    )

# Analysis Action Button
st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("🚀 Analyze Resume & Match Job Description", type="primary", use_container_width=True)

if analyze_btn:
    if not uploaded_file:
        st.error("⚠️ Please upload your resume PDF before running the analysis.")
    elif not jd_input or len(jd_input.strip().split()) < 6:
        st.error("⚠️ Please paste a complete target Job Description (or select one of the sample presets above).")
    else:
        with st.spinner("⏳ Analyzing resume alignment, matching keywords, and generating ATS improvements..."):
            pdf_bytes = uploaded_file.getvalue()
            try:
                report = asyncio.run(orchestrator.run_full_analysis(pdf_bytes, jd_input))
                st.session_state["report"] = report
                st.session_state["analyzed_file"] = uploaded_file.name
            except Exception as e:
                st.error(f"Error during analysis: {e}")

# Results Presentation
if "report" in st.session_state:
    report = st.session_state["report"]
    det_score = report.deterministic_score
    overall = det_score.overall_score
    
    st.markdown("---")
    
    # Match Scorecard Banner
    if overall >= 75:
        badge_html = '<span class="pill-green">🟢 Strong Match — Ready to Apply</span>'
        verdict = "Your resume has high alignment with the target role. A few targeted keyword additions will maximize your ATS score."
    elif overall >= 50:
        badge_html = '<span class="pill-amber">🟡 Moderate Match — Needs Keyword Alignment</span>'
        verdict = "Your profile shows relevant technical foundations, but is missing explicit evidence for several required tools and frameworks."
    else:
        badge_html = '<span class="pill-red">🔴 Low Match — Significant Gaps</span>'
        verdict = "Your resume lacks key technical skills and experience required for this specific role."

    st.markdown(f"""
    <div class="score-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
                <span style="font-size: 0.9rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700;">ATS Compatibility Score</span>
                <div style="font-size: 2.8rem; font-weight: 800; color: #f0f6fc; margin: 4px 0;">{overall:.0f}<span style="font-size: 1.4rem; color: #8b949e;"> / 100</span></div>
            </div>
            <div style="text-align: right;">
                {badge_html}
            </div>
        </div>
        <p style="color: #c9d1d9; font-size: 0.95rem; margin: 0;">{verdict}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Dimension Progress Cards
    cat_scores = det_score.category_scores
    skills_s = cat_scores.get("technical_skills").score if "technical_skills" in cat_scores else min(100.0, overall * 1.05)
    exp_s = cat_scores.get("experience").score if "experience" in cat_scores else overall
    prj_s = cat_scores.get("projects").score if "projects" in cat_scores else max(40.0, overall * 0.95)
    edu_s = cat_scores.get("education").score if "education" in cat_scores else 90.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"**💻 Tech Skills** &nbsp; `{skills_s:.0f}%`")
        st.progress(min(1.0, skills_s / 100))
    with c2:
        st.markdown(f"**🏢 Experience** &nbsp; `{exp_s:.0f}%`")
        st.progress(min(1.0, exp_s / 100))
    with c3:
        st.markdown(f"**🚀 Projects** &nbsp; `{prj_s:.0f}%`")
        st.progress(min(1.0, prj_s / 100))
    with c4:
        st.markdown(f"**🎓 Education** &nbsp; `{edu_s:.0f}%`")
        st.progress(min(1.0, edu_s / 100))

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Main Tabs
    tab_keywords, tab_rewrites, tab_action_plan, tab_interview = st.tabs([
        "🎯 Keyword & Skill Breakdown",
        "✍️ ATS Bullet Rewriter",
        "📋 Action Plan & Fixes",
        "🎙️ Tailored Interview Prep"
    ])

    # Tab 1: Keyword & Evidence Breakdown
    with tab_keywords:
        st.markdown("### 🔑 Keyword & Skill Alignment")
        st.caption("ATS systems scan for specific technical keywords. Here is how your resume stacks up against the job description:")
        
        # Collect matched vs missing entities
        matched_pills = []
        missing_pills = []
        for m in det_score.evidence_matrix:
            if m.status.value in ["STRONG", "PARTIAL"]:
                for ent in m.matched_entities:
                    if ent not in matched_pills:
                        matched_pills.append(ent)
            if m.status.value == "MISSING":
                req_title = m.requirement.replace("Proficiency in ", "").replace("Experience with ", "")
                if req_title not in missing_pills:
                    missing_pills.append(req_title)
                    
        k_col1, k_col2 = st.columns(2, gap="large")
        with k_col1:
            st.markdown("#### ✅ Matched Keywords Found in Your Resume")
            if matched_pills:
                pills_html = "".join([f'<span class="pill-green">✓ {p}</span>' for p in matched_pills])
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.write("No exact target keywords matched yet.")

        with k_col2:
            st.markdown("#### ⚠️ Missing High-Priority Keywords")
            if missing_pills:
                pills_html = "".join([f'<span class="pill-red">+ {p}</span>' for p in missing_pills])
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.markdown('<span class="pill-green">✓ All primary keywords covered!</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Detailed Requirement Match Table")
        
        table_rows = []
        for m in det_score.evidence_matrix:
            clean_evidence = ResumePDFParser.clean_evidence_text(m.best_matching_bullet)
            status_badge = "🟢 Strong" if m.status.value == "STRONG" else ("🟡 Partial" if m.status.value == "PARTIAL" else "🔴 Missing")
            cat_name = (m.category.value if hasattr(m.category, 'value') else str(m.category)).replace("_", " ").title()
            table_rows.append({
                "Job Requirement": m.requirement,
                "Category": cat_name,
                "Status": status_badge,
                "Match Relevance": f"{m.score * 100:.0f}%",
                "Your Resume Evidence": clean_evidence
            })
        
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Tab 2: ATS Bullet Rewriter
    with tab_rewrites:
        st.markdown("### ✍️ ATS-Optimized Bullet Rewrites")
        st.caption("We rewrite your existing resume bullets using strong action verbs and impact metrics while **strictly preserving your real technical skills (no fake claims)**.")
        
        if report.bullet_rewrites:
            for idx, rw in enumerate(report.bullet_rewrites, 1):
                clean_orig = ResumePDFParser.clean_evidence_text(rw.original_bullet)
                clean_rewritten = ResumePDFParser.clean_evidence_text(rw.rewritten_bullet)
                
                with st.container():
                    st.markdown(f"#### Improvement #{idx}: Aligning with *'{rw.target_requirement}'*")
                    
                    b_col1, b_col2 = st.columns(2, gap="medium")
                    with b_col1:
                        st.markdown('<div class="bullet-label">📄 Your Current Resume Bullet</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="bullet-content bullet-orig">{clean_orig}</div>', unsafe_allow_html=True)
                    
                    with b_col2:
                        st.markdown('<div class="bullet-label">✨ ATS-Optimized Version (Ready to copy)</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="bullet-content bullet-optimized">{clean_rewritten}</div>', unsafe_allow_html=True)
                    
                    st.caption(f"💡 **Why this is better:** {rw.rationale}")
                    st.text_area("📋 Copy optimized bullet:", value=clean_rewritten, height=68, key=f"copy_bullet_{idx}")
                    st.markdown("---")
        else:
            st.info("Upload a resume with project and work experience bullets to generate instant rewrites.")

    # Tab 3: Action Plan & Fixes
    with tab_action_plan:
        st.markdown("### 📋 Step-by-Step Action Plan")
        st.caption("Follow these prioritized recommendations to significantly boost your ATS match score:")
        
        ap_col1, ap_col2 = st.columns(2, gap="large")
        with ap_col1:
            st.markdown("#### 🎯 Priority Fixes")
            for gap in report.grounded_analysis.critical_gaps:
                st.warning(f"• {gap}")
            
            st.markdown("#### 💡 Resume Strategy Advice")
            for rec in report.grounded_analysis.honest_recommendations:
                st.info(f"• {rec}")
                
        with ap_col2:
            st.markdown("#### 📑 Section-by-Section Recommendations")
            for sec_name, tips in report.grounded_analysis.section_improvements.items():
                with st.expander(f"📌 {sec_name.upper()} Section Tips", expanded=True):
                    for tip in tips:
                        st.write(f"• {tip}")

    # Tab 4: Tailored Interview Prep
    with tab_interview:
        st.markdown("### 🎙️ Tailored Interview Preparation")
        st.caption("Custom questions prepared by analyzing the overlap between the job requirements and your resume claims:")
        
        iq_col1, iq_col2, iq_col3 = st.columns(3, gap="medium")
        
        with iq_col1:
            st.markdown("#### 1. 💻 Core Technical Questions")
            for q in report.interview_prep.technical_questions:
                with st.expander(f"❓ {q.question}"):
                    st.markdown(f"**Why they ask:** {q.context_or_reason}")
                    if q.sample_focus_points:
                        st.markdown("**Key Points to Mention:**")
                        for pt in q.sample_focus_points:
                            st.write(f"• {pt}")
                            
        with iq_col2:
            st.markdown("#### 2. 📂 Project & Claim Deep-Dives")
            for q in report.interview_prep.resume_deep_dives:
                with st.expander(f"❓ {q.question}"):
                    st.markdown(f"**Why they ask:** {q.context_or_reason}")
                    if q.sample_focus_points:
                        st.markdown("**Key Points to Mention:**")
                        for pt in q.sample_focus_points:
                            st.write(f"• {pt}")
                            
        with iq_col3:
            st.markdown("#### 3. 💡 Skill Gap Defense")
            for q in report.interview_prep.gap_questions:
                with st.expander(f"❓ {q.question}"):
                    st.markdown(f"**Why they ask:** {q.context_or_reason}")
                    if q.sample_focus_points:
                        st.markdown("**How to Answer:**")
                        for pt in q.sample_focus_points:
                            st.write(f"• {pt}")
