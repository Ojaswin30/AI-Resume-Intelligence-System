import asyncio
import sys
from pathlib import Path

# Add backend directory to sys.path so imports work seamlessly
backend_dir = Path(__file__).resolve().parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import streamlit as st
import pandas as pd
from app.services.orchestrator import ResumeIntelligenceOrchestrator
from app.core.config import settings
from app.core.rules_loader import load_scoring_rules

# Page Configuration
st.set_page_config(
    page_title="ResumeAI — Local SLM Resume Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished dark theme UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366F1, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #9CA3AF;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .badge-strong {
        color: #10B981;
        background-color: rgba(16, 185, 129, 0.15);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-partial {
        color: #F59E0B;
        background-color: rgba(245, 158, 11, 0.15);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-missing {
        color: #EF4444;
        background-color: rgba(239, 68, 68, 0.15);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sample JD Template
DEFAULT_SAMPLE_JD = """Senior AI Engineer (GenAI & Search)
Requirements:
- Strong proficiency in Python, FastAPI, and asynchronous backend engineering.
- Hands-on experience developing LLM applications using HuggingFace Transformers.
- Experience implementing RAG (Retrieval-Augmented Generation) and vector search using FAISS or similar vector stores.
- Familiarity with LangChain or LlamaIndex for orchestration.
- Knowledge of relational databases (PostgreSQL/SQL) and Docker containerization.
- Bachelor's or Master's degree in Computer Science or related engineering field."""

# Initialize Orchestrator in session state
@st.cache_resource
def get_orchestrator():
    return ResumeIntelligenceOrchestrator()

orchestrator = get_orchestrator()
scoring_rules = load_scoring_rules()

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.markdown("### ⚙️ Engine Settings")
    st.info("🔒 **100% Local Execution**\n\nEmbeddings run directly on your CPU via `all-MiniLM-L6-v2`. Your resume never leaves this machine.")
    
    st.markdown("#### ⚖️ Scoring Weights")
    weights = scoring_rules.get("scoring_weights", {})
    for cat, wt in weights.items():
        st.write(f"• **{cat.replace('_', ' ').title()}**: `{int(wt*100)}%`")
    
    st.markdown("---")
    st.caption("ResumeAI Local SLM Engine v2.0")

# Main Header
st.markdown('<div class="main-title">ResumeAI — Local SLM Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deterministic Evidence Matching • Anti-Hallucination Guardrails • Grounded Recommendations</div>', unsafe_allow_html=True)

# Layout Columns for Inputs
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("1. Candidate Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF format)", type=["pdf"])
    if uploaded_file:
        st.success(f"✓ File loaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

with col_right:
    st.subheader("2. Target Job Description (JD)")
    use_sample = st.checkbox("Use Sample AI Engineer JD", value=False)
    jd_default_val = DEFAULT_SAMPLE_JD if use_sample else ""
    jd_text = st.text_area("Paste the Job Description", value=jd_default_val, height=180, placeholder="Paste JD requirements here...")

# Action Button
st.markdown("---")
analyze_btn = st.button("🚀 Run Local Intelligence Analysis", type="primary", use_container_width=True)

if analyze_btn:
    if not uploaded_file:
        st.error("⚠️ Please upload a resume PDF file before running analysis.")
    elif not jd_text.strip():
        st.error("⚠️ Please paste or provide the target Job Description.")
    else:
        with st.spinner("⏳ Running deterministic matching and local SLM synthesis..."):
            pdf_bytes = uploaded_file.getvalue()
            # Run async orchestrator pipeline safely in Streamlit
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                report = loop.run_until_complete(orchestrator.run_full_analysis(pdf_bytes, jd_text))
                st.session_state["report"] = report
            finally:
                loop.close()

# Display Results if Available in Session State
if "report" in st.session_state:
    report = st.session_state["report"]
    det_score = report.deterministic_score
    
    st.markdown("## 📊 Evaluation Summary")
    
    # Summary Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric(label="🎯 Overall Match", value=f"{det_score.overall_score:.0f} / 100")
    with m_col2:
        skills_score = det_score.category_scores.get("technical_skills")
        st.metric(label="💻 Tech Skills (40%)", value=f"{skills_score.score:.0f}%" if skills_score else "N/A")
    with m_col3:
        exp_score = det_score.category_scores.get("experience")
        st.metric(label="🏢 Experience (25%)", value=f"{exp_score.score:.0f}%" if exp_score else "N/A")
    with m_col4:
        prj_score = det_score.category_scores.get("projects")
        st.metric(label="🚀 Projects (20%)", value=f"{prj_score.score:.0f}%" if prj_score else "N/A")
    with m_col5:
        edu_score = det_score.category_scores.get("education")
        st.metric(label="🎓 Education (10%)", value=f"{edu_score.score:.0f}%" if edu_score else "N/A")

    st.markdown("---")

    # Tabs for In-depth Findings
    tab_matrix, tab_recs, tab_rewrites, tab_interview = st.tabs([
        "📋 Evidence Matrix",
        "💡 Gaps & Recommendations",
        "✍️ Grounded Bullet Optimizer",
        "🎯 Tailored Interview Prep"
    ])

    # Tab 1: Evidence Matrix
    with tab_matrix:
        st.markdown("### Requirement vs. Resume Evidence")
        st.caption("Exact cosine similarity match scores against parsed candidate bullets.")
        
        matrix_data = []
        for m in det_score.evidence_matrix:
            status_icon = "🟢 STRONG" if m.status == "STRONG" else ("🟡 PARTIAL" if m.status == "PARTIAL" else "🔴 MISSING")
            matrix_data.append({
                "JD Requirement": m.requirement,
                "Category": m.category.value if hasattr(m.category, 'value') else str(m.category),
                "Status": status_icon,
                "Similarity": f"{m.score * 100:.0f}%",
                "Best Matching Resume Evidence": m.best_matching_bullet or "No direct evidence found"
            })
        
        df_matrix = pd.DataFrame(matrix_data)
        st.dataframe(df_matrix, width="stretch", hide_index=True)

    # Tab 2: Gaps & Grounded Recommendations
    with tab_recs:
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown("#### ✅ Honest Recommendations")
            for rec in report.grounded_analysis.honest_recommendations:
                st.success(rec)
        with r_col2:
            st.markdown("#### ⚠️ Critical Missing Gaps (Do Not Falsify)")
            for gap in report.grounded_analysis.critical_gaps:
                st.error(gap)

        st.markdown("#### 📑 Section-by-Section Optimizations")
        sec_cols = st.columns(len(report.grounded_analysis.section_improvements) or 1)
        for idx, (sec_name, tips) in enumerate(report.grounded_analysis.section_improvements.items()):
            with sec_cols[idx % len(sec_cols)]:
                st.markdown(f"**{sec_name.upper()}**")
                for t in tips:
                    st.write(f"• {t}")

    # Tab 3: Grounded Bullet Optimizer
    with tab_rewrites:
        st.markdown("### 🛡️ Fact-Preserved Bullet Rewriting")
        st.info("The **Claim Verifier Guardrail** ensures that rewritten bullets highlight relevant keywords without fabricating technologies or ungrounded metrics.")
        
        for rw in report.bullet_rewrites:
            with st.container(border=True):
                badge = "✅ Fact-Checked & Verified" if rw.verification_passed else "⚠️ Guardrail Warning"
                st.markdown(f"**Target Requirement:** `{rw.target_requirement}` &nbsp;|&nbsp; {badge}")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("ORIGINAL BULLET")
                    st.code(rw.original_bullet, language="text")
                with c2:
                    st.caption("OPTIMIZED VERSION")
                    st.code(rw.rewritten_bullet, language="text")
                
                st.caption(f"**Rationale:** {rw.rationale}")

    # Tab 4: Interview Prep
    with tab_interview:
        st.markdown("### 🎯 Role & Background Specific Interview Preparation")
        
        q_col1, q_col2, q_col3 = st.columns(3)
        with q_col1:
            st.markdown("#### 1. Core JD Questions")
            for q in report.interview_prep.technical_questions:
                with st.expander(q.question):
                    st.write(f"**Context:** {q.context_or_reason}")
                    if q.sample_focus_points:
                        st.write(f"**Key Points:** {', '.join(q.sample_focus_points)}")
        
        with q_col2:
            st.markdown("#### 2. Resume Deep-Dives")
            for q in report.interview_prep.resume_deep_dives:
                with st.expander(q.question):
                    st.write(f"**Context:** {q.context_or_reason}")
                    if q.sample_focus_points:
                        st.write(f"**Key Points:** {', '.join(q.sample_focus_points)}")
        
        with q_col3:
            st.markdown("#### 3. Gap Exploration")
            for q in report.interview_prep.gap_questions:
                with st.expander(q.question):
                    st.write(f"**Context:** {q.context_or_reason}")
                    if q.sample_focus_points:
                        st.write(f"**Key Points:** {', '.join(q.sample_focus_points)}")
