# 🎯 ResumeAI — Local SLM-Powered Resume Intelligence & Job Matching Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Privacy: 100% Local](https://img.shields.io/badge/Privacy-100%25%20Local%20Execution-success)](https://github.com)

A **privacy-first, local SLM-powered resume evaluation engine** with strict separation of concerns between **Deterministic Intelligence** (mathematical scoring, semantic evidence retrieval, claim verification guardrails) and **Generative Intelligence** (JD decomposition, honest gap analysis, fact-preserved bullet rewrites, and interview preparation).

---

## 🏛️ Core Architectural Principle

Instead of asking an LLM to generate subjective scores or dump full PDFs into a bloated prompt, the system cleanly separates responsibilities:

```text
┌──────────────────────────────────────────────┐
│          Deterministic Intelligence          │
│                                              │
│  • Structured PDF Section Parsing            │
│  • Embedding Cosine Similarity Matching      │
│  • Skill Taxonomy & Synonym Resolution       │
│  • Mathematical Weighted Scoring (0-100)     │
│  • Anti-Hallucination Claim Verification     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│           Generative Intelligence            │
│                 (Local SLM)                  │
│                                              │
│  • Atomic JD Requirement Decomposition       │
│  • Honest Gap Analysis & Recommendations     │
│  • Fact-Preserved Resume Bullet Rewriting    │
│  • Tailored 3-Tier Interview Question Prep   │
└──────────────────────────────────────────────┘
```

---

## 🔄 End-to-End Workflow

```text
                  ┌──────────────────────┐
                  │   Resume PDF         │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Resume Parser      │
                  │   PDF → Structured   │
                  │   Resume JSON        │
                  └──────────┬───────────┘
                             │
                             │
                  ┌──────────▼───────────┐
                  │                      │
                  │      JD Input        │
                  │                      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Local SLM          │
                  │                      │
                  │ JD Decomposition     │
                  │ Requirement Extract. │
                  └──────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   Requirement / Evidence     │
              │   Matching Engine             │
              │                              │
              │ • Skill matching              │
              │ • Semantic similarity         │
              │ • Experience matching         │
              │ • Education matching          │
              │ • Requirement weighting       │
              └──────────────┬───────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Deterministic Score  │
                  │                      │
                  │ Overall: 81/100      │
                  │ Skills: 87           │
                  │ Experience: 76       │
                  │ Education: 100       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Local SLM            │
                  │ Grounded Analysis    │
                  │                      │
                  │ • Gap analysis       │
                  │ • Recommendations    │
                  │ • Resume improvements│
                  │ • Interview questions│
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Claim Verifier       │
                  │                      │
                  │ Prevent invented     │
                  │ skills/experience    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Final Analysis JSON  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Streamlit / FastAPI  │
                  └──────────────────────┘
```

---

## ✨ Key Differentiators

1. **Deterministic Reproducibility**:
   Scoring is calculated mathematically based on category weights (Skills 40%, Experience 25%, Projects 20%, Education 10%, Certifications 5%) and importance multipliers rather than random LLM numbers.
2. **Semantic Evidence Matrix**:
   Evaluates each JD requirement against candidate resume bullets via `sentence-transformers` (`all-MiniLM-L6-v2`) cosine similarity and taxonomy aliases, categorizing each into **STRONG**, **PARTIAL**, or **MISSING**.
3. **Anti-Hallucination Guardrail (`ClaimVerifier`)**:
   When optimizing resume bullets, pure Python validation checks that every entity, tool, and metric in the rewrite is grounded in the candidate's original resume—rejecting fabricated claims.
4. **Honest Gap Recommendations**:
   Distinguishes between *missing evidence for existing experience* vs *skills the candidate should NOT falsely claim*.
5. **Tailored Interview Preparation**:
   Generates 3 targeted interview question sets:
   - **Core JD Questions**: Questions on the required technical stack.
   - **Resume Deep-Dives**: Behavioral & technical deep-dives into explicit projects on the candidate's resume.
   - **Gap Exploration**: Conceptual questions addressing the candidate's missing skill areas.
6. **100% Privacy-First & Local**:
   Runs on local CPU with sentence-transformers and Ollama (or OpenAI-compatible local endpoints like LM Studio / vLLM). Resumes and job descriptions never leave your laptop.

---

## 📂 Project Structure

```text
AI-Resume-Checker/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py              # REST API endpoints
│   │   ├── config/
│   │   │   ├── scoring_rules.yaml     # Configurable category weights & thresholds
│   │   │   ├── resume_rules.yaml      # Anti-hallucination & metric policies
│   │   │   ├── skill_taxonomy.json    # Skill synonyms & alias mapping
│   │   │   └── system_prompts.yaml    # Prompts for local SLM
│   │   ├── core/
│   │   │   ├── config.py              # Application settings
│   │   │   └── rules_loader.py        # YAML/JSON rules loader
│   │   ├── models/
│   │   │   └── schemas.py             # Strongly-typed Pydantic schemas
│   │   ├── services/
│   │   │   ├── pdf_parser.py          # PDF -> Structured Resume JSON
│   │   │   ├── embedding_engine.py    # Local SentenceTransformers + Cosine Similarity
│   │   │   ├── matching_engine.py     # Deterministic Requirement-Evidence Mapper
│   │   │   ├── scoring_engine.py      # Mathematical Weighted Scorer
│   │   │   ├── claim_verifier.py      # Anti-hallucination Guardrail
│   │   │   ├── llm_engine.py          # Local SLM client (Ollama/OpenAI-compatible)
│   │   │   └── orchestrator.py        # Pipeline Coordinator
│   │   ├── static/
│   │   │   └── index.html             # Built-in Tailwind SPA Dashboard
│   │   └── main.py                    # FastAPI server entrypoint
│   ├── requirements.txt               # Backend dependencies
│   └── .env.example                   # Environment configuration template
├── tests/
│   ├── test_orchestrator_init.py      # Import & initialization verification
│   ├── test_parser.py                 # Structured resume extraction tests
│   ├── test_scoring.py                # Deterministic scoring calculation tests
│   └── test_verifier.py               # Anti-hallucination guardrail tests
├── .streamlit/
│   └── config.toml                    # Streamlit server & dark theme configuration
├── streamlit_app.py                   # Streamlit Web Application
├── .gitignore                         # Git exclusion rules
└── README.md                          # Project Documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites

- Python 3.10, 3.11, or 3.12
- *(Optional)* [Ollama](https://ollama.com/) running a local model like `qwen2.5:3b`, `llama3.2`, or `phi3.5`. (If Ollama is not active, the system automatically uses built-in structured heuristic fallbacks).

### 2. Installation

Clone the repository and set up a virtual environment:

```powershell
# Navigate into project directory
cd E:\java\AI-Resume-Checker

# Create virtual environment
python -m venv backend\venv

# Activate virtual environment
# On Windows:
.\backend\venv\Scripts\activate
# On macOS/Linux:
source backend/venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Running the Application

You can run the application in two ways:

#### Option A: Streamlit Interactive UI (Recommended)

```powershell
streamlit run streamlit_app.py
```
Opens automatically in your browser at **`http://localhost:8501`**.

#### Option B: FastAPI REST Server + Web Dashboard

```powershell
python -m uvicorn app.main:app --reload --port 8000
```
Open **`http://localhost:8000`** for the web dashboard, or **`http://localhost:8000/docs`** for interactive Swagger API documentation.

---

## 🧪 Running the Test Suite

Execute the unit tests to verify the parser, scoring engine, and claim verifier:

```powershell
pytest tests -v
```

---

## ⚙️ Customizing Rules & Weights

All scoring formulas, alias taxonomies, and guardrails are defined in YAML/JSON configs:

* **Scoring Weights** ([`backend/app/config/scoring_rules.yaml`](backend/app/config/scoring_rules.yaml)):
  ```yaml
  scoring_weights:
    technical_skills: 0.40
    experience: 0.25
    projects: 0.20
    education: 0.10
    certifications: 0.05
  ```
* **Skill Taxonomy** ([`backend/app/config/skill_taxonomy.json`](backend/app/config/skill_taxonomy.json)):
  Add custom technical aliases (e.g. mapping `k8s` to `kubernetes`).
* **Anti-Hallucination Guardrail** ([`backend/app/config/resume_rules.yaml`](backend/app/config/resume_rules.yaml)):
  Enforce metric lineage and prohibit fabricating technologies.

---

## 📄 License

This project is licensed under the MIT License.
