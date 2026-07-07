from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from io import BytesIO
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re


# ============================================================
# App setup
# ============================================================

app = FastAPI(
    title="AI Resume Ranking Service",
    description="Ranks resumes against a job description using hybrid scoring.",
    version="1.0.0",
)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# Skill taxonomy
# In production, move this to a database/config file.
# ============================================================

SKILL_ALIASES: Dict[str, List[str]] = {
    "python": ["python"],
    "java": ["java"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "c++": ["c++", "cpp"],
    "c": ["c language"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite"],
    "nosql": ["nosql", "mongodb", "dynamodb", "cassandra"],
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "django": ["django"],
    "react": ["react", "react.js", "reactjs"],
    "node.js": ["node.js", "nodejs", "node"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud"],
    "linux": ["linux"],
    "git": ["git", "github", "gitlab"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "natural language processing": ["natural language processing", "nlp"],
    "computer vision": ["computer vision", "cv"],
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "llm": ["llm", "large language model", "large language models"],
    "langchain": ["langchain"],
    "llamaindex": ["llamaindex", "llama index"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "keras"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "spark": ["spark", "apache spark", "pyspark"],
    "airflow": ["airflow", "apache airflow"],
    "kafka": ["kafka", "apache kafka"],
    "faiss": ["faiss"],
    "chroma": ["chroma", "chromadb"],
    "pinecone": ["pinecone"],
    "weaviate": ["weaviate"],
    "elasticsearch": ["elasticsearch", "elastic search"],
    "data engineering": ["data engineering"],
    "data science": ["data science"],
    "mlops": ["mlops", "machine learning operations"],
    "ci/cd": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
    "rest api": ["rest api", "restful api", "rest"],
    "graphql": ["graphql"],
    "microservices": ["microservices", "microservice"],
    "system design": ["system design"],
    "distributed systems": ["distributed systems"],
    "security": ["security", "cybersecurity"],
    "iam": ["iam", "identity and access management"],
    "oauth": ["oauth", "oauth2"],
    "jwt": ["jwt", "json web token"],
}


DEGREE_LEVELS = {
    "high school": 1,
    "associate": 2,
    "bachelor": 3,
    "master": 4,
    "phd": 5,
}


DEGREE_PATTERNS = {
    "associate": [
        r"\bassociate\b",
        r"\ba\.s\.\b",
    ],
    "bachelor": [
        r"\bbachelor\b",
        r"\bbachelors\b",
        r"\bb\.s\.\b",
        r"\bbs\b",
        r"\bb\.tech\b",
        r"\bbtech\b",
        r"\bbe\b",
    ],
    "master": [
        r"\bmaster\b",
        r"\bmasters\b",
        r"\bm\.s\.\b",
        r"\bms\b",
        r"\bm\.eng\b",
        r"\bmeng\b",
    ],
    "phd": [
        r"\bphd\b",
        r"\bph\.d\.\b",
        r"\bdoctorate\b",
        r"\bdoctoral\b",
    ],
}


EDUCATION_FIELDS = [
    "computer science",
    "electrical engineering",
    "computer engineering",
    "electrical and computer engineering",
    "data science",
    "machine learning",
    "artificial intelligence",
    "software engineering",
    "information systems",
    "mathematics",
    "statistics",
    "physics",
    "business administration",
]


CERTIFICATION_PATTERNS = {
    "aws certified": r"\baws certified[\w\s\-]*",
    "azure certification": r"\bazure certified[\w\s\-]*",
    "google cloud certification": r"\bgoogle cloud certified[\w\s\-]*|\bgcp certified[\w\s\-]*",
    "pmp": r"\bpmp\b|project management professional",
    "cissp": r"\bcissp\b",
    "comptia security+": r"\bsecurity\+\b|\bcomptia security\+\b",
    "comptia network+": r"\bnetwork\+\b|\bcomptia network\+\b",
    "certified kubernetes administrator": r"\bcka\b|certified kubernetes administrator",
    "certified kubernetes application developer": r"\bckad\b|certified kubernetes application developer",
    "tensorflow developer certificate": r"tensorflow developer certificate",
}


# ============================================================
# Response schemas
# ============================================================

class EducationInfo(BaseModel):
    degrees: List[str] = Field(default_factory=list)
    highest_degree: Optional[str] = None
    fields: List[str] = Field(default_factory=list)


class ExperienceInfo(BaseModel):
    years: float = 0.0
    role_signals: List[str] = Field(default_factory=list)


class ExtractedProfile(BaseModel):
    skills: List[str]
    education: EducationInfo
    experience: ExperienceInfo
    certifications: List[str]


class ScoreBreakdown(BaseModel):
    final_score: float
    embedding_score: float
    skills_score: float
    experience_score: float
    education_score: float
    certification_score: float


class RankedCandidate(BaseModel):
    rank: int
    candidate_id: str
    filename: str
    score: float
    score_breakdown: ScoreBreakdown
    extracted_profile: ExtractedProfile
    strengths: List[str]
    gaps: List[str]


class JobRequirements(BaseModel):
    required_skills: List[str]
    preferred_skills: List[str]
    required_years_experience: float
    required_degree: Optional[str]
    required_certifications: List[str]


class RankResponse(BaseModel):
    job_requirements: JobRequirements
    total_candidates: int
    ranked_candidates: List[RankedCandidate]


# ============================================================
# Text parsing
# ============================================================

async def parse_uploaded_resume(file: UploadFile) -> str:
    filename = file.filename or ""
    extension = filename.lower().split(".")[-1]

    raw = await file.read()

    if not raw:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file {filename} is empty."
        )

    try:
        if extension == "pdf":
            return parse_pdf(raw)

        if extension == "docx":
            return parse_docx(raw)

        if extension in {"txt", "text"}:
            return parse_txt(raw)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse {filename}: {str(exc)}"
        )

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type for {filename}. Use PDF, DOCX, or TXT."
    )


def parse_pdf(raw: bytes) -> str:
    reader = PdfReader(BytesIO(raw))
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages)


def parse_docx(raw: bytes) -> str:
    doc = Document(BytesIO(raw))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def parse_txt(raw: bytes) -> str:
    return raw.decode("utf-8", errors="ignore")


# ============================================================
# Extraction utilities
# ============================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def phrase_exists(phrase: str, text: str) -> bool:
    phrase = phrase.lower().strip()

    if not phrase:
        return False

    escaped = re.escape(phrase)

    # For normal alphanumeric skills, use word boundaries.
    if re.match(r"^[a-z0-9\s]+$", phrase):
        pattern = rf"\b{escaped}\b"
    else:
        pattern = escaped

    return bool(re.search(pattern, text))


def extract_skills(text: str) -> List[str]:
    normalized = normalize_text(text)
    found = []

    for canonical_skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            if phrase_exists(alias, normalized):
                found.append(canonical_skill)
                break

    return sorted(set(found))


def extract_education(text: str) -> EducationInfo:
    normalized = normalize_text(text)

    degrees = []

    for degree, patterns in DEGREE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized):
                degrees.append(degree)
                break

    highest_degree = None

    if degrees:
        highest_degree = max(degrees, key=lambda d: DEGREE_LEVELS.get(d, 0))

    fields = []

    for field in EDUCATION_FIELDS:
        if phrase_exists(field, normalized):
            fields.append(field)

    return EducationInfo(
        degrees=sorted(set(degrees), key=lambda d: DEGREE_LEVELS.get(d, 0)),
        highest_degree=highest_degree,
        fields=sorted(set(fields)),
    )


def extract_years_of_experience(text: str) -> float:
    normalized = normalize_text(text)

    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+(?:of\s+)?(?:professional\s+)?experience",
        r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+(?:in|with|working)",
        r"experience\s+(?:of\s+)?(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)",
    ]

    values = []

    for pattern in patterns:
        matches = re.findall(pattern, normalized)
        for match in matches:
            try:
                values.append(float(match))
            except ValueError:
                pass

    # Fallback: catch general "5+ years" style references.
    fallback_matches = re.findall(r"\b(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\b", normalized)

    for match in fallback_matches:
        try:
            value = float(match)
            if value <= 50:
                values.append(value)
        except ValueError:
            pass

    return max(values) if values else 0.0


def extract_role_signals(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    role_keywords = [
        "engineer",
        "developer",
        "scientist",
        "analyst",
        "architect",
        "manager",
        "researcher",
        "intern",
        "consultant",
        "lead",
        "specialist",
    ]

    signals = []

    for line in lines:
        lower = line.lower()

        if any(keyword in lower for keyword in role_keywords):
            cleaned = re.sub(r"\s+", " ", line).strip()

            if 5 <= len(cleaned) <= 120:
                signals.append(cleaned)

    return signals[:8]


def extract_experience(text: str) -> ExperienceInfo:
    return ExperienceInfo(
        years=extract_years_of_experience(text),
        role_signals=extract_role_signals(text),
    )


def extract_certifications(text: str) -> List[str]:
    normalized = normalize_text(text)
    certifications = []

    for cert_name, pattern in CERTIFICATION_PATTERNS.items():
        if re.search(pattern, normalized):
            certifications.append(cert_name)

    return sorted(set(certifications))


def extract_profile(text: str) -> ExtractedProfile:
    return ExtractedProfile(
        skills=extract_skills(text),
        education=extract_education(text),
        experience=extract_experience(text),
        certifications=extract_certifications(text),
    )


# ============================================================
# Job description requirement extraction
# ============================================================

def extract_required_degree(job_description: str) -> Optional[str]:
    education = extract_education(job_description)
    return education.highest_degree


def extract_required_years(job_description: str) -> float:
    return extract_years_of_experience(job_description)


def extract_job_requirements(job_description: str) -> JobRequirements:
    skills = extract_skills(job_description)
    certs = extract_certifications(job_description)

    # Simple split between required and preferred.
    # Production systems can use an LLM or classifier here.
    normalized = normalize_text(job_description)

    required_skills = []
    preferred_skills = []

    for skill in skills:
        required_context_patterns = [
            rf"required[^.]*{re.escape(skill)}",
            rf"must have[^.]*{re.escape(skill)}",
            rf"need[^.]*{re.escape(skill)}",
            rf"minimum[^.]*{re.escape(skill)}",
        ]

        preferred_context_patterns = [
            rf"preferred[^.]*{re.escape(skill)}",
            rf"nice to have[^.]*{re.escape(skill)}",
            rf"bonus[^.]*{re.escape(skill)}",
            rf"plus[^.]*{re.escape(skill)}",
        ]

        is_required = any(re.search(pattern, normalized) for pattern in required_context_patterns)
        is_preferred = any(re.search(pattern, normalized) for pattern in preferred_context_patterns)

        if is_required:
            required_skills.append(skill)
        elif is_preferred:
            preferred_skills.append(skill)
        else:
            # Default to required because most JD skill mentions are meaningful.
            required_skills.append(skill)

    return JobRequirements(
        required_skills=sorted(set(required_skills)),
        preferred_skills=sorted(set(preferred_skills)),
        required_years_experience=extract_required_years(job_description),
        required_degree=extract_required_degree(job_description),
        required_certifications=certs,
    )


# ============================================================
# Scoring
# ============================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def overlap_score(candidate_items: List[str], required_items: List[str]) -> float:
    if not required_items:
        return 100.0

    candidate_set = set(candidate_items)
    required_set = set(required_items)

    matched = candidate_set.intersection(required_set)

    return 100.0 * len(matched) / len(required_set)


def score_experience(candidate_years: float, required_years: float) -> float:
    if required_years <= 0:
        return 100.0 if candidate_years > 0 else 70.0

    return min(100.0, 100.0 * candidate_years / required_years)


def score_education(candidate_degree: Optional[str], required_degree: Optional[str]) -> float:
    if not required_degree:
        return 100.0

    if not candidate_degree:
        return 0.0

    candidate_level = DEGREE_LEVELS.get(candidate_degree, 0)
    required_level = DEGREE_LEVELS.get(required_degree, 0)

    if candidate_level >= required_level:
        return 100.0

    # Partial credit if candidate is one level below.
    if candidate_level == required_level - 1:
        return 70.0

    return 40.0


def score_certifications(candidate_certs: List[str], required_certs: List[str]) -> float:
    if not required_certs:
        return 100.0

    return overlap_score(candidate_certs, required_certs)


def calculate_scores(
    job_embedding: np.ndarray,
    resume_embedding: np.ndarray,
    job_requirements: JobRequirements,
    candidate_profile: ExtractedProfile,
) -> ScoreBreakdown:
    raw_embedding_similarity = cosine_similarity(job_embedding, resume_embedding)
    embedding_score = clamp_score(raw_embedding_similarity * 100)

    skills_score = clamp_score(
        overlap_score(
            candidate_profile.skills,
            job_requirements.required_skills,
        )
    )

    experience_score = clamp_score(
        score_experience(
            candidate_profile.experience.years,
            job_requirements.required_years_experience,
        )
    )

    education_score = clamp_score(
        score_education(
            candidate_profile.education.highest_degree,
            job_requirements.required_degree,
        )
    )

    certification_score = clamp_score(
        score_certifications(
            candidate_profile.certifications,
            job_requirements.required_certifications,
        )
    )

    final_score = (
        0.45 * embedding_score +
        0.35 * skills_score +
        0.10 * experience_score +
        0.05 * education_score +
        0.05 * certification_score
    )

    return ScoreBreakdown(
        final_score=clamp_score(final_score),
        embedding_score=embedding_score,
        skills_score=skills_score,
        experience_score=experience_score,
        education_score=education_score,
        certification_score=certification_score,
    )


# ============================================================
# Strengths and gaps
# ============================================================

def build_strengths(
    profile: ExtractedProfile,
    job_requirements: JobRequirements,
    score_breakdown: ScoreBreakdown,
) -> List[str]:
    strengths = []

    matched_required_skills = sorted(
        set(profile.skills).intersection(set(job_requirements.required_skills))
    )

    if matched_required_skills:
        strengths.append(
            "Matches required skills: " + ", ".join(matched_required_skills[:10])
        )

    matched_preferred_skills = sorted(
        set(profile.skills).intersection(set(job_requirements.preferred_skills))
    )

    if matched_preferred_skills:
        strengths.append(
            "Matches preferred skills: " + ", ".join(matched_preferred_skills[:10])
        )

    if (
        job_requirements.required_years_experience > 0
        and profile.experience.years >= job_requirements.required_years_experience
    ):
        strengths.append(
            f"Meets experience requirement: {profile.experience.years:g} years found."
        )

    if (
        job_requirements.required_degree
        and profile.education.highest_degree
        and DEGREE_LEVELS.get(profile.education.highest_degree, 0)
        >= DEGREE_LEVELS.get(job_requirements.required_degree, 0)
    ):
        strengths.append(
            f"Meets education requirement: {profile.education.highest_degree} degree found."
        )

    matched_certs = sorted(
        set(profile.certifications).intersection(set(job_requirements.required_certifications))
    )

    if matched_certs:
        strengths.append(
            "Matches required certifications: " + ", ".join(matched_certs)
        )

    if score_breakdown.embedding_score >= 70:
        strengths.append("Resume is semantically aligned with the job description.")

    if not strengths:
        strengths.append("Some general semantic alignment found, but few explicit requirement matches.")

    return strengths


def build_gaps(
    profile: ExtractedProfile,
    job_requirements: JobRequirements,
) -> List[str]:
    gaps = []

    missing_required_skills = sorted(
        set(job_requirements.required_skills) - set(profile.skills)
    )

    if missing_required_skills:
        gaps.append(
            "Missing required skills: " + ", ".join(missing_required_skills[:10])
        )

    if (
        job_requirements.required_years_experience > 0
        and profile.experience.years < job_requirements.required_years_experience
    ):
        gaps.append(
            f"Experience appears below requirement: "
            f"{profile.experience.years:g} years found, "
            f"{job_requirements.required_years_experience:g} years required."
        )

    if job_requirements.required_degree:
        candidate_degree = profile.education.highest_degree

        if not candidate_degree:
            gaps.append(
                f"Required education not clearly found: {job_requirements.required_degree}."
            )
        elif DEGREE_LEVELS.get(candidate_degree, 0) < DEGREE_LEVELS.get(job_requirements.required_degree, 0):
            gaps.append(
                f"Education may be below requirement: "
                f"{candidate_degree} found, {job_requirements.required_degree} required."
            )

    missing_certs = sorted(
        set(job_requirements.required_certifications) - set(profile.certifications)
    )

    if missing_certs:
        gaps.append(
            "Missing required certifications: " + ", ".join(missing_certs)
        )

    if not gaps:
        gaps.append("No major gaps found from the parsed resume text.")

    return gaps


# ============================================================
# API endpoint
# ============================================================

@app.post("/rank", response_model=RankResponse)
async def rank_resumes(
    job_description: str = Form(...),
    resumes: List[UploadFile] = File(...),
    top_k: Optional[int] = Form(default=None),
):
    """
    Rank uploaded resumes against a job description.

    Request type:
    multipart/form-data

    Fields:
    - job_description: string
    - resumes: one or more PDF/DOCX/TXT files
    - top_k: optional integer
    """

    if not job_description.strip():
        raise HTTPException(status_code=400, detail="job_description cannot be empty.")

    if not resumes:
        raise HTTPException(status_code=400, detail="At least one resume must be uploaded.")

    job_requirements = extract_job_requirements(job_description)

    parsed_resumes = []

    for index, resume in enumerate(resumes):
        text = await parse_uploaded_resume(resume)

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"No readable text found in {resume.filename}."
            )

        parsed_resumes.append(
            {
                "candidate_id": f"candidate_{index + 1}",
                "filename": resume.filename or f"resume_{index + 1}",
                "text": text,
                "profile": extract_profile(text),
            }
        )

    all_texts = [job_description] + [item["text"] for item in parsed_resumes]
    embeddings = embedding_model.encode(all_texts, normalize_embeddings=True)

    job_embedding = embeddings[0]

    ranked = []

    for index, item in enumerate(parsed_resumes):
        resume_embedding = embeddings[index + 1]
        profile = item["profile"]

        score_breakdown = calculate_scores(
            job_embedding=job_embedding,
            resume_embedding=resume_embedding,
            job_requirements=job_requirements,
            candidate_profile=profile,
        )

        ranked.append(
            RankedCandidate(
                rank=0,
                candidate_id=item["candidate_id"],
                filename=item["filename"],
                score=score_breakdown.final_score,
                score_breakdown=score_breakdown,
                extracted_profile=profile,
                strengths=build_strengths(
                    profile=profile,
                    job_requirements=job_requirements,
                    score_breakdown=score_breakdown,
                ),
                gaps=build_gaps(
                    profile=profile,
                    job_requirements=job_requirements,
                ),
            )
        )

    ranked.sort(key=lambda candidate: candidate.score, reverse=True)

    if top_k is not None and top_k > 0:
        ranked = ranked[:top_k]

    for rank, candidate in enumerate(ranked, start=1):
        candidate.rank = rank

    return RankResponse(
        job_requirements=job_requirements,
        total_candidates=len(parsed_resumes),
        ranked_candidates=ranked,
    )


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "embedding_model": EMBEDDING_MODEL_NAME,
    }

# curl -X POST "http://127.0.0.1:8000/rank" \
#   -F "job_description=We need a Machine Learning Engineer with 3+ years of Python, FastAPI, PyTorch, RAG, Docker, AWS, and SQL experience. Master's degree preferred." \
#   -F "resumes=@resume_1.pdf" \
#   -F "resumes=@resume_2.docx" \
#   -F "top_k=5"