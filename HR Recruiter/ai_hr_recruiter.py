import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from typing import List, cast
import time


def invoke_with_retry(runnable, payload, tries=3, delay=2):
    """Groq sometimes returns broken JSON from function call. Retry few time."""
    last_err: Exception = RuntimeError("invoke_with_retry: no attempts made")
    for attempt in range(1, tries + 1):
        try:
            return runnable.invoke(payload)
        except Exception as e:
            last_err = e
            print(f"  (attempt {attempt}/{tries} failed: {e.__class__.__name__}, retrying...)")
            time.sleep(delay)
    raise last_err

# Check key exist. No key, no run.
if not os.getenv("GROQ_API_KEY"):
    raise SystemExit("No GROQ_API_KEY found. Set it in .env. Stop.")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)


def load_pdf_text(path: str) -> str:
    """Grab text from PDF. Die loud if path bad."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No file at: {path}")
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    if not pages:
        raise ValueError(f"PDF empty or unreadable: {path}")
    # Join with newline. Keep page break. No word-mash.
    return "\n\n".join(page.page_content for page in pages)


class CVDataExtraction(BaseModel):
    name: str = Field(description="candidate name")
    email: str = Field(description="candidate email")
    profile: str = Field(description="candidate profile description")
    skills: List[str] = Field(description="soft and technical skills")


class JobRequirements(BaseModel):
    technical_skills: List[str]
    soft_skills: List[str]
    experience: str


class MatchResult(BaseModel):
    score: int = Field(ge=0, le=100, description="match score, 0 to 100")
    matching_skills: List[str]
    missing_skills: List[str]
    strengths: List[str]
    weaknesses: List[str]


def run():
    cv_path = input("CV Path: ").strip()
    job_path = input("Job Requirements Path: ").strip()

    cv_text = load_pdf_text(cv_path)
    job_text = load_pdf_text(job_path)

    # Extract candidate
    cv_extractor = llm.with_structured_output(CVDataExtraction)
    cv_prompt = ChatPromptTemplate.from_messages([
        ("system", "You extract structured candidate info from a CV. Only use what's in the text."),
        ("human", "CV Text:\n{cv_text}")
    ])
    user_skills = cast(CVDataExtraction, invoke_with_retry(cv_prompt | cv_extractor, {"cv_text": cv_text}))

    # Extract job requirements
    job_extractor = llm.with_structured_output(JobRequirements)
    job_prompt = ChatPromptTemplate.from_messages([
        ("system", "You extract structured job requirements from a job description. Only use what's in the text."),
        ("human", "Job Description Text:\n{job_text}")
    ])
    job_requirements = cast(JobRequirements, invoke_with_retry(job_prompt | job_extractor, {"job_text": job_text}))

    # Match
    matching_llm = llm.with_structured_output(MatchResult)
    match_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert HR recruiter.

Compare the candidate with the job requirements.

Return:
    - score (0-100)
    - matching_skills
    - missing_skills
    - strengths
    - weaknesses
"""),
        ("human", """
Candidate Information:{candidate}
Job Requirements:{job}
""")
    ])

    chain = match_prompt | matching_llm

    result = cast(MatchResult, invoke_with_retry(chain, {
        "candidate": user_skills.model_dump_json(indent=2),
        "job": job_requirements.model_dump_json(indent=2)
    }))

    # Pretty print. No raw object dump.
    print(f"\nCandidate: {user_skills.name} ({user_skills.email})")
    print(f"Match Score: {result.score}/100\n")

    print("Matching Skills:")
    for s in result.matching_skills:
        print(f"  + {s}")

    print("\nMissing Skills:")
    for s in result.missing_skills:
        print(f"  - {s}")

    print("\nStrengths:")
    for s in result.strengths:
        print(f"  * {s}")

    print("\nWeaknesses:")
    for s in result.weaknesses:
        print(f"  ! {s}")


if __name__ == "__main__":
    try:
        run()
    except FileNotFoundError as e:
        print(f"File error: {e}")
    except ValueError as e:
        print(f"Data error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
