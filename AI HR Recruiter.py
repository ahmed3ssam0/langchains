from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.utils.pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from typing import List

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)

loader = PyPDFLoader(input("CV Path: "))
pages = loader.load_and_split()

text = " ".join(list(map(lambda page: page.page_content, pages)))

class CVDataExtraction(BaseModel):
    name: str = Field(description="candidate name")
    email: str = Field(description="candidate email")
    profile: str = Field(description="candidate profile description")
    skills: List[str] = Field(description="soft and technical skills")


structured_llm = llm.with_structured_output(CVDataExtraction)

user_skills = structured_llm.invoke(text)

loader = PyPDFLoader(input("Job Requirements: "))
pages = loader.load_and_split()

text = " ".join(list(map(lambda page: page.page_content, pages)))

class JobRequirements(BaseModel):
    technical_skills: List[str]
    soft_skills: List[str]
    experience: str

structured_llm = llm.with_structured_output(JobRequirements)
job_requirements = structured_llm.invoke(text)

class MatchResult(BaseModel):
    score: int
    matching_skills: List[str]
    missing_skills: List[str]
    strengths: List[str]
    weaknesses: List[str]

matching_llm = llm.with_structured_output(MatchResult)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert HR recruiter.

        Compare the candidate with the job requirements.

        Return:
            - score (0-100)
            - matching_skills
            - missing_skills
            - strengths
            - weaknesses
    """),

    (
        "human",
        """
        Candidate Information:{candidate}
        Job Requirements:{job}
        """
    )
])

chain = prompt | matching_llm

res = chain.invoke({
    "candidate": user_skills.model_dump_json(indent=2),
    "job": job_requirements.model_dump_json(indent=2)
})

print(res)
