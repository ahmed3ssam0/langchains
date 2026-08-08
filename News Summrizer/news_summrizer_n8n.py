"""
News Summarizer — API Endpoint (FastAPI)
Wrap news_summarizer.py logic. n8n hit this URL, get digest back.
Use Groq (openai/gpt-oss-20b) as model. Keys load from .env.

Setup:
    pip install fastapi uvicorn langchain langchain-groq langchain-community tavily-python python-dotenv

.env file need (same folder):
    GROQ_API_KEY=your_key_here
    TAVILY_API_KEY=your_key_here

Run local:
    uvicorn news_api:app --host 0.0.0.0 --port 8000

Then n8n HTTP Request node call:
    GET http://localhost:8000/digest/text?topic=AI
"""

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Query
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="News Summarizer API")

MODEL = "openai/gpt-oss-20b"
llm = ChatGroq(model=MODEL, temperature=0.3, api_key=os.environ.get("GROQ_API_KEY"))
search = TavilySearchResults(k=5)

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "You summarize news article in 2-3 short bullet. Neutral tone. No opinion."),
    ("user", "Title: {title}\n\nContent: {content}\n\nSummarize:")
])
summarize_chain = summarize_prompt | llm | StrOutputParser()

bias_prompt = ChatPromptTemplate.from_messages([
    ("system", "You rate the tone of a news summary: Left, Center, or Right leaning, "
               "plus one short reason. Be objective, not political yourself."),
    ("user", "Summary:\n{summary}\n\nRate tone:")
])
bias_chain = bias_prompt | llm | StrOutputParser()


def build_digest(topic: str, num_articles: int = 5):
    query = f"latest news today {topic}"
    articles = search.invoke(query)[:num_articles]

    digest = []
    for art in articles:
        title = art.get("title", "Untitled")
        content = art.get("content", "")[:2000]
        url = art.get("url", "")

        summary = summarize_chain.invoke({"title": title, "content": content})
        bias = bias_chain.invoke({"summary": summary})

        digest.append({
            "title": title,
            "url": url,
            "summary": summary,
            "bias_check": bias
        })
    return digest


@app.get("/digest")
def get_digest(topic: str = Query("AI technology"), num_articles: int = Query(5)):
    """Returns JSON digest — good for n8n to loop over."""
    digest = build_digest(topic, num_articles)
    return {"topic": topic, "articles": digest}


@app.get("/digest/text")
def get_digest_text(topic: str = Query("AI technology"), num_articles: int = Query(5)):
    """Returns one plain-text block — good for direct Telegram message body."""
    digest = build_digest(topic, num_articles)

    lines = [f"📰 Daily Digest: {topic}\n"]
    for i, item in enumerate(digest, 1):
        lines.append(f"{i}. {item['title']}")
        lines.append(f"{item['summary']}")
        lines.append(f"Tone: {item['bias_check']}")
        lines.append(f"{item['url']}\n")

    text = "\n".join(lines)
    return {"text": text}


@app.get("/health")
def health():
    return {"status": "ok"}