from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# --- Config ---
TOPIC = "AI technology"
NUM_ARTICLES = 5
MODEL = "openai/gpt-oss-20b"

llm = ChatGroq(model=MODEL, temperature=0.3)
search = TavilySearchResults(k=NUM_ARTICLES)

# --- Step 1: Pull news ---
def fetch_news(topic: str):
    query = f"latest news today {topic}"
    results = search.invoke(query)
    return results  # list of dicts: url, content, title

# --- Step 2: Summarize chain ---
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "You summarize news article in 2-3 short bullet. Neutral tone. No opinion."),
    ("user", "Title: {title}\n\nContent: {content}\n\nSummarize:")
])
summarize_chain = summarize_prompt | llm | StrOutputParser()

# --- Step 3: Bias-check chain ---
bias_prompt = ChatPromptTemplate.from_messages([
    ("system", "You rate the tone of a news summary: Left, Center, or Right leaning, "
               "plus one short reason. Be objective, not political yourself."),
    ("user", "Summary:\n{summary}\n\nRate tone:")
])
bias_chain = bias_prompt | llm | StrOutputParser()

# --- Step 4: Run pipeline ---
def build_digest(topic: str):
    articles = fetch_news(topic)
    digest = []
    for art in articles:
        title = art.get("title", "Untitled")
        content = art.get("content", "")[:2000]  # trim long content
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

if __name__ == "__main__":
    digest = build_digest(TOPIC)
    print(f"\n=== DAILY DIGEST: {TOPIC} ===\n")
    for i, item in enumerate(digest, 1):
        print(f"{i}. {item['title']}")
        print(f"   {item['url']}")
        print(f"   Summary: {item['summary']}")
        print(f"   Tone check: {item['bias_check']}")
        print()