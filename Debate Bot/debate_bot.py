from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
TOPIC = "Should AI development be paused for safety regulation?"
ROUNDS = 3
MODEL = "openai/gpt-oss-20b"

llm = ChatGroq(model=MODEL, temperature=0.7)

# --- Agent A: Pro side ---
pro_prompt = ChatPromptTemplate.from_messages([
    ("system", "You debate FOR this topic: {topic}\n"
               "Argue pro-side. Strong, concise, 3-4 sentence per turn. "
               "Respond directly to opponent last point when given."),
    ("user", "Debate history so far:\n{history}\n\nYour turn (Pro):")
])
pro_chain = pro_prompt | llm | StrOutputParser()

# --- Agent B: Con side ---
con_prompt = ChatPromptTemplate.from_messages([
    ("system", "You debate AGAINST this topic: {topic}\n"
               "Argue con-side. Strong, concise, 3-4 sentence per turn. "
               "Respond directly to opponent last point when given."),
    ("user", "Debate history so far:\n{history}\n\nYour turn (Con):")
])
con_chain = con_prompt | llm | StrOutputParser()

# --- Judge agent ---
judge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a neutral debate judge. Read full transcript. "
               "Score Pro and Con out of 10 on argument strength. "
               "Declare winner. Give 2-3 sentence reason."),
    ("user", "Topic: {topic}\n\nTranscript:\n{history}\n\nJudge:")
])
judge_chain = judge_prompt | llm | StrOutputParser()

# --- Run debate loop ---
def run_debate(topic: str, rounds: int):
    history = ""
    transcript = []

    for r in range(1, rounds + 1):
        pro_turn = pro_chain.invoke({"topic": topic, "history": history})
        history += f"\n[Round {r} - Pro]: {pro_turn}"
        transcript.append(("Pro", pro_turn))

        con_turn = con_chain.invoke({"topic": topic, "history": history})
        history += f"\n[Round {r} - Con]: {con_turn}"
        transcript.append(("Con", con_turn))

    verdict = judge_chain.invoke({"topic": topic, "history": history})
    return transcript, verdict

if __name__ == "__main__":
    print(f"\n=== DEBATE: {TOPIC} ===\n")
    transcript, verdict = run_debate(TOPIC, ROUNDS)

    for round_num, (side, text) in enumerate(transcript, 1):
        r = (round_num + 1) // 2
        print(f"[Round {r} - {side}]")
        print(text)
        print()

    print("=== JUDGE VERDICT ===")
    print(verdict)