"""
Debate Bot — Web App (Streamlit)
Same logic as debate_bot.py, now with UI.

Setup:
    pip install langchain langchain-openai streamlit

Env var need:
    OPENAI_API_KEY

Run:
    streamlit run debate_bot_web.py
"""

import os

import dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

MODEL = "openai/gpt-oss-20b"

# --- Page setup ---
st.set_page_config(page_title="Debate Bot", page_icon="⚔️", layout="centered")
st.title("⚔️ AI Debate Bot")
st.caption("Two agent argue opposite side. Judge score winner.")

# --- Sidebar config ---
with st.sidebar:
    st.header("Settings")
    api_key = dotenv.dotenv_values(".env").get("GROQ_API_KEY")
    rounds = st.slider("Rounds", 1, 5, 3)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

topic = st.text_input("Debate topic", "Should AI development be paused for safety regulation?")
start_btn = st.button("Start Debate", type="primary", use_container_width=True)

# --- Chains ---
def build_chains(api_key, temperature):
    llm = ChatGroq(model=MODEL, temperature=temperature, api_key=api_key)

    pro_prompt = ChatPromptTemplate.from_messages([
        ("system", "You debate FOR this topic: {topic}\n"
                   "Argue pro-side. Strong, concise, 3-4 sentence per turn. "
                   "Respond directly to opponent last point when given."),
        ("user", "Debate history so far:\n{history}\n\nYour turn (Pro):")
    ])
    con_prompt = ChatPromptTemplate.from_messages([
        ("system", "You debate AGAINST this topic: {topic}\n"
                   "Argue con-side. Strong, concise, 3-4 sentence per turn. "
                   "Respond directly to opponent last point when given."),
        ("user", "Debate history so far:\n{history}\n\nYour turn (Con):")
    ])
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a neutral debate judge. Read full transcript. "
                   "Score Pro and Con out of 10 on argument strength. "
                   "Declare winner. Give 2-3 sentence reason."),
        ("user", "Topic: {topic}\n\nTranscript:\n{history}\n\nJudge:")
    ])

    pro_chain = pro_prompt | llm | StrOutputParser()
    con_chain = con_prompt | llm | StrOutputParser()
    judge_chain = judge_prompt | llm | StrOutputParser()
    return pro_chain, con_chain, judge_chain

# --- Run debate ---
if start_btn:
    if not api_key:
        st.error("Need API key. Put in sidebar.")
        st.stop()

    pro_chain, con_chain, judge_chain = build_chains(api_key, temperature)
    history = ""

    progress = st.empty()

    for r in range(1, rounds + 1):
        progress.info(f"Round {r} of {rounds} — Pro thinking...")
        pro_turn = pro_chain.invoke({"topic": topic, "history": history})
        history += f"\n[Round {r} - Pro]: {pro_turn}"

        with st.chat_message("assistant", avatar="🔵"):
            st.markdown(f"**Round {r} — Pro**")
            st.write(pro_turn)

        progress.info(f"Round {r} of {rounds} — Con thinking...")
        con_turn = con_chain.invoke({"topic": topic, "history": history})
        history += f"\n[Round {r} - Con]: {con_turn}"

        with st.chat_message("assistant", avatar="🔴"):
            st.markdown(f"**Round {r} — Con**")
            st.write(con_turn)

    progress.info("Judge deliberating...")
    verdict = judge_chain.invoke({"topic": topic, "history": history})
    progress.empty()

    st.divider()
    st.subheader("⚖️ Judge Verdict")
    st.success(verdict)