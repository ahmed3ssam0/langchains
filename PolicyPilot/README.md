# PolicyPilot (LangGraph + RAG)

Not a dumb FAQ bot. This agent reads real company policy docs, reasons about
eligibility per case, and takes real action (refund / reship / cancel) when
policy clearly allows it — only escalating to a human when the policy is
ambiguous, high-risk (fraud), or over the auto-approve limit.

## How it works

1. **RAG layer** — policy docs (`policy_docs/*.md`) are chunked by section
   (so a rule never gets split in half) and embedded into a local Chroma DB.
2. **LangGraph agent** — a state machine, not a single LLM call:
   - `classify_intent` — billing / refund / shipping / account / other
   - `retrieve_policy` — pulls the relevant policy chunks for this message
   - `check_eligibility` — LLM reasons over policy + order data, outputs a
     structured decision (eligible? action? confidence? needs_human?)
   - branch: `take_action` (calls a real tool: refund/reship/cancel) or
     `escalate` (hands off to a human with a reason)
   - `respond` — final message back to customer, always references the
     policy reasoning used

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here   # or put it in a .env file

# ingest your business's real policy PDF
python ingest.py path/to/company_policies.pdf

# or a whole folder of PDFs/markdown files
python ingest.py path/to/policy_folder/

# or just run with no arg, uses the sample policy_docs/ folder
python ingest.py

python main.py        # runs the demo cases against whatever you ingested
```

`ingest.py` accepts:
- a single PDF file
- a folder (recursively grabs every `.pdf` and `.md` inside)
- nothing — falls back to the sample `policy_docs/` folder

Each PDF is split by page then chunked (~500 chars, 80 char overlap) so a
policy rule doesn't get cut off mid-sentence. The chunk's `category`
metadata is taken from the filename — rename the PDF to something like
`refund_policy.pdf` if you want cleaner category tags in retrieval.

## Why this is different from a normal support bot

- It doesn't just paste FAQ text — it **decides and acts** using tools.
- Every decision must cite a real policy chunk — no invented rules.
- Guardrails are explicit: dollar limits, fraud keywords, and low-confidence
  cases all route to `escalate` automatically.
- Swap `tools.py` mock functions for real API calls (Stripe, your order DB,
  Zendesk, etc.) and this becomes production-shaped.

## Extending it

- Add `ConversationBufferMemory` / a checkpointer for multi-turn threads.
- Add a `human_review` node with LangGraph's `interrupt` for true
  human-in-the-loop instead of just logging escalation.
- Swap Chroma for Pinecone/Weaviate for scale.
- Add per-category confidence thresholds instead of one global rule.