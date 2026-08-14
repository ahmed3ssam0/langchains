"""
LangGraph support agent.

Flow:
  classify_intent -> retrieve_policy -> check_eligibility
      -> (take_action | escalate) -> respond -> END

State carries everything nodes need, so each node stays a small pure-ish function.
"""

import os
import json
from typing import TypedDict, Optional, List, Dict, Any, Literal
from typing_extensions import NotRequired

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

import tools

LLM_MODEL = "llama-3.3-70b-versatile"  # check console.groq.com/docs/models for current list
llm = ChatGroq(model=LLM_MODEL, temperature=0)


def _text(content) -> str:
    """LLM .content can be str or a list of content blocks. Normalize to str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


# ---------- State ----------

class AgentState(TypedDict):
    # required — always set when the graph is invoked
    customer_message: str
    order_id: Optional[str]
    # filled in progressively by nodes as the graph runs
    intent: NotRequired[Optional[str]]
    retrieved_docs: NotRequired[List[Dict[str, Any]]]
    eligibility_decision: NotRequired[Optional[Dict[str, Any]]]
    action_result: NotRequired[Optional[Dict[str, Any]]]
    escalation_reason: NotRequired[Optional[str]]
    final_response: NotRequired[Optional[str]]


# ---------- Nodes ----------

def classify_intent(state: AgentState) -> AgentState:
    prompt = f"""Classify this customer support message into exactly one category:
billing, refund, shipping, account, other.

Message: "{state['customer_message']}"

Reply with only the category word, nothing else."""
    intent = _text(llm.invoke(prompt).content).strip().lower()
    state["intent"] = intent
    return state


def retrieve_policy(state: AgentState) -> AgentState:
    docs = tools.search_policy(state["customer_message"], k=4)
    state["retrieved_docs"] = docs
    return state


def check_eligibility(state: AgentState) -> AgentState:
    order_id = state.get("order_id")
    order_info = tools.check_order_status(order_id) if order_id else None
    policy_text = "\n\n".join(f"[{d['category']}] {d['content']}" for d in state.get("retrieved_docs", []))

    prompt = f"""You are a customer support policy engine. Decide what action to take.

Customer message: "{state['customer_message']}"
Order info: {json.dumps(order_info)}

Relevant policy excerpts:
{policy_text}

Respond ONLY with JSON, no other text, in this exact shape:
{{
  "eligible": true/false,
  "action": "refund" | "reship" | "cancel_subscription" | "info_only" | "none",
  "amount": number or null,
  "confidence": "high" | "medium" | "low",
  "reasoning": "short explanation citing the policy rule used",
  "needs_human": true/false,
  "escalation_reason": "string or null"
}}

Rules for you to follow:
- If the customer is asking a general question about policy (not requesting
  a specific action on a specific order), set "action": "info_only",
  "eligible": true, "needs_human": false, and put the actual answer to
  their question in "reasoning" — answer it directly using the excerpts.
- Only escalate an info_only question if it touches something in the
  handbook's "always escalate" list (fraud, account takeover, etc).
- If the policy is ambiguous, conflicting, involves fraud/unauthorized
  charges, or amount exceeds what the policy allows without review, set
  needs_human true and explain why.
- Only set eligible true if a specific policy rule clearly supports it.
- Never invent a policy rule that isn't in the excerpts above."""

    raw = _text(llm.invoke(prompt).content).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        decision = {
            "eligible": False,
            "action": "none",
            "amount": None,
            "confidence": "low",
            "reasoning": "Could not parse policy decision.",
            "needs_human": True,
            "escalation_reason": "Model output was not valid JSON.",
        }
    state["eligibility_decision"] = decision
    return state


def route_after_eligibility(state: AgentState) -> Literal["take_action", "escalate"]:
    decision: Dict[str, Any] = state.get("eligibility_decision") or {}
    if decision.get("action") == "info_only":
        return "take_action"
    if decision.get("needs_human") or not decision.get("eligible"):
        return "escalate"
    return "take_action"


def take_action(state: AgentState) -> AgentState:
    decision: Dict[str, Any] = state.get("eligibility_decision") or {}
    action = decision.get("action")
    order_id = state.get("order_id")

    if action in ("refund", "reship", "cancel_subscription") and not order_id:
        # policy said act, but we have no order to act on — don't guess, escalate instead
        state["action_result"] = {"success": False, "message": "No order_id provided for this action."}
        state["escalation_reason"] = "Action required an order_id but none was given."
        return state

    if action == "refund" and order_id:
        result = tools.issue_refund(order_id, decision.get("amount") or 0)
    elif action == "reship" and order_id:
        result = tools.reship_order(order_id)
    elif action == "cancel_subscription" and order_id:
        result = tools.cancel_subscription(order_id)
    else:
        result = {"success": True, "message": "No system action needed, informational answer only."}

    state["action_result"] = result
    return state


def escalate(state: AgentState) -> AgentState:
    decision = state.get("eligibility_decision") or {}
    reason = decision.get("escalation_reason") or "Needs human review."
    state["escalation_reason"] = reason
    return state


def respond(state: AgentState) -> AgentState:
    decision: Dict[str, Any] = state.get("eligibility_decision") or {}
    if state.get("escalation_reason"):
        state["final_response"] = (
            "I've reviewed this against our policy, but I want a human teammate to confirm "
            f"before proceeding. Reason: {state.get('escalation_reason')} "
            "I'm escalating this to our support team now, they'll follow up shortly."
        )
    else:
        action_result: Dict[str, Any] = state.get("action_result") or {}
        state["final_response"] = (
            f"{decision.get('reasoning', '')} "
            f"{action_result.get('message', '')}"
        ).strip()
    return state


# ---------- Build graph ----------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_policy", retrieve_policy)
    graph.add_node("check_eligibility", check_eligibility)
    graph.add_node("take_action", take_action)
    graph.add_node("escalate", escalate)
    graph.add_node("respond", respond)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "retrieve_policy")
    graph.add_edge("retrieve_policy", "check_eligibility")
    graph.add_conditional_edges(
        "check_eligibility",
        route_after_eligibility,
        {"take_action": "take_action", "escalate": "escalate"},
    )
    graph.add_edge("take_action", "respond")
    graph.add_edge("escalate", "respond")
    graph.add_edge("respond", END)

    return graph.compile()