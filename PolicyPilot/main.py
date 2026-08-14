"""
Interactive loop. Talk to the agent like a real customer chat.
Type 'quit' or 'exit' to stop.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from graph import build_graph, AgentState

agent = build_graph()

print("PolicyPilot — type 'quit' to exit.\n")

while True:
    message = input("You: ").strip()
    if message.lower() in ("quit", "exit"):
        print("Agent: Goodbye!")
        break
    if not message:
        continue

    order_id = input("Order ID (press Enter to skip): ").strip() or None

    state = AgentState(customer_message=message, order_id=order_id)
    result = agent.invoke(state)

    print(f"\nAgent: {result.get('final_response')}")
    print(f"  [intent: {result.get('intent')} | action: {(result.get('eligibility_decision') or {}).get('action')} | escalated: {bool(result.get('escalation_reason'))}]\n")