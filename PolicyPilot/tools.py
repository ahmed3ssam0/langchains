"""
Tools the agent can call. Mock DB stands in for real backend/API.
Every action tool returns a structured result so the graph can log what happened.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# --- Mock order database ---
MOCK_ORDERS = {
    "ORD1001": {"customer": "alice", "item": "Wireless Mouse", "amount": 29.99, "status": "delivered"},
    "ORD1002": {"customer": "bob", "item": "Yearly Subscription", "amount": 149.00, "status": "active"},
    "ORD1003": {"customer": "carol", "item": "4K Monitor", "amount": 420.00, "status": "delivered"},
}

_vectordb = None


def get_vectordb():
    global _vectordb
    if _vectordb is None:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return _vectordb


def search_policy(query: str, k: int = 3):
    """Retrieve relevant policy chunks for a query."""
    db = get_vectordb()
    results = db.similarity_search(query, k=k)
    return [
        {"content": r.page_content, "category": r.metadata.get("category"), "source": r.metadata.get("source")}
        for r in results
    ]


def check_order_status(order_id: str):
    """Look up an order in the mock DB."""
    order = MOCK_ORDERS.get(order_id)
    if not order:
        return {"found": False, "error": f"Order {order_id} not found"}
    return {"found": True, **order}


def issue_refund(order_id: str, amount: float):
    """Mock refund action. In real system, this calls payment provider API."""
    order = MOCK_ORDERS.get(order_id)
    if not order:
        return {"success": False, "error": f"Order {order_id} not found"}
    return {
        "success": True,
        "order_id": order_id,
        "refunded_amount": amount,
        "message": f"Refunded ${amount:.2f} to {order['customer']} for order {order_id}",
    }


def reship_order(order_id: str):
    """Mock reship action."""
    order = MOCK_ORDERS.get(order_id)
    if not order:
        return {"success": False, "error": f"Order {order_id} not found"}
    return {"success": True, "order_id": order_id, "message": f"Reshipped {order['item']} for order {order_id}"}


def cancel_subscription(order_id: str):
    """Mock subscription cancel action."""
    order = MOCK_ORDERS.get(order_id)
    if not order:
        return {"success": False, "error": f"Order {order_id} not found"}
    MOCK_ORDERS[order_id]["status"] = "cancelled"
    return {"success": True, "order_id": order_id, "message": f"Subscription {order_id} cancelled"}
