"""
Build the policy knowledge base (RAG) from a real business PDF (or a folder
of PDFs/markdown files). No more hardcoded sample docs — point this at
whatever policy PDF the business gives you.

Usage:
    python ingest.py path/to/policy.pdf
    python ingest.py path/to/policy_folder/          # loads every .pdf and .md inside
    python ingest.py                                  # defaults to policy_docs/ folder
"""

import os
import sys
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DEFAULT_POLICY_DIR = os.path.join(os.path.dirname(__file__), "policy_docs")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

HEADERS_TO_SPLIT_ON = [
    ("#", "doc_title"),
    ("##", "section"),
]

# PDFs have no markdown headers, so we lean on this for chunking instead.
# chunk_size kept smallish so a single policy rule doesn't get diluted with
# unrelated neighboring text when retrieved.
PDF_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def collect_input_paths(path: str) -> list[str]:
    """Resolve a file or folder argument into a flat list of pdf/md file paths."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        pdfs = glob.glob(os.path.join(path, "**", "*.pdf"), recursive=True)
        mds = glob.glob(os.path.join(path, "**", "*.md"), recursive=True)
        return sorted(pdfs + mds)
    raise FileNotFoundError(f"No such file or folder: {path}")


def load_pdf(path: str):
    """Load a PDF, one Document per page, then chunk it."""
    loader = PyPDFLoader(path)
    pages = loader.load()  # each page.metadata already has "page" number

    category = os.path.splitext(os.path.basename(path))[0]
    chunks = PDF_SPLITTER.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["source"] = path
        chunk.metadata["category"] = category
    return chunks


def load_markdown(path: str):
    """Load a markdown policy doc, split by section headers (keeps a rule whole)."""
    loader = TextLoader(path)
    raw_docs = loader.load()

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    all_chunks = []
    for doc in raw_docs:
        category = os.path.splitext(os.path.basename(path))[0].replace("_policy", "")
        split_docs = md_splitter.split_text(doc.page_content)
        for chunk in split_docs:
            chunk.metadata["source"] = path
            chunk.metadata["category"] = category
        all_chunks.extend(fallback_splitter.split_documents(split_docs))
    return all_chunks


def load_and_split(input_path: str):
    file_paths = collect_input_paths(input_path)
    if not file_paths:
        raise ValueError(f"No .pdf or .md files found under: {input_path}")

    all_chunks = []
    for path in file_paths:
        if path.lower().endswith(".pdf"):
            print(f"Loading PDF: {path}")
            all_chunks.extend(load_pdf(path))
        elif path.lower().endswith(".md"):
            print(f"Loading markdown: {path}")
            all_chunks.extend(load_markdown(path))
    return all_chunks


def build_vectorstore(input_path: str = DEFAULT_POLICY_DIR):
    chunks = load_and_split(input_path)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    # langchain-chroma persists automatically when persist_directory is set
    print(f"Ingested {len(chunks)} chunks from '{input_path}' into {DB_DIR}")
    return vectordb


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_POLICY_DIR
    build_vectorstore(target)
