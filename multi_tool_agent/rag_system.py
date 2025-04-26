# rag_system.py
import os
from typing import Dict
from agent import create_bank_agent, create_farewell_agent, create_root_agent
from database_qdrant import QdrantDB
from embedding import Embedder
from document_loader import load_pdf, chunk_text

# -----------------------------------------------------------------------------
# 1) Initialize core components
# -----------------------------------------------------------------------------
db = QdrantDB(
    url=os.getenv("QDRANT_URL", "http://localhost:6333/"),
    port=int(os.getenv("QDRANT_PORT", 6333)),
    vector_size=384,
    collection_name=os.getenv("QDRANT_COLLECTION", "rag_collection"),
)

embedder = Embedder(
    model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
)

bank_agent     = create_bank_agent()
farewell_agent = create_farewell_agent()
root_agent     = create_root_agent([bank_agent, farewell_agent])

# -----------------------------------------------------------------------------
# 2) Ingestion functions
# -----------------------------------------------------------------------------
def ingest_documents(docs: Dict[str, str]) -> None:
    """
    Ingest a dict of {doc_id: text} into Qdrant with embeddings + payload.
    """
    ids      = list(docs.keys())
    texts    = list(docs.values())
    vectors  = embedder.encode(texts)
    payloads = [{"text": txt} for txt in texts]
    db.insert(ids, vectors, payloads)

def ingest_pdf(path: str, 
               max_chars: int = 2000, 
               overlap_chars: int = 200) -> None:
    """
    Load a PDF, split into overlapping chunks, then ingest.
    """
    raw_text = load_pdf(path)
    chunks = chunk_text(raw_text, max_chars=max_chars, overlap_chars=overlap_chars)
    base = os.path.splitext(os.path.basename(path))[0]
    docs = {f"{base}_chunk{i}": chunk for i, chunk in enumerate(chunks, start=1)}
    ingest_documents(docs)
    print(f"✅ Ingested {len(chunks)} chunks from {path}")

def ingest_folder(folder: str) -> None:
    """
    Recursively ingest all PDFs in a folder.
    """
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                ingest_pdf(os.path.join(root, fname))

# -----------------------------------------------------------------------------
# 3) RAG query function
# -----------------------------------------------------------------------------
def answer_question(question: str, top_k: int = 5) -> str:
    """
    Embed the user question, retrieve top_k contexts, and dispatch
    to the root_agent for a grounded answer.
    """
    q_vec = embedder.encode([question])[0]
    hits = db.search(q_vec, top_k=top_k)
    contexts = [f"{hit.id}: {hit.payload['text']}" for hit in hits]
    prompt = (
        "You are a professional assistant grounded in the following contexts (with IDs):\n\n"
        + "\n\n---\n\n".join(contexts)
        + f"\n\nUser question: {question}"
    )
    return root_agent.run(prompt)

# -----------------------------------------------------------------------------
# 4) Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ingest_pdf("./docs/myRag.pdf")  # or ingest_folder("docs/")
    reply = answer_question("What was our total revenue in Q2?")
    print("Assistant reply:", reply)
