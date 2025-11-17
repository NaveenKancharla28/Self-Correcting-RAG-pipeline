import os
import sys
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional,Tuple, Dict,Any
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# Configurations
Embed_Model = "text-embedding-3-small"
LLM_Model = "gpt-4o-mini"
MAX_TOKENS = 4096
MAX_CHUNK_TOKENS = 600
CHUNK_OVERLAP = 50
MAX_TOKENS = 4096
CHUNK_OVERLAP = 50
TOP_K = 5
GAURDRAIL_TOP_K = 4
TARGET_SCORE = 0.75
MAX_ROUNDS = 3


@dataclass
class Chunk:
    doc_id: str
    text: str
    meta: Dict[str, Any]

class EvalResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str
    improvements: str


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")


# INGESTION 

def load_corpus() -> List[Chunk]:
    
    toy_docs = [
    (
        "doc_1",
        """
        Retrieval-Augmented Generation (RAG) augments an LLM with a retriever to fetch
        relevant context at query time. This reduces hallucination and improves factual
        accuracy. The core components are:
        - Retriever: finds candidate chunks.
        - Generator: produces the answer using retrieved context.
        - (Optional) Evaluator: scores the answer and triggers self-correction.
        """
    ),
    (
        "doc_2",
        """
        A self-correcting RAG can evaluate the answer against sources and iterate if weak.
        Typical roles: Retriever, Guardrail (filter), Answer Agent, Evaluator.
        The evaluator returns a score (0-1), rationale, and improvement suggestions.
        """
    ),
    (
        "doc_3",
        """
        FAISS is a library for efficient similarity search on dense vectors. It supports
        various indexes; IndexFlatL2 is a simple inner-product index. For production,
        use IndexIVFFlat or HNSW for speed on large corpora.
        """
    ),
    (
        "doc_4",
        """
        Chunking strategy matters. Fixed-size chunks (e.g., 600 tokens) with overlap
        (80 tokens) preserve context across boundaries. Metadata like doc_id and page
        number help traceability.
        """
    ),
    (
        "doc_5",
        """
        Guardrails filter low-relevance chunks before generation. A common threshold
        is cosine similarity > 0.7. This reduces noise and prevents the model from
        being distracted by irrelevant text.
        """
    ),
    (
        "doc_6",
        """
        Self-correction loop: generate → evaluate → if score < 0.78, refine prompt
        with evaluator feedback and retry (max 3 rounds). This boosts answer quality
        with minimal extra latency.
        """
    ),
]
    return toy_docs

def chunk_docs(pairs: List[Tuple[str, str]]) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_TOKENS, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks: List[Chunk] = []
    for doc_id, text in pairs:
        for i, piece in enumerate(splitter.split_text(text)):
            chunks.append(Chunk(doc_id=f"{doc_id}#p{i}", text=piece.strip(), meta={"source": doc_id}))
    return chunks