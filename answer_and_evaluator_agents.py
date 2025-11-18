import json
from typing import List
from first import Chunk, EvalResult, TOP_K
from LLM_Helpers import chat

def answer_agent(query: str, chunks: List[Chunk]) -> str:
    """
    Simulates an answer agent that retrieves relevant chunks and generates an answer.
    """
    # For simplicity, we concatenate top K chunks
    relevant_texts = [chunk.text for chunk in chunks[:TOP_K]]
    context = "\n".join(relevant_texts)
    answer = f"Based on the following context:\n{context}\nThe answer to your query '{query}' is ..."
    return answer


def evaluator_agent(query: str, answer: str, context_chunks: List[Chunk]) -> EvalResult:
    context = "\n\n".join([f"From {c.doc_id}]\n{c.text}" for c in context_chunks])
    
    system = (
        "You are an strict evaluator. Compare the ANSWER to CONTEXT."
        " Score factual consistency in [0,1]. 1 = fully supported, 0 = unsupported."
    )
    user = f"""
QUESTION: {query}


CONTEXT:
{context}


ANSWER:
{answer}


Return JSON with keys: score (0..1 float), rationale (why), improvements (how to fix).
"""
    raw = chat(system, user)
    try:
        data = json.loads(raw)
        return EvalResult(**data)
    except Exception:
        # Robust fallback: quick heuristic if JSON fails
        text = raw.strip()
        score = 0.0
        if "supported" in text.lower():
            score = 0.6
        return EvalResult(score=score, rationale=text[:500], improvements="Add more grounded citations.")