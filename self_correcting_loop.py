from typing import NamedTuple
from answer_and_evaluator_agents import answer_agent, evaluator_agent
from first import Chunk, EvalResult
from LLM_Helpers import chat
from chucking import FaissIndex
from chucking import embed_text

# Number of top results to keep after guardrail filtering
gaurdrail_TOP_K = 3

def gaurdrail_filter(query: str, hits, top_k: int):
    """
    Simple guardrail filter placeholder.
    Currently returns the first top_k hits; replace with real filtering logic.
    """
    return hits[:top_k]

def refine_query(original_q: str, eval_feedback: EvalResult) -> str:
    """Refine the original query using evaluator feedback."""
    system = "You are a query refinement agent." 
    user = (
        f"The original query was:\n{original_q}\n"
        f"Feedback from evaluator:\nScore: {eval_feedback.improvements}\n"
        "Rewrite a sharper query (max 20 words). "

    )
    return chat(system, user, temperature=0.0)

def run_round(query:str, index: FaissIndex, embed_dim: int) -> tuple[str, EvalResult, list[Chunk]]:
    #retrive
    query_vec = embed_text([query])
    hits = index.search(query_vec, top_k=5)

    # guardrail filter
    kept_tuples = gaurdrail_filter(query, hits, gaurdrail_TOP_K)
    kept = [chunk for chunk, score in kept_tuples]

    # answer agent
    answer = answer_agent(query, kept)

    #Evaluate
    eval_result = evaluator_agent(query, answer, kept)
    return answer, eval_result, kept  
