import json
from typing import List, Tuple
from first import Chunk
from LLM_Helpers import chat

# ----------------------------
# GUARDRAIL AGENT
# ----------------------------


def guardrail_filter(query: str, candidates: List[Tuple[Chunk, float]], k: int) -> List[Chunk]:
    """Ask an LLM to keep only the most relevant chunks for the query.
    Returns pruned list of Chunk.
    """
    packed = "\n\n".join([
        f"[C{i}] score={score:.3f} source={c.doc_id}\n{c.text}" for i, (c, score) in enumerate(candidates)
    ])
    system = "You are a retrieval guardrail. Pick the K most relevant chunks."
    user = f"""
Q: {query}
K: {k}
Chunks:
{packed}


Return JSON: {{"keep": [chunk_ids...]}} where chunk_ids are C0, C1, ...
"""
    out = chat(system, user)
    try:
        data = json.loads(out)
        keep = set(data.get("keep", []) )
        pruned: List[Chunk] = []
        for i, (c, _s) in enumerate(candidates):
            if f"C{i}" in keep:
                pruned.append(c)
        # fallback: if LLM response malformed
        if not pruned:
            pruned = [c for c,_ in candidates[:k]]
        return pruned[:k]
    except Exception:
        return [c for c,_ in candidates[:k]]