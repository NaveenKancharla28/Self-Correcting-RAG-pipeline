import sys
import json
import time
from typing import List, Tuple, Dict, Any
from first import Chunk, load_corpus, chunk_docs, MAX_ROUNDS, TARGET_SCORE
from chucking import FaissIndex, embed_text
from self_correcting_loop import run_round, refine_query


def build_index(chunks: List[Chunk]) -> Tuple[FaissIndex, int]:
    texts = [c.text for c in chunks]
    vecs = embed_text(texts)
    dim = vecs.shape[1]
    store = FaissIndex(dim)
    store.add(vecs, chunks)
    return store, dim


def self_correcting_answer(question: str) -> Dict[str, Any]:
    docs = load_corpus()
    chunks = chunk_docs(docs)
    index, dim = build_index(chunks)

    transcript = []
    q = question
    for round_i in range(1, MAX_ROUNDS + 1):
        start = time.time()
        ans, score, used = run_round(q, index, dim)
        took = time.time() - start
        transcript.append({
            "round": round_i,
            "question": q,
            "answer": ans,
            "score": score.model_dump(),
            "used_chunks": [u.doc_id for u in used],
            "latency_sec": round(took, 2),
        })
        if score.score >= TARGET_SCORE:
            break
        # Refine and try again
        q = refine_query(q, score)

    return {
        "final_answer": transcript[-1]["answer"],
        "final_score": transcript[-1]["score"],
        "rounds": transcript,
    }


# ----------------------------
# CLI entry
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python self_correcting_rag.py 'Your question here'\n")
        sys.exit(1)
    question = sys.argv[1]
    result = self_correcting_answer(question)
    print(json.dumps(result, indent=2))