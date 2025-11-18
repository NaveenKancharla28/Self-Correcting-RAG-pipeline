from openai import OpenAI
import first
from first import Chunk
import faiss
import numpy as np
from typing import List, Tuple

client = OpenAI(api_key=first.OPENAI_API_KEY)

def embed_text(text: List[str]) -> np.ndarray:
    #batched embedding
    resp = client.embeddings.create(model=first.Embed_Model, input=text)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # Normalize for inner product to behave like cosine
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    return vecs / norms

class FaissIndex:
    
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.payload: List[Chunk] = []

    def add(self, vecs: np.ndarray, items: List[Chunk]):
        assert vecs.shape[0] == len(items)
        self.index.add(vecs)
        self.payload.extend(items)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        D, I = self.index.search(query_vec, top_k)
        results: List[Tuple[Chunk, float]] = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            results.append((self.payload[idx], float(score)))
        return results