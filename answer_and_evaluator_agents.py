def answer_agent(query: str, chunks: List[Chunk]) -> str:
    """
    Simulates an answer agent that retrieves relevant chunks and generates an answer.
    """
    # For simplicity, we concatenate top K chunks
    relevant_texts = [chunk.text for chunk in chunks[:TOP_K]]
    context = "\n".join(relevant_texts)
    answer = f"Based on the following context:\n{context}\nThe answer to your query '{query}' is ..."
    return answer