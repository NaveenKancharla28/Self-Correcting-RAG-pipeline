# Self-Correcting RAG Pipeline

A production-ready implementation of a self-correcting Retrieval-Augmented Generation (RAG) system that iteratively refines answers based on quality evaluation.

## Overview

This project demonstrates an advanced RAG architecture that includes:

- **Semantic Retrieval**: FAISS-based vector search with cosine similarity
- **Guardrail Filtering**: LLM-powered relevance filtering to reduce noise
- **Answer Generation**: Context-aware response synthesis
- **Self-Correction Loop**: Automatic query refinement based on answer quality evaluation
- **Iterative Improvement**: Up to 3 rounds of refinement until target quality is achieved

## Features

- Modular architecture with separate agents for retrieval, filtering, generation, and evaluation
- Efficient vector search using FAISS with normalized embeddings
- OpenAI GPT-4 integration for answer generation and evaluation
- Configurable quality thresholds and iteration limits
- Detailed tracking of answer evolution across rounds
- Support for custom document corpora

## Architecture

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────────┐
│  Retrieval (FAISS Vector Search)        │
└──────┬──────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────┐
│  Guardrail Filter (Relevance Check)     │
└──────┬──────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────┐
│  Answer Agent (Context + Generation)    │
└──────┬──────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────┐
│  Evaluator Agent (Score 0-1)            │
└──────┬──────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────┐
│  Score >= Target? ──Yes──> Return       │
│         │                                │
│        No                                │
│         v                                │
│  Refine Query & Retry (Max 3 rounds)    │
└─────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/NaveenKancharla28/Self-Correcting-RAG-pipeline.git
cd Self-Correcting-RAG-pipeline
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

Run the pipeline with a question:

```bash
python main.py "What is RAG?"
```

### Example Output

```json
{
  "final_answer": "Retrieval-Augmented Generation (RAG) is a method...",
  "final_score": {
    "score": 1.0,
    "rationale": "The answer accurately describes RAG...",
    "improvements": "To enhance clarity, the answer could..."
  },
  "rounds": [
    {
      "round": 1,
      "question": "What is RAG?",
      "answer": "...",
      "score": {...},
      "used_chunks": ["doc_1#p0", "doc_2#p0"],
      "latency_sec": 3.48
    }
  ]
}
```

## Configuration

Key parameters can be adjusted in `first.py`:

- `MAX_CHUNK_TOKENS`: Maximum tokens per chunk (default: 600)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `TOP_K`: Number of chunks to retrieve (default: 5)
- `GAURDRAIL_TOP_K`: Chunks to keep after filtering (default: 4)
- `TARGET_SCORE`: Minimum acceptable answer quality (default: 0.75)
- `MAX_ROUNDS`: Maximum refinement iterations (default: 3)
- `LLM_Model`: OpenAI model for generation (default: "gpt-4o-mini")
- `Embed_Model`: OpenAI embedding model (default: "text-embedding-3-small")

## Project Structure

```
Self-Correcting-RAG-Pipeline/
├── first.py                          # Core configuration and data models
├── chucking.py                       # Text embedding and FAISS index
├── LLM_Helpers.py                    # OpenAI chat completion wrapper
├── Gaurdrail_agent.py                # Relevance filtering logic
├── answer_and_evaluator_agents.py   # Answer generation and evaluation
├── self_correcting_loop.py          # Query refinement and orchestration
├── main.py                          # Entry point and pipeline execution
├── .env                             # Environment variables (API keys)
└── README.md                        # This file
```

## How It Works

1. **Document Ingestion**: Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter`

2. **Vector Index Creation**: Chunks are embedded using OpenAI's embedding model and stored in a FAISS index with cosine similarity

3. **Retrieval**: For a given query, the system:
   - Embeds the query
   - Searches the FAISS index for top-k similar chunks
   - Applies guardrail filtering to remove low-relevance results

4. **Answer Generation**: An LLM synthesizes an answer from the filtered chunks

5. **Evaluation**: A separate evaluator agent scores the answer (0-1) based on:
   - Factual consistency with source context
   - Completeness
   - Relevance to the query

6. **Self-Correction**: If score < target threshold:
   - The evaluator provides improvement suggestions
   - The query is refined using the feedback
   - The process repeats (up to MAX_ROUNDS)

## Dependencies

- `openai`: OpenAI API client
- `faiss-cpu`: Vector similarity search
- `langchain-text-splitters`: Document chunking
- `pydantic`: Data validation
- `numpy`: Array operations
- `python-dotenv`: Environment variable management

## Examples

### Basic Question
```bash
python main.py "What is RAG?"
```

### Complex Question Requiring Refinement
```bash
python main.py "How does the self-correction mechanism work?"
```

### Domain-Specific Query
```bash
python main.py "What is the role of the evaluator in RAG?"
```

## Customization

### Adding Your Own Documents

Edit the `load_corpus()` function in `first.py` to load your documents:

```python
def load_corpus() -> List[Chunk]:
    docs = [
        ("doc_id_1", "Your document text here..."),
        ("doc_id_2", "Another document..."),
    ]
    return docs
```

### Adjusting Guardrail Logic

Modify `gaurdrail_filter()` in `self_correcting_loop.py` to implement custom filtering:

```python
def gaurdrail_filter(query: str, hits, top_k: int):
    # Your custom filtering logic
    return filtered_hits
```

## Performance Considerations

- **Embedding Cost**: Batch embeddings are created once during index building
- **Latency**: Each round involves 2-3 LLM calls (guardrail, answer, evaluation)
- **Quality vs Speed**: Adjust `TARGET_SCORE` to balance accuracy and response time
- **Index Size**: FAISS IndexFlatIP provides exact search; consider approximate methods for large corpora

## Troubleshooting

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### API Key Issues

Verify your `.env` file contains a valid OpenAI API key:
```bash
cat .env
```

### Module Not Found

Ensure you're in the project directory and the virtual environment is activated:
```bash
source venv/bin/activate
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Contact

Naveen Kancharla - [GitHub](https://github.com/NaveenKancharla28)

## Acknowledgments

- OpenAI for GPT and embedding models
- Facebook AI Research for FAISS
- LangChain for text splitting utilities
