# Agentic RAG System

An advanced Retrieval-Augmented Generation (RAG) system built with LangChain, capable of processing PDF and PPT files to provide educational explanations on any topic.

## Features

- **Document Processing**: Support for PDF and PPT files
- **Advanced Text Chunking**: Recursive character text splitting with overlap
- **State-of-the-Art Embeddings**: Using Ollama embeddings
- **Hybrid Retrieval**:
  - Dense retrieval with FAISS
  - Sparse retrieval with BM25
  - Cohere Reranker for result optimization
- **Streaming Responses**: Real-time response generation
- **Conversation History**: Maintains context across multiple queries
- **Performance Logging**: Tracks processing time for optimization

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Cohere API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
COHERE_API_KEY=your_cohere_api_key
```

## Usage

1. Start Ollama server locally

2. Run the system:
```python
from main import RAGSystem

# Initialize the system
rag_system = RAGSystem()

# Process documents
documents = [
    "path/to/your/document1.pdf",
    "path/to/your/document2.pptx"
]
chunks = rag_system.process_documents(documents)

# Initialize retriever
rag_system.initialize_retriever(chunks)

# Query the system
question = "What is the main topic of the documents?"
for response_chunk in rag_system.query(question):
    print(response_chunk, end="")
```

## Project Structure

```
rag_system/
├── config/
│   └── config.py           # Configuration settings
├── core/
│   ├── document_processor/
│   │   ├── pdf_processor.py
│   │   └── ppt_processor.py
│   ├── embedding/
│   │   └── ollama_embeddings.py
│   ├── retrieval/
│   │   └── hybrid_retriever.py
│   └── llm/
│       └── chat_model.py
├── utils/
│   ├── logger.py
│   └── timer.py
├── main.py
└── requirements.txt
```

## Performance Considerations

- Document processing time depends on file size and complexity
- Initial retriever setup requires embedding generation for all chunks
- Query response time varies based on:
  - Document corpus size
  - Query complexity
  - Number of retrieved documents
  - Reranking process

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 