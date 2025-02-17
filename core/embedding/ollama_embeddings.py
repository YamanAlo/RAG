from typing import List, Dict, Any
from langchain_ollama import OllamaEmbeddings
from config.config import EMBEDDING_MODEL

class EmbeddingService:
    def __init__(self):
        self.embedding_model = OllamaEmbeddings(
            model=EMBEDDING_MODEL
        )

    def get_embeddings(self, documents: List[Dict[str, Any]]) -> List[List[float]]:
        """Get embeddings for a list of documents."""
        # Extract just the text content for embedding
        texts = [doc['text'] for doc in documents]
        return self.embedding_model.embed_documents(texts)

    def get_query_embedding(self, text: str) -> List[float]:
        """Get embedding for a single query text."""
        return self.embedding_model.embed_query(text) 