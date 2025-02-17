from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
import faiss
from config.config import TOP_K_RETRIEVAL, RERANK_TOP_K, MIN_RELEVANCE_SCORE
import cohere
from config.config import COHERE_API_KEY
import logging
from langchain.docstore.document import Document
from langchain_cohere import CohereRerank

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Initialize the hybrid retriever with FAISS and BM25."""
        self.chunks = chunks
        self.texts = [chunk['text'] for chunk in chunks]
        
        # Initialize FAISS index with L2 normalization
        self.dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(self.dimension)
        normalized_embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(normalized_embeddings)
        self.index.add(normalized_embeddings)
        
        # Initialize BM25 with tokenized texts
        self.tokenized_texts = [text.split() for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_texts)
        
        # Initialize Cohere client
        self.co = cohere.Client(COHERE_API_KEY)
        
    def dense_search(self, query_embedding: List[float], top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Perform dense retrieval using FAISS with L2 normalization."""
        # Normalize query embedding
        query_embedding_np = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding_np)
        
        # Perform search
        distances, indices = self.index.search(query_embedding_np, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):  # Add safety check
                chunk = self.chunks[idx]
                results.append({
                    **chunk,  # Include all chunk metadata
                    'score': float(distances[0][i]),
                    'index': int(idx)
                })
        return results
    
    def sparse_search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Perform sparse retrieval using BM25 with improved tokenization."""
        try:
            # Tokenize query consistently with document tokenization
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top k indices
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                chunk = self.chunks[idx]
                results.append({
                    **chunk,  # Include all chunk metadata
                    'score': float(scores[idx]),
                    'index': int(idx)
                })
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {str(e)}")
            return []
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = RERANK_TOP_K) -> List[Dict[str, Any]]:
        """Rerank results using Cohere's reranking model."""
        if not documents:
            return []
            
        try:
            # Extract texts for reranking
            docs_for_rerank = [doc['text'] for doc in documents]
            
            # Perform reranking with Cohere
            rerank_results = self.co.rerank(
                query=query,
                documents=docs_for_rerank,
                top_n=min(top_k, len(docs_for_rerank)),
                model='rerank-v3.5'
            )
            
            # Format results and filter by minimum relevance score
            reranked_docs = []
            
            # Handle different response formats from Cohere
            if hasattr(rerank_results, 'results'):
                # New API format
                for result in rerank_results.results:
                    try:
                        relevance_score = float(result.relevance_score)
                        doc_index = result.index
                        if relevance_score >= MIN_RELEVANCE_SCORE:
                            original_doc = documents[doc_index].copy()  # Create a copy to avoid modifying original
                            original_doc['relevance_score'] = relevance_score
                            reranked_docs.append(original_doc)
                    except (AttributeError, ValueError, IndexError) as e:
                        logger.error(f"Error processing result in new format: {str(e)}")
                        continue
            else:
                # Legacy format or direct results
                for i, result in enumerate(rerank_results):
                    try:
                        if isinstance(result, (list, tuple)):
                            doc_index = int(result[0])
                            relevance_score = float(result[1])
                        else:
                            doc_index = i
                            relevance_score = float(result)
                            
                        if relevance_score >= MIN_RELEVANCE_SCORE:
                            original_doc = documents[doc_index].copy()  # Create a copy to avoid modifying original
                            original_doc['relevance_score'] = relevance_score
                            reranked_docs.append(original_doc)
                    except (ValueError, IndexError) as e:
                        logger.error(f"Error processing result in legacy format: {str(e)}")
                        continue
            
            # Sort by relevance score
            reranked_docs.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
            
            # Log reranking results
            logger.info(f"\nReranking Results:")
            for i, doc in enumerate(reranked_docs, 1):
                logger.info(f"Document {i} - Score: {doc.get('relevance_score', 0.0):.3f}")
                logger.info(f"Source: {doc.get('source', 'Unknown')}")
                logger.info(f"Content Preview: {doc.get('text', '')[:100]}...")
            
            if not reranked_docs:
                logger.warning("No documents passed the relevance threshold")
                return documents[:top_k]  # Fallback to top original documents
                
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            logger.info("Falling back to original document order")
            return documents[:top_k]  # Fallback to original order
    
    def hybrid_search(self, query: str, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """Perform hybrid search with improved result combination."""
        try:
            # Get results from both retrievers
            dense_results = self.dense_search(query_embedding)
            sparse_results = self.sparse_search(query)
            
            # Combine results with score normalization
            seen_indices = set()
            combined_results = []
            
            # Normalize scores to [0, 1] range
            all_results = dense_results + sparse_results
            if all_results:
                max_score = max(doc['score'] for doc in all_results)
                min_score = min(doc['score'] for doc in all_results)
                score_range = max_score - min_score if max_score != min_score else 1
                
                for result in all_results:
                    if result['index'] not in seen_indices:
                        seen_indices.add(result['index'])
                        # Normalize score
                        result['score'] = (result['score'] - min_score) / score_range
                        combined_results.append(result)
            
            # Rerank combined results
            reranked_results = self.rerank(query, combined_results)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return [] 