import pytest
from pathlib import Path
import json
from giskard import Dataset, Model, test_function, suite, GiskardClient
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from typing import List, Dict, Any
import time
from main import RAGSystem

class RAGEvaluator:
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def generate_test_queries(self, documents: List[Dict[str, Any]], num_queries: int = 50) -> List[Dict[str, Any]]:
        """Generate test queries from documents using Giskard."""
        test_queries = []
        
        for doc in documents:
            # Extract key information using Giskard's query generation
            text = doc['text']
            metadata = {
                'source': doc.get('source', 'unknown'),
                'chunk_index': doc.get('chunk_index', 0)
            }
            
            # Generate factual queries
            queries = self._generate_queries_from_text(text, num_queries // len(documents))
            
            for query in queries:
                test_queries.append({
                    'query': query,
                    'context': text,
                    'metadata': metadata,
                    'type': 'factual'
                })
                
        return test_queries
        
    def evaluate_retrieval(self, test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate retrieval performance."""
        results = {
            'precision': [],
            'recall': [],
            'f1': [],
            'retrieval_time': []
        }
        
        for test_case in test_queries:
            start_time = time.time()
            retrieved_docs = self.rag_system.retriever.hybrid_search(
                test_case['query'],
                self.rag_system.embedding_service.get_query_embedding(test_case['query'])
            )
            retrieval_time = time.time() - start_time
            
            # Calculate retrieval metrics
            relevant_docs = self._get_relevant_docs(test_case['context'], retrieved_docs)
            
            if retrieved_docs:
                precision = len(relevant_docs) / len(retrieved_docs)
                recall = len(relevant_docs) / 1  # Assuming 1 relevant document per query
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = recall = f1 = 0
                
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1'].append(f1)
            results['retrieval_time'].append(retrieval_time)
            
        # Calculate averages
        return {
            'avg_precision': np.mean(results['precision']),
            'avg_recall': np.mean(results['recall']),
            'avg_f1': np.mean(results['f1']),
            'avg_retrieval_time': np.mean(results['retrieval_time'])
        }
        
    def evaluate_answer_quality(self, test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate answer quality using ROUGE and BLEU scores."""
        results = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
            'bleu': [],
            'response_time': []
        }
        
        for test_case in test_queries:
            start_time = time.time()
            response = ""
            for chunk in self.rag_system.query(test_case['query']):
                if chunk:
                    response += chunk
            response_time = time.time() - start_time
            
            # Calculate ROUGE scores
            rouge_scores = self.scorer.score(test_case['context'], response)
            results['rouge1'].append(rouge_scores['rouge1'].fmeasure)
            results['rouge2'].append(rouge_scores['rouge2'].fmeasure)
            results['rougeL'].append(rouge_scores['rougeL'].fmeasure)
            
            # Calculate BLEU score
            reference = test_case['context'].split()
            hypothesis = response.split()
            bleu_score = sentence_bleu([reference], hypothesis)
            results['bleu'].append(bleu_score)
            
            results['response_time'].append(response_time)
            
        # Calculate averages
        return {
            'avg_rouge1': np.mean(results['rouge1']),
            'avg_rouge2': np.mean(results['rouge2']),
            'avg_rougeL': np.mean(results['rougeL']),
            'avg_bleu': np.mean(results['bleu']),
            'avg_response_time': np.mean(results['response_time'])
        }
        
    def evaluate_performance(self, test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate system performance metrics."""
        results = {
            'retrieval_times': [],
            'response_times': [],
            'memory_usage': [],
            'token_usage': []
        }
        
        for test_case in test_queries:
            # Measure retrieval time
            start_time = time.time()
            _ = self.rag_system.retriever.hybrid_search(
                test_case['query'],
                self.rag_system.embedding_service.get_query_embedding(test_case['query'])
            )
            retrieval_time = time.time() - start_time
            
            # Measure response time
            start_time = time.time()
            response = ""
            for chunk in self.rag_system.query(test_case['query']):
                if chunk:
                    response += chunk
            response_time = time.time() - start_time
            
            results['retrieval_times'].append(retrieval_time)
            results['response_times'].append(response_time)
            
        return {
            'avg_retrieval_time': np.mean(results['retrieval_times']),
            'avg_response_time': np.mean(results['response_times']),
            'p95_retrieval_time': np.percentile(results['retrieval_times'], 95),
            'p95_response_time': np.percentile(results['response_times'], 95)
        }
        
    def _generate_queries_from_text(self, text: str, num_queries: int) -> List[str]:
        """Generate test queries from text using Giskard's query generation."""
        # Initialize Giskard client
        client = GiskardClient()
        
        # Create dataset from text
        dataset = Dataset(
            text,
            name="rag_test_dataset",
            target="query"
        )
        
        # Generate queries using Giskard's test generation
        generated_tests = client.generate_tests(
            dataset=dataset,
            model=Model(
                self.rag_system.query,
                model_type="text_generation",
                name="rag_model"
            ),
            test_types=["semantic", "factual"],
            n_tests=num_queries
        )
        
        return [test.inputs['query'] for test in generated_tests]
        
    def _get_relevant_docs(self, context: str, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Determine which retrieved documents are relevant."""
        relevant_docs = []
        for doc in retrieved_docs:
            # Calculate similarity between context and retrieved doc
            rouge_scores = self.scorer.score(context, doc['text'])
            if rouge_scores['rouge1'].fmeasure > 0.3:  # Threshold for relevance
                relevant_docs.append(doc)
        return relevant_docs
        
    def save_evaluation_results(self, results: Dict[str, Any], output_file: str = "evaluation_results.json"):
        """Save evaluation results to a file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
# Test functions
@pytest.fixture
def rag_evaluator():
    rag_system = RAGSystem()
    return RAGEvaluator(rag_system)

def test_retrieval_performance(rag_evaluator):
    """Test retrieval performance metrics."""
    # Generate test queries
    test_queries = rag_evaluator.generate_test_queries(rag_evaluator.rag_system.chunks)
    
    # Evaluate retrieval
    results = rag_evaluator.evaluate_retrieval(test_queries)
    
    # Assert minimum performance thresholds
    assert results['avg_precision'] >= 0.6
    assert results['avg_recall'] >= 0.6
    assert results['avg_f1'] >= 0.6
    assert results['avg_retrieval_time'] <= 2.0  # seconds

def test_answer_quality(rag_evaluator):
    """Test answer quality metrics."""
    # Generate test queries
    test_queries = rag_evaluator.generate_test_queries(rag_evaluator.rag_system.chunks)
    
    # Evaluate answer quality
    results = rag_evaluator.evaluate_answer_quality(test_queries)
    
    # Assert minimum quality thresholds
    assert results['avg_rouge1'] >= 0.4
    assert results['avg_rouge2'] >= 0.2
    assert results['avg_rougeL'] >= 0.3
    assert results['avg_bleu'] >= 0.2

def test_system_performance(rag_evaluator):
    """Test system performance metrics."""
    # Generate test queries
    test_queries = rag_evaluator.generate_test_queries(rag_evaluator.rag_system.chunks)
    
    # Evaluate performance
    results = rag_evaluator.evaluate_performance(test_queries)
    
    # Assert performance thresholds
    assert results['avg_retrieval_time'] <= 2.0  # seconds
    assert results['avg_response_time'] <= 5.0  # seconds
    assert results['p95_retrieval_time'] <= 3.0  # seconds
    assert results['p95_response_time'] <= 7.0  # seconds 