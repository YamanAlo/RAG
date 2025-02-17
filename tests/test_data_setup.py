import shutil
from pathlib import Path
import json
from typing import List, Dict, Any
from giskard import Dataset, Model
import pandas as pd
from main import RAGSystem

class TestDataManager:
    def __init__(self):
        self.test_data_dir = Path("test_data")
        self.test_docs_dir = self.test_data_dir / "documents"
        self.test_queries_file = self.test_data_dir / "test_queries.json"
        self.rag_system = RAGSystem()
        
    def setup_test_environment(self):
        """Set up the test environment and directories."""
        # Create test directories
        self.test_data_dir.mkdir(exist_ok=True)
        self.test_docs_dir.mkdir(exist_ok=True)
        
    def copy_documents_to_test(self, source_dir: str | Path = "../documents"):
        """Copy documents from source directory to test directory."""
        source_dir = Path(source_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
            
        # Copy PDF files
        for pdf_file in source_dir.glob("**/*.pdf"):
            dest_file = self.test_docs_dir / pdf_file.name
            shutil.copy2(pdf_file, dest_file)
            
        # Copy PPT files
        for ppt_file in source_dir.glob("**/*.ppt*"):
            dest_file = self.test_docs_dir / ppt_file.name
            shutil.copy2(ppt_file, dest_file)
            
    def generate_test_dataset(self) -> Dataset:
        """Generate test dataset using Giskard's scanner."""
        # Process documents and get chunks
        chunks = self.rag_system.process_documents(self.test_docs_dir)
        
        # Create a DataFrame from chunks
        df = pd.DataFrame([{
            'text': chunk['text'],
            'source': chunk['source'],
            'chunk_index': chunk['chunk_index']
        } for chunk in chunks])
        
        # Create Giskard dataset
        dataset = Dataset(
            df,
            name="rag_test_dataset",
            target="text"  # The column containing the text to generate questions from
        )
        
        return dataset
        
    def generate_test_queries(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Generate test queries using Giskard's scanner."""
        # Create model wrapper for RAG system
        model = Model(
            self.rag_system.query,
            model_type="text_generation",
            name="rag_model"
        )
        
        # Initialize scanner
        text_scanner = TextGenerationScanner()
        
        # Generate test suite
        scan_results = scanner.scan(
            model,
            dataset,
            scanners=[text_scanner],
            n_samples=50  # Number of test cases to generate
        )
        
        # Convert scan results to test queries
        test_queries = []
        for result in scan_results.results:
            for test in result.tests:
                test_queries.append({
                    'query': test.inputs['text'],  # The generated question
                    'context': test.inputs.get('context', ''),  # Original context
                    'source': test.metadata.get('source', 'unknown'),
                    'type': test.test_type,
                    'expected_behavior': test.expected_behavior
                })
        
        # Save test queries
        with open(self.test_queries_file, 'w') as f:
            json.dump(test_queries, f, indent=2)
            
        return test_queries
        
    def generate_test_types(self) -> List[Dict[str, Any]]:
        """Generate different types of test cases."""
        return [
            # Factual Questions
            {"type": "factual", "weight": 0.4},
            
            # Semantic Understanding
            {"type": "semantic", "weight": 0.3},
            
            # Edge Cases
            {"type": "edge_case", "weight": 0.1},
            
            # Complex Queries
            {"type": "complex", "weight": 0.2}
        ]
        
    def load_test_queries(self) -> List[Dict[str, Any]]:
        """Load existing test queries or generate new ones."""
        if self.test_queries_file.exists():
            with open(self.test_queries_file, 'r') as f:
                return json.load(f)
        
        # Generate new test queries
        dataset = self.generate_test_dataset()
        return self.generate_test_queries(dataset)
        
    def clean_test_environment(self):
        """Clean up test environment."""
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
            
def setup_test_data():
    """Main function to set up test data."""
    manager = TestDataManager()
    
    print("Setting up test environment...")
    manager.setup_test_environment()
    
    print("Copying documents to test directory...")
    manager.copy_documents_to_test()
    
    print("Generating test dataset...")
    dataset = manager.generate_test_dataset()
    
    print("Generating test queries...")
    test_queries = manager.generate_test_queries(dataset)
    
    print("\nTest Setup Complete!")
    print(f"Test documents directory: {manager.test_docs_dir}")
    print(f"Test queries file: {manager.test_queries_file}")
    print(f"Number of test queries: {len(test_queries)}")
    
    # Print sample test queries
    print("\nSample Test Queries:")
    for i, query in enumerate(test_queries[:3], 1):
        print(f"\n{i}. Type: {query['type']}")
        print(f"   Query: {query['query']}")
        print(f"   Source: {query['source']}")
    
if __name__ == "__main__":
    setup_test_data() 