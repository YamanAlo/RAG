import time
from pathlib import Path
from typing import List, Dict, Any, Generator
import logging
from core.document_processor.pdf_processor import PDFProcessor
from core.document_processor.ppt_processor import PPTProcessor
from core.embedding.ollama_embeddings import EmbeddingService
from core.retrieval.hybrid_retriever import HybridRetriever
from core.llm.chat_model import ChatModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.ppt_processor = PPTProcessor()
        self.embedding_service = EmbeddingService()
        self.chat_model = ChatModel()
        self.retriever = None
        self.chat_history: List[Dict[str, str]] = []
        
    def get_documents_from_folder(self, folder_path: str | Path = "documents") -> List[Path]:
        """Get all PDF and PPT files from the documents folder."""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            logger.error(f"Documents folder not found: {folder_path}")
            raise FileNotFoundError(f"Documents folder not found: {folder_path}")
            
        # Get all PDF and PPT files
        pdf_files = list(folder_path.glob("**/*.pdf"))
        ppt_files = list(folder_path.glob("**/*.ppt*"))  # Matches both .ppt and .pptx
        
        all_files = pdf_files + ppt_files
        if not all_files:
            logger.warning(f"No PDF or PPT files found in {folder_path}")
            raise ValueError(f"No PDF or PPT files found in {folder_path}")
            
        logger.info(f"Found {len(pdf_files)} PDF files and {len(ppt_files)} PPT files")
        return all_files
        
    def process_documents(self, file_paths: List[str | Path] = None) -> List[str]:
        """Process multiple documents and return chunks."""
        if file_paths is None:
            try:
                file_paths = self.get_documents_from_folder()
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Error getting documents: {str(e)}")
                raise
                
        all_chunks = []
        total_start_time = time.time()
        
        for file_path in file_paths:
            file_path = Path(file_path)
            start_time = time.time()
            
            try:
                logger.info(f"Processing {file_path}")
                if file_path.suffix.lower() == '.pdf':
                    chunks = self.pdf_processor.process_document(file_path)
                elif file_path.suffix.lower() in ['.ppt', '.pptx']:
                    chunks = self.ppt_processor.process_document(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                    
                all_chunks.extend(chunks)
                processing_time = time.time() - start_time
                logger.info(f"Processed {file_path} in {processing_time:.2f} seconds")
                logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
                
        total_time = time.time() - total_start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Total chunks extracted: {len(all_chunks)}")
        
        return all_chunks
        
    def initialize_retriever(self, chunks: List[str]):
        """Initialize the retriever with processed documents."""
        start_time = time.time()
        
        try:
            embeddings = self.embedding_service.get_embeddings(chunks)
            self.retriever = HybridRetriever(chunks, embeddings)
            
            initialization_time = time.time() - start_time
            logger.info(f"Initialized retriever in {initialization_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error initializing retriever: {str(e)}")
            raise
            
    def query(self, question: str) -> Generator[str, None, None]:
        """Process a query and return streaming response."""
        if not self.retriever:
            raise ValueError("Retriever not initialized. Please process documents first.")
            
        start_time = time.time()
        
        try:
            # Get query embedding
            query_embedding = self.embedding_service.get_query_embedding(question)
            
            # Perform hybrid search
            retrieved_docs = self.retriever.hybrid_search(question, query_embedding)
            
            if not retrieved_docs:
                yield "I couldn't find any relevant information in the documents to answer your question."
                return
            
            # Generate response
            full_response = ""
            for chunk in self.chat_model.generate_response(
                query=question,
                retrieved_docs=retrieved_docs,
                chat_history=self.chat_history
            ):
                if chunk:
                    full_response += chunk
                    yield chunk
            
            # Only update history if we got a response
            if full_response:
                self.chat_history.append({"role": "user", "content": question})
                self.chat_history.append({"role": "assistant", "content": full_response})
            
            query_time = time.time() - start_time
            logger.info(f"Processed query in {query_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            yield f"I apologize, but I encountered an error while processing your query: {str(e)}"
            
def main():
    # Initialize the RAG system
    rag_system = RAGSystem()
    
    try:
        # Process all documents in the documents folder
        logger.info("Processing documents from the documents folder...")
        chunks = rag_system.process_documents()
        
        # Initialize retriever
        logger.info("Initializing retriever...")
        rag_system.initialize_retriever(chunks)
        
        logger.info("System ready for queries!")
        print("\nWelcome to the RAG System!")
        print("You can ask questions about the documents in the 'documents' folder.")
        print("Type 'quit', 'exit', or 'q' to end the session.\n")
        
        # Example conversation
        while True:
            try:
                question = input("\nYour question: ")
                if question.lower().strip() in ['quit', 'exit', 'q']:
                    print("\nThank you for using the RAG System. Goodbye!")
                    break
                
                if not question.strip():
                    print("Please enter a valid question.")
                    continue
                
                print("\nAssistant: ", end="", flush=True)
                
                # Track if we've received any response
                received_response = False
                full_response = ""
                
                for response_chunk in rag_system.query(question):
                    if response_chunk:
                        received_response = True
                        full_response += response_chunk
                        print(response_chunk, end="", flush=True)
                
                if not received_response:
                    print("I apologize, but I couldn't generate a response based on the available documents.")
                
                print("\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                print(f"\nI apologize, but an error occurred: {str(e)}")
                print("Please try asking your question again.")
                continue
                
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Please make sure you have:")
        print("1. Created a 'documents' folder")
        print("2. Added PDF or PPT files to the folder")
        print("3. Set up all required environment variables")
        return

if __name__ == "__main__":
    main() 