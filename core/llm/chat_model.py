from typing import List, Dict, Any, Generator
from langchain_ollama import ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from config.config import LLM_MODEL, TEMPERATURE, SYSTEM_PROMPT
import logging

logger = logging.getLogger(__name__)

class ChatModel:
    def __init__(self):
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            streaming=True,
            num_ctx=4096  # Increase context window
        )
        
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string."""
        formatted_docs = []
        
        # Log retrieved documents for debugging
        logger.info(f"\n{'='*50}\nRetrieved Documents:")
        for i, doc in enumerate(retrieved_docs, 1):
            # Add source information if available
            source = doc.get('source', f'Document {i}')
            text = doc.get('text', '').strip()
            score = doc.get('relevance_score', 0.0)
            
            # Log document details
            logger.info(f"\nDocument {i}:")
            logger.info(f"Source: {source}")
            logger.info(f"Relevance Score: {score}")
            logger.info(f"Content: {text[:200]}...")  # Show first 200 chars
            
            if text:
                formatted_docs.append(f"{source}:\n{text}")
                
        logger.info(f"{'='*50}\n")
        return "\n\n".join(formatted_docs)
    
    def format_chat_history(self, chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format chat history into message format."""
        formatted_history = []
        for message in chat_history:
            if message['role'] == 'user':
                formatted_history.append(HumanMessage(content=message['content']))
            elif message['role'] == 'assistant':
                formatted_history.append(AIMessage(content=message['content']))
        return formatted_history
    
    def generate_response(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]], 
        chat_history: List[Dict[str, str]] = None
    ) -> Generator[str, None, None]:
        """Generate streaming response using the LLM."""
        if chat_history is None:
            chat_history = []
            
        try:
            # Format context and history
            context = self.format_context(retrieved_docs)
            formatted_history = self.format_chat_history(chat_history)
            
            # Create system message with context
            system_content = SYSTEM_PROMPT.format(
                context=context,
                chat_history=formatted_history,
                question=query
            )
            
            # Prepare messages
            messages = [
                SystemMessage(content=system_content),
                *formatted_history,
                HumanMessage(content=query)
            ]
            
            # Generate streaming response
            current_response = ""
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                elif isinstance(chunk, str):
                    content = chunk
                else:
                    continue
                    
                if content:
                    # Ensure proper spacing between words
                    content = content.replace('  ', ' ').strip()  # Remove double spaces
                    if current_response and not current_response.endswith(' ') and not content.startswith(' '):
                        content = ' ' + content  # Add space between chunks if needed
                    current_response += content
                    yield content
                    
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            yield f"I apologize, but I encountered an error while generating the response: {str(e)}" 