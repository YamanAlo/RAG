import gradio as gr
import json
import tempfile
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Generator
from main import RAGSystem
import time

# Initialize the RAG system
rag_system = RAGSystem()
processed_files_dict = {}  # Track processed files with their chunks

# Constants
RETRY_ATTEMPTS = 2
RETRY_DELAY = 1  # seconds

def process_uploaded_files(files: List[tempfile._TemporaryFileWrapper]) -> str:
    """Process uploaded files and initialize the retriever."""
    if not files:
        return "Please upload at least one file."
        
    try:
        # Get file paths from temporary files
        file_paths = [file.name for file in files]
        new_files = []
        
        # Check which files are new
        for file_path in file_paths:
            file_name = Path(file_path).name
            if file_name not in processed_files_dict:
                new_files.append(file_path)
                
        if not new_files:
            return "✅ All files have already been processed."
        
        # Process only new documents
        new_chunks = rag_system.process_documents(new_files)
        
        # Add new chunks to existing ones and reinitialize retriever
        all_chunks = []
        for chunks in processed_files_dict.values():
            all_chunks.extend(chunks)
        all_chunks.extend(new_chunks)
        
        # Update processed files dictionary
        chunk_size = len(new_chunks) // len(new_files)  # Approximate chunks per file
        start_idx = 0
        for file_path in new_files:
            file_name = Path(file_path).name
            end_idx = start_idx + chunk_size
            processed_files_dict[file_name] = new_chunks[start_idx:end_idx]
            start_idx = end_idx
        
        # Initialize retriever with all chunks
        rag_system.initialize_retriever(all_chunks)
        
        return f"✅ Successfully processed {len(new_files)} new files and extracted {len(new_chunks)} chunks."
    except Exception as e:
        return f"❌ Error processing files: {str(e)}"

def get_processed_files() -> str:
    """Get list of currently processed files."""
    if not processed_files_dict:
        return "No files have been processed yet."
    return "Currently processed files:\n" + "\n".join(f"- {file}" for file in processed_files_dict.keys())

def retry_with_delay(func, *args, max_attempts=RETRY_ATTEMPTS, delay=RETRY_DELAY):
    """Retry a function with delay on failure."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            return func(*args)
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(delay)
    raise last_error

def fix_spacing(text: str) -> str:
    """Fix spacing issues in text."""
    # First, normalize spaces
    text = ' '.join(text.split())
    
    # Fix spacing around punctuation
    punctuation_fixes = {
        ' .': '.',
        ' ,': ',',
        ' !': '!',
        ' ?': '?',
        ' :': ':',
        ' ;': ';',
        ' )': ')',
        '( ': '(',
        ' "': '"',
        '" ': '"',
        " '": "'",
        "' ": "'",
    }
    
    for space_punct, punct in punctuation_fixes.items():
        text = text.replace(space_punct, punct)
    
    # Add space after punctuation if followed by a letter
    for punct in '.,!?:;':
        text = text.replace(f"{punct}(?=[a-zA-Z])", f"{punct} ")
    
    # Fix multiple spaces again
    text = ' '.join(text.split())
    
    return text.strip()

def chat_response(message: str, history: List[List[str]]) -> str:
    """Generate response for user message."""
    if not message.strip():
        return None
        
    try:
        # Combine response chunks into a single string
        response = ""
        for chunk in rag_system.query(message):
            if chunk:
                response += chunk
                
        # Fix spacing issues
        response = fix_spacing(response)
        
        return response
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

def clear_chat_history():
    """Clear the chat history and reset processed files."""
    global processed_files_dict
    processed_files_dict = {}
    rag_system.chat_history = []
    return None

def export_chat_history():
    """Export chat history as JSON file."""
    if not rag_system.chat_history:
        return "No chat history to export."
        
    try:
        # Create exports directory if it doesn't exist
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = export_dir / f"chat_history_{timestamp}.json"
        
        # Export to JSON
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(rag_system.chat_history, f, indent=2, ensure_ascii=False)
        
        return f"✅ Chat history exported to: {filename}"
    except Exception as e:
        return f"❌ Error exporting chat history: {str(e)}"

def user_message(message: str, history: List[List[str]]) -> Generator[Tuple[List[List[str]], str], None, None]:
    """Format user message and get response with improved error handling and typing indicator."""
    if not message.strip():
        yield history, ""
        return
        
    try:
        # Add user message to history immediately
        history = history + [[message, ""]]
        yield history, ""  # Show user message immediately
        
        # Show typing indicator
        history[-1][1] = "🤔 Thinking..."
        yield history, ""
        
        # Check if asking about processed files
        if any(keyword in message.lower() for keyword in ["what files", "which files", "list files", "show files"]):
            history[-1][1] = get_processed_files()
            yield history, ""
            return
            
        # Check if files are processed
        if not processed_files_dict:
            history[-1][1] = "❌ Please upload and process some files first."
            yield history, ""
            return
        
        # Stream assistant response with retry
        response = ""
        try:
            for chunk in retry_with_delay(rag_system.query, message):
                if chunk:
                    response += chunk
                    history[-1][1] = fix_spacing(response)
                    yield history, ""
        except Exception as e:
            if "context window" in str(e).lower():
                history[-1][1] = "⚠️ The conversation is getting too long. Consider clearing history to start fresh."
            else:
                history[-1][1] = f"❌ Error: {str(e)}\nTry rephrasing your question or uploading relevant documents."
            yield history, ""
            return
            
    except Exception as e:
        history[-1][1] = f"❌ Error: {str(e)}"
        yield history, ""

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📚 RAG System Demo
        Upload PDF or PPT files and ask questions about their content!
        
        Tips:
        - Clear chat history for long sessions
        - Be specific in your questions
        """
    )
    
    with gr.Row():
        # Left column - File Upload
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="Upload Files",
                file_types=[".pdf", ".ppt", ".pptx"],
                file_count="multiple",
                type="filepath"
            )
            process_button = gr.Button("Process Files", variant="primary")
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                show_copy_button=True
            )
            
        # Right column - Chat Interface
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat History",
                height=400,
                show_copy_button=True,
                bubble_full_width=False
            )
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask a question about the uploaded documents...",
                lines=2,
                show_copy_button=True
            )
            with gr.Row():
                submit = gr.Button("Submit", variant="primary")
                clear = gr.Button("Clear History", variant="secondary")
                export = gr.Button("Export History", variant="secondary")
    
    # Set up event handlers
    process_button.click(
        process_uploaded_files,
        inputs=[file_upload],
        outputs=[status_text],
        api_name="process_files"
    )
    
    # Don't clear message box after submission
    msg.submit(
        user_message,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg],
        api_name="chat",
        queue=True,  # Enable queuing for streaming
        show_progress=False  # Don't show progress bar
    ).then(
        lambda x: x,  # Keep the message
        inputs=[msg],
        outputs=[msg]
    )
    
    submit.click(
        user_message,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg],
        api_name="chat_button",
        queue=True,  # Enable queuing for streaming
        show_progress=False  # Don't show progress bar
    ).then(
        lambda x: x,  # Keep the message
        inputs=[msg],
        outputs=[msg]
    )
    
    clear.click(
        clear_chat_history,
        outputs=[chatbot],
        api_name="clear_history"
    )
    
    export.click(
        export_chat_history,
        outputs=[status_text],
        api_name="export_history"
    )

if __name__ == "__main__":
    demo.queue(max_size=20)  # Enable queue with max 20 requests
    demo.launch(
        server_name="0.0.0.0",  # Required for external access
        server_port=7860,
        share=True,  # Enable sharing
        max_threads=60,  # Limit concurrent processing
        show_error=True,
        ssl_verify=False,  # Disable SSL verification for local sharing
          
    ) 