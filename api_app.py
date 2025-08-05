# --- 1. Imports ---
from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import json
import os
import re
import requests
import hashlib
import time
from pathlib import Path
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
API_TOKEN = os.getenv('API_TOKEN', '49092b2a30dc77e80c88e0550254ddd7928dea77103e0f05ad669ba81de92b04')
PORT = int(os.getenv('PORT', 8000))
HOST = os.getenv('HOST', '0.0.0.0')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# --- 2. Pydantic Schemas ---
class HackRxRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]
    status: str = "processing"

# --- 3. Configuration ---
CACHE_DIR = Path("./temp_indexes")
CACHE_DIR.mkdir(exist_ok=True)


# --- 4. FastAPI Application ---
app = FastAPI(title="HackRx Document Query-Retrieval System")

# Global variables for lazy loading
processor = None
loading_in_progress = False

async def initialize_processor():
    """Lazy load the processor and models in the background"""
    global processor, loading_in_progress
    
    if processor is not None or loading_in_progress:
        return
    
    loading_in_progress = True
    
    try:
        # Import here to avoid loading all dependencies at startup
        from sentence_transformers import SentenceTransformer
        import faiss
        import pypdf
        import docx
        from email.parser import BytesParser
        from email.policy import default
        from io import BytesIO
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        import google.generativeai as genai
        import pickle
        
        # Initialize Google Generative AI
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyDXuvJKPzcXAcEhIsJi2M-kT7mfc8Q9MYQ')
        GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.0-flash')
        
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            print(f"Gemini API configured successfully.")
        except Exception as e:
            print(f"Could not configure Gemini API. Error: {e}")
        
        class DocumentProcessor:
            @staticmethod
            def download_from_blob(url: str) -> bytes:
                """Download a document from a blob storage URL"""
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    return response.content
                except Exception as e:
                    raise ValueError(f"Failed to download document from URL: {e}")

            @staticmethod
            def detect_document_type(path_or_url: str) -> str:
                """Detect document type from extension or content"""
                if isinstance(path_or_url, str):
                    if path_or_url.lower().endswith('.pdf'):
                        return "pdf"
                    elif path_or_url.lower().endswith(('.docx', '.doc')):
                        return "docx"
                    elif path_or_url.lower().endswith(('.eml', '.msg')):
                        return "email"
                # Default to PDF if we can't determine
                return "pdf"

            @staticmethod
            def process_pdf(content: bytes) -> List[Dict]:
                """Process PDF content and extract text by page"""
                reader = pypdf.PdfReader(BytesIO(content))
                docs = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():  # Only add non-empty pages
                        docs.append({
                            "content": text, 
                            "metadata": {"page": i + 1}
                        })
                return docs

            @staticmethod
            def process_docx(content: bytes) -> List[Dict]:
                """Process DOCX content and extract text"""
                doc = docx.Document(BytesIO(content))
                full_text = "\n".join([para.text for para in doc.paragraphs])
                return [{"content": full_text, "metadata": {"page": 1}}]

            @staticmethod
            def process_email(content: bytes) -> List[Dict]:
                """Process email content and extract text"""
                parser = BytesParser(policy=default)
                msg = parser.parsebytes(content)
                
                # Extract header information
                header_text = f"From: {msg['from']}\nTo: {msg['to']}\nSubject: {msg['subject']}\n\n"
                
                # Extract body
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        if content_type == "text/plain":
                            body += part.get_content()
                else:
                    body = msg.get_content()
                
                full_text = header_text + body
                return [{"content": full_text, "metadata": {"page": 1}}]

            @classmethod
            def load_and_process_document(cls, document_url: str) -> List[Dict]:
                """Load document from URL and process based on type"""
                # Handle URL
                if document_url.startswith(('http://', 'https://')):
                    content = cls.download_from_blob(document_url)
                    source_name = document_url.split('/')[-1].split('?')[0]  # Extract filename from URL
                else:
                    # Assume it's a local file path
                    file_path = Path(document_url)
                    if not file_path.exists():
                        raise FileNotFoundError(f"No file at path: {document_url}")
                    with open(file_path, "rb") as f:
                        content = f.read()
                    source_name = file_path.name
                    
                # Detect document type
                document_type = cls.detect_document_type(document_url)
                    
                # Process based on document type
                if document_type == "pdf":
                    docs = cls.process_pdf(content)
                elif document_type == "docx":
                    docs = cls.process_docx(content)
                elif document_type == "email":
                    docs = cls.process_email(content)
                else:
                    raise ValueError(f"Unsupported document type: {document_type}")
                    
                # Add source to metadata
                for doc in docs:
                    doc["metadata"]["source"] = source_name
                    
                return docs


        class DynamicRAGProcessor:
            def __init__(self):
                print("Initializing Dynamic RAG Processor...")
                
                # Load embedding model - use a lightweight model for Render
                EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  # This is a small model (384 dim)
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                print(f"Loaded embedding model: {EMBEDDING_MODEL_NAME} with dimension: {self.embedding_dim}")
                
                try:
                    # Use the model name from environment variable
                    self.llm = genai.GenerativeModel(GEMINI_MODEL_NAME)
                    print(f"Gemini model '{GEMINI_MODEL_NAME}' initialized successfully.")
                except Exception as e:
                    print(f"CRITICAL ERROR: Failed to initialize Gemini model. Error: {e}")
                    self.llm = None
                    
                print("Processor Initialized.")

            def _get_or_create_faiss_index(self, doc_path: str, documents: List[Dict]) -> tuple:
                """Create or load a FAISS index for the given document"""
                doc_hash = hashlib.md5(doc_path.encode()).hexdigest()
                index_path = CACHE_DIR / f"{doc_hash}.faiss"
                metadata_path = CACHE_DIR / f"{doc_hash}.pkl"

                if index_path.exists() and metadata_path.exists():
                    print(f"CACHE HIT: Loading FAISS index for '{doc_path}' from disk.")
                    index = faiss.read_index(str(index_path))
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    return index, metadata

                print(f"CACHE MISS: Creating new FAISS index for '{doc_path}'.")
                chunked_docs = self._split_documents(documents)
                if not chunked_docs:
                    raise ValueError("Document is empty or could not be read.")

                print(f"Generating embeddings for {len(chunked_docs)} chunks...")
                embeddings = self.embedding_model.encode([doc['content'] for doc in chunked_docs])
                print(f"Creating FAISS index with {len(chunked_docs)} embeddings...")
                
                index = faiss.IndexFlatL2(self.embedding_dim)
                index.add(np.array(embeddings, dtype=np.float32))

                print(f"Saving FAISS index to disk: {index_path}")
                faiss.write_index(index, str(index_path))
                with open(metadata_path, 'wb') as f:
                    pickle.dump(chunked_docs, f)
                return index, chunked_docs

            def _split_documents(self, documents: List[Dict]) -> List[Dict]:
                """Split documents into smaller chunks for better retrieval"""
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunked_docs = []
                for doc in documents:
                    for i, chunk_text in enumerate(text_splitter.split_text(doc['content'])):
                        chunked_docs.append({
                            "content": chunk_text, 
                            "metadata": {**doc['metadata'], "chunk_id": i}
                        })
                return chunked_docs

            def _llm_evaluate(self, query: str, context_clauses: List[Dict]) -> str:
                """Generate a direct answer based on the retrieved context"""
                if not self.llm:
                    raise ValueError("Gemini model is not initialized.")

                if not context_clauses:
                    return "I couldn't find relevant information to answer your question in the provided document."

                context_str = "\n".join([f"- (Source: {c['metadata'].get('source', 'Unknown')}, Page: {c['metadata'].get('page', 'N/A')}) {c['content']}" for c in context_clauses])
                prompt = f"""
                You are an AI assistant specializing in insurance, legal, HR, and compliance documents.
                
                Based ONLY on the context clauses provided below, answer the user's question directly, concisely and accurately.
                If the information to answer the question is not available in the context, state that clearly.
                Do not use the phrase "based on the context" or refer to the source documents in your answer.
                Provide a complete, standalone answer that directly addresses the question.

                [CONTEXT CLAUSES]:
                {context_str}

                [USER QUESTION]:
                {query}

                Answer:
                """
                
                try:
                    response = self.llm.generate_content(prompt)
                    
                    # Defensive check for safety blocks
                    if not response.parts:
                        raise ValueError(f"Gemini response was blocked. Reason: {response.prompt_feedback.block_reason}")
                        
                    answer_text = response.text.strip()
                    # Remove any prefixes like "Answer:" or "Based on the context"
                    answer_text = re.sub(r'^(Answer:|Based on the context:?)\s*', '', answer_text, flags=re.IGNORECASE)
                    return answer_text
                    
                except (AttributeError, ValueError) as e:
                    raise ValueError(f"Invalid or blocked response from Gemini API. Details: {e}")

            def process_request(self, request: HackRxRequest) -> Dict:
                try:
                    doc_url = request.documents
                    
                    # Load and process the document
                    print(f"Processing document: {doc_url}")
                    documents = DocumentProcessor.load_and_process_document(doc_url)
                    
                    # Use FAISS for vector storage
                    print("Using FAISS for vector storage")
                    index, metadata = self._get_or_create_faiss_index(doc_url, documents)
                    
                    all_answers = []
                    for question in request.questions:
                        print(f"Processing question: {question}")
                        # Generate embedding for the query
                        query_embedding = self.embedding_model.encode([question])
                        
                        # Search the FAISS index
                        top_k = min(5, len(metadata))  # In case we have fewer than 5 chunks
                        _, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
                        
                        # Get the relevant chunks
                        retrieved_clauses = []
                        for i in indices[0]:
                            if i < len(metadata):  # Safety check
                                retrieved_clauses.append(metadata[i])
                        
                        # Generate direct answer using LLM
                        answer = self._llm_evaluate(question, retrieved_clauses)
                        all_answers.append(answer)
                            
                    return {"answers": all_answers, "status": "complete"}
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return {"answers": [f"An error occurred: {str(e)}"], "status": "error"}
        
        # Initialize processor
        global processor
        processor = DynamicRAGProcessor()
        
    except Exception as e:
        print(f"Error initializing processor: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loading_in_progress = False


# --- 5. Authorization Middleware ---
def verify_api_key(authorization: str = Header(...)):
    """Simple API key verification"""
    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer':
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        
        # Use the token from environment variables
        if token != API_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        return token
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")


@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(request: HackRxRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    """Process a HackRx request asynchronously"""
    # Ensure processor is initialized
    if processor is None:
        if not loading_in_progress:
            # Start initialization in background
            background_tasks.add_task(initialize_processor)
        return HackRxResponse(
            answers=["Your request is being processed. The models are still loading. Please try again in a minute."],
            status="initializing"
        )
    
    # Process the request
    result = processor.process_request(request)
    return HackRxResponse(**result)


@app.get("/")
async def read_root(background_tasks: BackgroundTasks):
    """Root endpoint that triggers lazy loading of models"""
    # Start initialization in background if needed
    if processor is None and not loading_in_progress:
        background_tasks.add_task(initialize_processor)
    
    return {
        "status": "API is running",
        "endpoints": {
            "query": "/hackrx/run"
        },
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "user": "nischal2805",
        "model_status": "loaded" if processor else "initializing" if loading_in_progress else "not_loaded"
    }
    

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_status": "loaded" if processor else "initializing" if loading_in_progress else "not_loaded"
    }


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    print("API starting up... Models will be loaded when needed.")
    # Note: We're using lazy loading, so we don't initialize the processor here


# Run the application if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_app:app", host=HOST, port=PORT, reload=DEBUG)
