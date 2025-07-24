import os
import io
import json
import re
import time
import tempfile
import shutil
from typing import List, Dict, Optional
import requests
import asyncio
from pathlib import Path

# FastAPI and web components
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Document processing
from doclayout_yolo import YOLOv10
import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
import docx
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# LangChain and Vector Database
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from groq import Groq
import os
import pytesseract

# Configure system paths
def configure_system_paths():
    """Configure Tesseract and Poppler paths"""
    
    # Tesseract configuration
    tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_cmd):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        print(f"✅ Tesseract configured at: {tesseract_cmd}")
    
    # Poppler configuration
    poppler_path = r"C:\Program Files\Poppler\poppler-24.08.0\Library\bin"
    if os.path.exists(poppler_path):
        current_path = os.environ.get('PATH', '')
        if poppler_path not in current_path:
            os.environ['PATH'] = current_path + os.pathsep + poppler_path
        print(f"✅ Poppler configured at: {poppler_path}")
    
    return True

# Configure paths at startup
configure_system_paths()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
HACKRX_AUTH_TOKEN = "95f763f2e367cc7e5f72304cb9e9b84229f97f2a5b2b08f14b5034e8328596ec"

# Paths
TEMP_DIR = "./temp_docs"
CHROMA_DB_PATH = "./chroma_db"

CLASS_NAMES = [
    'caption', 'footnote', 'formula', 'list_item', 'page_footer', 'page_header',
    'picture', 'section_header', 'table', 'text', 'title'
]
TEXT_CLASSES = ['text', 'title', 'section_header', 'list_item', 'caption', 'footnote']

class QueryRequest(BaseModel):
    documents: str  # Blob URL
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]
#authenticator
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the Bearer token"""
    if credentials.credentials != HACKRX_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

#Document Processor
class DocumentProcessor:
    """Handle multiple document formats with AI-powered layout detection"""
    
    def __init__(self):
        self.model = None
        self.embeddings = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def initialize_models(self):
        """Initialize AI models"""
        try:
            print("🤖 Loading DocLayout-YOLO model...")
            try:
                self.model = YOLOv10.from_pretrained("juliozhao/DocLayout-YOLO-DocStructBench")
            except Exception as e:
                print(f"Fallback loading: {e}")
                filepath = hf_hub_download(
                    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
                )
                self.model = YOLOv10(filepath)
            
            print("🔤 Loading FREE embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            print("✅ All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def download_document(self, url: str) -> bytes:
        """Download document from blob URL"""
        try:
            print(f"📥 Downloading document from: {url[:100]}...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            print(f"✅ Downloaded {len(response.content)} bytes")
            return response.content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    
    def detect_file_type(self, content: bytes) -> str:
        """Detect file type from content"""
        if content.startswith(b'%PDF'):
            return 'pdf'
        elif content.startswith(b'PK'):
            # Check if it's a DOCX (ZIP-based format)
            try:
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(content)
                    tmp.flush()
                    doc = docx.Document(tmp.name)
                    return 'docx'
            except:
                pass
        
        # Try to parse as email
        try:
            content_str = content.decode('utf-8', errors='ignore')
            if 'From:' in content_str and 'Subject:' in content_str:
                return 'email'
        except:
            pass
        
        return 'unknown'
    
    def process_pdf_with_yolo(self, content: bytes) -> str:
        """Process PDF using DocLayout-YOLO + OCR"""
        import tempfile
        import time
        
        # Create temporary file
        tmp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        tmp_path = tmp_file.name
        
        try:
            # Write and close file properly
            tmp_file.write(content)
            tmp_file.close()  # Explicitly close before processing
            
            # Small delay to ensure file is released
            time.sleep(0.1)
            
            # Convert PDF to images
            images = convert_from_path(tmp_path)
            full_text = ""
            
            for page_num, image in enumerate(images, 1):
                print(f"   📖 Processing page {page_num}/{len(images)}")
                
                # Convert to OpenCV format
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # AI Layout Detection
                results = self.model(img_cv)[0]
                
                # Extract text blocks
                text_blocks = []
                for box in results.boxes:
                    class_idx = int(box.cls[0])
                    class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else 'unknown'
                    confidence = float(box.conf[0])
                    
                    if class_name in TEXT_CLASSES and confidence > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Add padding and crop
                        padding = 5
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(img_cv.shape[1], x2 + padding)
                        y2 = min(img_cv.shape[0], y2 + padding)
                        
                        cropped_img = img_cv[y1:y2, x1:x2]
                        
                        if cropped_img.size > 0:
                            text = pytesseract.image_to_string(cropped_img, lang='eng').strip()
                            if text:
                                text_blocks.append({
                                    'text': text,
                                    'class': class_name,
                                    'y_position': y1,
                                    'confidence': confidence
                                })
                
                # Sort by position and combine
                text_blocks.sort(key=lambda x: x['y_position'])
                page_text = ""
                
                for block in text_blocks:
                    if block['class'] in ['title', 'section_header']:
                        page_text += f"\n=== {block['text'].upper()} ===\n"
                    elif block['class'] == 'list_item':
                        page_text += f"• {block['text']}\n"
                    else:
                        page_text += f"{block['text']}\n"
                
                full_text += page_text + f"\n{'-'*50} Page {page_num} End {'-'*50}\n\n"
            
            return full_text.strip()
            
        finally:
            # Always try to clean up
            try:
                os.unlink(tmp_path)
            except:
                pass  # Ignore cleanup errors
    
    def process_docx(self, content: bytes) -> str:
        """Process DOCX files"""
        try:
            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file.write(content)
                tmp_file.flush()
                
                doc = docx.Document(tmp_file.name)
                full_text = ""
                
                for paragraph in doc.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        # Detect if it's a heading
                        if paragraph.style.name.startswith('Heading'):
                            full_text += f"\n=== {text.upper()} ===\n"
                        else:
                            full_text += f"{text}\n"
                
                # Process tables
                for table in doc.tables:
                    full_text += "\n=== TABLE DATA ===\n"
                    for row in table.rows:
                        row_text = " | ".join([cell.text.strip() for cell in row.cells])
                        if row_text.strip():
                            full_text += f"{row_text}\n"
                    full_text += "=== END TABLE ===\n\n"
                
                return full_text.strip()
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process DOCX: {str(e)}")
    
    def process_email(self, content: bytes) -> str:
        """Process email files"""
        try:
            content_str = content.decode('utf-8', errors='ignore')
            msg = email.message_from_string(content_str)
            
            full_text = ""
            
            # Extract headers
            full_text += f"=== EMAIL METADATA ===\n"
            full_text += f"From: {msg.get('From', 'Unknown')}\n"
            full_text += f"To: {msg.get('To', 'Unknown')}\n"
            full_text += f"Subject: {msg.get('Subject', 'No Subject')}\n"
            full_text += f"Date: {msg.get('Date', 'Unknown')}\n"
            full_text += f"=== EMAIL CONTENT ===\n\n"
            
            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True)
                        if body:
                            full_text += body.decode('utf-8', errors='ignore')
            else:
                body = msg.get_payload(decode=True)
                if body:
                    full_text += body.decode('utf-8', errors='ignore')
            
            return full_text.strip()
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process email: {str(e)}")
    
    def process_document(self, url: str) -> List[Document]:
        """Main document processing pipeline"""
        print(f"🔄 Processing document from URL...")
        
        # Download document
        content = self.download_document(url)
        
        # Detect file type
        file_type = self.detect_file_type(content)
        print(f"📄 Detected file type: {file_type}")
        
        # Process based on type
        if file_type == 'pdf':
            if self.model is None:
                raise HTTPException(status_code=500, detail="DocLayout-YOLO model not loaded")
            text = self.process_pdf_with_yolo(content)
        elif file_type == 'docx':
            text = self.process_docx(content)
        elif file_type == 'email':
            text = self.process_email(content)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from document")
        
        print(f"✅ Extracted {len(text)} characters")
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        print(f"📊 Created {len(chunks)} text chunks")
        
        # Create documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": url,
                    "chunk_id": i,
                    "file_type": file_type,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        
        return documents

#QA system

class HackRxIntelligentQA:
    """Complete intelligent QA system for insurance documents"""
    
    def __init__(self):
        self.groq_client = None
        self.doc_processor = DocumentProcessor()
        
    def initialize(self):
        """Initialize all components"""
        print("🚀 Initializing HackRx 6.0 Intelligent QA System...")
        
        # Initialize Groq
        try:
            self.groq_client = Groq(api_key=GROQ_API_KEY)
            print("⚡ Groq client initialized")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize Groq: {str(e)}")
        
        # Initialize document processor
        if not self.doc_processor.initialize_models():
            raise HTTPException(status_code=500, detail="Failed to initialize AI models")
        
        print("✅ HackRx system ready!")
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create vector database from documents"""
        print("🔄 Creating vector database...")
        
        try:
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.doc_processor.embeddings,
                persist_directory=None  # In-memory for faster processing
            )
            print("✅ Vector database created")
            return vectorstore
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create vector database: {str(e)}")
    
    def extract_query_intent(self, question: str) -> Dict[str, str]:
        """Extract intent and entities from question using Groq"""
        
        extraction_prompt = f"""
        You are an expert insurance query analyzer. Extract key information from this question and return ONLY a JSON object.

        Question: "{question}"

        Extract these fields (use "not specified" if not found):
        {{
            "query_type": "coverage/waiting_period/limits/benefits/definition/procedure",
            "medical_procedure": "specific medical procedure or condition mentioned",
            "policy_aspect": "what aspect of policy they're asking about",
            "time_related": "any time periods, waiting periods, or durations mentioned",
            "amount_related": "any amounts, limits, percentages, or financial aspects",
            "urgency": "emergency/routine/general"
        }}

        Return only the JSON object, no other text.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean JSON
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
            
            return json.loads(content)
            
        except Exception as e:
            print(f"⚠️ Intent extraction failed: {e}")
            return {
                "query_type": "general",
                "medical_procedure": "not specified",
                "policy_aspect": "coverage",
                "time_related": "not specified",
                "amount_related": "not specified",
                "urgency": "general"
            }
    
    def generate_answer(self, question: str, context: str, intent: Dict) -> str:
        """Generate comprehensive answer using Groq"""
        
        insurance_prompt = f"""
        You are an expert insurance policy analyst. Analyze the policy documents and answer the customer's question accurately.

        CUSTOMER QUESTION: {question}

        EXTRACTED INTENT:
        - Query Type: {intent.get('query_type', 'general')}
        - Medical Procedure: {intent.get('medical_procedure', 'not specified')}
        - Policy Aspect: {intent.get('policy_aspect', 'coverage')}

        POLICY CONTEXT:
        {context}

        INSTRUCTIONS:
        1. Answer based STRICTLY on the provided policy context
        2. Be specific and cite exact policy terms when possible
        3. If information is not in the context, state "not specified in provided policy documents"
        4. Include relevant details like waiting periods, conditions, limits, or exclusions
        5. Be clear and direct - this is for insurance claims processing

        Provide a comprehensive but concise answer:
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": insurance_prompt}],
                temperature=0.2,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"⚠️ Answer generation failed: {e}")
            return f"Unable to process the question '{question}' due to technical issues. Please try rephrasing or contact support."
    
    def answer_question(self, question: str, vectorstore: Chroma) -> str:
        """Process single question and return answer"""
        
        # Extract intent
        intent = self.extract_query_intent(question)
        
        # Search for relevant context
        try:
            # Enhanced search query based on intent
            search_query = question
            if intent.get('medical_procedure') != 'not specified':
                search_query += f" {intent['medical_procedure']}"
            if intent.get('query_type') != 'general':
                search_query += f" {intent['query_type']}"
            
            search_results = vectorstore.similarity_search(search_query, k=5)
            
            if not search_results:
                return "No relevant information found in the policy documents for this question."
            
            # Combine context
            context = "\n\n".join([doc.page_content for doc in search_results])
            
            # Generate answer
            answer = self.generate_answer(question, context, intent)
            
            return answer
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            return f"Error processing the question. Please try again or contact support."
    
    def process_queries(self, document_url: str, questions: List[str]) -> List[str]:
        """Main processing pipeline"""
        start_time = time.time()
        
        # Process document
        documents = self.doc_processor.process_document(document_url)
        
        # Create vector database
        vectorstore = self.create_vectorstore(documents)
        
        # Process all questions
        answers = []
        for i, question in enumerate(questions, 1):
            print(f"🤔 Processing question {i}/{len(questions)}: {question[:50]}...")
            answer = self.answer_question(question, vectorstore)
            answers.append(answer)
        
        total_time = time.time() - start_time
        print(f"⚡ Processed {len(questions)} questions in {total_time:.2f} seconds")
        
        return answers

#FastAPI application

app = FastAPI(
    title="HackRx 6.0 - Intelligent Document QA System",
    description="AI-powered document processing and question answering for insurance domains",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global QA system instance
qa_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the QA system on startup"""
    global qa_system
    try:
        print("🚀 Starting HackRx 6.0 system...")
        qa_system = HackRxIntelligentQA()
        qa_system.initialize()
        print("✅ HackRx 6.0 system ready!")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "system": "HackRx 6.0 - Intelligent Document QA",
        "version": "1.0.0",
        "components": {
            "doclayout_yolo": "✅ Loaded",
            "free_embeddings": "✅ Loaded", 
            "groq_llm": "✅ Connected",
            "vectorstore": "✅ Ready"
        }
    }

@app.post("/hackrx/run", response_model=QueryResponse)
async def hackrx_run(
    request: QueryRequest,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """
    Main HackRx endpoint for document processing and question answering
    """
    try:
        print(f"📥 Received HackRx request with {len(request.questions)} questions")
        
        if qa_system is None:
            raise HTTPException(status_code=500, detail="QA system not initialized")
        
        # Process queries
        answers = qa_system.process_queries(request.documents, request.questions)
        
        print(f"✅ Successfully processed all questions")
        
        return QueryResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {
                "qa_system": "✅" if qa_system is not None else "❌",
                "groq_client": "✅" if qa_system and qa_system.groq_client else "❌",
                "doc_processor": "✅" if qa_system and qa_system.doc_processor else "❌",
                "yolo_model": "✅" if qa_system and qa_system.doc_processor.model else "❌",
                "embeddings": "✅" if qa_system and qa_system.doc_processor.embeddings else "❌"
            }
        }
        
        # Quick functionality test
        if qa_system:
            try:
                test_response = qa_system.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": "Health check"}],
                    max_tokens=10
                )
                health_status["groq_test"] = "✅ Working"
            except:
                health_status["groq_test"] = "❌ Failed"
        
        return health_status
        
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    print("🏆 HackRx 6.0 - Complete AI Insurance Document QA System")
    print("=" * 70)
    print("🎯 Features:")
    print("   ✅ DocLayout-YOLO: Advanced document layout detection")
    print("   ✅ Multi-format: PDF, DOCX, Email support")
    print("   ✅ FREE Embeddings: HuggingFace all-MiniLM-L6-v2")
    print("   ✅ Groq LLM: Lightning-fast intelligent responses")
    print("   ✅ FastAPI: Production-ready REST API")
    print("   ✅ HTTPS Ready: Secure deployment support")
    print("=" * 70)
    
    # Create necessary directories
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )