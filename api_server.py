"""
FastAPI server for the Agentic RAG system.
Provides REST endpoints for PDF ingestion and querying.
"""

import os
import shutil
import json
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from agentic_rag import RAGOrchestrator

app = FastAPI(
    title="Agentic RAG API",
    description="REST API for PDF-based RAG with reranking",
    version="1.0.0"
)

# Global RAG orchestrator instance
rag_orchestrator: Optional[RAGOrchestrator] = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    success: bool
    message: str = ""

class IngestResponse(BaseModel):
    success: bool
    message: str
    filename: str = ""

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Agentic RAG API is running", "status": "healthy"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Ingest a PDF file into the RAG system.
    
    Args:
        file: PDF file to be processed
        
    Returns:
        IngestResponse with success status and message
    """
    global rag_orchestrator
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize RAG orchestrator if not exists
        if rag_orchestrator is None:
            rag_orchestrator = RAGOrchestrator()
        
        # Ingest the PDF
        rag_orchestrator.ingest(temp_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return IngestResponse(
            success=True,
            message=f"Successfully ingested PDF: {file.filename}",
            filename=file.filename
        )
        
    except Exception as e:
        # Clean up temporary file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return IngestResponse(
            success=False,
            message=f"Error ingesting PDF: {str(e)}",
            filename=file.filename or ""
        )

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    Args:
        request: QueryRequest containing the question
        
    Returns:
        QueryResponse with the answer and status
    """
    global rag_orchestrator
    
    try:
        # Check if RAG system is initialized
        if rag_orchestrator is None:
            raise HTTPException(
                status_code=400, 
                detail="No PDF has been ingested yet. Please ingest a PDF first."
            )
        
        # Get answer from RAG system
        answer = rag_orchestrator.query(request.question)
        
        return QueryResponse(
            answer=answer,
            success=True,
            message="Query processed successfully"
        )
        
    except Exception as e:
        return QueryResponse(
            answer="",
            success=False,
            message=f"Error processing query: {str(e)}"
        )

@app.post("/query_stream")
async def query_rag_stream(request: QueryRequest):
    """Stream the answer to a query as it's generated."""
    global rag_orchestrator

    # Check initialization
    if rag_orchestrator is None:
        raise HTTPException(
            status_code=400,
            detail="No PDF has been ingested yet. Please ingest a PDF first."
        )

    def token_generator():
        try:
            for token in rag_orchestrator.query_stream(request.question):
                yield token
        except Exception as e:
            # Send error message and terminate stream
            yield f"\n[error] {str(e)}\n"

    # Stream plain text chunks; clients can display incrementally
    return StreamingResponse(token_generator(), media_type="text/plain")

@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket endpoint for streaming RAG queries.
    
    Expected message format: Raw text question
    Response format: Raw text tokens streamed directly
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client (raw text question)
            question = await websocket.receive_text()
            question = question.strip()
            
            if not question:
                await websocket.send_text("[ERROR] Question cannot be empty")
                continue
            
            if not rag_orchestrator:
                await websocket.send_text("[ERROR] RAG system not initialized. Please ingest a PDF first.")
                continue
            
            # Stream the response (raw tokens like query_stream)
            try:
                for token in rag_orchestrator.query_stream(question):
                    await websocket.send_text(token)
                
                # Send completion signal
                await websocket.send_text("\n\n--- Response Complete ---\n")
                
            except Exception as e:
                await websocket.send_text(f"[ERROR] {str(e)}")
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")

@app.get("/status")
async def get_status():
    """Get the current status of the RAG system."""
    global rag_orchestrator
    
    return {
        "initialized": rag_orchestrator is not None,
        "ready_for_queries": rag_orchestrator is not None,
        "message": "RAG system is ready" if rag_orchestrator else "No PDF ingested yet"
    }

@app.delete("/reset")
async def reset_rag():
    """Reset the RAG system (clear all ingested data)."""
    global rag_orchestrator
    
    rag_orchestrator = None
    
    return {
        "success": True,
        "message": "RAG system has been reset"
    }

def initialize_rag_with_existing_pdf():
    """Initialize RAG system with the existing PDF file."""
    global rag_orchestrator
    
    pdf_path = "pdf/Genting.pdf"
    if os.path.exists(pdf_path):
        print(f"📚 Found existing PDF: {pdf_path}")
        print("🔄 Initializing RAG system...")
        
        try:
            rag_orchestrator = RAGOrchestrator()
            rag_orchestrator.ingest(pdf_path)
            print("✅ PDF successfully ingested and ready for queries!")
            return True
        except Exception as e:
            print(f"❌ Error ingesting PDF: {str(e)}")
            return False
    else:
        print(f"📝 No PDF found at {pdf_path}")
        print("   You can upload PDFs via POST /ingest")
        return False

if __name__ == "__main__":
    print("🚀 Starting Agentic RAG API Server...")
    
    # Auto-ingest existing PDF
    initialize_rag_with_existing_pdf()
    
    print("❓ Ask questions via POST /query")
    print("🔍 Check status via GET /status")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )