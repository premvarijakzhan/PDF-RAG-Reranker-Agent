# Agentic RAG API Setup

This project now includes a FastAPI backend and terminal client for your RAG system.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Environment
Make sure you have your OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

### 3. Start the API Server
```bash
python api_server.py
```
The server will start at `http://localhost:8000`

### 4. Use the Terminal Client

#### Interactive Mode
```bash
python terminal_client.py
```

#### Command Line Mode
```bash
# Upload a PDF
python terminal_client.py --upload path/to/your/document.pdf

# Ask a question
python terminal_client.py --ask "What is this document about?"

# Use custom server URL
python terminal_client.py --url http://localhost:8000
```

## API Endpoints

### Health Check
- **GET** `/` - Check if server is running

### PDF Ingestion
- **POST** `/ingest` - Upload and process a PDF file
  - Form data: `file` (PDF file)

### Querying
- **POST** `/query` - Ask questions about ingested PDF
  - JSON body: `{"question": "your question here"}`

### System Management
- **GET** `/status` - Check RAG system status
- **DELETE** `/reset` - Reset the system (clear all data)

## Terminal Client Commands

When running in interactive mode:
- `upload <pdf_path>` - Upload and ingest a PDF
- `ask <question>` - Ask a question
- `status` - Check system status
- `reset` - Reset the RAG system
- `help` - Show available commands
- `quit` or `exit` - Exit the application

## Example Usage

1. Start the server:
   ```bash
   python api_server.py
   ```

2. In another terminal, start the client:
   ```bash
   python terminal_client.py
   ```

3. Upload a PDF:
   ```
   rag> upload pdf/straitstrading.pdf
   ```

4. Ask questions:
   ```
   rag> ask What is the main business of this company?
   ```

## Architecture

- **`api_server.py`** - FastAPI backend that wraps your RAG orchestrator
- **`terminal_client.py`** - Command-line client for interacting with the API
- **`agentic_rag.py`** - Your existing RAG implementation (unchanged)
- **`streamlit_app.py`** - Your existing Streamlit UI (still works)

The FastAPI backend maintains a persistent RAG orchestrator instance, so you don't need to reinitialize it for each query.