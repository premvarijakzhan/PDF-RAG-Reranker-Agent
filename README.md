# Agentic ReRanking RAG

An agentic Retrieval-Augmented Generation (RAG) pipeline with parallel retrieval, modular agents, and LLM-based reranking for robust, transparent, and high-quality question answering over PDFs and other documents.


## Table of Contents

- [Agentic ReRanking RAG](#agentic-reranking-rag)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Configuration](#configuration)
  - [Usage](#usage)
    - [Python Module](#python-module)
    - [Streamlit App](#streamlit-app)
    - [FastAPI Backend \& Terminal Client](#fastapi-backend--terminal-client)
    - [CLI \& Async App](#cli--async-app)
  - [API Endpoints](#api-endpoints)
  - [Extending \& Customizing](#extending--customizing)
  - [Documentation](#documentation)
  - [License](#license)

## Features

- **Agentic Design**: Modular agents for PDF loading, embedding, retrieval, question answering, and reranking coordinated by a central orchestrator.
- **Parallel Retrieval & QA**: Generate diverse context sets via perturbed embeddings, then answer in parallel for efficiency.
- **LLM-based Reranking**: A final RankingAgent selects the best answer with rationale, improving accuracy and transparency.
- **Multiple Interfaces**: Use as a Python module, a Streamlit web app, a FastAPI backend with terminal client, or an async CLI application.
- **Extensible Pipeline**: Swap or extend agents (e.g., HTML loader, CSV loader), tune hyperparameters, and integrate external sources.

## Project Structure

```text
PDF-RAG-Reranker-Agent/
├── agentic_rag.py        # Core agentic RAG implementation
├── streamlit_app.py      # Streamlit interface for PDF ingestion and QA
├── api_server.py         # FastAPI backend server
├── terminal_client.py    # Terminal client for API interaction
├── multi/                # Advanced async CLI and multi-agent app
│   └── agentic-rag-app.py
├── doc/                  # Diagrams, tutorial, and design docs
├── pdf/                  # PDF files (auto-ingested on server start)
│   └── .pdf
├── input/                # Sample PDF inputs
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project metadata
└── .env                  # Environment variables (e.g., OPENAI_API_KEY)
``` 

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Access to a terminal or command prompt

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/agentic-rag.git
   cd agentic-rag
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .\.venv\\Scripts\\activate # Windows
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Configuration

1. Copy the `.env` template or create a `.env` file in the project root.
2. Add your OpenAI API key:
   ```env
   OPENAI_API_KEY=sk-...
   ```

## Usage

### Python Module

Use the `RAGOrchestrator` class to ingest PDFs and run queries in your Python code:

```python
from agentic_rag import RAGOrchestrator

rag = RAGOrchestrator(n_candidates=3, k=5)
rag.ingest('path/to/document.pdf')
answer = rag.query('What is the main finding?')
print(answer)
```

### Streamlit App

Launch a web interface to upload PDFs and ask questions:

```bash
streamlit run streamlit_app.py
```

- Upload a PDF
- Click **Ingest PDF**
- Enter questions and view answers interactively

### FastAPI Backend & Terminal Client

Launch a REST API server with automatic PDF ingestion and use a terminal client for interaction:

#### Start the API Server
```bash
python api_server.py
```
The server automatically ingests `pdf/straitstrading.pdf` on startup and runs at `http://localhost:8000`

#### Use the Terminal Client

**Interactive Mode:**
```bash
python terminal_client.py
```

**Command Line Mode:**
```bash
# Ask a question directly
python terminal_client.py --ask "What is the main business of this company?"

# Use custom server URL
python terminal_client.py --url http://localhost:8000
```

**Terminal Client Commands (Interactive Mode):**
- `ask <question>` - Ask a question about the ingested PDF
- `status` - Check system status
- `reset` - Reset the RAG system
- `help` - Show available commands
- `quit` or `exit` - Exit the application

### CLI & Async App

An advanced asynchronous CLI app is available in the `multi/` folder, supporting query decomposition, ChromaDB storage, and external sources.

```bash
cd multi
python agentic-rag-app.py
```

Type `/help` in the prompt for commands such as `/index`, `/search`, and `/stats`.

## API Endpoints

The FastAPI backend provides the following REST endpoints:

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

**Example API Usage:**
```bash
# Check server health
curl http://localhost:8000/

# Ask a question
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is this document about?"}'

# Check system status
curl http://localhost:8000/status
```

## Extending & Customizing

- Adjust hyperparameters (`chunk_size`, `chunk_overlap`, `n_candidates`, `k`) in constructors.
- Add new agents for different data types (HTML, CSV, code).
- Modify the `RankingAgent` prompt or logic for custom reranking strategies.
- Integrate UIs or APIs for real-time inspection of candidate contexts and rankings.

"# PDF-RAG-Reranker-Agent" 
# PDF-RAG-Reranker-Agent
