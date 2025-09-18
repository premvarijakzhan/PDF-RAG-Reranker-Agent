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
    - [CLI \& Async App](#cli--async-app)
  - [Extending \& Customizing](#extending--customizing)
  - [Documentation](#documentation)
  - [License](#license)

## Features

- **Agentic Design**: Modular agents for PDF loading, embedding, retrieval, question answering, and reranking coordinated by a central orchestrator.
- **Parallel Retrieval & QA**: Generate diverse context sets via perturbed embeddings, then answer in parallel for efficiency.
- **LLM-based Reranking**: A final RankingAgent selects the best answer with rationale, improving accuracy and transparency.
- **Multiple Interfaces**: Use as a Python module, a Streamlit web app, or an async CLI application.
- **Extensible Pipeline**: Swap or extend agents (e.g., HTML loader, CSV loader), tune hyperparameters, and integrate external sources.

## Project Structure

```text
PDF-RAG-Reranker-Agent/
├── agentic_rag.py        # Core agentic RAG implementation
├── streamlit_app.py      # Streamlit interface for PDF ingestion and QA
├── multi/                # Advanced async CLI and multi-agent app
│   └── agentic-rag-app.py
├── doc/                  # Diagrams, tutorial, and design docs
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

### CLI & Async App

An advanced asynchronous CLI app is available in the `multi/` folder, supporting query decomposition, ChromaDB storage, and external sources.

```bash
cd multi
python agentic-rag-app.py
```

Type `/help` in the prompt for commands such as `/index`, `/search`, and `/stats`.

## Extending & Customizing

- Adjust hyperparameters (`chunk_size`, `chunk_overlap`, `n_candidates`, `k`) in constructors.
- Add new agents for different data types (HTML, CSV, code).
- Modify the `RankingAgent` prompt or logic for custom reranking strategies.
- Integrate UIs or APIs for real-time inspection of candidate contexts and rankings.

"# PDF-RAG-Reranker-Agent" 
# PDF-RAG-Reranker-Agent
