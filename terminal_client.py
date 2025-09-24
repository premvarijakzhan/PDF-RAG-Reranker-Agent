"""
Terminal client for the Agentic RAG API.
Provides a command-line interface to interact with the FastAPI backend.
"""

import os
import sys
import requests
import json
import asyncio
import websockets
from typing import Optional
import argparse
from pathlib import Path

class RAGTerminalClient:
    """Terminal client for interacting with the Agentic RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def check_server_health(self) -> bool:
        """Check if the API server is running."""
        try:
            response = self.session.get(f"{self.base_url}/")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
    
    def get_status(self) -> dict:
        """Get the current status of the RAG system."""
        try:
            response = self.session.get(f"{self.base_url}/status")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def ingest_pdf(self, pdf_path: str) -> dict:
        """Upload and ingest a PDF file."""
        if not os.path.exists(pdf_path):
            return {"success": False, "message": f"File not found: {pdf_path}"}
        
        if not pdf_path.lower().endswith('.pdf'):
            return {"success": False, "message": "File must be a PDF"}
        
        try:
            with open(pdf_path, 'rb') as f:
                files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
                response = self.session.post(f"{self.base_url}/ingest", files=files)
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"Error uploading PDF: {str(e)}"}
    
    def query(self, question: str) -> dict:
        """Send a query to the RAG system."""
        try:
            data = {"question": question}
            response = self.session.post(f"{self.base_url}/query", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"Error querying: {str(e)}"}

    def query_stream(self, question: str):
        """Stream the answer from the RAG system (generator yielding text chunks)."""
        try:
            with self.session.post(f"{self.base_url}/query_stream", json={"question": question}, stream=True) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        yield chunk
        except requests.exceptions.RequestException as e:
            yield f"\n[error] {str(e)}\n"
    
    async def query_websocket(self, question: str):
        """Stream the answer from the RAG system via WebSocket."""
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/query"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                # Send the question
                message = json.dumps({"question": question})
                await websocket.send(message)
                
                # Receive and yield responses
                async for response in websocket:
                    try:
                        data = json.loads(response)
                        
                        if data.get("type") == "token":
                            yield data.get("content", "")
                        elif data.get("type") == "error":
                            yield f"\n[error] {data.get('content', 'Unknown error')}\n"
                            break
                        elif data.get("type") == "done":
                            break
                            
                    except json.JSONDecodeError:
                        yield f"\n[error] Invalid response format\n"
                        break
                        
        except Exception as e:
            yield f"\n[error] WebSocket error: {str(e)}\n"
    
    def reset(self) -> dict:
        """Reset the RAG system."""
        try:
            response = self.session.delete(f"{self.base_url}/reset")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"Error resetting: {str(e)}"}

def print_banner():
    """Print the application banner."""
    print("=" * 60)
    print("🤖 Agentic RAG Terminal Client")
    print("📚 Upload PDFs and ask questions via command line")
    print("=" * 60)

def print_help():
    """Print available commands."""
    print("\n📋 Available Commands:")
    print("  upload <pdf_path>  - Upload and ingest a PDF file")
    print("  ask <question>     - Ask a question about the ingested PDF")
    print("  ask-stream <question> - Ask a question and stream the answer live (HTTP)")
    print("  ask-ws <question>  - Ask a question and stream the answer via WebSocket")
    print("  status            - Check system status")
    print("  reset             - Reset the RAG system")
    print("  help              - Show this help message")
    print("  quit/exit         - Exit the application")
    print()

def interactive_mode(client: RAGTerminalClient):
    """Run the client in interactive mode."""
    print_banner()
    
    # Check server connection
    if not client.check_server_health():
        print("❌ Cannot connect to API server at", client.base_url)
        print("   Make sure the server is running with: python api_server.py")
        return
    
    print(f"✅ Connected to API server at {client.base_url}")
    
    # Show initial status
    status = client.get_status()
    if "error" not in status:
        if status.get("initialized"):
            print("📚 RAG system is ready for queries")
        else:
            print("📝 No PDF ingested yet. Use 'upload <pdf_path>' to get started")
    
    print_help()
    
    while True:
        try:
            user_input = input("🤖 rag> ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            
            if command in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            elif command == 'help':
                print_help()
            
            elif command == 'status':
                status = client.get_status()
                if "error" in status:
                    print(f"❌ Error: {status['error']}")
                else:
                    print(f"📊 Status: {status['message']}")
                    print(f"   Initialized: {status['initialized']}")
                    print(f"   Ready for queries: {status['ready_for_queries']}")
            
            elif command == 'reset':
                print("🔄 Resetting RAG system...")
                result = client.reset()
                if result.get("success"):
                    print("✅ RAG system reset successfully")
                else:
                    print(f"❌ Error: {result.get('message', 'Unknown error')}")
            
            elif command == 'upload':
                if len(parts) < 2:
                    print("❌ Usage: upload <pdf_path>")
                    continue
                
                pdf_path = parts[1]
                print(f"📤 Uploading and ingesting: {pdf_path}")
                
                result = client.ingest_pdf(pdf_path)
                if result.get("success"):
                    print(f"✅ {result['message']}")
                else:
                    print(f"❌ {result['message']}")
            
            elif command == 'ask':
                if len(parts) < 2:
                    print("❌ Usage: ask <question>")
                    continue
                
                question = parts[1]
                print(f"🤔 Thinking about: {question}")
                
                result = client.query(question)
                if result.get("success"):
                    print(f"\n💡 Answer:")
                    print(f"   {result['answer']}")
                    print()
                else:
                    print(f"❌ {result.get('message', 'Unknown error')}")

            elif command == 'ask-stream':
                if len(parts) < 2:
                    print("❌ Usage: ask-stream <question>")
                    continue
                question = parts[1]
                print(f"📡 Streaming answer for: {question}")
                print()
                # Stream tokens and print without newlines for live effect
                for token in client.query_stream(question):
                    sys.stdout.write(token)
                    sys.stdout.flush()
                print("\n")

            elif command == 'ask-ws':
                if len(parts) < 2:
                    print("❌ Usage: ask-ws <question>")
                    continue
                question = parts[1]
                print(f"🔌 WebSocket streaming answer for: {question}")
                print()
                
                # Run the async WebSocket query
                async def run_websocket_query():
                    async for token in client.query_websocket(question):
                        sys.stdout.write(token)
                        sys.stdout.flush()
                
                try:
                    asyncio.run(run_websocket_query())
                    print("\n")
                except Exception as e:
                    print(f"\n❌ WebSocket error: {str(e)}")
            
            else:
                print(f"❌ Unknown command: {command}")
                print("   Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Agentic RAG Terminal Client")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000",
        help="API server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--upload",
        help="Upload a PDF file and exit"
    )
    parser.add_argument(
        "--ask",
        help="Ask a question and exit"
    )
    parser.add_argument(
        "--ask-stream",
        help="Ask a question and stream the answer live, then exit"
    )
    parser.add_argument(
        "--ask-ws",
        help="Ask a question and stream the answer via WebSocket, then exit"
    )
    
    args = parser.parse_args()
    
    client = RAGTerminalClient(args.url)
    
    # Non-interactive mode
    if args.upload or args.ask or args.ask_stream or args.ask_ws:
        if not client.check_server_health():
            print(f"❌ Cannot connect to API server at {args.url}")
            sys.exit(1)
        
        if args.upload:
            result = client.ingest_pdf(args.upload)
            if result.get("success"):
                print(f"✅ {result['message']}")
            else:
                print(f"❌ {result['message']}")
                sys.exit(1)
        
        if args.ask:
            result = client.query(args.ask)
            if result.get("success"):
                print(result['answer'])
            else:
                print(f"❌ {result.get('message', 'Unknown error')}")
                sys.exit(1)

        if args.ask_stream:
            for token in client.query_stream(args.ask_stream):
                sys.stdout.write(token)
                sys.stdout.flush()
            print()

        if args.ask_ws:
            async def run_websocket_cli():
                async for token in client.query_websocket(args.ask_ws):
                    sys.stdout.write(token)
                    sys.stdout.flush()
            
            try:
                asyncio.run(run_websocket_cli())
                print()
            except Exception as e:
                print(f"❌ WebSocket error: {str(e)}")
                sys.exit(1)
    
    else:
        # Interactive mode
        interactive_mode(client)

if __name__ == "__main__":
    main()