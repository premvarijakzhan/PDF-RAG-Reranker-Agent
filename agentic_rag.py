
import os
import faiss
import tiktoken
import openai
from dotenv import load_dotenv
from typing import List, Tuple
from PyPDF2 import PdfReader

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o-mini"
ENC = tiktoken.get_encoding("cl100k_base")

def num_tokens(text:str) -> int:
    return len(ENC.encode(text))

class PDFLoaderAgent:
    """Load a PDF and split into ~500-token chunks."""
    def __init__(self, chunk_size:int=500, chunk_overlap:int=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split(self, path:str) -> List[str]:
        print(f"[PDFLoaderAgent] Loading and splitting PDF: {path}")
        reader = PdfReader(path)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        tokens = ENC.encode(full_text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk = ENC.decode(tokens[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        print(f"[PDFLoaderAgent] Total chunks created: {len(chunks)}")
        return chunks

class EmbeddingAgent:
    """Embed text chunks and build/upsert into a FAISS index."""
    def __init__(self, dim:int=1536):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)

    def embed(self, texts:List[str]) -> List[List[float]]:
        print(f"[EmbeddingAgent] Creating embeddings for {len(texts)} texts")
        response = openai.embeddings.create(model=EMBED_MODEL, input=texts)
        embeddings = [item.embedding for item in response.data]
        print(f"[EmbeddingAgent] Created embeddings")
        return embeddings

    def add_to_index(self, texts:List[str]):
        print(f"[EmbeddingAgent] Adding {len(texts)} embeddings to index")
        embs = self.embed(texts)
        import numpy as np
        vecs = np.array(embs, dtype="float32")
        self.index.add(vecs)
        print(f"[EmbeddingAgent] Added embeddings to index. Total vectors now: {self.index.ntotal}")

class RetrievalAgent:
    """Retrieve multiple sets of top-k similar chunks from FAISS for candidate diversity."""
    def __init__(self, index:faiss.IndexFlatL2):
        self.index = index

    def retrieve_candidates(self, query:str, texts:List[str], n_candidates:int=3, k:int=5) -> List[List[str]]:
        # For diversity, perturb the query embedding slightly for each candidate
        print(f"[RetrievalAgent] Retrieving {n_candidates} sets of top {k} chunks for query: {query}")
        base_emb = EmbeddingAgent().embed([query])[0]
        import numpy as np
        candidates = []
        for i in range(n_candidates):
            perturbed_emb = np.array(base_emb, dtype="float32") + np.random.normal(0, 0.01, len(base_emb))
            D, I = self.index.search(np.array([perturbed_emb], dtype="float32"), k)
            retrieved = [texts[j] for j in I[0] if j < len(texts)]
            candidates.append(retrieved)
        print(f"[RetrievalAgent] Created {len(candidates)} candidate sets")
        return candidates

class QAAgent:
    """Answer questions given retrieved context."""
    def __init__(self, model:str=CHAT_MODEL):
        self.model = model

    def answer(self, question:str, context:List[str]) -> str:
        print(f"[QAAgent] Answering question with model {self.model}")
        context_str = '---\n'.join(context)
        prompt = (
            "You are an expert assistant. Use the following context to answer the question.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\nAnswer:"
        )
        print(f"[QAAgent] Sending prompt to model. Prompt length: {len(prompt)} characters")
        resp = openai.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":prompt}],
            temperature=0.2,
            max_tokens=500
        )
        answer = resp.choices[0].message.content.strip()
        print(f"[QAAgent] Received answer of length {len(answer)}")
        return answer

    def answer_parallel(self, question:str, candidate_contexts:List[List[str]]) -> List[str]:
        """Generate answers to the question in parallel for multiple context sets."""
        print(f"[QAAgent] Generating answers in parallel for {len(candidate_contexts)} candidates.")
        from concurrent.futures import ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.answer, question, ctx) for ctx in candidate_contexts]
            for fut in futures:
                results.append(fut.result())
        return results

class RankingAgent:
    """Rank/score multiple answer candidates given the question and their context."""
    def __init__(self, model:str=CHAT_MODEL):
        self.model = model

    def rank(self, question:str, candidate_answers:List[str], candidate_contexts:List[List[str]]) -> Tuple[str, int]:
        """Returns the best answer and its index, and explains why."""
        print("[RankingAgent] Ranking candidates with LLM self-eval.")

        # Print all candidates and their answers
        print("\n[RankingAgent] All candidate contexts and answers:")
        for idx, (ctx, ans) in enumerate(zip(candidate_contexts, candidate_answers), 1):
            print(f"\nCandidate #{idx} Context:\n----------------------")
            for chunk in ctx:
                print(chunk)
                print('---')
            print(f"Candidate #{idx} Answer: {ans}\n----------------------")

        ranking_prompt = f"""
You are an expert assistant judging a RAG system. Given several candidate answers (each with their retrieval context) to the same question, first select the single most accurate/supportable candidate, then explain briefly why you chose it.\n\nOutput exactly this format:\nCandidate #N\nReason: <reason>\n\nBest Answer:\n<full text>\n\nQuestion: {question}\n"""
        summary = ""
        for idx, (ctx, ans) in enumerate(zip(candidate_contexts, candidate_answers), 1):
            ctx_part = "\n".join(ctx)
            summary += f"\nCandidate #{idx}:\nContext:\n{ctx_part}\nAnswer:\n{ans}\n"
        full_prompt = ranking_prompt + summary

        resp = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": full_prompt}],
            temperature=0.2,
            max_tokens=350
        )
        response_text = resp.choices[0].message.content.strip()

        # Print the LLM's output decision and reason
        print("\n[RankingAgent] LLM Decision and Reason:\n----------------------\n" + response_text + "\n----------------------")

        import re
        m = re.search(r"Candidate #(\d+)\s*\nReason:([^\n]*)\n+Best Answer:\n(.+)", response_text, re.DOTALL)
        if m:
            cand_idx = int(m.group(1)) - 1
            reason = m.group(2).strip()
            answer = m.group(3).strip()
            print(f"[RankingAgent] Selected candidate #{cand_idx+1}.")
            print(f"[RankingAgent] Reasoning: {reason}")
        else:
            cand_idx = 0
            answer = candidate_answers[0]
            print("[RankingAgent] Could not parse ranking output, returning first candidate.")
            print("[RankingAgent] LLM output was:\n" + response_text)
        return answer, cand_idx

class RAGOrchestrator:
    """Fully agentic and parallel RAG orchestrator."""
    def __init__(self, n_candidates:int=3, k:int=5, auto_load_pdf:bool=True):
        print("[RAGOrchestrator] Initializing agents")
        self.loader = PDFLoaderAgent()
        self.embedder = EmbeddingAgent()
        self.text_chunks: List[str] = []
        self.retriever: RetrievalAgent = None
        self.qa = QAAgent()
        self.ranker = RankingAgent()
        self.n_candidates = n_candidates
        self.k = k
        
        # Auto-load PDF from pdf folder if enabled
        if auto_load_pdf:
            pdf_path = os.path.join(os.path.dirname(__file__), "pdf", "Genting.pdf")
            if os.path.exists(pdf_path):
                print(f"[RAGOrchestrator] Auto-loading PDF: {pdf_path}")
                self.ingest(pdf_path)
            else:
                print(f"[RAGOrchestrator] PDF not found at {pdf_path}, skipping auto-load")

    def ingest(self, pdf_path:str):
        print(f"[RAGOrchestrator] Ingesting PDF: {pdf_path}")
        self.text_chunks = self.loader.load_and_split(pdf_path)
        self.embedder.add_to_index(self.text_chunks)
        self.retriever = RetrievalAgent(self.embedder.index)
        print(f"[RAGOrchestrator] Ingestion complete with {len(self.text_chunks)} chunks")

    def query(self, question:str) -> str:
        print(f"[RAGOrchestrator] Querying for question: {question}")
        # Step 1: Retrieval
        candidate_contexts = self.retriever.retrieve_candidates(question, self.text_chunks, n_candidates=self.n_candidates, k=self.k)
        # Step 2: QA in parallel
        candidate_answers = self.qa.answer_parallel(question, candidate_contexts)
        # Step 3: Ranking
        final_answer, chosen_idx = self.ranker.rank(question, candidate_answers, candidate_contexts)
        print(f"[RAGOrchestrator] Final answer selected from candidate #{chosen_idx+1}.")
        return final_answer