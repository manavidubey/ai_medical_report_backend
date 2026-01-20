import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class RAGChatbot:
    def __init__(self):
        print("Initializing RAG Engine...")
        # Load embedding model (lightweight)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load local LLM for generation (using flan-t5-small for speed on CPU)
        # In a real setup, we might use a larger model or external API
        self.generator = pipeline("text2text-generation", model="google/flan-t5-small")
        
        self.index = None
        self.chunks = []
        
    def ingest_report(self, text):
        """
        Splits report into chunks and builds FAISS index.
        """
        # 1. Chunking (Simple sliding window or paragraph based)
        # Simple approach: Split by double newlines or sentences if too long
        self.chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
        
        # fallback if chunks are too large
        final_chunks = []
        for c in self.chunks:
            if len(c.split()) > 200:
                # split by sentences roughly
                subs = c.split('. ')
                final_chunks.extend(subs)
            else:
                final_chunks.append(c)
        self.chunks = final_chunks
        
        if not self.chunks: return
        
        # 2. Embed
        embeddings = self.encoder.encode(self.chunks)
        
        # 3. Index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print(f"RAG Index built with {len(self.chunks)} chunks.")

    def search(self, query, top_k=3):
        """
        Retrieves relevant context.
        """
        if not self.index: return []
        
        q_emb = self.encoder.encode([query])
        D, I = self.index.search(np.array(q_emb).astype('float32'), k=top_k)
        
        results = []
        for idx in I[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

    def answer_question(self, question):
        """
        End-to-end RAG pipeline.
        """
        if not self.index:
            return "Please upload and analyze a report first."
            
        context_chunks = self.search(question)
        context = " ".join(context_chunks)
        
        if self.is_unsafe(question):
            return {
                "answer": "I cannot provide a medical diagnosis or prescription. Please consult a qualified doctor for professional medical advice.",
                "sources": []
            }

        # Prompt Engineering for Flan-T5
        prompt = (
            f"You are a helpful and empathetic medical assistant. "
            f"Answer the question based strictly on the context provided. "
            f"Be professional but warm. Use simple language where possible.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        
        # Generate
        output = self.generator(prompt, max_length=200, do_sample=False)
        return {
            "answer": output[0]['generated_text'],
            "sources": context_chunks
        }

    def is_unsafe(self, question):
        """
        Basic guardrail against direct diagnosis requests.
        """
        triggers = ["diagnose me", "do i have cancer", "prescribe", "medication for me", "treatment plan"]
        q_lower = question.lower()
        return any(t in q_lower for t in triggers)

    def generate_doctor_questions(self):
        """
        Generates 3-5 smart questions for the patient to ask their doctor.
        """
        if not self.chunks:
            return "Please upload a report first."
            
        # Use simple heuristic or LLM summary chunks
        context = " ".join(self.chunks[:5]) # Use first few chunks as summary context
        
        prompt = (
            f"Based on this medical report, list 3 important questions the patient should ask their doctor. "
            f"Focus on abnormal values or risks.\n\n"
            f"Context: {context}\n\n"
            f"Questions:"
        )
        
        output = self.generator(prompt, max_length=150, do_sample=True) # Sample for variety
        return output[0]['generated_text']
