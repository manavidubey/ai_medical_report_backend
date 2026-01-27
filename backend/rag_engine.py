import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class MedicalAgent:
    """
    Agentic AI that uses a ReAct (Reason+Act) loop to solve complex medical queries.
    Tools: Report Search, Reference Lookup, Severity Check.
    """
    def __init__(self):
        print("Initializing Medical Agent & RAG Memory...")
        self.encoder = None
        self.generator = None
        self.index = None
        self.chunks = []
        
    def load_models(self):
        if self.encoder is None:
            print("Loading Agent Encoder...")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
        if self.generator is None:
            print("Loading Agent Brain (Flan-T5)...")
            # Upgrade to 'base' for better reasoning (Small is too weak)
            self.generator = pipeline("text2text-generation", model="google/flan-t5-base")

    def ingest_report(self, text):
        self.load_models()
        self.chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
        
        # Optimize chunk size
        final_chunks = []
        for c in self.chunks:
            if len(c.split()) > 200:
                subs = c.split('. ')
                final_chunks.extend(subs)
            else:
                final_chunks.append(c)
        self.chunks = final_chunks
        
        if not self.chunks: return
        
        embeddings = self.encoder.encode(self.chunks)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print(f"Agent Memory Updated: {len(self.chunks)} facts indexed.")

    # --- TOOLS ---
    def search_report(self, query):
        """Tool: Search the patient's report for specific info."""
        if not self.index: return "No report loaded."
        q_emb = self.encoder.encode([query])
        D, I = self.index.search(np.array(q_emb).astype('float32'), k=3)
        results = [self.chunks[i] for i in I[0] if i < len(self.chunks)]
        return "\n".join(results)

    def check_guidelines(self, query):
        """Tool: Check general medical guidelines (Simulated Knowledge Base)."""
        # In a real LangChain app, this would query an external vector DB.
        # Here we use heuristic knowledge.
        q_lower = query.lower()
        if "diabetes" in q_lower or "glucose" in q_lower:
            return "Guideline: Normal Fasting Glucose is 70-99 mg/dL. >126 implies Diabetes."
        if "bp" in q_lower or "pressure" in q_lower:
            return "Guideline: Normal BP is <120/80. >140/90 is Hypertension."
        if "fever" in q_lower:
            return "Guideline: Body temp > 100.4F (38C) is considered fever."
        return "No specific guideline found in local KB."

    # --- AGENT BRAIN (ReAct Loop) ---
    def answer_question(self, question):
        self.load_models()
        
        # 1. Thought Step: Simple routing (Zero-Shot Classification heuristic)
        # We classify if we need to search the report or answer generally.
        tool_to_use = "search_report"
        if "normal" in question.lower() or "range" in question.lower() or "standard" in question.lower():
            tool_to_use = "check_guidelines"
            
        print(f"Agent Thought: User asks '{question}'. Using Tool: [{tool_to_use}]")
        
        # 2. Action Step
        context = ""
        if tool_to_use == "search_report":
            context = self.search_report(question)
        elif tool_to_use == "check_guidelines":
            context = self.check_guidelines(question)
            
        # 3. Observation & Synthesis
        # Combine context with a specific prompt
        prompt = (
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer the question using the context:"
        )
        
        output = self.generator(prompt, max_length=200, do_sample=False)
        return {
            "answer": output[0]['generated_text'],
            "sources": [context[:100] + "..."] if context else []
        }

    def generate_doctor_questions(self):
        if not self.chunks: return "Please upload a report first."
        self.load_models()
        context = " ".join(self.chunks[:5])
        prompt = (
            f"List 3 questions to ask a doctor based on this report.\n"
            f"Context: {context}\n"
            f"Questions:"
        )
        output = self.generator(prompt, max_length=150, do_sample=True)
        return output[0]['generated_text']
