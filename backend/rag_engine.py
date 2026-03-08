import os
import json

class MedicalAgent:
    """
    Agentic AI that uses OpenAI to solve medical queries.
    Replaced local FAISS/Transformers with OpenAI for Vercel/Lightweight local run.
    """
    def __init__(self):
        self.report_text = ""
        
    def ingest_report(self, text):
        """Stores the report text for context."""
        self.report_text = text
        print(f"Agent Memory Updated: Report ingested.")

    def check_guidelines(self, query):
        """Heuristic knowledge tool."""
        q_lower = query.lower()
        if "diabetes" in q_lower or "glucose" in q_lower:
            return "Guideline: Normal Fasting Glucose is 70-99 mg/dL. >126 implies Diabetes."
        if "bp" in q_lower or "pressure" in q_lower:
            return "Guideline: Normal BP is <120/80. >140/90 is Hypertension."
        if "fever" in q_lower:
            return "Guideline: Body temp > 100.4F (38C) is considered fever."
        return "No specific guideline found in local KB."

    def answer_question(self, question):
        """Uses OpenAI with report context to answer questions."""
        from ai_engine import get_openai_client
        client = get_openai_client()
        
        if not client:
            return {"answer": "AI service unavailable (Check OpenAI API Key).", "sources": []}

        try:
            # Combine context
            context = self.report_text[:12000] # Limit context size
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant. Use the provided report context to answer the user's question accurately. If the answer isn't in the report, use your general medical knowledge but specify it's general info."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                ],
                max_tokens=300,
            )
            
            answer = response.choices[0].message.content
            return {
                "answer": answer,
                "sources": ["Current Medical Report"] if self.report_text else []
            }
        except Exception as e:
            print(f"Agent Error: {e}")
            return {"answer": f"Error generating answer: {str(e)}", "sources": []}

    def generate_doctor_questions(self):
        """Generates follow-up questions for a clinician."""
        from ai_engine import get_openai_client
        client = get_openai_client()
        
        if not client or not self.report_text:
            return "Unable to generate questions at this time."

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Generate 3 professional questions a patient should ask their doctor based on this report."},
                    {"role": "user", "content": f"Report Content: {self.report_text[:8000]}"}
                ],
                max_tokens=200,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
