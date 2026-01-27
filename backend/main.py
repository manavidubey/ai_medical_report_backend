from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
import shutil
import os
import io
import json
import pdfplumber
import docx
from ai_engine import (
    analyze_full_report,
    analyze_medical_image,
    explain_to_patient
)
from report_generator import create_pdf
from rag_engine import MedicalAgent
from gtts import gTTS
from pdf_processor import process_pdf

app = FastAPI()

# CORS logic
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agents
medical_agent = MedicalAgent()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)



def extract_text_from_file(filepath):
    """
    Extracts text using the robust PDF pipeline (Text/OCR).
    """
    ext = os.path.splitext(filepath)[1].lower()
    text = ""
    
    try:
        if ext == '.pdf':
            with open(filepath, "rb") as f:
                file_bytes = f.read()
            # Use the new robust processor
            text = process_pdf(file_bytes)
            
        elif ext == '.docx':
            doc = docx.Document(filepath)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            # Fallback for txt/md
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        print(f"Error extracting text from {filepath}: {e}")
        return ""
        
    return text

@app.post("/analyze")
async def analyze_report(
    current_file: UploadFile = File(...),
    previous_file: Optional[UploadFile] = File(None),
    location: Optional[str] = Form(None)
):
    try:
        # Check if Image
        if current_file.content_type.startswith('image/'):
            print(f"Processing Image: {current_file.filename}")
            image_bytes = await current_file.read()
            result = analyze_medical_image(image_bytes)
            
            # Wrap in standard format for frontend compatibility
            return JSONResponse({
                "report_type": "Visual",
                "summary": result.get("finding", "Visual Analysis Complete"),
                "radiology": {
                    "abnormalities": [],
                    "visual_findings": result.get("predictions", [])
                },
                "sections": {"findings": result.get("note", "")},
                "patient_view": {
                    "findings": "Visual analysis performed.",
                    "impression": "AI has analyzed the visual patterns.",
                    "explanation": "Image classification results are screening estimates."
                },
                "risks": [],
                "labs": [],
                "recommendations": []
            })

        # Save files
        curr_path = os.path.join(UPLOAD_DIR, current_file.filename)
        with open(curr_path, "wb") as f:
            shutil.copyfileobj(current_file.file, f)
            
        prev_path = None
        if previous_file:
            prev_path = os.path.join(UPLOAD_DIR, previous_file.filename)
            with open(prev_path, "wb") as f:
                shutil.copyfileobj(previous_file.file, f)
            
        # Extract Text
        curr_text = extract_text_from_file(curr_path)
        if not curr_text.strip():
            return JSONResponse({"error": "Could not extract text from the file. Please ensure it is a readable PDF/image."}, status_code=400)
            
        prev_text = None
        if prev_path:
            prev_text = extract_text_from_file(prev_path)

        # Analyze
        result = analyze_full_report(curr_text, prev_text, location)
        
        # Ingest into RAG
        medical_agent.ingest_report(curr_text)
        
        return JSONResponse(result)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response = medical_agent.answer_question(request.question)
    return response

@app.post("/generate-pdf")
async def generate_pdf_endpoint(
    report_data: dict,
    filename: Optional[str] = "medical_report.pdf"
):
    try:
        # Generate
        pdf_path = create_pdf(report_data, filename)
        return FileResponse(pdf_path, media_type='application/pdf', filename=filename)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/generate-questions")
async def generate_questions_endpoint():
    response = medical_agent.generate_doctor_questions()
    return {"questions": response}

@app.post("/generate-referral")
async def generate_referral_endpoint(
    report_data: dict,
    filename: Optional[str] = "Referral_Letter.pdf"
):
    try:
        from report_generator import create_referral_letter
        
        pdf_path = create_referral_letter(report_data, filename)
        return FileResponse(pdf_path, media_type='application/pdf', filename=filename)
    except Exception as e:
        print(f"Referral Generation Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
