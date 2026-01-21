import fitz  # PyMuPDF
import cv2
import numpy as np
import pdfplumber
import io
import os
# Lazy load OCR model
ocr_model = None

def get_ocr_model():
    global ocr_model
    if ocr_model is None:
        try:
            from paddleocr import PaddleOCR
            print("Loading PaddleOCR Model (English)...")
            # use_angle_cls=True for better orientation detection
            # lang='en' for English medical reports
            ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        except ImportError:
            print("Error: PaddleOCR not installed. Please install 'paddleocr' and 'paddlepaddle'.")
            return None
    return ocr_model

def detect_pdf_type(file_bytes):
    """
    Detects if PDF is Text-Based (selectable) or Scanned (requires OCR).
    Returns: "text_pdf" or "scanned_pdf"
    """
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            # Check first few pages
            total_text_len = 0
            for page in pdf.pages[:3]:
                text = page.extract_text()
                if text:
                    total_text_len += len(text.strip())
            
            # Heuristic: If we extracted > 100 chars, it's likely text-based
            if total_text_len > 100:
                print("PDF Detection: Text-Based")
                return "text_pdf"
            
            print("PDF Detection: Scanned/Image-Based (Low text count)")
            return "scanned_pdf"
    except Exception as e:
        print(f"Error detecting PDF type: {e}. Defaulting to scanned.")
        return "scanned_pdf"

def preprocess_image_for_ocr(image):
    """
    Standard preprocessing for medical docs:
    Grayscale -> Thresholding -> Denoising
    """
    # Convert RGB to Grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Adaptive Thresholding (good for varying lighting in scanners)
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Simple binarization is often safer for clean digital scans
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    return binary

def extract_text_with_layout(file_bytes):
    """
    Extracts text using pdfplumber's layout preservation.
    Best for digital medical reports with tables.
    """
    text_content = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                # layout=True helps preserve column structure visually
                text = page.extract_text(layout=True)
                if text:
                    text_content += text + "\n"
    except Exception as e:
        print(f"Layout Extraction Error: {e}")
    return text_content

def extract_text_with_ocr(file_bytes):
    """
    Converts PDF pages to images and runs PaddleOCR.
    Best for scanned reports or images wrapped in PDF.
    """
    text_content = ""
    ocr = get_ocr_model()
    
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Render page to image (300 DPI = zoom_x=4, zoom_y=4 roughly)
            mat = fitz.Matrix(2, 2) 
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to numpy array (RGB)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4: # RGBA
                img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
            
            # Preprocess
            # processed_img = preprocess_image_for_ocr(img_data)
            # PaddleOCR handles raw inputs quite well, but grayscale can help speed
            
            # Run OCR
            # result structure: [[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)]]
            result = ocr.ocr(img_data, cls=True)
            
            if not result or result[0] is None:
                continue
                
            # Reconstruct lines
            # Sort boxes by Y-coordinate primarily (rows), then X-coordinate (columns)
            # This is critical for medical tables
            
            boxes = [line[0] for line in result[0]]
            txts = [line[1][0] for line in result[0]]
            scores = [line[1][1] for line in result[0]]
            
            # Simple line reconstruction
            # We add texts sequentially. For strictly table reconstruction, we need more logic.
            # For now, we rely on the line-by-line nature of OCR output.
            page_text = "\n".join(txts)
            
            text_content += page_text + "\n"
            
    except Exception as e:
        print(f"OCR Extraction Error: {e}")
        
    return text_content

def process_pdf(file_bytes):
    """
    Main entry point. Automatically routes to best extraction method.
    """
    pdf_type = detect_pdf_type(file_bytes)
    
    if pdf_type == "text_pdf":
        print("Using Layout Extraction (pdfplumber)...")
        return extract_text_with_layout(file_bytes)
    else:
        print("Using OCR Extraction (PaddleOCR)...")
        return extract_text_with_ocr(file_bytes)
