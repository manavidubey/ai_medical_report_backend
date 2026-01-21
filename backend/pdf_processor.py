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

from difflib import SequenceMatcher

class DocumentStitcher:
    """
    Intelligently stitches PDF pages by removing repeating headers/footers
    and maintaining document continuity.
    """
    def __init__(self, min_similarity=0.85):
        self.min_similarity = min_similarity

    def _get_clean_lines(self, text):
        return [line.strip() for line in text.split('\n') if len(line.strip()) > 5]

    def identify_repeating_mask(self, pages_text, num_lines_to_check=5, reverse=False):
        """
        Identifies lines that repeat across pages (conceptually).
        Returns a set of 'signature' strings that are headers/footers.
        """
        if len(pages_text) < 2: return set()
        
        signatures = set()
        
        # Compare Page i with Page i+1
        for i in range(len(pages_text) - 1):
            p1_lines = self._get_clean_lines(pages_text[i])
            p2_lines = self._get_clean_lines(pages_text[i+1])
            
            if reverse:
                p1_lines = p1_lines[::-1]
                p2_lines = p2_lines[::-1]
                
            check_len = min(len(p1_lines), len(p2_lines), num_lines_to_check)
            
            for k in range(check_len):
                line1 = p1_lines[k]
                line2 = p2_lines[k]
                
                ratio = SequenceMatcher(None, line1, line2).ratio()
                if ratio > self.min_similarity:
                    # Add one of them to signatures
                    signatures.add(line2) # Use the later one as stricter reference
                    
        return signatures

    def stitch(self, pages_text):
        if not pages_text: return ""
        if len(pages_text) == 1: return pages_text[0]
        
        # 1. Detect Headers (Top of pages)
        header_sigs = self.identify_repeating_mask(pages_text, num_lines_to_check=6, reverse=False)
        
        # 2. Detect Footers (Bottom of pages)
        footer_sigs = self.identify_repeating_mask(pages_text, num_lines_to_check=4, reverse=True)
        
        stitched_doc = []
        
        for i, page in enumerate(pages_text):
            lines = page.split('\n')
            cleaned_page_lines = []
            
            for line in lines:
                clean_line = line.strip()
                if len(clean_line) < 3:
                    cleaned_page_lines.append(line)
                    continue
                    
                is_header = any(SequenceMatcher(None, clean_line, h).ratio() > self.min_similarity for h in header_sigs)
                is_footer = any(SequenceMatcher(None, clean_line, f).ratio() > self.min_similarity for f in footer_sigs)
                
                # Rule: Keep Header on Page 1 (Context), Remove on others
                if is_header:
                    if i == 0:
                        cleaned_page_lines.append(line)
                    continue # Skip this line on subseq pages (or P1 if we want strictly no repeats, but P1 is usually reliable)
                    
                # Rule: Remove Footers everywhere (usually page nums etc)
                if is_footer:
                    continue
                    
                cleaned_page_lines.append(line)
                
            stitched_doc.append("\n".join(cleaned_page_lines))
            
        return "\n".join(stitched_doc)

def extract_text_with_layout(file_bytes):
    """
    Extracts text using pdfplumber's layout preservation.
    Best for digital medical reports with tables.
    """
    pages_text = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                # layout=True helps preserve column structure visually
                text = page.extract_text(layout=True)
                if text:
                    pages_text.append(text)
    except Exception as e:
        print(f"Layout Extraction Error: {e}")
        
    # Stitch pages
    stitcher = DocumentStitcher()
    return stitcher.stitch(pages_text)

def extract_text_with_ocr(file_bytes):
    """
    Converts PDF pages to images and runs PaddleOCR.
    Best for scanned reports or images wrapped in PDF.
    """
    pages_text = []
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
            
            # Run OCR
            if ocr:
                result = ocr.ocr(img_data, cls=True)
                if not result or result[0] is None:
                    pages_text.append("")
                    continue
                    
                # Sort boxes by Y-coordinate primarily (rows), then X-coordinate (columns)
                # This is critical for medical tables
                # PaddleOCR result: [ [ [[x1,y1]..], (text, conf) ], ... ]
                lines_data = sorted(result[0], key=lambda x: x[0][0][1]) # Check Y1 of box
                
                txts = [line[1][0] for line in lines_data]
                page_text = "\n".join(txts)
                pages_text.append(page_text)
            else:
                 pages_text.append("[OCR Failed - No Model]")
            
    except Exception as e:
        print(f"OCR Extraction Error: {e}")
    
    # Stitch pages
    stitcher = DocumentStitcher()
    return stitcher.stitch(pages_text)

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
