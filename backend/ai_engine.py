import re
import json
import os
import time
import io
import requests
import concurrent.futures
import statistics
from PIL import Image
import spacy
from transformers import pipeline
from deepdiff import DeepDiff
try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False
    print("DuckDuckGo Search not installed. Using mock data.")

try:
    import googlemaps
    HAS_GOOGLE_MAPS = True
except ImportError:
    HAS_GOOGLE_MAPS = False

# Global Model Variables (Lazy Loaded)
summarizer = None
explainer = None
nlp = None
llama_model = None
llama_processor = None
clip_classifier = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        print("Loading Summarization Model...")
        try:
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except Exception as e:
            print(f"Error loading summarizer: {e}")
            return None
    return summarizer

def get_explainer():
    global explainer
    if explainer is None:
        print("Loading Explainer Model...")
        try:
            explainer = pipeline("text2text-generation", model="google/flan-t5-small")
        except Exception as e:
            print(f"Warning: Could not load explainer model: {e}")
            explainer = None
    return explainer

def get_nlp():
    global nlp
    if nlp is None:
        print("Loading Scispacy Medical Model...")
        try:
            # Use the installed small scientific model
            nlp = spacy.load("en_core_sci_sm")
        except OSError:
            print("Warning: en_core_sci_sm not found, falling back to en_core_web_sm. Please install the model.")
            try:
                nlp = spacy.load("en_core_web_sm")
            except:
                try:
                    from spacy.cli import download
                    download("en_core_web_sm")
                    nlp = spacy.load("en_core_web_sm")
                except Exception as e:
                    print(f"Error loading SpaCy: {e}")
    return nlp

def get_vision_models():
    """
    Lazy loads vision models. returns (llama_model, llama_processor, clip_classifier)
    """
    global llama_model, llama_processor, clip_classifier
    
    if llama_model is None and clip_classifier is None:
        print("Loading Visual AI Models...")

        # A. Try Loading Llama 3.2 Vision (11B)
        # This requires HF_TOKEN and significant RAM (>20GB) or VRAM.
        # Disabled by default for Render Free Tier / Standard deployments to prevent timeouts
        load_llama = False 
        
        if load_llama:
            try:
                model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
                print(f"Attempting to load {model_id}...")
                
                # Use device_map="auto" to offload to standard RAM if GPU/MPS is full
                # torch_dtype=torch.bfloat16 is standard for Llama 3 on recent hardware
                llama_model = MllamaForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                llama_processor = AutoProcessor.from_pretrained(model_id)
                print("SUCCESS: Llama 3.2 Vision Loaded!")
                
            except Exception as e:
                print(f"Llama 3.2 Load Failed (Falling back to CLIP): {e}")
                # Common reasons: No HF_TOKEN, OOM, or model access not granted.

        # B. Load CLIP (Zero-Shot) - Always load as fallback or primary if Llama fails
        try:
            clip_classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
            if not llama_model:
                print("Using CLIP as Primary Vision Model.")
        except Exception as e:
            print(f"Warning: Could not load CLIP model: {e}")
            
    return llama_model, llama_processor, clip_classifier


def analyze_medical_image(image_bytes):
    """
    Analyzes an X-Ray/MRI image using Llama 3.2 (if avail) or CLIP.
    """
    # 1. Try Llama 3.2 Vision (Deep Analysis)
    llama_model, llama_processor, clip_classifier = get_vision_models()
    
    if llama_model and llama_processor:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Simple Medical Prompt
            prompt = "<|image|><|begin_of_text|>Analyze this medical image. Describe findings and impression."
            inputs = llama_processor(image, prompt, return_tensors="pt").to(llama_model.device)
            
            output = llama_model.generate(**inputs, max_new_tokens=150)
            description = llama_processor.decode(output[0])
            
            # Cleanup output (remove prompt parts)
            description = description.replace(prompt, "").replace("<|begin_of_text|>", "").strip()
            
            return {
                "modality": "Medical Imaging (Llama 3.2 Analysis)",
                "predictions": [{"label": "Detailed Analysis", "confidence": 100}],
                "finding": description,
                "note": "Generated by Llama 3.2 Vision (11B). Validate clinically."
            }
        except Exception as e:
            print(f"Llama Inference Failed: {e}. Falling back to CLIP.")
            # Fallthrough to CLIP

    # 2. Fallback: CLIP (Modality Detection)
    if clip_classifier:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            candidate_labels = [
                "chest x-ray", "brain mri", "abdominal ct scan", 
                "fetal ultrasound", "bone fracture", "medical document",
                "random noise", "random object"
            ]
            
            predictions = clip_classifier(image, candidate_labels=candidate_labels)
            
            formatted_preds = []
            is_medical_image = False
            top_label = predictions[0]['label']
            top_score = predictions[0]['score'] * 100
            
            for p in predictions[:3]:
                label = p['label']
                score = round(p['score'] * 100, 1)
                
                if any(x in label for x in ['x-ray', 'mri', 'ct scan', 'ultrasound', 'fracture', 'medical']):
                    if score > 20: is_medical_image = True
                    
                formatted_preds.append({"label": label.title(), "confidence": score})
                
            finding = f"Appearance consistent with {top_label.title()}"
            if top_label in ["random noise", "random object"] or top_score < 40:
                 finding = "Image classification uncertain. May not be a standard medical scan."
                 is_medical_image = False
                
            return {
                "modality": top_label.title() if is_medical_image else "Unknown",
                "predictions": formatted_preds,
                "finding": finding,
                "note": "AI modality detection (CLIP). For screening only."
            }
            
        except Exception as e:
            return {"error": f"CLIP Analysis Error: {str(e)}"}

    return {"error": "No Vision Models Available."}

REFERENCE_RANGES = {
    # 1. COMPLETE BLOOD COUNT (CBC + DIFFERENTIAL)
    "hemoglobin": {"min": 12.0, "max": 17.5, "unit": "g/dL"}, # Combined M/F
    "rbc count": {"min": 4.1, "max": 5.9, "unit": "million/uL"},
    "wbc count": {"min": 4000, "max": 11000, "unit": "/uL"},
    "platelets": {"min": 150000, "max": 450000, "unit": "/uL"},
    "hematocrit": {"min": 36, "max": 53, "unit": "%"},
    "mcv": {"min": 80, "max": 100, "unit": "fL"},
    "mch": {"min": 27, "max": 33, "unit": "pg"},
    "mchc": {"min": 32, "max": 36, "unit": "g/dL"},
    "rdw": {"min": 11.5, "max": 14.5, "unit": "%"},
    "neutrophils": {"min": 40, "max": 70, "unit": "%"},
    "lymphocytes": {"min": 20, "max": 40, "unit": "%"},
    "monocytes": {"min": 2, "max": 8, "unit": "%"},
    "eosinophils": {"min": 1, "max": 4, "unit": "%"},
    "basophils": {"min": 0, "max": 1, "unit": "%"},

    # 2. LIPID PROFILE (ADVANCED)
    "cholesterol": {"min": 0, "max": 200, "unit": "mg/dL"},
    "total cholesterol": {"min": 0, "max": 200, "unit": "mg/dL"},
    "ldl-c": {"min": 0, "max": 100, "unit": "mg/dL"},
    "ldl": {"min": 0, "max": 100, "unit": "mg/dL"},
    "hdl-c": {"min": 40, "max": 100, "unit": "mg/dL"}, # Using Male lower bound as generic min
    "hdl": {"min": 40, "max": 100, "unit": "mg/dL"},
    "triglycerides": {"min": 0, "max": 150, "unit": "mg/dL"},
    "vldl": {"min": 5, "max": 40, "unit": "mg/dL"},
    "non-hdl cholesterol": {"min": 0, "max": 130, "unit": "mg/dL"},
    "apob": {"min": 0, "max": 90, "unit": "mg/dL"},
    "lp(a)": {"min": 0, "max": 30, "unit": "mg/dL"},
    "cholesterol/hdl ratio": {"min": 0, "max": 4.5, "unit": "ratio"},

    # 3. LIVER FUNCTION TEST (LFT)
    "ast": {"min": 10, "max": 40, "unit": "U/L"},
    "sgot": {"min": 10, "max": 40, "unit": "U/L"},
    "alt": {"min": 7, "max": 56, "unit": "U/L"},
    "sgpt": {"min": 7, "max": 56, "unit": "U/L"},
    "alp": {"min": 44, "max": 147, "unit": "U/L"},
    "ggt": {"min": 9, "max": 48, "unit": "U/L"},
    "total bilirubin": {"min": 0.3, "max": 1.2, "unit": "mg/dL"},
    "direct bilirubin": {"min": 0.0, "max": 0.3, "unit": "mg/dL"},
    "indirect bilirubin": {"min": 0.2, "max": 0.9, "unit": "mg/dL"},
    "albumin": {"min": 3.5, "max": 5.0, "unit": "g/dL"},
    "globulin": {"min": 2.0, "max": 3.5, "unit": "g/dL"},
    "total protein": {"min": 6.0, "max": 8.3, "unit": "g/dL"},
    "a/g ratio": {"min": 1.1, "max": 2.5, "unit": "ratio"},

    # 4. KIDNEY FUNCTION TEST (KFT)
    "creatinine": {"min": 0.59, "max": 1.35, "unit": "mg/dL"}, # Combined range
    "urea": {"min": 15, "max": 40, "unit": "mg/dL"},
    "bun": {"min": 7, "max": 20, "unit": "mg/dL"},
    "uric acid": {"min": 2.4, "max": 7.0, "unit": "mg/dL"},
    "egfr": {"min": 90, "max": 200, "unit": "mL/min/1.73m²"}, # Min is critical

    # 5. DIABETES & METABOLIC
    "fasting glucose": {"min": 70, "max": 99, "unit": "mg/dL"},
    "glucose": {"min": 70, "max": 99, "unit": "mg/dL"}, # Default to fasting range standard
    "post-prandial glucose": {"min": 0, "max": 140, "unit": "mg/dL"},
    "random glucose": {"min": 0, "max": 200, "unit": "mg/dL"},
    "hba1c": {"min": 0, "max": 5.7, "unit": "%"},
    "insulin": {"min": 2, "max": 25, "unit": "uIU/mL"},
    "homa-ir": {"min": 0, "max": 2.0, "unit": "index"},

    # 6. THYROID PANEL (FULL)
    "tsh": {"min": 0.4, "max": 4.0, "unit": "mIU/L"},
    "total t3": {"min": 80, "max": 200, "unit": "ng/dL"},
    "total t4": {"min": 5.0, "max": 12.0, "unit": "ug/dL"},
    "free t3": {"min": 2.3, "max": 4.2, "unit": "pg/mL"},
    "free t4": {"min": 0.8, "max": 1.8, "unit": "ng/dL"},
    "anti-tpo": {"min": 0, "max": 35, "unit": "IU/mL"},
    "anti-tg": {"min": 0, "max": 20, "unit": "IU/mL"},

    # 7. ELECTROLYTES & MINERALS
    "sodium": {"min": 135, "max": 145, "unit": "mmol/L"},
    "potassium": {"min": 3.5, "max": 5.1, "unit": "mmol/L"},
    "chloride": {"min": 98, "max": 107, "unit": "mmol/L"},
    "calcium": {"min": 8.6, "max": 10.2, "unit": "mg/dL"},
    "ionized calcium": {"min": 1.12, "max": 1.32, "unit": "mmol/L"},
    "phosphorus": {"min": 2.5, "max": 4.5, "unit": "mg/dL"},
    "magnesium": {"min": 1.7, "max": 2.2, "unit": "mg/dL"},

    # 8. IRON STUDIES
    "serum iron": {"min": 60, "max": 170, "unit": "ug/dL"},
    "ferritin": {"min": 11, "max": 336, "unit": "ng/mL"},
    "tibc": {"min": 240, "max": 450, "unit": "ug/dL"},
    "transferrin saturation": {"min": 20, "max": 50, "unit": "%"},

    # 9. VITAMINS & NUTRITION
    "vitamin d": {"min": 30, "max": 100, "unit": "ng/mL"},
    "vitamin b12": {"min": 200, "max": 900, "unit": "pg/mL"},
    "folate": {"min": 2.7, "max": 17, "unit": "ng/mL"},
    "vitamin a": {"min": 20, "max": 60, "unit": "ug/dL"},
    "vitamin e": {"min": 5, "max": 20, "unit": "ug/mL"},
    "vitamin k": {"min": 0.2, "max": 3.2, "unit": "ng/mL"},

    # 10. INFLAMMATION & INFECTION
    "crp": {"min": 0, "max": 1.0, "unit": "mg/L"},
    "hs-crp": {"min": 0, "max": 3.0, "unit": "mg/L"},
    "esr": {"min": 0, "max": 20, "unit": "mm/hr"},
    "procalcitonin": {"min": 0, "max": 0.1, "unit": "ng/mL"},
    "il-6": {"min": 0, "max": 7, "unit": "pg/mL"},
    "d-dimer": {"min": 0, "max": 500, "unit": "ng/mL"},

    # 11. CARDIAC MARKERS
    "troponin i": {"min": 0, "max": 0.04, "unit": "ng/mL"},
    "troponin t": {"min": 0, "max": 0.01, "unit": "ng/mL"},
    "ck-mb": {"min": 0, "max": 5, "unit": "ng/mL"},
    "bnp": {"min": 0, "max": 100, "unit": "pg/mL"},
    "nt-probnp": {"min": 0, "max": 125, "unit": "pg/mL"},

    # 12. HORMONES
    "cortisol": {"min": 5, "max": 25, "unit": "ug/dL"},
    "acth": {"min": 10, "max": 60, "unit": "pg/mL"},
    "prolactin": {"min": 4, "max": 25, "unit": "ng/mL"},
    "lh": {"min": 2, "max": 15, "unit": "IU/L"},
    "fsh": {"min": 3, "max": 10, "unit": "IU/L"},
    "testosterone": {"min": 15, "max": 1000, "unit": "ng/dL"}, # Wide range M/F
    "estradiol": {"min": 30, "max": 400, "unit": "pg/mL"},

    # 13. AUTOIMMUNE
    "rf": {"min": 0, "max": 14, "unit": "IU/mL"},
    "anti-ccp": {"min": 0, "max": 20, "unit": "U/mL"},
    "dsdna": {"min": 0, "max": 30, "unit": "IU/mL"},
    "complement c3": {"min": 90, "max": 180, "unit": "mg/dL"},
    "complement c4": {"min": 10, "max": 40, "unit": "mg/dL"},

    # 14. TUMOR MARKERS
    "afp": {"min": 0, "max": 10, "unit": "ng/mL"},
    "cea": {"min": 0, "max": 3, "unit": "ng/mL"},
    "ca-125": {"min": 0, "max": 35, "unit": "U/mL"},
    "ca-19-9": {"min": 0, "max": 37, "unit": "U/mL"},
    "psa": {"min": 0, "max": 4.0, "unit": "ng/mL"},

    # 15. COAGULATION
    "pt": {"min": 11, "max": 13.5, "unit": "sec"},
    "inr": {"min": 0.8, "max": 1.2, "unit": "ratio"},
    "aptt": {"min": 25, "max": 35, "unit": "sec"},
    "fibrinogen": {"min": 200, "max": 400, "unit": "mg/dL"},

    # 16. URINE
    "ph": {"min": 4.5, "max": 8.0, "unit": "pH"},
    "specific gravity": {"min": 1.005, "max": 1.030, "unit": "sg"},

    # 17. PANCREATIC
    "amylase": {"min": 30, "max": 110, "unit": "U/L"},
    "lipase": {"min": 0, "max": 160, "unit": "U/L"},

    # 18. ABG
    "ph blood": {"min": 7.35, "max": 7.45, "unit": "pH"},
    "pao2": {"min": 75, "max": 100, "unit": "mmHg"},
    "paco2": {"min": 35, "max": 45, "unit": "mmHg"},
    "hco3": {"min": 22, "max": 26, "unit": "mEq/L"},
    "o2 saturation": {"min": 95, "max": 100, "unit": "%"},

    # 19. IRON STUDIES
    "iron": {"min": 60, "max": 170, "unit": "ug/dL"},
    "ferritin": {"min": 30, "max": 300, "unit": "ng/mL"},
    "tibc": {"min": 250, "max": 450, "unit": "ug/dL"},
}

# Simple Medical Dictionary for Patient Translation
MEDICAL_DICTIONARY = {
    "osteopenia": "lower than normal bone density (warning sign for osteoporosis)",
    "pneumonia": "infection inflaming air sacs in one or both lungs",
    "hyperlipidemia": "high levels of fat particles (lipids) in the blood",
    "hypertension": "high blood pressure",
    "tachycardia": "faster than normal heart rate",
    "anemia": "lack of enough healthy red blood cells",
    "erythema": "redness of the skin",
    "edema": "swelling caused by excess fluid",
    "lesion": "damage or abnormal change in tissue",
    "benign": "not harmful in effect (not cancer)",
    "malignant": "very virulent or infectious (cancerous)",
    "stenosis": "narrowing of a passage in the body",
    "acute": "sudden and severe",
    "chronic": "persisting for a long time",
    "fracture": "broken bone"
}

SYNONYMS = {
    "hb": "hemoglobin",
    "hgb": "hemoglobin",
    "total cholesterol": "cholesterol",
    "t. chol": "cholesterol",
    "t chol": "cholesterol",
    "fbs": "glucose",
    "blood sugar": "glucose",
    "f.b.s": "glucose",
    "glucose - fasting": "glucose",
    "cre": "creatinine",
    "creat": "creatinine",
    "s. creatinine": "creatinine",
    "s creatinine": "creatinine",
    "sgot": "ast",
    "asparts aminotransferase": "ast",
    "sgpt": "alt",
    "alanine aminotransferase": "alt",
    "trigs": "triglycerides",
    "pcv": "hematocrit",
    "hct": "hematocrit",
    "plt": "platelets",
    "platelet count": "platelets",
    "a1c": "hba1c",
    "hba1c": "hba1c",
    "glycated hemoglobin": "hba1c",
    "glycosylated hemoglobin": "hba1c",
    "tlc": "wbc count",
    "total leukocyte count": "wbc count",
    "total leucocyte count": "wbc count",
    "dlc": "differential", # Generic mapping
    "neutrophils": "neutrophils", # Ensure self-mapping exists
    "polymorphs": "neutrophils",
    "lymphocytes": "lymphocytes",
    "monocytes": "monocytes",
    "eosinophils": "eosinophils",
    "basophils": "basophils",
    "r.b.c": "rbc count",
    "rbc": "rbc count",
    "w.b.c": "wbc count",
    "wbc": "wbc count",
    "tec": "eosinophils",
    "absolute eosinophil count": "eosinophils", # Approximation
    "iron serum": "iron",
    "unsaturated iron binding capacity": "uibc",
    "total iron binding capacity": "tibc"
}

def normalize_name(name):
    clean = name.lower().strip().replace(":", "").replace(".", "")
    return SYNONYMS.get(clean, clean)

def extract_sections(text):
    """
    Extract high-level document sections (history/findings/impression/labs/advice).

    This is used by tests and downstream logic; it must never return None.
    """
    if not text:
        return {
            "history": "",
            "findings": "",
            "impression": "",
            "labs": "",
            "advice": "",
            "full_text": ""
        }

    try:
        sections = ClinicalParser().parse(text)
        if not isinstance(sections, dict):
            sections = {"full_text": text}
    except Exception:
        sections = {"full_text": text}

    # Ensure stable keys for callers/tests
    for k in ["history", "findings", "impression", "labs", "advice"]:
        sections.setdefault(k, "")
        if sections[k] is None:
            sections[k] = ""
    sections.setdefault("full_text", text)
    return sections


class ClinicalParser:
    """
    Robust regex-based parser to identify clinical sections.
    Preserves order and handles variations in headers.
    """
    def __init__(self):
        # Maps standard keys to list of potential regex variations
        self.section_patterns = {
            "labs": [r"laboratory data", r"investigation", r"haematology", r"biochemistry", r"lab results", r"test name", r"doctor summary"],
            "history": [r"clinical history", r"history", r"clinical indication", r"reason for exam"],
            "findings": [r"findings", r"technique", r"examination", r"procedure", r"comments", r"report"],
            "impression": [r"impression", r"conclusion", r"diagnosis", r"summary", r"opinion"],
            "advice": [r"advice", r"suggested correlation", r"recommendation"]
        }

    def parse(self, text):
        found_sections = {}
        # We search for all headers and their positions
        headers_found = []
        
        text_lower = text.lower()
        
        for key, patterns in self.section_patterns.items():
            for pat in patterns:
                # Look for Header followed by colon or newline
                # Use word boundary to avoid partial matches
                matches = list(re.finditer(rf"\b{pat}\b[:\s]", text_lower))
                for m in matches:
                    headers_found.append({
                        "key": key,
                        "start": m.start(),
                        "name": m.group().strip().strip(':'), # Capture actual text used
                        "raw_start": m.start()
                    })
        
        # Sort by position in document
        headers_found.sort(key=lambda x: x['start'])
        
        # Filter overlapping/duplicates (keep first occurrence of a type if close?)
        # For simple robustness, we just take distinct distinct sorted regions
        
        if not headers_found:
             return {"full_text": text} # No structure detected, treat as one block
            
        for i, header in enumerate(headers_found):
            start = header['start'] + len(header['name']) + 1 # +1 for colon assumption/space
            
            end = len(text)
            if i + 1 < len(headers_found):
                end = headers_found[i+1]['start']
            
            content = text[start:end].strip()
            # If multiple headers of same key (e.g. History... Findings... History...), append or overwrite?
            # Append is safer for merged docs.
            if header['key'] in found_sections:
                found_sections[header['key']] += "\n" + content
            else:
                found_sections[header['key']] = content
                
        # Debug Log
        print(f"Detected {len(found_sections)} sections: {list(found_sections.keys())}")
        return found_sections

def extract_demographics(text):
    """
    Extracts Patient Age, Gender, and potentially Name.
    """
    demographics = {"age": None, "gender": "Unknown", "name": "Unknown"}
    
    # 1. AGE
    age_match = re.search(r"(?:Age|yg|y\.o\.|years? old)[:\s]*(\d{1,3})", text, re.IGNORECASE)
    if age_match:
        demographics["age"] = int(age_match.group(1))
        
    # 2. GENDER
    if re.search(r"(?:Sex|Gender)[:\s]*(?:Male|M\b)", text, re.IGNORECASE):
        demographics["gender"] = "Male"
    elif re.search(r"(?:Sex|Gender)[:\s]*(?:Female|F\b)", text, re.IGNORECASE):
        demographics["gender"] = "Female"
    elif re.search(r"\bMale\b", text, re.IGNORECASE) and not re.search(r"\bFemale\b", text, re.IGNORECASE):
        demographics["gender"] = "Male"
    elif re.search(r"\bFemale\b", text, re.IGNORECASE):
        demographics["gender"] = "Female"
        
    # 3. NAME
    # Patterns: "Patient Name: John Doe", "Name: John Doe", "Patient: John Doe"
    name_match = re.search(r"(?:Patient Name|Name|Patient)[:\s]+([A-Za-z\s\.]+)(?:\n|$|\s{2,})", text, re.IGNORECASE)
    if name_match:
        raw_name = name_match.group(1).strip()
        # Filter out if it captured too much (e.g. "Patient: 45 Male")
        if not any(char.isdigit() for char in raw_name) and len(raw_name) > 2 and len(raw_name) < 40:
            demographics["name"] = raw_name.title()

    # 4. DATE
    date_match = re.search(r"(?:Date|Report Date|Collected)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text, re.IGNORECASE)
    if date_match:
        demographics["date"] = date_match.group(1)

    # 5. LOCATION (Area/City)
    loc_match = re.search(r"(?:Location|City|Area|Address)[:\s]+([A-Za-z\s]{2,30})(?:\n|$|,)", text, re.IGNORECASE)
    if loc_match:
        demographics["location"] = loc_match.group(1).strip()
            
    return demographics

def extract_vitals(text):
    """
    Extracts BP, Heart Rate, BMI, Temperature.
    """
    vitals = []
    
    # 1. Blood Pressure (e.g. 120/80 mmHg)
    bp_match = re.search(r"\b(\d{2,3})[/\\](\d{2,3})\b(?:mmHg)?", text, re.IGNORECASE)
    if bp_match:
        systolic = int(bp_match.group(1))
        diastolic = int(bp_match.group(2))
        # Sanity check
        if 50 < systolic < 250 and 30 < diastolic < 150:
            status = "Normal"
            if systolic > 130 or diastolic > 80: status = "Elevated"
            if systolic > 140 or diastolic > 90: status = "High (Hypertension)"
            vitals.append({"name": "Blood Pressure", "value": f"{systolic}/{diastolic}", "unit": "mmHg", "status": status})

    # 2. Heart Rate / Pulse
    hr_match = re.search(r"(?:Heart Rate|Pulse|HR)[:\s]+(\d{2,3})", text, re.IGNORECASE)
    if hr_match:
        hr = int(hr_match.group(1))
        status = "Normal"
        if hr > 100: status = "High (Tachycardia)"
        elif hr < 60: status = "Low (Bradycardia)"
        vitals.append({"name": "Heart Rate", "value": hr, "unit": "bpm", "status": status})

    # 3. BMI
    bmi_match = re.search(r"(?:BMI|Body Mass Index)[:\s]+(\d{1,2}(?:\.\d)?)", text, re.IGNORECASE)
    if bmi_match:
        bmi = float(bmi_match.group(1))
        status = "Normal"
        if bmi >= 25: status = "Overweight"
        if bmi >= 30: status = "Obese"
        vitals.append({"name": "BMI", "value": bmi, "unit": "kg/m²", "status": status})

    return vitals

def extract_medications(text):
    """
    Extracts list of medications using heuristic keywords and list format detection.
    """
    meds = []
    
    # Locate section
    # Search for "Current Medications", "Rx", "Treatment" followed by list items
    
    # Simple strategy: Find "Medications:" header and grab lines until next header
    start_match = re.search(r"(?:Current Medications|Medications|Tx|Rx)[:\n]", text, re.IGNORECASE)
    if start_match:
        start_idx = start_match.end()
        # Look for next double newline or Header-like pattern
        sub_text = text[start_idx:]
        lines = sub_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Stop if we hit a likely new header (All caps, ending in colon?)
            if re.match(r"^[A-Z\s]+:$", line) or len(meds) > 15: # Safety break
                break
                
            # If line is short and looks like a list item
            # Remove bullets
            clean_line = re.sub(r"^[\d\-\.\*\)]+\s*", "", line)
            
            # Heuristic: Valid med line is usually < 50 chars, has letters
            if 3 < len(clean_line) < 60 and re.search(r"[a-zA-Z]", clean_line):
                # Check for common dosage keywords to confirm it's a med
                if re.search(r"(?:mg|mcg|ml|tablet|capsule|daily|qn|bd|od|tds)", clean_line, re.IGNORECASE):
                    meds.append(clean_line)
                # Or if implies a list
                elif line.startswith("-") or line[0].isdigit():
                     meds.append(clean_line)
                     
    return meds

def process_clinical_document(text, previous_text=None, user_location=None):
    """
    Orchestrator: Differentiated Processing Strategy with optional Comparison.
    """
    print("\n--- Processing Clinical Document ---")
    parser = ClinicalParser()
    sections = parser.parse(text)
    
    # Data Containers
    processed_data = {
        "report_type": detect_report_type(text),
        "raw_sections": sections, 
        "structured_labs": [],
        "lab_abnormalities": [],
        "section_summaries": {},
        "risks": [], # Mapped from clinical_risks
        "recommendations": [],
        "demographics": {},
        "vitals": [],
        "medications": [],
        "specialists": [],
        "summary": "",
        "patient_view": {"findings": "", "impression": ""}
    }
    
    # 0. Advanced IQ: Demographics & Vitals
    processed_data["demographics"] = extract_demographics(text[:5000]) 
    processed_data["vitals"] = extract_vitals(text) 
    processed_data["medications"] = extract_medications(text)
    
    # 1. Process Lab Data
    all_labs = extract_labs_regex(text) 
    processed_data["structured_labs"] = all_labs
    processed_data["lab_abnormalities"] = [l for l in all_labs if l['status'] != "Normal"]
    
    # Risky Business
    risks = calculate_risk(all_labs)
    processed_data["risks"] = risks # Matches App.jsx
    processed_data["recommendations"] = generate_recommendations(all_labs, risks)
    print("DEBUG: Getting Specialists...")
    # Use user provided location first, else falling back to extracted info
    final_location = user_location or processed_data["demographics"].get("location")
    processed_data["specialists"] = get_recommended_specialists(risks, "Lab", final_location)
    print(f"DEBUG: Specialists Done for {final_location}.")
    
    # Enrich Abnormal Labs with "Deep Insight"
    processed_data["structured_labs"] = enrich_abnormalities(all_labs)
    processed_data["lab_abnormalities"] = [l for l in processed_data["structured_labs"] if l['status'] != "Normal"]

    # 2. Process Narrative Sections
    for sec_key in ["history", "findings", "impression"]:
        content = sections.get(sec_key, "")
        if not content: continue
        
        word_count = len(content.split())
        print(f"Processing Section '{sec_key}' ({word_count} words)...")
        
        if word_count < 50:
            processed_data["section_summaries"][sec_key] = content
        elif word_count > 800:
            processed_data["section_summaries"][sec_key] = summarize_text(content)
        else:
             processed_data["section_summaries"][sec_key] = content # Keep original if medium size

    # 3. Final Synthesis & Patient View
    processed_data["summary"] = synthesize_report(processed_data)
    
    # --- ENHANCED PATIENT VIEW ---
    patient_narrative = synthesize_patient_view(processed_data)
    processed_data["patient_view"] = {
        "findings": patient_narrative["findings"],
        "impression": patient_narrative["impression"],
        "explanation": "Simplified medical overview for patients."
    }
    
    # 4. Comparison (Optional)
    if previous_text:
        prev_labs = extract_labs_regex(previous_text)
        # Simple diff logic can be added here
        print("Comparing with previous report...")
        # Access trend analysis function if available
        processed_data["structured_labs"] = compare_labs(processed_data["structured_labs"], prev_labs)
        # processed_data["comparison"] = comparison

    return processed_data 

def clean_ads_and_noise(text):
    """
    Best-effort removal of common footer/boilerplate noise from reports.
    """
    clean = text or ""
    noise_patterns = [
        r"This test has been performed at.*",
        r"TATA 1MG.*",
        r"Powered by.*",
        r"Page \d+ of \d+",
        r"Sample collected at.*",
        r"Order Medicines.*",
        r"EXPLORE NOW.*",
        r"Terms and Conditions.*",
        r"support@.*",
        r"care@.*",
        r"www\..*",
        r"Reference\s*:\s*American Diabetes Association.*",
        r"\d{2,}\s*or\s*(?:above|below).*Normal.*",
        r"\d{2,}\s*to\s*\d{2,}.*Pre-Diabetes.*",
        r"\d{2,}\s*or\s*above.*Diabetes.*",
        r"It is advised to read this in conjunction.*",
        r"Clinical Impression of clinically significant parameters.*"
    ]
    for pat in noise_patterns:
        clean = re.sub(pat, "", clean, flags=re.IGNORECASE)
    return clean.strip()


def synthesize_report(processed_data):
    """
    Enhanced Professional Clinical Summary.
    Structure: Overview -> Impression -> Lab Abnormalities -> Imaging -> Plan
    """
    processed_data = processed_data or {}
    report_lines = []
    
    # 0. Overview
    demo = processed_data.get("demographics", {})
    name = demo.get("name", "Unknown Patient")
    date = demo.get("date", "Unknown Date")
    report_lines.append(f"REPORT OVERVIEW")
    report_lines.append(f"Patient: {name} | Date: {date}")
    report_lines.append("---")

    # 1. Clinical Impression
    summaries = processed_data.get("section_summaries", {})
    impression = summaries.get("impression") or summaries.get("findings")
    if impression:
        clean_imp = clean_ads_and_noise(impression)
        if clean_imp:
            report_lines.append("CLINICAL IMPRESSION")
            report_lines.append(clean_imp[:600])
            report_lines.append("---")

    # 2. Vital Indicators (Abnormal Labs)
    abnormal_labs = processed_data.get("lab_abnormalities", [])
    if abnormal_labs:
        report_lines.append("VITAL INDICATORS (ABNORMAL)")
        for lab in abnormal_labs:
            name = lab.get("name", "Test")
            val = lab.get("value", "")
            unit = lab.get("unit", "")
            status = lab.get("status", "")
            why = lab.get("insight_why", "Clinically significant outlier.")
            
            # Cleaner bullet: [Test] - [Value] [Unit] ([Status])
            report_lines.append(f"• {name}: {val} {unit} ({status}) — {why}")
        report_lines.append("---")

    # 3. Imaging & Measurements
    if "findings" in summaries or "impression" in summaries:
        text_block = (summaries.get("findings","") + " " + summaries.get("impression","")).lower()
        sizes = re.findall(r"(\d+(?:\.\d+)?\s*[xX]\s*\d+(?:\.\d+)?(?:\s*[xX]\s*\d+(?:\.\d+)?)?\s*(?:cm|mm))", text_block)
        if sizes:
             report_lines.append("IMAGING / MEASUREMENTS")
             for s in list(set(sizes)):
                 report_lines.append(f"• Physical Finding: {s} — Clinical correlation advised.")
             report_lines.append("---")

    # 4. Next Steps & Recommendations
    recs = processed_data.get("recommendations", [])
    if recs:
        report_lines.append("NEXT STEPS & RECOMMENDATIONS")
        for i, rec in enumerate(recs, 1):
            advice = rec.get("advice", "") if isinstance(rec, dict) else rec
            report_lines.append(f"{i}. {advice}")

    final_text = "\n".join(report_lines).strip()
    # Strip markdown bold per user preference
    final_text = final_text.replace("**", "")
    return final_text if final_text else "No clinical findings detected for summary."

def analyze_full_report(current_text, previous_text=None):
    # 1. Run New Orchestrator
    processed_data = process_clinical_document(current_text)
    
    # 2. Detect Type (Legacy Logic kept for Specialist routing)
    report_type = detect_report_type(current_text)
    
    # 3. Generate Synthesis (Replaces blind summary)
    full_narrative_report = synthesize_report(processed_data)
    
    # 4. Entities (Common)
    nlp_model = get_nlp()
    if nlp_model:
        doc = nlp_model(current_text)
        entities = [{"text": e.text, "label": "ENTITY"} for e in doc.ents]
    else:
        entities = []
    entities = [dict(t) for t in {tuple(d.items()) for d in entities}]
    
    # 5. Construct Result (Frontend Compatible)
    labs = processed_data["structured_labs"]
    risks = processed_data["clinical_risks"]
    recommendations = processed_data["recommendations"]
    follow_up_tests = get_follow_up_tests(labs, risks)
    raw_sections = processed_data["raw_sections"]
    section_abstractions = processed_data["section_summaries"] 
    # Frontend likely expects 'sections' key to be raw text for display tabs
    
    result = {
        "report_type": report_type,
        "summary": full_narrative_report,
        "sections": raw_sections, 
        "entities": entities
    }

    if report_type == "Radiology":
        rad_analysis = analyze_radiology(current_text, raw_sections) # Pass raw sections
        
        # Patient Explanation for Radiology
        patient_impression = explain_to_patient(raw_sections.get('impression', ''))
        
        # Specialists
        specialists = ["Primary Care Physician"]
        for ab in rad_analysis.get('abnormalities', []):
            if "fracture" in ab.lower() or "bone" in ab.lower():
                 specialists.append("Orthopedist")
            if "tumor" in ab.lower() or "mass" in ab.lower():
                 specialists.append("Oncologist")
        specialists = list(set(specialists))

        result.update({
            "radiology": rad_analysis,
            "patient_view": {
                "findings": "See imaging details below.",
                "impression": patient_impression,
                "explanation": "Imaging reports describe visual findings. We've highlighted key abnormalities."
            },
            "risks": [], 
            "labs": [], 
            "recommendations": [],
            "follow_up_tests": [],
            "specialists": specialists 
        })
        
        if rad_analysis['abnormalities']:
            result['risks'].append({"type": "Imaging Finding", "status": "Observation", "detail": "Abnormalities detected in scan."})
            
    else:
        # Lab / General
        specialists = get_recommended_specialists(risks, "Lab")
        
        # Patient View (driven by raw text or summaries? Summaries are safer/simpler)
        # But 'explain_to_patient' expects text. Let's use raw sections for now to be safe.
        patient_findings = explain_to_patient(raw_sections.get('findings', ''))
        patient_impression = explain_to_patient(raw_sections.get('impression', ''))
        
        severity = {
            "findings": detect_severity(raw_sections.get('findings', '')),
            "impression": detect_severity(raw_sections.get('impression', ''))
        }
        
        comparison = {}
        if previous_text:
            prev_labs = extract_labs_regex(previous_text)
            labs = compare_labs(labs, prev_labs)
            comparison = {
                "previous_labs": prev_labs,
                "diff_summary": f"Compared against previous report." 
            }
            
        result.update({
            "patient_view": {
                "findings": patient_findings,
                "impression": patient_impression,
                "explanation": "Simplified medical terms are shown in parentheses."
            },
            "severity": severity,
            "labs": labs,
            "risks": risks,
            "recommendations": recommendations,
            "follow_up_tests": follow_up_tests,
            "comparison": comparison,
            "specialists": specialists
        })

    return result

def simplify_text(text):
    """Fallback manual simplification if LLM is unavailable."""
    if not text: return ""
    replacements = {
        "hematology": "Blood Test",
        "cardiology": "Heart Health",
        "pathology": "Lab Results",
        "radiology": "Imaging/Scans",
        "acute": "Recent/Sudden",
        "chronic": "Long-term",
        "hypolipidemic": "Cholesterol-lowering",
        "nephropathy": "Kidney issues"
    }
    for k, v in replacements.items():
        text = text.replace(k, v).replace(k.title(), v)
    return text

def synthesize_patient_view(processed_data):
    """
    Constructs a clear, layman's summary and health status.
    """
    all_labs = processed_data.get("structured_labs", [])
    # Only count actual medical alerts
    abnormal_labs = [l for l in all_labs if l['status'] != "Normal"]
    risks = processed_data.get("risks", [])
    
    # 1. Findings
    if not abnormal_labs:
        findings = "Your laboratory results are within the normal reference range. All tested indicators appear stable."
    else:
        top_labs = [f"**{l['name']}** ({l['status']})" for l in abnormal_labs[:3]]
        findings = f"Your report identified {len(abnormal_labs)} indicators that are currently outside the standard range, including {', '.join(top_labs)}."

    # 2. Health Status & Timeline
    urgent_flags = [r for r in risks if r.get('level') == 'Severe']
    
    if len(abnormal_labs) == 0 and not urgent_flags:
        impression = "Overall Health Status: **Excellent / Healthy**."
        next_test = "Next routine health screening is recommended in **12 months**."
    elif len(abnormal_labs) <= 2 and not urgent_flags:
        impression = "Overall Health Status: **Good (with minor variations)**."
        next_test = "Follow-up testing is advised in **6 months** to monitor these values."
    else:
        impression = "Overall Health Status: **Needs Clinical Correlation**."
        next_test = "Please schedule a follow-up with your physician within **1-3 months** for a full evaluation."

    full_impression = f"{impression}\n\nRecommendation: {next_test}"
    
    # Strip markdown bold per user preference
    findings = findings.replace("**", "")
    full_impression = full_impression.replace("**", "")
    
    return {
        "findings": findings,
        "impression": full_impression
    }

def explain_to_patient(text):
    """
    Uses LLM to generate a patient-friendly explanation.
    """
    if not text or not text.strip():
        return "No significant findings to explain."
        
    model = get_explainer()
    if not model:
        return simplify_text(text)
        
    try:
        # Prompt Engineering for Flan-T5
        prompt = f"Explain this medical finding to a patient in simple words: {text}"
        output = model(prompt, max_length=128, do_sample=False)
        return output[0]['generated_text']
    except Exception as e:
        print(f"Explainer Error: {e}")
        return simplify_text(text)

def detect_severity(text):
    """
    Detects severity from adjectives in the text.
    """
    text_lower = text.lower()
    if any(x in text_lower for x in ["critical", "severe", "emergency", "acute distress"]):
        return {"level": "Severe", "color": "red"}
    elif any(x in text_lower for x in ["moderate", "elevated", "abnormal"]):
        return {"level": "Moderate", "color": "orange"}
    elif any(x in text_lower for x in ["mild", "trace", "minor", "borderline"]):
        return {"level": "Mild", "color": "yellow"}
    
    return {"level": "Normal", "color": "green"}


def parse_complex_value(val_str):
    """
    Converts strings like '220 x 10^3', '1.2E+03', '4.8 million', or '11.0 k/uL' into absolute floats.
    """
    if not val_str:
        return 0.0
    
    # 0. Handle "Trends" or interleaved data
    # If string contains multiple numbers separated by spaces, take the first one
    # This prevents catching last year's results on the same line.
    first_block = re.split(r"\s{2,}", val_str.strip())[0].strip()
    
    # 0.1 Handle numeric values preceded by '-' (common in change columns)
    # We want to skip if it's JUST a change column, but usually the result is first.
    
    clean_val = first_block.replace(" ", "").lower()
    
    try:
        # Multipliers (million, k, mili, billion)
        multiplier = 1.0
        if 'million' in clean_val or 'mili' in clean_val: multiplier = 1000000.0
        elif 'billion' in clean_val: multiplier = 1000000000.0
        elif 'k' in clean_val or '^3' in clean_val or ('x' in clean_val and '10^' not in clean_val):
             multiplier = 1000.0

        # 0.2 Handle standalone ^ multiplier (e.g. "6.261 ^3")
        if '^' in clean_val:
            caret_match = re.search(r"\^(\d+)", clean_val)
            if caret_match:
                exp = int(caret_match.group(1))
                if '10^' not in clean_val:
                    # Treat ^3 as 10^3 if no base is provided
                    multiplier = 10 ** exp

        # 1. Handle "220x10^3" or "5.5*10^3"
        sci_match = re.search(r"(\d+(?:\.\d+)?)\s*[x\*]\s*10\^(\d+)", first_block, re.IGNORECASE)
        if sci_match:
            base = float(sci_match.group(1))
            exp = int(sci_match.group(2))
            return base * (10 ** exp) * multiplier
            
        # 2. Handle "1.2e+03"
        if 'e' in clean_val and ('+' in clean_val or '-' in clean_val):
            e_match = re.search(r"(\d+(?:\.\d+)?e[\+\-]?\d+)", clean_val)
            if e_match:
                return float(e_match.group(1)) * multiplier
            
        # 3. Standard extraction
        # Allow < or > but strip for math
        numeric_match = re.search(r"(\d+(?:\.\d+)?)", clean_val)
        if numeric_match:
            return float(numeric_match.group(1)) * multiplier
            
        return 0.0
    except (ValueError, TypeError):
        return 0.0

def is_significant_test(name, val_str=None):
    """Filters out noise that looks like a lab test but isn't."""
    if not name: return False
    name_low = name.lower().strip()
    
    # Comprehensive Noise List for Institutional Reports
    noise_words = [
        "date", "collected", "reported", "page", "result", "reference", "ref", 
        "range", "units", "status", "flag", "method", "specimen", "investigation",
        "order", "visit", "patient", "customer", "referred", "barcode",
        "sample type", "report status", "desirable", "borderline", "optimal", 
        "technique", "prepared for", "basic info", "summary", "electronically",
        "click here", "verified", "interpreted", "disclaimer", "pregnant",
        "age", "gender", "sex", "name", "doctor", "history", "findings", "impression",
        "high risk", "low risk", "desirable", "normal", "abnormal"
    ]
    
    # 1. Exact matches or starters
    if any(name_low == n or name_low.startswith(f"{n} ") or name_low.startswith(f"{n}:") for n in noise_words):
        return False
        
    # 2. Pattern based (e.g. MGP791564 or 15338405)
    if re.match(r"[a-zA-Z]{2,}\d{4,}", name_low) or re.match(r"^\d{5,}", name_low): # Mixed or long numeric ID
        return False
        
    # 3. Validation
    if len(name_low) < 2: return False
    if not name_low[0].isalnum(): return False
    
    return True

def extract_labs_regex(text):
    """
    Improved regex to catch labs, units, and ranges.
    Prioritizes Lab Report Range > System Default.
    Supports Unit Inference if unit is missing.
    """
    labs = []
    
    # Pattern: Name: Value Unit? [Range?]
    # Groups: 1=Name, 2=Value, 3=Unit(Optional), 4=Range(Optional)
    # Improved value part to capture keywords like million/k
    # Improved value part to capture keywords like million/k and scaling signs (^, x)
    V_VAL = r"(?:[<>≈]?\s*\d+(?:\.\d+)?(?:\s*[x\*]\s*10\^\d+|[eE][\+\-]?\d+|\s*[\^x\*]\s*\d+)?(?:\s*million|billion|k|mili)?)"
    # Improved Range regex to capture units and multipliers (million, k/uL, etc.)
    V_UNIT = r"(?:[<>≈]?\s*\d+(?:\.\d+)?(?:\s*[x\*]\s*10\^\d+|[eE][\+\-]?\d+)?(?:\s*[a-zA-Z%\^/0-9\*]+)?)"
    RANGE_REGEX = r"(?:[\(\[]?(" + V_UNIT + r"\s*[-–]\s*" + V_UNIT + r"|[<>≈]\s*" + V_UNIT + r"|" + V_UNIT + r")[\)\]]?)"
    
    patterns = [
        # 1. Standard: Name: Value Unit Range? (Requires Colon)
        re.compile(r"([A-Za-z0-9 \t\(\)\-\.]{2,60}?):\s*(" + V_VAL + r")\s*([a-zA-Z%\^/0-9\*]*)\s*(?:" + RANGE_REGEX + ")?"),
        
        # 2. Tabular: Name (2+ spaces) Value (Flexible separator for institutional results)
        re.compile(r"(?m)^\s*([A-Za-z][A-Za-z0-9 \(\)\-\.]{2,40}?)(?:\t|\s{2,})([<>≈]?\s*\d+(?:\.\d+)?(?:\s*million|billion|k|mili)?[\s\d\.\-–\%\^x\*]*)\s*([a-zA-Z%\^/0-9\*]*)\s*(?:" + RANGE_REGEX + ")?"),
        
        # 3. Anchor Hunt: Look for known tests even with single space (Institutional Fallback)
        # Matches: "Hemoglobin 14.5 g/dL" or "RBC 4.8 million"
        # We only do this for the top 50 most common labs to avoid noise.
    ]
    
    # 4. Supplemental "Anchor Hunt" for critical labs (if they were missed by regex)
    # We'll run this manually after the regex loop below.
    
    found_names = set()

    for i, pattern in enumerate(patterns):
        # Process in chunks to avoid regex hangs on huge files
        chunk_size = 50000
        for start_idx in range(0, len(text), chunk_size):
            chunk = text[start_idx : start_idx + chunk_size + 500]
            
            for match in pattern.finditer(chunk):
                raw_name = ""
                value = 0.0
                unit = ""
                raw_range = None
                
                # Check for 4 groups (Name, Val, Unit, Range)
                if len(match.groups()) == 4:
                    raw_name = match.group(1).strip()
                    val_str = match.group(2).strip()
                    value = parse_complex_value(val_str)
                    unit = match.group(3).strip() if match.group(3) else ""
                    raw_range = match.group(4)
                else:
                    # Legacy or 3-group match
                    raw_name = match.group(1).strip()
                    val_str = match.group(2).strip()
                    value = parse_complex_value(val_str)
                    unit = match.group(3).strip() if match.group(3) else ""
                    raw_range = None

                # Defensive: never accept multi-line names (usually a header + analyte merged)
                if "\n" in raw_name or "\r" in raw_name:
                    continue

                if not is_significant_test(raw_name):
                    continue

                # Ad / Noise Filter
                raw_lower = match.group(0).lower()
                if any(x in raw_lower for x in ["discount", "package", "offer", "visit us", "www.", "iso", "nabl", "cap accredited", "technology", "powered by"]):
                    continue

                normalized = normalize_name(raw_name)
                system_ref = REFERENCE_RANGES.get(normalized)

                # Unit Inference Logic
                if not unit and system_ref:
                    # If unit missing but we know the test, infer it
                    unit = system_ref.get('unit', '')

                # Validation: If unit still invalid/missing and not in KB, skip
                known_units = ["mg/dl", "g/dl", "u/l", "mmol/l", "%", "x10^3/ul", "fl", "pg", "iu/l", "ng/ml", "ug/dl", "ratio", "/ul", "cells/ul"]
                if not system_ref and unit.lower() not in known_units and not (unit == "" and value > 0):
                    # Keep it looser now: if it looks like a test (Name + Value), keep it unless obviously junk.
                    # Only skip if no unit AND no range AND name is suspicious?
                    if len(raw_name.split()) > 6:
                        continue  # Name too long, probably text

                if normalized in found_names:
                    continue

                status = "Normal"
                ref_source = "Unknown"
                ref_str = "N/A"

                # 1. Try Lab Provided Range (Primary)
                lab_min = None
                lab_max = None

                if raw_range:
                    try:
                        # Handle "10 - 20"
                        if "-" in raw_range or "–" in raw_range:
                            parts = re.split(r"[-–]", raw_range)
                            if len(parts) == 2:
                                # Multiplier check for range (e.g. "4.1 - 5.9 million")
                                # We apply the multiplier if found in either part or the whole string
                                multiplier = 1.0
                                if 'million' in raw_range.lower(): multiplier = 1000000.0
                                elif 'billion' in raw_range.lower(): multiplier = 1000000000.0
                                elif 'k' in raw_range.lower(): multiplier = 1000.0

                                # parse_complex_value handles internal multipliers too.
                                # But we handle the "range-wide" multiplier here.
                                v_min = parse_complex_value(parts[0].strip())
                                v_max = parse_complex_value(parts[1].strip())
                                
                                # Optimization: only apply multiplier if parse didn't already
                                lab_min = v_min * (multiplier if v_min < 1000 and multiplier > 1 else 1.0)
                                lab_max = v_max * (multiplier if v_max < 1000 and multiplier > 1 else 1.0)
                                
                                ref_str = raw_range
                                ref_source = "Lab Report"
                        # Handle "< 100" or "> 50"
                        elif "<" in raw_range:
                            val = parse_complex_value(raw_range.replace("<", "").strip())
                            lab_max = val
                            lab_min = 0 
                            ref_str = raw_range
                            ref_source = "Lab Report"
                        elif ">" in raw_range:
                            val = parse_complex_value(raw_range.replace(">", "").strip())
                            lab_min = val
                            lab_max = float('inf')
                            ref_str = raw_range
                            ref_source = "Lab Report"
                    except Exception:
                        pass  # parsing failed, fall back

                # 2. Fallback to System Default (Secondary)
                if ref_source == "Unknown" and system_ref:
                    # --- SCALING FIX ---
                    # If the system unit implies a multiplier but the value might be absolute, 
                    # we must scale the reference values.
                    sys_unit = system_ref.get('unit', '').lower()
                    sys_mult = 1.0
                    if 'million' in sys_unit: sys_mult = 1000000.0
                    elif 'billion' in sys_unit: sys_mult = 1000000000.0
                    elif sys_unit.startswith('k/'): sys_mult = 1000.0
                    
                    lab_min = system_ref['min'] * sys_mult
                    lab_max = system_ref['max'] * sys_mult
                    ref_str = f"{system_ref['min']} - {system_ref['max']} {system_ref['unit']}"
                    ref_source = "System Default"

                # 3. Calculate Status
                if lab_min is not None and lab_max is not None:
                    if value < lab_min:
                        status = "Low"
                    elif value > lab_max:
                        status = "High"

                labs.append({
                    "name": raw_name,
                    "normalized_name": normalized,
                    "value": value,
                    "unit": unit,
                    "status": status,
                    "reference": ref_str,
                    "source": ref_source
                })
                found_names.add(normalized)
                
    # --- 4. ANCHOR HUNT (Institutional Fallback) ---
    # Many reports (1mg/LabCorp) use single spaces in tabular data.
    # We hunt for critical markers from our knowledge base.
    crit_markers = list(REFERENCE_RANGES.keys()) + list(SYNONYMS.keys())
    # Strip very short common words to avoid noise
    crit_markers = [m for m in crit_markers if len(m) > 3]

    for marker in crit_markers:
        norm_marker = normalize_name(marker)
        if norm_marker in found_names: continue
        
        # Match: Marker + optional junk + [Value] + [Unit?] + [Range?]
        # The range capture uses our V_UNIT logic
        m_pat = rf"\b{re.escape(marker)}\b\s*([<>≈]?\s*\d+(?:\.\d+)?(?:\s*million|mili|k)?)\s*([a-zA-Z%\^/0-9\*]*)[\s\t]*({RANGE_REGEX})?"
        match = re.search(m_pat, text, re.IGNORECASE)
        
        if match:
            raw_name = marker
            val_str = match.group(1).strip()
            unit = match.group(2).strip()
            raw_range = match.group(3)
            
            value = parse_complex_value(val_str)
            if value == 0 and not val_str: continue # skip junk
            
            normalized = normalize_name(raw_name)
            
            status = "Normal"
            ref_str = "N/A"
            ref_source = "System Default"
            lab_min, lab_max = None, None

            # 1. Try Captured Range
            if raw_range:
                try:
                    if "-" in raw_range or "–" in raw_range:
                         parts = re.split(r"[-–]", raw_range)
                         v_min = parse_complex_value(parts[0].strip())
                         v_max = parse_complex_value(parts[1].strip())
                         
                         multiplier = 1.0
                         if 'million' in raw_range.lower() or 'mili' in raw_range.lower(): multiplier = 1000000.0
                         elif 'k' in raw_range.lower(): multiplier = 1000.0
                         
                         lab_min = v_min * (multiplier if v_min < 1000 and multiplier > 1 else 1.0)
                         lab_max = v_max * (multiplier if v_max < 1000 and multiplier > 1 else 1.0)
                         ref_str = raw_range
                         ref_source = "Lab Report"
                except: pass

            # 2. Fallback to System
            if lab_min is None and norm_marker in REFERENCE_RANGES:
                system_ref = REFERENCE_RANGES[norm_marker]
                sys_unit = system_ref.get('unit', '').lower()
                sys_mult = 1.0
                if 'million' in sys_unit or 'mili' in sys_unit: sys_mult = 1000000.0
                elif sys_unit.startswith('k/'): sys_mult = 1000.0
                
                lab_min = system_ref['min'] * sys_mult
                lab_max = system_ref['max'] * sys_mult
                ref_str = f"{system_ref['min']}-{system_ref['max']} {system_ref['unit']}"
                if not unit: unit = system_ref.get('unit', '')

            if lab_min is not None and lab_max is not None:
                if value < lab_min: status = "Low"
                elif value > lab_max: status = "High"
            
            labs.append({
                "name": raw_name.title(),
                "normalized_name": normalized,
                "value": value,
                "unit": unit,
                "status": status,
                "reference": ref_str,
                "source": ref_source
            })
            found_names.add(normalized)
        
    return labs

def calculate_risk(labs):
    risks = []
    lab_map = {l['normalized_name']: l for l in labs}
    
    # 1. Cardiovascular Risk
    chol = lab_map.get('cholesterol')
    ldl = lab_map.get('ldl')
    hdl = lab_map.get('hdl')
    if chol and chol['value'] > 240:
        risks.append({"type": "Cardiovascular", "status": "High Risk", "level": "Severe", "condition": "High Total Cholesterol", "detail": f"Cholesterol {chol['value']} mg/dL exceeds high risk threshold."})
    elif chol and chol['value'] > 200:
        risks.append({"type": "Cardiovascular", "status": "Moderate Risk", "level": "Moderate", "condition": "Elevated Cholesterol", "detail": "Total Cholesterol > 200 mg/dL."})
    if ldl and ldl['value'] > 160:
        risks.append({"type": "Cardiovascular", "status": "High Risk", "level": "Severe", "condition": "High LDL", "detail": "Bad cholesterol (LDL) is significantly elevated."})
    if hdl and hdl['value'] < 40:
         risks.append({"type": "Cardiovascular", "status": "Elevated Risk", "level": "Moderate", "condition": "Low HDL", "detail": "Good cholesterol (HDL) is low."})

    # 2. Diabetes & Metabolic
    gluc = lab_map.get('glucose')
    hba1c = lab_map.get('hba1c')
    if hba1c:
        if hba1c['value'] >= 6.5:
            risks.append({"type": "Diabetes", "status": "Likely Diabetic", "level": "Severe", "condition": "Diabetes", "detail": f"HbA1c {hba1c['value']}% indicates diabetic range."})
        elif hba1c['value'] >= 5.7:
            risks.append({"type": "Diabetes", "status": "Prediabetic", "level": "Moderate", "condition": "Prediabetes", "detail": f"HbA1c {hba1c['value']}% indicates early insulin resistance."})
    elif gluc and gluc['value'] > 126:
        risks.append({"type": "Diabetes", "status": "High Risk", "level": "Severe", "condition": "Hyperglycemia", "detail": "Fasting Glucose > 126 mg/dL."})
        
    # 3. Kidney Health
    creat = lab_map.get('creatinine')
    if creat and creat['value'] > 1.4:
         risks.append({"type": "Kidney Health", "status": "Elevated Risk", "level": "Moderate", "condition": "Reduced Filtration", "detail": "Creatinine level suggests potential kidney stress."})

    # 4. Inflammation
    esr = lab_map.get('esr')
    crp = lab_map.get('crp')
    if (crp and crp['value'] > 10) or (esr and esr['value'] > 40):
         risks.append({"type": "Inflammation", "status": "Systemic Stress", "level": "Moderate", "condition": "Inflammation", "detail": "Elevated inflammatory markers detected."})

    # 5. Iron & Anemia
    iron = lab_map.get('iron')
    ferr = lab_map.get('ferritin')
    hb = lab_map.get('hemoglobin')
    if hb and hb['status'] == 'Low':
        if (iron and iron['value'] < 50) or (ferr and ferr['value'] < 20):
            risks.append({"type": "Hematology", "status": "Deficiency", "level": "Severe", "condition": "Iron Deficiency Anemia", "detail": "Low hemoglobin combined with low iron/ferritin stores."})
        else:
            risks.append({"type": "Hematology", "status": "Observation", "level": "Moderate", "condition": "Anemia", "detail": "Hemoglobin is below normal range."})

    return risks

def generate_recommendations(labs, risks):
    recs = []
    lab_map = {l['normalized_name']: l for l in labs}

    if any(r['type'] == 'Cardiovascular' for r in risks):
        recs.append({"action": "Diet", "advice": "Reduce saturated fats and cholesterol intake. Consider Mediterranean diet."})
        recs.append({"action": "Exercise", "advice": "Aim for 150 minutes of moderate aerobic activity per week."})
    
    if any(r['type'] == 'Diabetes' for r in risks):
         recs.append({"action": "Diet", "advice": "Monitor carbohydrate intake. Avoid sugary beverages."})
         recs.append({"action": "Monitoring", "advice": "Check HbA1c to assess long-term glucose control."})

    if not recs:
        recs.append({"action": "General", "advice": "Maintain current healthy lifestyle."})
        recs.append({"action": "Screening", "advice": "Routine annual physical."})
    else:
        recs.append({"action": "Follow-up", "advice": "Repeat lipids/metabolic panel in 3-6 months."})

    return recs

def chunk_text(text, chunk_size=3000, overlap=200):
    """
    Splits text into chunks with overlap for processing long documents.
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
        
    return chunks

def summarize_text(text):
    """
    Recursively summarizes long text (Map-Reduce style).
    Optimized for Speed: High-Density Extract for medical reports.
    """
    model = get_summarizer()
    if not model: return "Summary unavailable."
    
    # 1. Faster Threshold: Only summarize if > 1000 words
    word_count = len(text.split())
    if word_count < 300:
        return text[:500] # Too short to summarize effectively
        
    # 2. Extract Key Sections (Faster than processing 19 pages of tables)
    # We look for Impression/Findings/History which contain the "meat"
    lines = text.split('\n')
    key_lines = []
    capture = False
    for line in lines:
        l_low = line.lower()
        if any(k in l_low for k in ["impression", "findings", "history", "diagnosis", "conclusion"]):
            capture = True
        if capture and len(key_lines) < 200: # Limit focus
            key_lines.append(line)
        if len(line) > 100 and any(k in l_low for k in ["disclaimer", "performed at"]):
            capture = False
            
    focused_text = "\n".join(key_lines) if key_lines else text[:10000]
    
    # 3. Direct Summary if manageable
    if len(focused_text) < 4000:
        try:
            summary = model(focused_text, max_length=120, min_length=40, do_sample=False, truncation=True, batch_size=4)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Summary Error (Direct): {e}")
            return focused_text[:300] + "..."

    # 4. Long Document Strategy (Density Sampling)
    print(f"Long clinical text detected ({len(text)} chars). Using High-Density Extract.")
    chunks = chunk_text(focused_text, chunk_size=3000, overlap=100)
    
    chunk_summaries = []
    # Speedup: Process with Batching
    BATCH_SIZE = 4 
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        try:
            results = model(batch, max_length=80, min_length=20, do_sample=False, truncation=True, batch_size=len(batch))
            for res in results:
                chunk_summaries.append(res['summary_text'])
        except: continue

    combined = " ".join(chunk_summaries)
    try:
         final = model(combined[:3500], max_length=150, min_length=50, do_sample=False, truncation=True)
         return final[0]['summary_text']
    except:
         return combined[:500] + "..."

def get_follow_up_tests(labs, risks):
    tests = []
    lab_map = {l['normalized_name']: l for l in labs}
    
    # 1. Lipid / Cardiovascular
    if any(r['type'] == 'Cardiovascular' for r in risks):
        tests.append({
            "test": "Lipid Fractionation",
            "reason": "Detailed breakdown of cholesterol particles to assess risk more accurately."
        })
        tests.append({
            "test": "Lipoprotein(a)",
            "reason": "Genetic marker for cardiovascular risk."
        })
        tests.append({
            "test": "ApoB",
            "reason": "Direct measure of atherogenic particles."
        })

    # 2. Diabetes / Metabolic
    glucose = lab_map.get('glucose')
    hba1c = lab_map.get('hba1c')
    if (glucose and glucose['status'] == 'High') or (hba1c and hba1c['status'] == 'High'):
        tests.append({
            "test": "C-Peptide & Insulin",
            "reason": "Distinguish between Type 1 and Type 2 diabetes / assess insulin resistance."
        })
        tests.append({
            "test": "Urine Microalbumin",
            "reason": "Screen for early kidney damage due to sugar levels."
        })

    # 3. Anemia
    hemoglobin = lab_map.get('hemoglobin')
    if hemoglobin and hemoglobin['status'] == 'Low':
        tests.append({
            "test": "Iron Studies (Ferritin, TIBC)",
            "reason": "Determine if anemia is due to iron deficiency."
        })
        tests.append({
            "test": "Vitamin B12 & Folate",
            "reason": "Rule out nutritional deficiency."
        })
        tests.append({
            "test": "Reticulocyte Count",
            "reason": "Check bone marrow response to anemia."
        })

    # 4. Thyroid
    tsh = lab_map.get('tsh')
    if tsh and tsh['status'] != 'Normal':
        tests.append({
            "test": "Free T3 & Free T4",
            "reason": "Confirm extent of thyroid dysfunction."
        })
        tests.append({
            "test": "Thyroid Antibodies (TPO)",
            "reason": "Check for autoimmune causes like Hashimoto's."
        })

    # 5. Inflammation / Infection
    wbc = lab_map.get('wbc')
    if wbc and wbc['status'] == 'High':
        tests.append({
            "test": "WBC Differential",
            "reason": "Identify if infection is bacterial, viral, or other."
        })
        tests.append({
            "test": "CRP & ESR",
            "reason": "General markers of inflammation in the body."
        })

    # 6. Liver
    alt = lab_map.get('alt')
    ast = lab_map.get('ast')
    if (alt and alt['status'] == 'High') or (ast and ast['status'] == 'High'):
        tests.append({
            "test": "Hepatitis Panel",
            "reason": "Rule out viral hepatitis."
        })
        tests.append({
            "test": "Abdominal Ultrasound",
            "reason": "Check for fatty liver or structural issues."
        })
        
    return tests

def compare_labs(current_labs, previous_labs):
    """
    Matches labs by normalized name and calculates trends.
    """
    prev_map = {l['normalized_name']: l for l in previous_labs}
    
    for lab in current_labs:
        norm = lab['normalized_name']
        if norm in prev_map:
            prev = prev_map[norm]
            lab['previous_value'] = prev['value']
            
            diff = lab['value'] - prev['value']
            lab['change'] = round(diff, 2)
            
            # Trend Logic
            # For most things, lower is better if it was High, higher is better if Low.
            # Simplified Logic:
            if abs(diff) < 0.1:
                lab['trend'] = 'Stable'
            elif diff > 0:
                lab['trend'] = 'Increasing'
            else:
                lab['trend'] = 'Decreasing'
                
            # Heuristic for "Better/Worse" (Very basic)
            if lab['status'] == 'High' and diff < 0:
                lab['trend_context'] = 'Improving'
            elif lab['status'] == 'Low' and diff > 0:
                lab['trend_context'] = 'Improving'
            elif lab['status'] == 'Normal':
                lab['trend_context'] = 'Stable'
            else:
                lab['trend_context'] = 'Worsening'
                
    return current_labs

def analyze_full_report(current_text, previous_text=None):
    # 1. Structural Parsing
    sections = extract_sections(current_text)
    
    # 2. Lab Extraction
    labs = extract_labs_regex(current_text)
    
    # 3. Risk & Recs
    risks = calculate_risk(labs)
    recommendations = generate_recommendations(labs, risks)
    follow_up_tests = get_follow_up_tests(labs, risks)
    
    # 4. Intelligence Layers
    # Patient-Friendly Translation of Findings/Impression
    patient_findings = explain_to_patient(sections['findings'])
    patient_impression = explain_to_patient(sections['impression'])
    
    # Severity Detection per section
    severity = {
        "findings": detect_severity(sections['findings']),
        "impression": detect_severity(sections['impression'])
    }
    
    # 5. Summary (Technical)
    summary = summarize_text(current_text)
    
    # Entities
    nlp_model = get_nlp()
    if nlp_model:
        doc = nlp_model(current_text)
        entities = [{"text": e.text, "label": "ENTITY"} for e in doc.ents]
    else:
        entities = []
    entities = [dict(t) for t in {tuple(d.items()) for d in entities}]

def detect_report_type(text):
    """
    Determines if the report is 'Lab' or 'Radiology' based on keywords.
    """
    text_lower = text.lower()
    
    # Radiology Signals
    rad_keywords = [
        "mri", "ct scan", "computed tomography", "x-ray", "ultrasound", 
        "sonography", "technique:", "exam:", "examination:", "contrast", 
        "sagittal", "coronal", "axial", "t1", "t2", "echo", "doppler"
    ]
    
    # Lab Signals
    lab_keywords = [
        "hemoglobin", "glucose", "cholesterol", "creatinine", 
        "reference range", "units", "u/l", "mg/dl", "g/dl"
    ]
    
    rad_score = sum(1 for k in rad_keywords if k in text_lower)
    lab_score = sum(1 for k in lab_keywords if k in text_lower)
    
    if rad_score > lab_score:
        return "Radiology"
    return "Lab"

def analyze_radiology(text, sections):
    """
    Specialized analysis for imaging reports.
    """
    findings = sections.get('findings', '')
    impression = sections.get('impression', '')
    
    # NLP Extraction for Abnormalities
    nlp_model = get_nlp()
    if nlp_model:
        doc = nlp_model(findings + " " + impression)
        
        # Heuristic: Find sentences with 'alert' words
        abnormalities = []
        alert_terms = [
            "mass", "nodule", "fracture", "herniation", "stenosis", 
            "lesion", "cyst", "tumor", "infarct", "hemorrhage", 
            "effusion", "opacification", "tear", "rupture", "dislocation"
        ]
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if any(term in sent_text.lower() for term in alert_terms):
                # Check if it is negated? (Basic check)
                if "no " not in sent_text.lower() and "unremarkable" not in sent_text.lower():
                    abnormalities.append(sent_text)
    else:
        abnormalities = []
                
    # Deduplicate
    abnormalities = list(set(abnormalities))
    
    return {
        "modality": "Unknown Imaging", # Could extract "MRI of Brain" here later
        "abnormalities": abnormalities,
        "is_normal": len(abnormalities) == 0,
        "sizes": [] # Could be extracted if analyze_radiology was called on text with sizes
    }


    return list(specialists)

def get_recommended_specialists(risks, report_type="Lab", location=None):
    """
    Maps risks/abnormalities to medical specialists with Location Search Link.
    """
    specialists = set()
    
    # 1. Logic based on Risks (Lab/General)
    for risk in risks:
        r_type = risk.get('type', '').lower()
        detail = risk.get('detail', '').lower()
        
        if "cardiovascular" in r_type or "cholesterol" in detail:
            specialists.add("Cardiologist")
        elif "diabetes" in r_type or "glucose" in detail or "thyroid" in detail:
            specialists.add("Endocrinologist")
        elif "kidney" in r_type or "creatinine" in detail:
            specialists.add("Nephrologist")
        elif "anemia" in r_type or "hemoglobin" in detail:
            specialists.add("Hematologist")
        elif "liver" in r_type or "hepatitis" in detail:
            specialists.add("Hepatologist")
            
    # Default fallbacks
    if not specialists:
        if report_type == "Radiology":
            specialists.add("Orthopedist")
            specialists.add("Radiologist")
        else:
            specialists.add("General Physician")

    # Step 2: Use ThreadPoolExecutor to fetch specialist details in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Create a map of role to future object
        # Each specialist role search is run in a separate thread
        future_to_role = {executor.submit(fetch_specialist_data, role, location): role for role in specialists}
        
        for future in concurrent.futures.as_completed(future_to_role):
            role = future_to_role[future]
            try:
                data = future.result()
                results.append(data)
            except Exception as exc:
                print(f"Parallel Search Error for {role}: {exc}")
                # Fallback for failed threads
                results.append({
                    "role": role,
                    "link": f"https://www.google.com/maps/search/{role}+doctor".replace(' ', '+'),
                    "profiles": []
                })

    return results

def fetch_specialist_data(role, location):
    """Helper to fetch all data for a single specialist role (Optimized for Threading)."""
    loc_display = location if location else "your area"
    
    # 1. Generate Map Link
    query = f"{role} near {loc_display}"
    map_link = f"https://www.google.com/maps/search/{query.replace(' ', '+')}"
    
    # 2. Fetch Real Profiles (This hits APIs, slow part)
    real_profiles = fetch_real_doctors(role, loc_display)
    
    return {
        "role": role,
        "link": map_link,
        "profiles": real_profiles
    }

def fetch_real_doctors(role, location):
    """
    Orchestrates fetching real doctor profiles from multiple sources.
    Prioritizes structured data (Overpass) for speed.
    """
    try:
        # Step 1: Try Overpass API (Community Driven / OpenStreetMap)
        # Fast & no API key required.
        doctors = fetch_overpass_doctors(role, location)
        if doctors: return doctors
    except:
        pass
        
    # Fallback to empty if both fail
    return []


def geocode_locale(locale):
    """Converts city/area name to lat,lon using Nominatim (Fast Timeout)."""
    if not locale or locale == "your area":
        return 18.5204, 73.8567 # Default to Pune, India coordinates as fallback
    
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={locale}&format=json&limit=1"
        headers = {'User-Agent': 'MedicalReportAssistant/1.0'}
        # REDUCED TIMEOUT for speed
        resp = requests.get(url, headers=headers, timeout=3)
        data = resp.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except Exception as e:
        print(f"Geocoding Error: {e}")
    return 18.5204, 73.8567

def fetch_overpass_doctors(role, locale):
    """Finds doctors nearby using Overpass API (Fast Timeout)."""
    lat, lon = geocode_locale(locale)
    
    # Mapping common roles to OSM speciality tags
    osm_specialities = {
        "Cardiologist": "cardiology",
        "Endocrinologist": "endocrinology",
        "Nephrologist": "nephrology",
        "Hematologist": "hematology",
        "Hepatologist": "hepatology",
        "Orthopedist": "orthopaedics",
        "Radiologist": "radiology",
        "General Physician": "general_practice"
    }
    spec = osm_specialities.get(role, "")
    
    # Build Query (Around 5km for speed)
    spec_filter = f'["healthcare:speciality"="{spec}"]' if spec else ""
    query = f"""
    [out:json];
    (
      node["amenity"="doctor"]{spec_filter}(around:5000,{lat},{lon});
      node["healthcare"="doctor"]{spec_filter}(around:5000,{lat},{lon});
    );
    out;
    """
    
    try:
        url = "https://overpass-api.de/api/interpreter"
        # 4s timeout for Overpass vs old 10s
        resp = requests.post(url, data={'data': query}, timeout=4)
        data = resp.json()
        
        doctors = []
        for element in data.get('elements', [])[:3]:
            tags = element.get('tags', {})
            name = tags.get('name', f"Medical Center ({role})")
            addr = tags.get('addr:street', tags.get('addr:full', 'Nearby Medical Facility'))
            doctors.append({
                "name": name,
                "rating": "Community Verified",
                "reviews": addr,
                "image": f"https://ui-avatars.com/api/?name={name.replace(' ','+')}&background=random",
                "link": f"https://www.openstreetmap.org/node/{element['id']}"
            })
        if doctors: return doctors
    except: pass
    return []

def enrich_abnormalities(labs):
    """
    Generates structured AI insights for abnormal values using LLM or Rule Base.
    """
    explainer = get_explainer() # flan-t5-small
    
    for lab in labs:
        if lab['status'] == "Normal":
            continue
            
        # Context for AI
        name = lab['name']
        val = lab['value']
        status = lab['status']
        
        # We can use the LLM to generate these fields, or a smart dictionary for speed/accuracy.
        # Given latency concerns, let's use a Hybrid Template + LLM fill.
        
        # 1. WHAT
        lab['insight_what'] = f"{name} is {status} ({val})."
        
        # 2. WHY (Simplified Terms)
        # Expanded Layman Dictionary
        LAY_TERMS = {
            "hemoglobin": "Red blood cell protein that carries oxygen.",
            "glucose": "Blood sugar level.",
            "cholesterol": "Fat-like substance in blood.",
            "hba1c": "Average blood sugar over past 3 months.",
            "creatinine": "Waste product filtered by kidneys.",
            "tsh": "Hormone controlling thyroid energy.",
            "wbc count": "Immune cells fighting infection.",
            "platelets": "Cells that help blood clot.",
            "alt": "Liver enzyme indicating health.",
            "ast": "Liver enzyme indicating health.",
            "triglycerides": "Type of fat in the blood.",
            "sodium": "Electrolyte balancing water levels.",
            "potassium": "Electrolyte vital for heart function.",
            "calcium": "Mineral for bone strength.",
            "vitamin d": "Vitamin for bones and immunity.",
            "ferritin": "Stored iron levels.",
            "uric acid": "Waste product linked to gout.",
            "esr": "Inflammation marker.",
            "crp": "Inflammation marker."
        }
        
        norm_name = normalize_name(name)
        simple_name = LAY_TERMS.get(norm_name, name)
        
        # Smart Insight based on Status (Focus on clinical meaning, avoid redundant text)
        if status == "High":
            if "glucose" in norm_name: reason = "Elevated sugar level; indicates risk of diabetes."
            elif "cholesterol" in norm_name: reason = "High cholesterol level; linked to cardiovascular risk."
            elif "creatinine" in norm_name: reason = "Elevated waste product; kidneys may require evaluation."
            elif "wbc" in norm_name: reason = "High white cell count; suggests underlying infection/inflammation."
            elif "lymphocytes" in norm_name: reason = "Elevated count; often seen in viral infections."
            elif "eosinophils" in norm_name: reason = "High count; common in allergic reactions or parasitic issues."
            elif "glycosylated" in norm_name or "hba1c" in norm_name: reason = "Indicator of elevated long-term blood sugar levels."
            else: reason = f"Value is higher than standard clinical range."
        elif status == "Low":
            if "hemoglobin" in norm_name: reason = "Decreased red blood cell protein; indicator for anemia."
            elif "vitamin" in norm_name: reason = "Clinical deficiency detected."
            elif "platelets" in norm_name: reason = "Low platelet count; increased risk of easy bruising/bleeding."
            elif "neutrophils" in norm_name: reason = "Decreased count; potential reduction in infection-fighting ability."
            elif "rbc" in norm_name: reason = "Low red blood cell count; often linked to anemia or nutritional gaps."
            else: reason = f"Value is lower than standard clinical range."
        else:
            reason = f"Abnormal level detected for {simple_name}."

        lab['insight_why'] = reason
        
        # 3. ACTION
        lab['insight_action'] = "Consult doctor."
        if status == "High" and "glucose" in norm_name: lab['insight_action'] = "Limit sugar/carbs."
        if status == "High" and "cholesterol" in norm_name: lab['insight_action'] = "Low-fat diet & exercise."
        if status == "Low" and "hemoglobin" in norm_name: lab['insight_action'] = "Iron-rich foods."
        
        # 4. MONITOR
        lab['insight_monitor'] = "Retest in 3 months."
        
        # 5. NEXT TESTS
        lab['insight_next_steps'] = "Standard panel follow-up."
        
    return labs

def analyze_full_report(current_text, previous_text=None, location=None):
    """
    Main entry point for report analysis.
    Delegates to the production-grade 'process_clinical_document' orchestrator.
    """
    print("Analyze Request Received. Delegating to Clinical Orchestrator.")
    try:
        return process_clinical_document(current_text, previous_text, user_location=location)
    except Exception as e:
        print(f"Orchestrator Error: {e}")
        return {
            "report_type": "Error",
            "summary": f"An error occurred: {str(e)}",
            "sections": {},
            "entities": [],
            "risks": [],
            "labs": [],
            "demographics": {},
            "vitals": [],
            "medications": []
        }
