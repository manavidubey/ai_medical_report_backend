
import sys
import os
import io
from PIL import Image

# Ensure backend imports work
sys.path.append(os.path.abspath("./backend"))

from ai_engine import analyze_medical_image

def create_dummy_image():
    # Create a 224x224 RGB image (noise)
    img = Image.new('RGB', (224, 224), color = 'gray')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

print("--- Testing Visual AI Engine ---")

try:
    print("1. Creating dummy image...")
    img_bytes = create_dummy_image()
    
    print("2. Running Analysis (calling analyze_medical_image)...")
    result = analyze_medical_image(img_bytes)
    
    print("\n--- Result ---")
    print(result)
    
    if "error" in result:
        print("\nFAIL: Engine returned error.")
        sys.exit(1)
        
    if "predictions" not in result:
        print("\nFAIL: No predictions returned.")
        sys.exit(1)
        
    # Since we sent a gray image, we expect "random noise" or "random object" to likely be high, 
    # or low confidence on medical terms.
    top_label = result['predictions'][0]['label']
    print(f"\nTop Prediction: {top_label} ({result['predictions'][0]['confidence']}%)")
    
    # Check if format matches our V13 expectations
    if "modality" in result and "finding" in result:
         print("PASS: Structure is correct (Modality/Finding present).")
    else:
         print("FAIL: Missing V13 structure.")
         sys.exit(1)

except Exception as e:
    print(f"\nCRITICAL FAIL: {e}")
    sys.exit(1)
