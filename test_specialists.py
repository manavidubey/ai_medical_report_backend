
import sys
import os
sys.path.append(os.path.abspath("./backend"))

from ai_engine import get_recommended_specialists

print("testing Specialist Mapping Logic...")

# Test Case 1: Cardiovascular
risks_cardio = [{"type": "Cardiovascular", "detail": "High LDL"}]
specs_cardio = get_recommended_specialists(risks_cardio, "Lab")
print(f"Cardio Risks -> {specs_cardio}")
if "Cardiologist" in specs_cardio:
    print("PASS: Cardiologist identified.")
else:
    print("FAIL: Cardiologist missing.")

# Test Case 2: Diabetes & Kidney
risks_complex = [
    {"type": "Diabetes", "detail": "High Glucose"},
    {"type": "Kidney Health", "detail": "High Creatinine"}
]
specs_complex = get_recommended_specialists(risks_complex, "Lab")
print(f"Complex Risks -> {specs_complex}")
if "Endocrinologist" in specs_complex and "Nephrologist" in specs_complex:
    print("PASS: Multiple specialists identified.")
else:
    print("FAIL: Specialists missing.")

# Test Case 3: Radiology Fracture
specs_rad = get_recommended_specialists([], "Radiology")
print(f"Radiology (Default) -> {specs_rad}")
if "Orthopedist" in specs_rad:
    print("PASS: Orthopedist identified (default fallback).")
