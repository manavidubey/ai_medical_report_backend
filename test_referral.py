
import sys
import os
sys.path.append(os.path.abspath("./backend"))

from report_generator import create_referral_letter

sample_data = {
    "report_type": "Lab",
    "summary": "Patient shows elevated cholesterol levels and mild anemia.",
    "risks": [
        {"type": "Cardiovascular", "status": "Moderate Risk", "detail": "LDL > 160"},
        {"type": "Anemia", "status": "Observation", "detail": "Hemoglobin slightly low"}
    ],
    "recommendations": [
        {"action": "Diet", "advice": "Reduce saturated fat intake."},
        {"action": "Follow-up", "advice": "Retest lipids in 3 months."}
    ],
    "labs": []
}

print("Testing Referral Letter Generation...")
try:
    filename = "test_referral_letter.pdf"
    create_referral_letter(sample_data, filename)
    if os.path.exists(filename):
        print(f"SUCCESS: Generated {filename}")
    else:
        print("FAIL: File not created.")
except Exception as e:
    print(f"FAIL: {e}")
