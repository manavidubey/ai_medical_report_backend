import sys
import os

# Ensure backend dir is in path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from backend.ai_engine import analyze_full_report

sample_text = """
Patient: John Doe
Date: 2024-01-01
Clinical History: 45 year old male with fatigue.

Findings:
The patient reports feeling tired. 
Labs:
Cholesterol: 245 mg/dL
TSH: 6.5 uIU/mL
Hemoglobin: 11.0 g/dL

Impression:
Hyperlipidemia and Hypothyroidism. Anemia.
"""

print("Starting Analysis...")
try:
    result = analyze_full_report(sample_text)
    print("Analysis Successful!")
    print("Follow-up Tests:", result.get('follow_up_tests'))
except Exception as e:
    print("Analysis Failed!")
    print(e)
    import traceback
    traceback.print_exc()
