
import sys
import os

# Ensure backend imports work
sys.path.append(os.path.abspath("./backend"))

from ai_engine import detect_report_type, analyze_radiology, extract_sections

# Sample MRI Report Text
sample_mri = """
MRI LUMBAR SPINE 

HISTORY: Lower back pain radiating to left leg.
TECHNIQUE: Sagittal T1, T2, STIR and axial T1, T2 images were obtained.

FINDINGS:
There is straightening of the lumbar lordosis.
L4-L5: There is a diffuse disc bulge with a superimposed left paracentral disc protrusion.
This results in moderate left lateral recess stenosis and impingement of the traversing L5 nerve root.
L5-S1: Mild facet hypertrophy. No significant stenosis.
The conus medullaris ends at L1 and is unremarkable.
No evidence of fracture or destructive osseous lesion.
Paraspinal soft tissues are normal.

IMPRESSION:
1. L4-L5 disc protrusion causing moderate left lateral recess stenosis and nerve root impingement.
2. Mild facet arthropathy at L5-S1.
"""

print(f"Testing Analysis on:\n{sample_mri[:100]}...\n")

# 1. Test Detection
print("--- 1. Report Detection ---")
r_type = detect_report_type(sample_mri)
print(f"Detected Type: {r_type}")

if r_type != "Radiology":
    print("FAIL: Should be Radiology")
    sys.exit(1)
print("PASS: Correctly identified as Radiology")

# 2. Test Section Extraction
print("\n--- 2. Section Extraction ---")
sections = extract_sections(sample_mri)
print(f"Findings Length: {len(sections['findings'])}")
print(f"Impression Length: {len(sections['impression'])}")

# 3. Test Radiology Analysis
print("\n--- 3. NLP Analysis ---")
result = analyze_radiology(sample_mri, sections)
print(f"Abnormalities Found: {len(result['abnormalities'])}")
for abnormal in result['abnormalities']:
    print(f" - {abnormal}")

# Assertions
expected_terms = ["bulge", "stenosis", "impingement", "hypertrophy"]
found_count = 0
for item in result['abnormalities']:
    if any(x in item.lower() for x in expected_terms):
        found_count += 1

if found_count > 0:
    print("\nPASS: Found significant abnormalities (herniation/stenosis etc)")
else:
    print("\nFAIL: Did not extract expected abnormalities.")

print("Verification Complete.")
