from ai_engine import process_clinical_document

def test_advanced_intelligence():
    print("Testing Advanced Clinical Intelligence (Phase 4)...\n")
    
    # Simulate a comprehensive report content
    text = """
    HOSPITAL ADMISSION RECORD
    Patient Name: John Doe   Age: 58   Gender: Male
    Date: 2024-10-10
    
    VITALS:
    BP: 145/90 mmHg
    Heart Rate: 102 bpm
    BMI: 28.5 (Overweight)
    
    CURRENT CITATIONS:
    Medications:
    1. Atorvastatin 20mg daily
    2. Metformin 500mg BD
    3. Aspirin 75mg
    
    CLINICAL HISTORY:
    Patient compliant with meds.
    
    LABORATORY DATA:
    Hemoglobin: 13.5
    """
    
    print(">>> Processing Report...")
    result = process_clinical_document(text)
    
    # 1. Demographics Check
    print("\n--- Demographics ---")
    demos = result.get("demographics", {})
    print(f"Detected: {demos}")
    if demos["age"] == 58 and demos["gender"] == "Male":
        print("SUCCESS: Demographics extracted.")
    else:
        print("FAILURE: Demographics mismatch.")

    # 2. Vitals Check
    print("\n--- Vitals ---")
    vitals = result.get("vitals", [])
    for v in vitals:
        print(f"- {v['name']}: {v['value']} ({v['status']})")
        
    bp = next((v for v in vitals if v['name'] == 'Blood Pressure'), None)
    hr = next((v for v in vitals if v['name'] == 'Heart Rate'), None)
    
    if bp and hr and bp['status'] == "High (Hypertension)" and hr['status'] == "High (Tachycardia)":
        print("SUCCESS: Vitals extracted and analyzed correctly.")
    else:
        print("FAILURE: Vitals analysis incorrect.")

    # 3. Medications Check
    print("\n--- Medications ---")
    meds = result.get("medications", [])
    print(f"Rx List: {meds}")
    
    if len(meds) >= 3 and "Atorvastatin 20mg daily" in meds:
        print("SUCCESS: Medications list extracted.")
    else:
        print("FAILURE: Medication extraction failed.")

if __name__ == "__main__":
    test_advanced_intelligence()
