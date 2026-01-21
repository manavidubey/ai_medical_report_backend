from ai_engine import process_clinical_document, ClinicalParser

def test_production_upgrade():
    print("Testing Production-Grade Upgrade...\n")
    
    # Test 1: Narrative Heavy (Referral Letter Style)
    text_narrative = """
    CLINICAL HISTORY:
    Patient is a 45 year old male presenting with chronic cough for 3 months.
    History of smoking. No fever.
    
    FINDINGS:
    Chest X-ray shows right lower lobe consolidation consistent with pneumonia.
    Small pleural effusion noted on the left value.
    Heart size is normal. No pneumothorax.
    Use of accessory muscles observed.
    
    IMPRESSION:
    1. Right lower lobe pneumonia.
    2. Mild pleural effusion.
    Suggest correlation with clinical symptoms.
    """
    
    print(">>> Testing Narrative Report processing...")
    result = process_clinical_document(text_narrative)
    
    print("Detected Sections:", list(result["raw_sections"].keys()))
    if "findings" in result["section_summaries"]:
        print("Findings Summary Generated:", result["section_summaries"]["findings"])
        print("SUCCESS: Narrative Section Summarized.")
    else:
        print("FAILURE: Findings section skipped.")

    # Test 2: Lab Heavy
    text_labs = """
    LABORATORY DATA:
    Hemoglobin: 10.5 g/dL (Low)
    WBC Count: 12000 /uL (High)
    Platelets: 160000 /uL
    
    IMPRESSION:
    Anemia and Leukocytosis.
    """
    
    print("\n>>> Testing Lab Report processing...")
    result_lab = process_clinical_document(text_labs)
    
    print("Detected Sections:", list(result_lab["raw_sections"].keys()))
    print("Structured Labs:", len(result_lab["structured_labs"]))
    if len(result_lab["structured_labs"]) >= 3:
        print("SUCCESS: Labs extracted structurally.")
    else:
        print(f"FAILURE: Labs extraction count {len(result_lab['structured_labs'])}")
        
    print("\nProduction Verification Complete.")

if __name__ == "__main__":
    test_production_upgrade()
