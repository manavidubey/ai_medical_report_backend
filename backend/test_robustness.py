from ai_engine import extract_labs_regex, SYNONYMS

def test_robustness():
    print("Testing Universal Robustness...")
    
    # TestCase 1: Common Abbreviation (TLC)
    text1 = "TLC: 8000 /uL"
    labs = extract_labs_regex(text1)
    if any(l['normalized_name'] == 'wbc count' for l in labs):
        print("SUCCESS: TLC mapped to WBC Count.")
    else:
        print(f"FAILURE: TLC mapping failed. Found: {labs}")

    # TestCase 2: Missing Unit (Inference)
    # "Hemoglobin 14.2" without unit -> should infer g/dL
    text2 = "Hemoglobin 14.2" 
    labs2 = extract_labs_regex(text2)
    if labs2 and labs2[0]['unit'] == 'g/dL':
        print("SUCCESS: Unit Inference worked (g/dL inferred).")
    else:
        print(f"FAILURE: Unit Inference failed. Found: {labs2}")
        
    # TestCase 3: Loose Layout (Name... Value) matches?
    # Regex expects roughly continuous, but let's test.
    text3 = "Platelet Count : 250000"
    labs3 = extract_labs_regex(text3)
    if labs3:
         print("SUCCESS: Platelet Count extracted.")
    else:
         print("FAILURE: Platelet Count failed.")

if __name__ == "__main__":
    test_robustness()
