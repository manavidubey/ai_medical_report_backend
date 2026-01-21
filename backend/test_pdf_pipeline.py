from pdf_processor import process_pdf, detect_pdf_type
import os

# Use an existing PDF for testing
TEST_PDF = "Referral_Letter.pdf"

if not os.path.exists(TEST_PDF):
    print(f"Skipping test: {TEST_PDF} not found.")
else:
    print(f"Testing with {TEST_PDF}...")
    with open(TEST_PDF, "rb") as f:
        file_bytes = f.read()

    # 1. Detect Type
    pdf_type = detect_pdf_type(file_bytes)
    print(f"Detected Type: {pdf_type}")

    # 2. Extract Text
    text = process_pdf(file_bytes)
    print(f"Extracted Text Length: {len(text)}")
    print("First 200 chars:")
    print(text[:200])

    if len(text) > 50:
        print("SUCCESS: PDF Pipeline verification passed!")
    else:
        print("FAILURE: Extracted text is too short.")

def test_stitcher():
    print("\nTesting Document Stitcher Logic...")
    from pdf_processor import DocumentStitcher
    
    # Simulate 2 pages with repeated header/footer and content spill
    page1 = """
    APOLLO HOSPITALS
    Patient: John Doe
    Page 1 of 2
    
    COMPLETE BLOOD COUNT
    Hemoglobin: 14.5
    WBC: 6000
    """
    
    page2 = """
    APOLLO HOSPITALS
    Patient: John Doe
    Page 2 of 2
    
    Platelets: 250000
    """
    
    stitcher = DocumentStitcher()
    stitched = stitcher.stitch([page1.strip(), page2.strip()])
    
    print("--- Stitched Output ---")
    print(stitched)
    print("-----------------------")
    
    # Verification: "APOLLO HOSPITALS" should appear once (at top), not twice.
    if stitched.count("APOLLO HOSPITALS") == 1:
        print("SUCCESS: Header removed from Page 2.")
    else:
        print(f"FAILURE: Header count is {stitched.count('APOLLO HOSPITALS')}")

if __name__ == "__main__":
    test_stitcher()
    # Existing PDF test (commented out if file missing, or kept if present)
    if os.path.exists(TEST_PDF):
        with open(TEST_PDF, "rb") as f:
             process_pdf(f.read()) # Just smoke test integration
