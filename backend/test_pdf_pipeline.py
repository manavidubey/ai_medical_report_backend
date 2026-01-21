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
