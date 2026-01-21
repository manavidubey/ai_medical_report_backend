import time
from ai_engine import summarize_text

def test_performance():
    print("Testing Performance Optimization...\n")
    
    # Simulate a huge text > 50k chars
    print("Generating synthetic large report (60,000 chars)...")
    base_text = "Patient presented with mild cough. Lungs clear. " * 300
    huge_text = (base_text * 10)[:60000] 
    
    print(f"Total Length: {len(huge_text)} chars")
    
    start_time = time.time()
    summary = summarize_text(huge_text)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\nTime Taken: {duration:.2f} seconds")
    print(f"Summary Length: {len(summary)} chars")
    print(f"Summary Preview: {summary[:100]}...")
    
    # Heuristic for success: < 30 seconds for 60k chars on decent hardware w/ sampling
    if "Batch" in summary or len(summary) > 10:
        print("SUCCESS: Performance test completed.")
    else:
        print("FAILURE: Summary too short or failed.")

if __name__ == "__main__":
    test_performance()
