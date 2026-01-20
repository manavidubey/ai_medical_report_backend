import argparse
import os
import json
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def read_file(file_path):
    """Reads content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        sys.exit(1)

def analyze_report(current_text, previous_text=None, api_key=None, model="gpt-4"):
    """
    Performs deep analysis on the medical report using OpenAI API.
    """
    if not api_key:
        print("Error: OpenAI API Key not found. Please set OPENAI_API_KEY environment variable or pass it to the script.")
        return

    client = OpenAI(api_key=api_key)

    system_prompt = """You are an expert AI Medical Analyst designed for comprehensive, deep-dive clinical analysis. 
    Your goal is to extract every meaningful detail from the provided medical reports and synthesize them into a highly structured, professional summary.

    INSTRUCTIONS:
    1. **Full Extraction**: Do not summarize superficially. Extract ALL available patient metadata, clinical history, specimen details (size, location, weight, color, consistency), microscopic findings, and immunophenotyping results.
    2. **Comparative Analysis**: If a previous report is provided, meticulously compare every metric. Highlight subtle changes in size, progression of disease, or appearance of new features. Mention what has remained stable.
    3. **Clinical Interpretation**: Synthesize the findings into a coherent 'Diagnosis/Impression'. If the report has a conclusion, restate it clearly but add context if findings support it.
    4. **Output Format**: Return ONLY valid JSON structured exactly as follows:
    {
      "patient_demographics": "Age, Sex, ID, etc.",
      "clinical_history": "Summary of history provided...",
      "specimen_details": {
        "source": "...",
        "gross_description": "..."
      },
      "microscopic_findings": "Detailed histology...",
      "special_studies": "IHC, molecular tests, etc. (if any)",
      "diagnosis": "Final diagnosis or impression...",
      "comparative_analysis": "Detailed comparison with previous report (if available)...",
      "key_changes_list": ["Change 1", "Change 2", "Change 3"],
      "recommendations_or_notes": "Any follow-up suggested or critical notes..."
    }
    """

    user_content = f"### CURRENT REPORT ###\n{current_text}\n"
    if previous_text:
        user_content += f"\n### PREVIOUS REPORT ###\n{previous_text}\n"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.2  # Lower temperature for more analytical/precise output
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during API call: {e}"


def process_directory(input_dir, output_dir, api_key, model):
    """
    Batch processes all text files in a directory.
    """
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.txt', '.md'))]
    
    if not files:
        print(f"No text files (.txt, .md) found in {input_dir}")
        return

    print(f"Found {len(files)} files in '{input_dir}'. Starting batch processing...")

    for filename in files:
        file_path = os.path.join(input_dir, filename)
        print(f"\nProcessing: {filename}...")
        
        content = read_file(file_path)
        
        # In batch mode, we treat each file as a independent 'Current Report' for now
        # unless strict naming logic is added later.
        analysis_json_str = analyze_report(content, previous_text=None, api_key=api_key, model=model)
        
        try:
            if analysis_json_str and analysis_json_str.strip().startswith("{"):
                data = json.loads(analysis_json_str)
                
                # Construct output filename
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_analysis.json"
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"  -> Saved analysis to: {os.path.join('output_reports', output_filename)}")
            else:
                print(f"  -> Failed to generate valid JSON for {filename}")
        except Exception as e:
            print(f"  -> Error processing result for {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="AI Medical Report Analyzer (CLI)")
    parser.add_argument("--current", help="Path to the current medical report text file")
    parser.add_argument("--previous", help="Path to the previous medical report text file")
    parser.add_argument("--dir", help="Directory to scan for batch processing (default: ./input_reports)", default="input_reports")
    parser.add_argument("--batch", action="store_true", help="Force batch mode (processes all files in --dir)")
    parser.add_argument("--key", help="OpenAI API Key (optional if invalid in env)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI Model to use (default: gpt-4o)")

    args = parser.parse_args()

    # Determine API Key
    api_key = args.key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter OpenAI API Key: ").strip()

    # Mode Selection
    # If explicit files are given, run single mode.
    # If --batch is set OR no files are given, run batch mode.
    if args.current:
        print(f"--- Starting Single Analysis using {args.model} ---")
        print(f"Reading current report: {args.current}")
        current_text = read_file(args.current)
        
        previous_text = None
        if args.previous:
            print(f"Reading previous report: {args.previous}")
            previous_text = read_file(args.previous)

        print("Analyzing... (this may take a moment)")
        result_json_str = analyze_report(current_text, previous_text, api_key, args.model)
        
        # Display and save single result
        try:
            if result_json_str and result_json_str.strip().startswith("{"):
                data = json.loads(result_json_str)
                print("\n" + "="*50)
                print(" ANALYSIS RESULTS")
                print("="*50)
                print(json.dumps(data, indent=2))
                
                # Save to file
                output_filename = "analysis_result.json"
                with open(output_filename, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"\n[Saved full JSON output to {output_filename}]")
            else:
                print("\nRaw Output (Non-JSON or Error):")
                print(result_json_str)
        except json.JSONDecodeError:
            print("Failed to decode JSON from response.")
            print(result_json_str)
            
    else:
        # Batch Mode
        input_dir = args.dir
        output_dir = "output_reports"
        process_directory(input_dir, output_dir, api_key, args.model)

if __name__ == "__main__":
    main()

