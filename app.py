import streamlit as st
from openai import OpenAI
import os
import json

# Page Configuration
st.set_page_config(
    page_title="Medical Report Assistant",
    page_icon="fq",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #ced4da;
    }
    h1 {
        color: #2c3e50; 
        font-family: 'Helvetica-Neue', sans-serif;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        border-radius: 0.5rem;
        border: 1px solid transparent;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for Configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("Configuration")
    
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API Key here.")
    
    st.info("This tool provides AI-powered summaries and comparisons of medical reports. Please ensure you comply with HIPAA and data privacy regulations when using this tool.")

# Main Content
st.title("Medical Report Assistant")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Report")
    current_report = st.text_area("Paste the current medical report here:", height=300)

with col2:
    st.subheader("Previous Report (Optional)")
    previous_report = st.text_area("Paste the previous medical report here (if available):", height=300)

def analyze_reports(current, previous, key):
    client = OpenAI(api_key=key)
    
    system_prompt = """You are an AI medical report assistant.
    1. Extract key details from the current report: Patient info, Specimen info, Findings, Diagnosis/Impression.
    2. Compare with previous report(s) if provided.
    3. Provide a clear, structured summary in plain English.
    4. Output MUST be valid JSON with the following structure:
    {
      "current_report_summary": "summary text...",
      "comparison_with_previous": "highlight changes...",
      "key_changes": "list key changes...",
      "optional_recommendations": "recommendations..."
    }
    """
    
    user_content = f"Current Report:\n{current}\n\n"
    if previous:
        user_content += f"Previous Report:\n{previous}"
    else:
        user_content += "Previous Report: None"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Or gpt-4 if available/preferred
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

if st.button("Analyze Reports", type="primary"):
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    elif not current_report:
        st.warning("Please enter at least the Current Report.")
    else:
        with st.spinner("Analyzing reports..."):
            result = analyze_reports(current_report, previous_report, api_key)
            
            if result.startswith("Error"):
                st.error(result)
            else:
                try:
                    data = json.loads(result)
                    
                    st.markdown("### Analysis Result")
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>Current Report Summary</h4>
                        <p>{data.get('current_report_summary', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if previous_report:
                        st.markdown("#### Comparison")
                        st.info(data.get('comparison_with_previous', 'No comparison provided.'))
                        
                        st.markdown("#### Key Changes")
                        st.write(data.get('key_changes', 'None'))
                    
                    if data.get('optional_recommendations'):
                        st.markdown("#### Recommendations")
                        st.warning(data.get('optional_recommendations'))
                        
                    with st.expander("View Raw JSON"):
                        st.json(data)
                        
                except json.JSONDecodeError:
                    st.error("Failed to parse the response from the AI. Raw output:")
                    st.text(result)

