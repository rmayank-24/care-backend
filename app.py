import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from datetime import datetime
import io
import base64
from PIL import Image
import numpy as np  # Added for EasyOCR
import pdfplumber
import tempfile
import os
import re

# --- CONFIGURATION (UPDATED TO PORT 8501) ---
CLINICAL_ANALYSIS_URL = "http://127.0.0.1:8501/api/v1/clinical-analysis"
ONLINE_RESOURCES_URL = "http://127.0.0.1:8501/api/v1/online-resources"
USAGE_STATS_URL = "http://127.0.0.1:8501/api/v1/usage-stats"
RECENT_QUERIES_URL = "http://127.0.0.1:8501/api/v1/recent-queries"
DEBUG_VECTOR_URL = "http://127.0.0.1:8501/api/v1/debug/vector-store"
DEBUG_RETRIEVAL_URL = "http://127.0.0.1:8501/api/v1/debug/retrieval"
DEBUG_VECTOR_MEDS_URL = "http://127.0.0.1:8501/api/v1/debug/vector-store-meds"
DEBUG_RETRIEVAL_MEDS_URL = "http://127.0.0.1:8501/api/v1/debug/retrieval-meds"
# --- END OF PORT UPDATE ---

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="C.A.R.E. Clinical Analysis",
    page_icon="🏥",
    layout="wide",
)

# --- HELPER FUNCTIONS ---
def get_usage_stats():
    """Fetch usage statistics from the backend."""
    try:
        # --- ADD TIMEOUT ---
        response = requests.get(USAGE_STATS_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching usage statistics: {e}")
        return None

def get_recent_queries(limit=10):
    """Fetch recent queries from the backend."""
    try:
        # --- ADD TIMEOUT ---
        response = requests.get(f"{RECENT_QUERIES_URL}?limit={limit}", timeout=10)
        response.raise_for_status()
        return response.json().get("recent_queries", [])
    except Exception as e:
        st.error(f"Error fetching recent queries: {e}")
        return []

def debug_vector_store():
    """Debug the vector store."""
    try:
        # --- ADD TIMEOUT ---
        response = requests.get(DEBUG_VECTOR_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error debugging vector store: {e}")
        return None

def debug_vector_store_meds():
    """Debug the medication vector store."""
    try:
        # --- ADD TIMEOUT ---
        response = requests.get(DEBUG_VECTOR_MEDS_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error debugging medication vector store: {e}")
        return None

def debug_retrieval(query="chest pain"):
    """Debug document retrieval."""
    try:
        # --- ADD TIMEOUT ---
        response = requests.post(f"{DEBUG_RETRIEVAL_URL}?query={query}", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error in debug retrieval: {e}")
        return None

def debug_retrieval_meds(query="medication for hypertension"):
    """Debug medication document retrieval."""
    try:
        # --- ADD TIMEOUT ---
        response = requests.post(f"{DEBUG_RETRIEVAL_MEDS_URL}?query={query}", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error in debug medication retrieval: {e}")
        return None

def get_online_resources(disease_name):
    """Fetch online resources for a specific disease."""
    try:
        payload = {"disease_name": disease_name}
        # --- ADD TIMEOUT (longer for external search) ---
        response = requests.post(ONLINE_RESOURCES_URL, json=payload, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching online resources: {e}")
        return None

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None
    return text

def extract_text_from_image(image_file):
    """Extract text from uploaded image file using EasyOCR."""
    try:
        import easyocr
        
        # Initialize the reader. This downloads models on the first run.
        # It might take a minute the very first time it runs.
        with st.spinner("Initializing EasyOCR (downloading models on first run)..."):
            reader = easyocr.Reader(['en']) 

        # Open the image using PIL
        image = Image.open(image_file)
        
        # Convert the PIL image to a numpy array
        image_np = np.array(image)

        # Use EasyOCR to read the text
        with st.spinner("Extracting text from image..."):
            results = reader.readtext(image_np)

        # Combine the detected text into a single string
        extracted_text = "\n".join([result[1] for result in results])
        
        if not extracted_text.strip():
            st.warning("No text could be extracted from the image. Please ensure the image is clear and contains readable text.")
            return None
            
        return extracted_text

    except ImportError:
        st.error("""
        **EasyOCR is not installed.**
        
        Please install it by running the following command in your terminal:
        ```bash
        pip install easyocr
        ```
        Then restart your Streamlit app.
        """)
        return None
    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
        return None

def extract_disease_name(analysis_result):
    """Extract disease name from analysis result."""
    try:
        # Try to extract from enhanced result first
        if 'enhanced_result' in st.session_state and st.session_state.enhanced_result:
            enhanced = st.session_state.enhanced_result
            if enhanced.get("success") and enhanced.get("data"):
                primary = enhanced.get("data", {}).get("primary_diagnosis", {})
                if primary and primary.get("disease_name"):
                    return primary.get("disease_name")
        
        # Try to extract from legacy analysis
        if 'analysis_result' in st.session_state:
            analysis = st.session_state.analysis_result.get("analysis", {})
            
            # Check if analysis is a dictionary with primary_diagnosis
            if isinstance(analysis, dict) and "primary_diagnosis" in analysis:
                primary = analysis["primary_diagnosis"]
                if isinstance(primary, dict) and "disease_name" in primary:
                    return primary["disease_name"]
            
            # Try to extract from the first item if it's a list
            if isinstance(analysis, list) and len(analysis) > 0:
                first_item = analysis[0]
                if isinstance(first_item, dict) and "primary_diagnosis" in first_item:
                    primary = first_item["primary_diagnosis"]
                    if isinstance(primary, dict) and "disease_name" in primary:
                        return primary["disease_name"]
        
        # If all else fails, try to extract from the raw JSON
        if 'analysis_result' in st.session_state:
            raw_json = json.dumps(st.session_state.analysis_result)
            # Look for disease_name patterns in the JSON
            disease_match = re.search(r'"disease_name":\s*"([^"]+)"', raw_json)
            if disease_match:
                return disease_match.group(1)
        
        return "Unknown"
    except Exception as e:
        st.error(f"Error extracting disease name: {e}")
        return "Unknown"

# --- EXAMPLE REPORTS ---
example_reports = {
    "Cardiac Case": "Patient is a 58-year-old female with a history of type 2 diabetes and obesity. She presents to the emergency department with acute onset of severe substBurnal chest pain that began 3 hours ago. The pain is described as crushing, radiates to her jaw and left arm, and is associated with diaphoresis and nausea. On examination, she is tachycardic (HR 110), hypertensive (BP 160/95), and has an S4 gallop. ECG shows ST-segment elevation in leads II, III, and aVF. Initial cardiac enzymes show elevated troponin I at 5.2 ng/mL.",
    "Neurological Case": "Patient is a 72-year-old male with atrial fibrillation and hypertension who was found by his family with sudden onset of right-sided weakness and speech difficulty. Symptoms began approximately 2 hours prior to presentation. On examination, he has right facial droop, dysarthria, right hemiparesis (3/5 strength), and sensory loss. NIH Stroke Scale score is 12. Non-contrast CT head shows no acute hemorrhage. The patient is outside the window for thrombolytic therapy.",
    "Respiratory Case": "Patient is a 45-year-old female with no significant past medical history who presents with 5 days of progressively worsening cough, fever, and shortness of breath. She reports productive cough with yellowish sputum and pleuritic chest pain. Temperature is 38.9°C, heart rate 105, respiratory rate 24, oxygen saturation 91% on room air. Chest examination reveals decreased breath sounds and crackles in the right lower lobe. Chest X-ray shows a right lower lobe consolidation consistent with pneumonia.",
    "Gastrointestinal Case": "Patient is a 38-year-old male with a history of gallstones and occasional alcohol use who presents with acute onset of severe epigastric pain radiating to the back. The pain began after a large meal and has been constant for 6 hours. He reports associated nausea and vomiting. On examination, he is tachycardic (HR 115) with epigastric tenderness and guarding. Laboratory studies show serum lipase of 850 U/L (normal < 60), amylase of 420 U/L, and white blood cell count of 14,500/μL. Abdominal CT shows edematous pancreas with peripancreatic fat stranding.",
    "Endocrine Case": "Patient is a 24-year-old female with type 1 diabetes who presents with 2 days of nausea, vomiting, and abdominal pain. She reports running out of insulin 3 days ago. On examination, she is tachycardic (HR 120), hypotensive (BP 90/60), and tachypneic (respiratory rate 28) with deep, labored breathing (Kussmaul respirations). She is lethargic but responsive. Laboratory studies show glucose of 580 mg/dL, bicarbonate of 12 mEq/L, anion gap of 24, and positive urine ketones. Blood gas shows pH 7.20."
}

# --- STREAMLIT UI ---
st.title("🏥 C.A.R.E. Clinical Analysis System")
st.markdown("Enter a clinical report or upload a file (PDF/Image) to get a comprehensive analysis including diagnosis, tests, treatment, and prevention.")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Clinical Analysis", "Knowledge Base Usage", "Recent Queries", "Debug"])

with tab1:
    st.header("Clinical Report Analysis")
    
    # Add file upload section
    st.subheader("Upload Medical Report")
    uploaded_file = st.file_uploader(
        "Upload a medical report (PDF or Image)",
        type=["pdf", "png", "jpg", "jpeg", "tiff"]
    )
    
    # Process uploaded file
    extracted_text = ""
    if uploaded_file is not None:
        file_type = uploaded_file.type
        
        if file_type == "application/pdf":
            with st.spinner("Extracting text from PDF..."):
                extracted_text = extract_text_from_pdf(uploaded_file)
        elif file_type.startswith("image/"):
            # Use the new EasyOCR function
            extracted_text = extract_text_from_image(uploaded_file)
        
        if extracted_text:
            st.success("Text extracted successfully!")
            st.session_state.report_text = extracted_text
        else:
            # The error message is now handled inside the extraction function
            pass
    
    # Example report selector
    st.subheader("Load Example Report")
    selected_example = st.selectbox("Choose an example report:", ["None"] + list(example_reports.keys()))
    
    if selected_example != "None":
        st.session_state.report_text = example_reports[selected_example]
    
    # Report input
    report_input = st.text_area(
        "Clinical Report",
        height=300,
        key="report_text",
        placeholder="Enter clinical report text here or upload a file above..."
    )
    
    # Single Analyze button
    if st.button("Analyze Report", type="primary"):
        if not report_input or len(report_input) < 20:
            st.warning("Please enter a clinical report with at least 20 characters.")
        else:
            with st.spinner("Analyzing report... (This may take up to 30 seconds)"):
                try:
                    # Get the enhanced analysis for structured display
                    payload = {"report_text": report_input}
                    # --- ADD TIMEOUT (longer for AI) ---
                    enhanced_response = requests.post(CLINICAL_ANALYSIS_URL, json=payload, timeout=60)
                    enhanced_response.raise_for_status()
                    enhanced_result = enhanced_response.json()
                    
                    if enhanced_result.get("success"):
                        st.session_state.enhanced_result = enhanced_result
                        st.success("Analysis complete!")
                    else:
                        st.error("Analysis failed. Please check the backend logs.")
                        st.session_state.enhanced_result = None
                
                except requests.exceptions.RequestException as e:
                    st.error(f"A network or API error occurred: {e}")
                    if e.response is not None:
                        st.text_area("Error Response Body", value=e.response.text, height=200)
    
    # More Options Section - Always visible after analysis
    if 'enhanced_result' in st.session_state and st.session_state.enhanced_result:
        st.divider()
        
        # Display enhanced analysis if available
        st.subheader("📊 Enhanced Analysis View")
        data = st.session_state.enhanced_result.get("data", {})
        
        # Display primary diagnosis
        primary = data.get("primary_diagnosis", {})
        if primary:
            st.subheader("🔍 Primary Diagnosis")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Disease:** {primary.get('disease_name', 'Unknown')}")
                st.markdown(f"**Summary:** {primary.get('summary', 'No summary available')}")
            with col2:
                st.metric("Confidence Score", f"{primary.get('confidence_score', 0):.2%}")
            
            # Display clinical presentation
            clinical_presentation = primary.get('clinical_presentation', {})
            if clinical_presentation:
                st.markdown("### Clinical Presentation")
                col1, col2 = st.columns(2)
                
                with col1:
                    if clinical_presentation.get('common_symptoms'):
                        st.markdown("**Common Symptoms:**")
                        for symptom in clinical_presentation.get('common_symptoms', []):
                            # --- ROBUST FIX: Use concatenation ---
                            st.markdown("- " + str(symptom))
                
                with col2:
                    if clinical_presentation.get('key_findings'):
                        st.markdown("**Key Findings:**")
                        for finding in clinical_presentation.get('key_findings', []):
                            # --- ROBUST FIX: Use concatenation ---
                            st.markdown("- " + str(finding))
            
            # Display diagnostic approach
            diagnostic_approach = primary.get('diagnostic_approach', {})
            if diagnostic_approach:
                st.markdown("### Diagnostic Approach")
                col1, col2 = st.columns(2)
                
                with col1:
                    if diagnostic_approach.get('initial_tests'):
                        st.markdown("**Initial Tests:**")
                        for test in diagnostic_approach.get('initial_tests', []):
                            # --- ROBUST FIX: Use concatenation ---
                            st.markdown("- " + str(test))
                
                with col2:
                    if diagnostic_approach.get('confirmatory_tests'):
                        st.markdown("**Confirmatory Tests:**")
                        for test in diagnostic_approach.get('confirmatory_tests', []):
                            # --- ROBUST FIX: Use concatenation ---
                            st.markdown("- " + str(test))
            
            # Display treatment plan
            treatment_plan = primary.get('treatment_plan', {})
            if treatment_plan:
                st.markdown("### Treatment Plan")
                col1, col2 = st.columns(2)
                
                with col1:
                    if treatment_plan.get('first_line'):
                        st.markdown("**First-line Treatment:**")
                        for treatment in treatment_plan.get('first_line', []):
                            # --- ROBUST FIX: Use concatenation ---
                            st.markdown("- " + str(treatment))
                
                with col2:
                    if treatment_plan.get('advanced_options'):
                        st.markdown("**Advanced Options:**")
                        for treatment in treatment_plan.get('advanced_options', []):
                            # --- ROBUST FIX: Use concatenation ---
                            st.markdown("- " + str(treatment))
            
            # --- ADD SUGGESTED MEDICATIONS ---
            suggested_meds = data.get('suggested_medications', [])
            if suggested_meds:
                st.markdown("### Suggested Medications")
                st.dataframe(suggested_meds)
            # --- END OF ADDITION ---
            
            # Display prevention strategies
            if primary.get('prevention_strategies'):
                st.markdown("### Prevention Strategies")
                for strategy in primary.get('prevention_strategies', []):
                    # --- ROBUST FIX: Use concatenation ---
                    st.markdown("- " + str(strategy))
        
        # Display differential diagnoses
        differentials = data.get("differential_diagnoses", [])
        if differentials:
            st.subheader("🔬 Differential Diagnoses")
            for diff in differentials:
                # --- ROBUST FIX: Build title safely ---
                title = (
                    str(diff.get('disease_name', 'Unknown'))
                    + " (Confidence: {:.2f})".format(diff.get('confidence_score', 0))
                )
                with st.expander(title):
                    # --- ROBUST FIX: Use concatenation ---
                    st.markdown("**Summary:** " + str(diff.get('summary', 'No summary available')))
                    st.markdown("**Relevance:** " + str(diff.get('relevance_explanation', 'No explanation available')))
                    
                    # Show additional details
                    col1, col2 = st.columns(2)
                    with col1:
                        if diff.get('key_symptoms'):
                            st.markdown("**Key Symptoms:**")
                            for symptom in diff.get('key_symptoms', []):
                                # --- ROBUST FIX: Use concatenation ---
                                st.markdown("- " + str(symptom))
                        
                        if diff.get('diagnostic_tests'):
                            st.markdown("**Diagnostic Tests:**")
                            for test in diff.get('diagnostic_tests', []):
                                # --- ROBUST FIX: Use concatenation ---
                                st.markdown("- " + str(test))
                    
                    with col2:
                        if diff.get('treatment_approach'):
                            st.markdown("**Treatment Approach:**")
                            for treatment in diff.get('treatment_approach', []):
                                # --- ROBUST FIX: Use concatenation ---
                                st.markdown("- " + str(treatment))
        
        # Display clinical recommendations
        recommendations = data.get("clinical_recommendations", [])
        if recommendations:
            st.subheader("💡 Clinical Recommendations")
            for rec in recommendations:
                # --- ROBUST FIX: Use concatenation ---
                st.markdown("- " + str(rec))
        
        # --- ADD KNOWLEDGE BASE SOURCES ---
        st.subheader("📚 Retrieved Knowledge Base Context")
        kb_sources = data.get('knowledge_base_sources', [])
        if not kb_sources:
            st.info("No documents were retrieved from the disease knowledge base for this query.")
        else:
            for i, doc in enumerate(kb_sources):
                # --- ROBUST FIX: Build title safely ---
                title = (
                    "**Document {}:** `".format(i + 1)
                    + str(doc.get('disease', 'Unknown Disease'))
                    + "` (Score: {:.2f})".format(doc.get('score', 0))
                )
                with st.expander(title):
                    # --- ROBUST FIX: Use concatenation ---
                    st.caption("Source URL: " + str(doc.get('source_url', 'N/A')))
        
        med_sources = data.get('medication_sources', [])
        if not med_sources:
            st.info("No documents were retrieved from the medication knowledge base for this query.")
        else:
            for i, doc in enumerate(med_sources):
                # --- ROBUST FIX: Build title safely ---
                title = (
                    "**Medication Doc {}:** `".format(i + 1)
                    + str(doc.get('brand_name', 'Unknown'))
                    + "` (Score: {:.2f})".format(doc.get('score', 0))
                )
                with st.expander(title):
                    # --- ROBUST FIX: Use concatenation ---
                    st.caption("Generic Name: " + str(doc.get('generic_name', 'N/A')))
        # --- END OF ADDITION ---
        
        # Online Resources Section
        st.subheader("🌐 Online Resources")
        
        # Extract disease name using the improved function
        disease_name = primary.get('disease_name', 'Unknown')
        
        # Display the disease name with an option to edit it
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Current Disease:** {disease_name}")
        with col2:
            # Add a button to refresh the disease name extraction
            if st.button("Refresh Disease Name"):
                disease_name = extract_disease_name(st.session_state.analysis_result)
                st.rerun()
        
        # Add a text input to manually override the disease name
        manual_disease = st.text_input("Or enter disease name manually:", value=disease_name)
        
        if st.button("Find Online Resources", type="secondary"):
            if manual_disease and manual_disease != "Unknown":
                with st.spinner(f"Searching for online resources about {manual_disease}..."):
                    resources_result = get_online_resources(manual_disease)
                    if resources_result and resources_result.get("success"):
                        st.session_state.resources_result = resources_result
                    else:
                        st.error("Failed to fetch online resources. Please try again.")
            else:
                st.warning("Please enter a valid disease name to search for resources.")
        
        # Display resources if available
        if 'resources_result' in st.session_state:
            resources = st.session_state.resources_result
            if resources.get("success") and resources.get("resources"):
                st.markdown(f"### Found {resources.get('count', 0)} Resources")
                
                for i, resource in enumerate(resources.get("resources", [])):
                    # --- ROBUST FIX: Build title safely ---
                    title = (
                        "{}. ".format(i + 1)
                        + str(resource.get('title', 'No Title'))
                    )
                    with st.expander(title):
                        st.markdown(f"**URL:** [{resource.get('url', 'No URL')}]({resource.get('url', '#')})")
                        # --- ROBUST FIX: Use concatenation ---
                        st.markdown("**Description:** " + str(resource.get('snippet', 'No description available')))
                        
                        # Add a button to open the URL in a new tab
                        if resource.get('url'):
                            url = resource.get('url')
                            st.markdown("[Open in New Tab](" + url + "){:target='_blank'}")
            else:
                st.info("No resources found or an error occurred.")

with tab2:
    st.header("Knowledge Base Usage Analytics")
    
    # Refresh button
    if st.button("Refresh Usage Statistics"):
        st.rerun()
    
    # Get usage statistics
    usage_stats = get_usage_stats()
    
    if usage_stats and "error" not in usage_stats:
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", usage_stats.get("total_queries", 0))
        
        with col2:
            st.metric("Total Documents in KB", usage_stats.get("total_documents_in_kb", 0))
        
        with col3:
            st.metric("Unique Documents Used", usage_stats.get("unique_documents_used", 0))
        
        with col4:
            coverage = usage_stats.get("coverage_percentage", 0)
            st.metric("KB Coverage", f"{coverage:.2f}%")
        
        # Create visualizations
        st.subheader("Document Usage Frequency")
        
        # Get top 10 most used documents
        doc_usage = usage_stats.get("document_usage_frequency", {})
        if doc_usage:
            # Sort by frequency
            sorted_docs = sorted(doc_usage.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Create a dataframe for visualization
            doc_df = pd.DataFrame(sorted_docs, columns=["Document ID", "Usage Count"])
            
            # Create bar chart
            fig = px.bar(
                doc_df, 
                x="Usage Count", 
                y="Document ID", 
                orientation='h',
                title="Top 10 Most Used Documents",
                color="Usage Count",
                color_continuous_scale=px.colors.sequential.Blues
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Disease Usage Frequency")
        
        # Get top 10 most referenced diseases
        disease_usage = usage_stats.get("disease_usage_frequency", {})
        if disease_usage:
            # Sort by frequency
            sorted_diseases = sorted(disease_usage.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Create a dataframe for visualization
            disease_df = pd.DataFrame(sorted_diseases, columns=["Disease", "Usage Count"])
            
            # Create bar chart
            fig = px.bar(
                disease_df, 
                x="Usage Count", 
                y="Disease", 
                orientation='h',
                title="Top 10 Most Referenced Diseases",
                color="Usage Count",
                color_continuous_scale=px.colors.sequential.Reds
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Coverage percentage gauge
        st.subheader("Knowledge Base Coverage")
        coverage = usage_stats.get("coverage_percentage", 0)
        fig = px.pie(
            values=[coverage, 100-coverage],
            names=["Used Documents", "Unused Documents"],
            title=f"Knowledge Base Coverage: {coverage:.2f}%",
            color_discrete_map={"Used Documents": "#1f77b4", "Unused Documents": "#d3d3d3"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No usage data available yet. Analyze some reports to see statistics.")

with tab3:
    st.header("Recent Queries and Retrieved Documents")
    
    # Get recent queries
    recent_queries = get_recent_queries()
    
    if recent_queries:
        for query in recent_queries:
            with st.expander(f"Query from {query.get('timestamp', 'Unknown time')}"):
                st.write(f"Query ID: {query.get('query_id', 'Unknown')}")
                
                retrieved_docs = query.get("retrieved_documents", [])
                if retrieved_docs:
                    st.write("Retrieved Documents:")
                    for i, doc in enumerate(retrieved_docs):
                        # --- ROBUST FIX: Build title safely ---
                        st.write(
                            "{}. **Disease:** ".format(i + 1)
                            + str(doc.get('disease', 'Unknown'))
                        )
                        st.write(f"   **Document ID:** {doc.get('id', 'Unknown')}")
                        # --- ROBUST FIX: Use concatenation ---
                        st.write("   **Content Preview:** " + str(doc.get('content_preview', 'No preview available')))
                        st.write("---")
                else:
                    st.write("No documents were retrieved for this query.")
    else:
        st.info("No recent queries available.")

with tab4:
    st.header("Debug Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vector Store Status")
        if st.button("Check Vector Store"):
            debug_info = debug_vector_store()
            if debug_info:
                st.json(debug_info)
    
    with col2:
        st.subheader("Test Document Retrieval")
        test_query = st.text_input("Test Query", value="chest pain")
        if st.button("Test Retrieval"):
            retrieval_result = debug_retrieval(test_query)
            if retrieval_result:
                st.json(retrieval_result)
    
    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Medication Vector Store Status")
        if st.button("Check Medication Vector Store"):
            debug_info_meds = debug_vector_store_meds()
            if debug_info_meds:
                st.json(debug_info_meds)
                
    with col4:
        st.subheader("Test Medication Document Retrieval")
        test_query_meds = st.text_input("Test Meds Query", value="medication for hypertension")
        if st.button("Test Meds Retrieval"):
            retrieval_result_meds = debug_retrieval_meds(test_query_meds)
            if retrieval_result_meds:
                st.json(retrieval_result_meds)
