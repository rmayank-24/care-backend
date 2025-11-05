import os
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import json
from datetime import datetime
import uuid
import pinecone
from pinecone import ServerlessSpec  # Import the spec

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain v0.2+ modular imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain.agents import Tool

# --- CONFIGURATION ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a local cache folder to ensure the model is saved consistently
MODEL_CACHE_PATH = '.model_cache'
# --- Use the correct index names ---
DISEASE_INDEX_NAME = 'care-mini' 
MED_INDEX_NAME = 'care-meds'

# Create a directory for usage tracking if it doesn't exist
USAGE_TRACKING_DIR = "usage_tracking"
os.makedirs(USAGE_TRACKING_DIR, exist_ok=True)

# --- APPLICATION STATE ---
app_state = {}

# --- ONLINE RESOURCES FUNCTION ---
def search_online_resources(disease_name, max_results=5):
    """
    Search for online resources related to a disease using LangChain's search tools.
    Returns a list of dictionaries with title, url, and snippet.
    """
    try:
        # Initialize LangChain search tools
        search = DuckDuckGoSearchRun()
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        
        # Create a list of tools
        tools = [
            Tool(
                name="Web Search",
                func=search.run,
                description="Search the web for information about medical conditions"
            ),
            Tool(
                name="Wikipedia",
                func=wikipedia.run,
                description="Search Wikipedia for medical information"
            )
        ]
        
        # Create a search query focused on medical information
        query = f"{disease_name} clinical guidelines treatment diagnosis symptoms"
        
        # Use the search tool to get results
        search_results = search.run(query)
        
        # Use Wikipedia to get additional information
        wiki_results = wikipedia.run(disease_name)
        
        # Process the results
        results = []
        
        # Process web search results
        if search_results:
            # Split the results into lines and process each line
            lines = search_results.split('\n')
            for line in lines[:max_results]:
                if line.strip():
                    # Try to extract URL and title from the result
                    if 'http' in line:
                        parts = line.split('http')
                        if len(parts) > 1:
                            title = parts[0].strip()
                            url = 'http' + parts[1].split()[0]
                            snippet = line.strip()
                            
                            results.append({
                                "title": title,
                                "url": url,
                                "snippet": snippet[:200] + "..." if len(snippet) > 200 else snippet
                            })
        
        # Add Wikipedia result if available
        if wiki_results and len(results) < max_results:
            # Extract the first paragraph from Wikipedia
            lines = wiki_results.split('\n')
            if lines:
                snippet = lines[0].strip()
                if snippet:
                    # Create a Wikipedia URL
                    wiki_url = f"https://en.wikipedia.org/wiki/{disease_name.replace(' ', '_')}"
                    
                    results.append({
                        "title": f"{disease_name} - Wikipedia",
                        "url": wiki_url,
                        "snippet": snippet[:200] + "..." if len(snippet) > 200 else snippet
                    })
        
        # If we still don't have enough results, add fallback resources
        if len(results) < max_results:
            fallback_results = create_fallback_resources(disease_name)
            results.extend(fallback_results[:max_results - len(results)])
        
        return results[:max_results]
        
    except Exception as e:
        logger.error(f"Error searching with LangChain tools: {e}")
        # Return fallback resources if search fails
        return create_fallback_resources(disease_name)

def create_fallback_resources(disease_name):
    """Create fallback resources when web scraping fails."""
    return [
        {
            "title": f"{disease_name} - Mayo Clinic",
            "url": f"https://www.mayoclinic.org/search/search-results?q={disease_name.replace(' ', '%20')}",
            "snippet": f"Find comprehensive information about {disease_name} including symptoms, causes, diagnosis, and treatment options from Mayo Clinic."
        },
        {
            "title": f"{disease_name} - MedlinePlus",
            "url": f"https://medlineplus.gov/search?query={disease_name.replace(' ', '%20')}",
            "snippet": f"Learn about {disease_name} from the National Library of Medicine's trusted health information resource."
        },
        {
            "title": f"{disease_name} - NIH",
            "url": f"https://www.nih.gov/search-results?term={disease_name.replace(' ', '%20')}",
            "snippet": f"Explore research and clinical information about {disease_name} from the National Institutes of Health."
        },
        {
            "title": f"{disease_name} - WebMD",
            "url": f"https://www.webmd.com/search/search_results/default.aspx?query={disease_name.replace(' ', '%20')}",
            "snippet": f"Get information about {disease_name} symptoms, treatments, medications, and more from WebMD."
        },
        {
            "title": f"{disease_name} - Healthline",
            "url": f"https://www.healthline.com/search?q={disease_name.replace(' ', '%20')}",
            "snippet": f"Find evidence-based information about {disease_name} including causes, symptoms, diagnosis, and treatment options."
        }
    ]

# --- USAGE TRACKING FUNCTIONS ---
def track_document_usage(retrieved_docs: List[Any], query_id: str):
    """Track which documents were retrieved for a specific query."""
    usage_data = {
        "query_id": query_id,
        "timestamp": datetime.now().isoformat(),
        "retrieved_documents": []
    }
    
    for doc in retrieved_docs:
        doc_data = {
            "id": doc.metadata.get("id", "unknown"),
            "disease": doc.metadata.get("disease", "unknown"),
            "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
            # Add score if available (it will be for similarity_score_threshold)
            "score": doc.metadata.get("score", "N/A") 
        }
        usage_data["retrieved_documents"].append(doc_data)
    
    # Save to file
    with open(f"{USAGE_TRACKING_DIR}/{query_id}.json", "w") as f:
        json.dump(usage_data, f, indent=2)
    
    return usage_data

def get_usage_statistics():
    """Analyze usage statistics from all tracked queries."""
    usage_files = [f for f in os.listdir(USAGE_TRACKING_DIR) if f.endswith(".json")]
    
    if not usage_files:
        return {"error": "No usage data available"}
    
    # Track document usage frequency
    doc_usage_count = {}
    disease_usage_count = {}
    total_queries = len(usage_files)
    
    for file in usage_files:
        with open(f"{USAGE_TRACKING_DIR}/{file}", "r") as f:
            data = json.load(f)
            
            for doc in data["retrieved_documents"]:
                doc_id = doc["id"]
                disease = doc["disease"]
                
                doc_usage_count[doc_id] = doc_usage_count.get(doc_id, 0) + 1
                disease_usage_count[disease] = disease_usage_count.get(disease, 0) + 1
    
    # Calculate percentages
    doc_usage_percentage = {k: (v / total_queries) * 100 for k, v in doc_usage_count.items()}
    disease_usage_percentage = {k: (v / total_queries) * 100 for k, v in disease_usage_count.items()}
    
    # Get total documents in the knowledge base
    try:
        # For Pinecone, we need to use the index directly
        pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(DISEASE_INDEX_NAME)
        stats = index.describe_index_stats()
        total_docs = stats.get('total_vector_count', 0)
    except:
        total_docs = 0
    
    # Calculate coverage
    used_docs = len(doc_usage_count)
    coverage_percentage = (used_docs / total_docs) * 100 if total_docs > 0 else 0
    
    return {
        "total_queries": total_queries,
        "total_documents_in_kb": total_docs,
        "unique_documents_used": used_docs,
        "coverage_percentage": coverage_percentage,
        "document_usage_frequency": doc_usage_count,
        "document_usage_percentage": doc_usage_percentage,
        "disease_usage_frequency": disease_usage_count,
        "disease_usage_percentage": disease_usage_percentage
    }


# --- HELPER FUNCTIONS ---
def format_docs(docs: List[Any]) -> str:
    """Format documents for the context, including their scores if available."""
    formatted_docs = []
    for d in docs:
        score = d.metadata.get('score', 'N/A')
        doc_string = f"[Score: {score}] {d.page_content}"
        formatted_docs.append(doc_string)
    return "\n\n---\n\n".join(formatted_docs)

format_docs_lambda = RunnableLambda(format_docs)

def create_tracked_retriever(vector_store: PineconeVectorStore, usage_key: str):
    """Creates a retriever that tracks usage."""
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 5}
    )
    
    def tracked_retriever_func(query: str, config: RunnableConfig):
        logger.info(f"Retrieving documents for {usage_key} with query: {query[:100]}...")
        try:
            # Use the score-based retriever
            docs_with_scores = retriever.invoke(query)
            
            logger.info(f"Retrieved {len(docs_with_scores)} documents for {usage_key} meeting score threshold.")
            
            query_id = str(uuid.uuid4())
            
            # Process docs
            processed_docs = []
            if docs_with_scores and isinstance(docs_with_scores[0], tuple):
                processed_docs = []
                for doc, score in docs_with_scores:
                    doc.metadata['score'] = score
                    processed_docs.append(doc)
            else:
                processed_docs = docs_with_scores

            # Store in app_state for the final response
            if "retrieved_context" not in app_state:
                app_state["retrieved_context"] = {}
            app_state["retrieved_context"][usage_key] = processed_docs
            
            # Track usage for analytics (only for primary disease)
            if usage_key == "disease":
                app_state["last_query_id"] = query_id
                track_document_usage(processed_docs, query_id)
                
            return processed_docs
        except Exception as e:
            logger.error(f"Error during document retrieval for {usage_key}: {e}")
            return []
    
    return RunnableLambda(tracked_retriever_func)


# --- FASTAPI LIFESPAN MANAGEMENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the FastAPI app."""
    logger.info("Starting C.A.R.E. Backend...")
    
    try:
        # 1. Initialize the LLM
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
            
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest", 
            temperature=0.2, 
            google_api_key=gemini_api_key
        )

        # 2. Initialize the embedding model
        logger.info(f"Loading embedding model from cache: {MODEL_CACHE_PATH}")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=MODEL_CACHE_PATH,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 3. Connect to Pinecone
        logger.info(f"Connecting to Pinecone...")
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in .env file.")
        pc = pinecone.Pinecone(api_key=pinecone_api_key)
        
        # 4. Connect to DISEASE Vector Store
        logger.info(f"Connecting to DISEASE index: {DISEASE_INDEX_NAME}")
        if DISEASE_INDEX_NAME not in pc.list_indexes().names():
            logger.error(f"Disease index '{DISEASE_INDEX_NAME}' not found.")
            raise ValueError(f"Pinecone index '{DISEASE_INDEX_NAME}' not found")
        
        disease_vector_store = PineconeVectorStore(
            index_name=DISEASE_INDEX_NAME,
            embedding=embeddings,
            text_key="text_content"
        )
        app_state["disease_vector_store"] = disease_vector_store
        logger.info("Connected to DISEASE index.")

        # 5. Connect to MEDICATION Vector Store
        logger.info(f"Connecting to MEDICATION index: {MED_INDEX_NAME}")
        if MED_INDEX_NAME not in pc.list_indexes().names():
            logger.error(f"Medication index '{MED_INDEX_NAME}' not found.")
            raise ValueError(f"Pinecone index '{MED_INDEX_NAME}' not found")

        med_vector_store = PineconeVectorStore(
            index_name=MED_INDEX_NAME,
            embedding=embeddings,
            text_key="text_content" 
        )
        app_state["med_vector_store"] = med_vector_store
        logger.info("Connected to MEDICATION index.")

        # 6. Define RAG Chain 1: Clinical Diagnosis
        logger.info("Initializing Clinical Diagnosis RAG chain...")
        
        # --- Prompt for Chain 1 ---
        diag_template = """
        You are a highly specialized medical AI, C.A.R.E. Your primary function is to analyze clinical reports and provide a comprehensive, evidence-based diagnosis in JSON format.
        You MUST use the provided "KNOWLEDGE BASE CONTEXT" to ground your answer.
        **Crucially, if a document in the context is clearly irrelevant to the PATIENT REPORT (e.g., context is about "Lead Poisoning" but the report is about "Food Poisoning"), you MUST ignore that specific document.**
        Synthesize information from relevant context, do not copy phrases. Create coherent sentences.
        Eliminate all redundancy. Ensure all lists contain distinct, meaningful items.
        If context is missing information, use your medical knowledge to fill gaps.

        Your response MUST NOT include a "suggested_medications" section. You will only provide the diagnosis and treatment plan.

        KNOWLEDGE BASE CONTEXT:
        {context}

        PATIENT REPORT:
        {report}

        JSON OUTPUT:
        """
        diag_prompt = ChatPromptTemplate.from_template(diag_template)
        
        # --- Schema for Chain 1 ---
        diag_output_schema = {
            "title": "ClinicalAnalysis",
            "description": "Comprehensive analysis of a clinical report with disease information.",
            "type": "object",
            "properties": {
                "primary_diagnosis": {
                    "type": "object",
                    "properties": {
                        "disease_name": {"type": "string"},
                        "confidence_score": {"type": "number", "description": "A score from 0.0 to 1.0"},
                        "summary": {"type": "string", "description": "A concise, well-written paragraph describing the condition"},
                        "clinical_presentation": {
                            "type": "object",
                            "properties": {
                                "common_symptoms": {"type": "array", "items": {"type": "string"}},
                                "key_findings": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "diagnostic_approach": {
                            "type": "object",
                            "properties": {
                                "initial_tests": {"type": "array", "items": {"type": "string"}},
                                "confirmatory_tests": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "treatment_plan": {
                            "type": "object",
                            "properties": {
                                "first_line": {"type": "array", "items": {"type": "string"}},
                                "advanced_options": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "prevention_strategies": {"type": "array", "items": {"type": "string"}},
                        "relevance_explanation": {"type": "string", "description": "Explanation of why this diagnosis fits the patient"}
                    },
                    "required": ["disease_name", "confidence_score", "summary", "clinical_presentation", "diagnostic_approach", "treatment_plan", "prevention_strategies", "relevance_explanation"]
                },
                "differential_diagnoses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "disease_name": {"type": "string"},
                            "confidence_score": {"type": "number"},
                            "summary": {"type": "string"},
                            "key_symptoms": {"type": "array", "items": {"type": "string"}},
                            "diagnostic_tests": {"type": "array", "items": {"type": "string"}},
                            "treatment_approach": {"type": "array", "items": {"type": "string"}},
                            "relevance_explanation": {"type": "string"}
                        },
                        "required": ["disease_name", "confidence_score", "summary", "key_symptoms", "diagnostic_tests", "treatment_approach", "relevance_explanation"]
                    }
                },
                "clinical_recommendations": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["primary_diagnosis", "differential_diagnoses", "clinical_recommendations"]
        }
        
        structured_diag_llm = llm.with_structured_output(diag_output_schema)
        
        disease_retriever = create_tracked_retriever(disease_vector_store, "disease")

        app_state["diag_rag_chain"] = (
            RunnableParallel(
                {
                    "context": disease_retriever | format_docs_lambda,
                    "report": RunnablePassthrough()
                }
            )
            | diag_prompt
            | structured_diag_llm
        )
        logger.info("Clinical Diagnosis RAG chain initialized successfully!")

        # 7. Define RAG Chain 2: Medication Suggestion
        logger.info("Initializing Medication Suggestion RAG chain...")

        # --- Prompt for Chain 2 ---
        med_template = """
        You are a clinical pharmacologist AI. Your task is to suggest first-line medications for a given disease, based on a KNOWLEDGE BASE of drug information.
        
        Analyze the "KNOWLEDGE BASE CONTEXT" and the "DISEASE NAME".
        You must only suggest relevant medications from the context.
        Filter out any irrelevant drugs (e.g., homeopathic remedies, sanitizers) and focus on common, appropriate prescription or OTC drugs.
        Provide up to 5 distinct medication suggestions.

        KNOWLEDGE BASE CONTEXT:
        {context}

        DISEASE NAME:
        {disease_name}

        JSON OUTPUT:
        """
        med_prompt = ChatPromptTemplate.from_template(med_template)
        
        # --- Schema for Chain 2 ---
        med_output_schema = {
            "title": "MedicationSuggestions",
            "description": "A list of suggested medications for a given disease.",
            "type": "object",
            "properties": {
                "suggested_medications": {
                    "type": "array",
                    "description": "A list of suggested medications for the primary diagnosis.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "drug_name": {"type": "string", "description": "The generic or brand name (e.g., Amoxicillin, Ibuprofen)"},
                            "dosage": {"type": "string", "description": "A typical dosage (e.g., '500 mg', '2 tablets')"},
                            "frequency": {"type": "string", "description": "How often to take it (e.g., 'every 12 hours', 'as needed')"}
                        },
                        "required": ["drug_name", "dosage", "frequency"]
                    }
                }
            },
            "required": ["suggested_medications"]
        }
        
        structured_med_llm = llm.with_structured_output(med_output_schema)
        
        med_retriever = create_tracked_retriever(med_vector_store, "medication")
        
        app_state["med_rag_chain"] = (
            RunnableParallel(
                {
                    "context": med_retriever | format_docs_lambda,
                    "disease_name": RunnablePassthrough()
                }
            )
            | med_prompt
            | structured_med_llm
        )
        logger.info("Medication Suggestion RAG chain initialized successfully!")
        
        logger.info("All chains initialized. Application startup complete.")

    except Exception as e:
        logger.error(f"Failed to initialize RAG chains: {e}", exc_info=True)
    
    yield
    
    logger.info("Shutting down C.A.R.E. Backend...")
    app_state.clear()


# --- API SETUP ---
app = FastAPI(
    title="C.A.R.E. Backend API",
    description="API for the Clinical Assistant for Reasoning and Evaluation",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https.emrpro.netlify.app",
        "https://emr-frontend1.onrender.com", 
        "http://localhost:3000", 
        "http://localhost:8501",
        "https://new-emr-pqlz.onrender.com"  # <-- ADDED
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class ReportRequest(BaseModel):
    report_text: str = Field(..., min_length=20, description="The unstructured clinical report text.")

class ResourceRequest(BaseModel):
    disease_name: str = Field(..., description="The name of the disease to search for resources")

# --- API ENDPOINTS ---

@app.post("/api/v1/clinical-analysis")
async def clinical_analysis(request: ReportRequest):
    """
    Analyzes a clinical report using the new "chain of chains" logic.
    """
    diag_chain = app_state.get("diag_rag_chain")
    med_chain = app_state.get("med_rag_chain")
    
    if not diag_chain or not med_chain:
        raise HTTPException(status_code=503, detail="RAG chains are not available. The service might be starting up or has encountered an error.")
    
    # Clear any context from previous calls
    app_state["retrieved_context"] = {}
    
    try:
        # --- Step 1: Run Diagnosis Chain ---
        logger.info("Invoking Diagnosis RAG chain...")
        diag_response = await diag_chain.ainvoke(request.report_text)
        logger.info("Diagnosis chain complete.")

        # --- FIX: Handle list-based (agent) response ---
        if isinstance(diag_response, list) and len(diag_response) > 0:
            logger.info("Response is a list, attempting to extract 'args'.")
            diag_response_data = diag_response[0].get('args', diag_response[0])
        elif isinstance(diag_response, dict):
            diag_response_data = diag_response
        else:
            logger.error(f"Unexpected diagnosis response type: {type(diag_response)}")
            raise HTTPException(status_code=500, detail="Failed to get a valid diagnosis from the AI model.")
        
        if not diag_response_data or "primary_diagnosis" not in diag_response_data:
            logger.error(f"Invalid diagnosis data: {diag_response_data}")
            raise HTTPException(status_code=500, detail="Failed to get a valid diagnosis from the AI model.")
            
        primary_disease = diag_response_data["primary_diagnosis"].get("disease_name", "Unknown")
        
        # --- Step 2: Run Medication Chain ---
        logger.info(f"Invoking Medication RAG chain for: {primary_disease}")
        med_response = await med_chain.ainvoke(primary_disease)
        logger.info("Medication chain complete.")
        
        # --- FIX: Handle list-based (agent) response for meds ---
        if isinstance(med_response, list) and len(med_response) > 0:
            logger.info("Medication response is a list, attempting to extract 'args'.")
            med_response_data = med_response[0].get('args', med_response[0])
        elif isinstance(med_response, dict):
            med_response_data = med_response
        else:
            logger.warning(f"Unexpected medication response type: {type(med_response)}")
            med_response_data = {"suggested_medications": []}
            
        if not med_response_data:
            logger.warning("Medication chain returned empty data.")
            med_response_data = {"suggested_medications": []}
            
        # --- Step 3: Combine Results ---
        logger.info("Combining diagnosis and medication results...")
        
        # Combine all data into the final structured response
        final_data = {
            **diag_response_data,  # Add all fields from diagnosis
            "suggested_medications": med_response_data.get("suggested_medications", []) # Add the new field
        }
        
        # Retrieve the context documents stored in app_state
        disease_docs = app_state.get("retrieved_context", {}).get("disease", [])
        med_docs = app_state.get("retrieved_context", {}).get("medication", [])

        result = {
            "success": True,
            "data": {
                **final_data, # Spread the combined data
                "knowledge_base_sources": [
                    {
                        "disease": doc.metadata.get("disease", "Unknown"),
                        "source_url": doc.metadata.get("source_url", ""),
                        "score": doc.metadata.get("score", "N/A")
                    } for doc in disease_docs
                ],
                "medication_sources": [
                    {
                        "brand_name": doc.metadata.get("brand_name", "Unknown"),
                        "generic_name": doc.metadata.get("generic_name", "Unknown"),
                        "score": doc.metadata.get("score", "N/A")
                    } for doc in med_docs
                ]
            },
            "metadata": {
                "query_id": app_state.get("last_query_id", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error during 'chain of chains' invocation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the report.")


@app.post("/api/v1/online-resources")
async def get_online_resources(request: ResourceRequest):
    """
    Search for online resources related to a specific disease using LangChain's search tools.
    """
    try:
        disease_name = request.disease_name
        logger.info(f"Searching for online resources for: {disease_name}")
        
        resources = search_online_resources(disease_name)
        
        logger.info(f"Found {len(resources)} resources for {disease_name}")
        
        return {
            "success": True,
            "disease": disease_name,
            "resources": resources,
            "count": len(resources)
        }
    except Exception as e:
        logger.error(f"Error fetching online resources: {e}", exc_info=True)
        fallback = create_fallback_resources(request.disease_name)
        return {
            "success": True,
            "disease": request.disease_name,
            "resources": fallback,
            "count": len(fallback),
            "note": "Using fallback resources due to search error"
        }

@app.post("/api/v1/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyzes a clinical report from an uploaded file (PDF or image).
    Extracts text and calls the main /clinical-analysis endpoint.
    """
    if "diag_rag_chain" not in app_state:
        raise HTTPException(status_code=503, detail="RAG chain is not available.")
    
    file_type = file.content_type
    if file_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg", "image/tiff"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or image file.")
    
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        try:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            
            extracted_text = ""
            if file_type == "application/pdf":
                import pdfplumber
                with pdfplumber.open(temp_file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\n"
            elif file_type.startswith("image/"):
                from PIL import Image
                import pytesseract
                image = Image.open(temp_file_path)
                extracted_text = pytesseract.image_to_string(image)
            
            if not extracted_text or len(extracted_text.strip()) < 20:
                raise HTTPException(status_code=400, detail="Could not extract sufficient text from the file.")
            
            logger.info("File processed, calling /clinical-analysis endpoint...")
            
            # --- THIS IS THE FIX ---
            # Instead of re-running the chain, just call the main endpoint
            # This ensures the full "chain of chains" logic is applied
            analysis_request = ReportRequest(report_text=extracted_text)
            response_data = await clinical_analysis(analysis_request)
            
            # Add extracted text to the response metadata
            if isinstance(response_data, dict) and "metadata" in response_data:
                 response_data["metadata"]["extracted_text"] = extracted_text
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error during file analysis: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/v1/usage-stats")
async def get_usage_stats():
    """Get statistics about knowledge base usage."""
    return get_usage_statistics()

@app.get("/api/v1/recent-queries")
async def get_recent_queries(limit: int = 10):
    """Get information about recent queries and their retrieved documents."""
    usage_files = [f for f in os.listdir(USAGE_TRACKING_DIR) if f.endswith(".json")]
    
    usage_files.sort(key=lambda x: os.path.getmtime(f"{USAGE_TRACKING_DIR}/{x}"), reverse=True)
    
    recent_queries = []
    for file in usage_files[:limit]:
        with open(f"{USAGE_TRACKING_DIR}/{file}", "r") as f:
            data = json.load(f)
            recent_queries.append(data)
    
    return {"recent_queries": recent_queries}

# --- DEBUG ENDPOINTS ---

def get_pinecone_stats(index_name: str):
    """Helper function to get stats for a Pinecone index."""
    try:
        pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(index_name)
        stats = index.describe_index_stats()

        serializable_stats = {}
        for key in ["dimension", "index_fullness", "metric", "total_vector_count", "vector_type"]:
            value = stats.get(key)
            if isinstance(value, (str, int, float, bool, type(None))):
                serializable_stats[key] = value

        namespaces = stats.get("namespaces", {})
        if isinstance(namespaces, dict):
            namespace_info = {}
            for ns_name, ns_data in namespaces.items():
                if isinstance(ns_data, dict):
                    vector_count = ns_data.get("vector_count")
                    if isinstance(vector_count, (int, float)):
                        namespace_info[ns_name] = {"vector_count": vector_count}
            serializable_stats["namespaces"] = namespace_info
        
        return serializable_stats
        
    except Exception as e:
        logger.error(f"Error debugging vector store {index_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def debug_retrieval_helper(vector_store: PineconeVectorStore, query: str):
    """Helper function to test retrieval for a given vector store."""
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5, "k": 5}
        )
        docs = await retriever.ainvoke(query)
        
        result = {
            "query": query,
            "retrieved_count": len(docs),
            "documents": []
        }
        
        for doc in docs:
            serializable_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    serializable_metadata[key] = value
            
            result["documents"].append({
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": serializable_metadata
            })
        
        return result
    except Exception as e:
        logger.error(f"Error in debug retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Disease Debug Endpoints ---
@app.get("/api/v1/debug/vector-store")
async def debug_vector_store():
    """Debug endpoint to check the DISEASE vector store."""
    return get_pinecone_stats(DISEASE_INDEX_NAME)

@app.post("/api/v1/debug/retrieval")
async def debug_retrieval(query: str = "chest pain"):
    """Debug endpoint to test DISEASE document retrieval."""
    vector_store = app_state.get("disease_vector_store")
    if not vector_store:
        raise HTTPException(status_code=503, detail="Disease vector store not initialized")
    return await debug_retrieval_helper(vector_store, query)

# --- Medication Debug Endpoints ---
@app.get("/api/v1/debug/vector-store-meds")
async def debug_vector_store_meds():
    """Debug endpoint to check the MEDICATION vector store."""
    return get_pinecone_stats(MED_INDEX_NAME)

@app.post("/api/v1/debug/retrieval-meds")
async def debug_retrieval_meds(query: str = "medication for hypertension"):
    """Debug endpoint to test MEDICATION document retrieval."""
    vector_store = app_state.get("med_vector_store")
    if not vector_store:
        raise HTTPException(status_code=503, detail="Medication vector store not initialized")
    return await debug_retrieval_helper(vector_store, query)
