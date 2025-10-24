import os
import json
import pinecone
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

# --- CONFIGURATION ---
DATA_FILE = 'mayo_keywords_final.json'
# --- THIS IS THE FIX ---
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_CACHE_PATH = '.model_cache'
BATCH_SIZE = 100
# --- THIS IS THE FIX ---
PINECONE_INDEX_NAME = 'care-mini' 

def create_document_text(entry):
    """Combines all relevant keywords from a disease entry into a single text block."""
    fields_to_include = [
        "symptoms_causes", "diagnosis", "tests", "treatment", "prevention"
    ]
    
    text_parts = [f"Disease: {entry.get('disease', '')}"]
    
    for field in fields_to_include:
        keywords = entry.get(field)
        if isinstance(keywords, list) and keywords:
            text_parts.append(f"{field.replace('_', ' ').title()}: {', '.join(keywords)}")
        elif isinstance(keywords, str) and keywords:
             text_parts.append(f"{field.replace('_', ' ').title()}: {keywords}")

    return ". ".join(text_parts)

def main():
    """Main function to ingest data into Pinecone."""
    print(f"--- Starting Data Ingestion for Pinecone index: '{PINECONE_INDEX_NAME}' ---")
    load_dotenv()

    # 1. Load the source data
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: Data file '{DATA_FILE}' not found.")
        return

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Successfully loaded {len(data)} entries from '{DATA_FILE}'.")

    # 2. Load the embedding model
    print(f"⏳ Loading embedding model ('{EMBEDDING_MODEL}')...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODEL_CACHE_PATH)
        print("✅ Embedding model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load embedding model. Error: {e}")
        return

    # 3. Connect to Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("❌ Error: PINECONE_API_KEY not found in .env file.")
        return

    print("⏳ Connecting to Pinecone...")
    pc = pinecone.Pinecone(api_key=pinecone_api_key)
    
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"❌ Error: Pinecone index '{PINECONE_INDEX_NAME}' not found.")
        print("Please create the index in your Pinecone dashboard with the following settings:")
        print(f"  - Name: {PINECONE_INDEX_NAME}")
        print(f"  - Dimension: 384 (for {EMBEDDING_MODEL})")
        print(f"  - Metric: cosine")
        return
    
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # --- Check dimension ---
    stats = index.describe_index_stats()
    if stats.get('dimension') != 384:
        print(f"❌ Error: Pinecone index dimension is {stats.get('dimension')}, but model requires 384.")
        print("Please delete the index and recreate it with dimension 384.")
        return
        
    print(f"✅ Connected to index '{PINECONE_INDEX_NAME}'.")
    print(stats)

    # 4. Prepare documents and generate embeddings
    print("⏳ Preparing documents...")
    documents_to_embed = [create_document_text(entry) for entry in data]
    
    print(f"⏳ Generating {len(documents_to_embed)} embeddings... (This may take a while)")
    embeddings = model.encode(
        documents_to_embed,
        batch_size=32,
        show_progress_bar=True
    )

    # 5. Format for Pinecone and upsert in batches
    print("⏳ Formatting data for Pinecone upsert...")
    vectors_to_upsert = []
    for i, entry in enumerate(data):
        doc_id = f"doc_{i}"
        vector = embeddings[i].tolist()
        
        metadata = {k: (v if v is not None else "") for k, v in entry.items()}
        metadata["text_content"] = documents_to_embed[i]
        
        vectors_to_upsert.append((doc_id, vector, metadata))

    print(f"⏳ Upserting {len(vectors_to_upsert)} vectors in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, len(vectors_to_upsert), BATCH_SIZE)):
        batch = vectors_to_upsert[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)

    print("\n✅ Verification:")
    stats = index.describe_index_stats()
    print(stats)
    print(f"🎉 --- Data Ingestion Complete ---")
    print(f"Successfully added {stats.get('total_vector_count', 'N/A')} vectors to the '{PINECONE_INDEX_NAME}' index.")

if __name__ == '__main__':
    main()