import os
import json
import pinecone
from pinecone import ServerlessSpec # Correct import for v6.0.0
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

# --- CONFIGURATION ---
DATA_FILE = 'drug_data.json' # Make sure your file is named this
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_CACHE_PATH = '.model_cache'
BATCH_SIZE = 100
PINECONE_INDEX_NAME = 'care-meds' # The new index for medications

def create_document_text(entry):
    """Combines all relevant fields from a drug entry into a single text block."""
    try:
        text_parts = []
        
        brand_name = entry.get('brand_name')
        generic_name = entry.get('generic_name')
        
        if brand_name:
            text_parts.append(f"Brand Name: {brand_name}")
        if generic_name:
            text_parts.append(f"Generic Name: {generic_name}")

        # Add active ingredients
        active_ingredients = entry.get('active_ingredients', [])
        if active_ingredients:
            ingredients_list = [ing.get('name', 'Unknown') for ing in active_ingredients]
            text_parts.append(f"Active Ingredients: {', '.join(ingredients_list)}")

        # Add pharmaceutical classes
        pharm_class = entry.get('openfda', {}).get('pharm_class_epc', []) + \
                      entry.get('openfda', {}).get('pharm_class_pe', []) + \
                      entry.get('openfda', {}).get('pharm_class_cs', [])
        if pharm_class:
            text_parts.append(f"Pharmaceutical Classes: {', '.join(list(set(pharm_class)))}")

        # Add product type
        product_type = entry.get('product_type')
        if product_type:
            text_parts.append(f"Product Type: {product_type}")

        return ". ".join(text_parts)
    except Exception as e:
        print(f"Warning: Error processing entry {entry.get('product_ndc')}: {e}")
        return ""

def main():
    """Main function to ingest medication data into Pinecone."""
    print(f"--- Starting Medication Data Ingestion for Pinecone index: '{PINECONE_INDEX_NAME}' ---")
    load_dotenv()

    # 1. Load the source data
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: Data file '{DATA_FILE}' not found.")
        return

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract the 'results' list from the JSON
    drug_entries = data.get('results', [])
    if not drug_entries:
        print("❌ Error: 'results' key not found in JSON or is empty.")
        return
        
    print(f"✅ Successfully loaded {len(drug_entries)} drug entries from '{DATA_FILE}'.")

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
        print(f"⚠️ Warning: Pinecone index '{PINECONE_INDEX_NAME}' not found. Creating it...")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,  # Dimension for all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"✅ Successfully created index '{PINECONE_INDEX_NAME}'.")
        except Exception as e:
            print(f"❌ Error creating Pinecone index: {e}")
            return
    
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Check dimension
    stats = index.describe_index_stats()
    if stats.get('dimension') != 384:
        print(f"❌ Error: Pinecone index dimension is {stats.get('dimension')}, but model requires 384.")
        print("Please delete the index and recreate it with dimension 384.")
        return
        
    print(f"✅ Connected to index '{PINECONE_INDEX_NAME}'.")

    # 4. Prepare documents and generate embeddings
    print("⏳ Preparing documents...")
    documents_to_embed = []
    valid_entries = []
    
    for entry in drug_entries:
        doc_text = create_document_text(entry)
        if doc_text: # Only add if we successfully created text
            documents_to_embed.append(doc_text)
            valid_entries.append(entry)

    print(f"⏳ Generating {len(documents_to_embed)} embeddings... (This may take a while)")
    embeddings = model.encode(
        documents_to_embed,
        batch_size=32,
        show_progress_bar=True
    )

    # 5. Format for Pinecone and upsert in batches
    print("⏳ Formatting data for Pinecone upsert...")
    vectors_to_upsert = []
    for i, entry in enumerate(valid_entries):
        doc_id = entry.get('product_ndc', f"doc_{i}") # Use product_ndc as ID
        vector = embeddings[i].tolist()
        
        # Create metadata
        metadata = {
            "brand_name": entry.get('brand_name', ''),
            "generic_name": entry.get('generic_name', ''),
            "product_type": entry.get('product_type', ''),
            "dosage_form": entry.get('dosage_form', ''),
            "text_content": documents_to_embed[i] # Store the text for retrieval
        }
        
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
