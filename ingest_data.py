import os
import json
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_FILE = 'mayo_keywords_final.json'
VECTOR_DB_PATH = 'clinical_db'
COLLECTION_NAME = 'diseases'
EMBEDDING_MODEL = 'BAAI/bge-m3'
BATCH_SIZE = 100
# Define a local cache folder to ensure the model is saved consistently
MODEL_CACHE_PATH = '.model_cache'

def create_document_text(entry):
    """Combines all relevant keywords from a disease entry into a single text block."""
    fields_to_include = [
        "symptoms_causes", "diagnosis", "tests", "treatment", "prevention"
    ]
    
    # Start with the disease name for strong context
    text_parts = [f"Disease: {entry.get('disease', '')}"]
    
    for field in fields_to_include:
        # Check for both list and string types for keywords
        keywords = entry.get(field)
        if isinstance(keywords, list) and keywords:
            text_parts.append(f"{field.replace('_', ' ').title()}: {', '.join(keywords)}")
        elif isinstance(keywords, str) and keywords:
             text_parts.append(f"{field.replace('_', ' ').title()}: {keywords}")

    return ". ".join(text_parts)

def main():
    """Main function to ingest data into the vector database."""
    print("--- Starting Data Ingestion for Project C.A.R.E. ---")

    # 1. Load the source data
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: Data file '{DATA_FILE}' not found. Please place it in the same directory.")
        return

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Successfully loaded {len(data)} entries from '{DATA_FILE}'.")

    # 2. Load the embedding model, using the dedicated cache folder
    print(f"⏳ Loading the embedding model ('{EMBEDDING_MODEL}')... This will download the model on the first run.")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODEL_CACHE_PATH)
        print("✅ Embedding model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load embedding model. Check your internet connection. Error: {e}")
        return

    # 3. Set up the vector database, ensuring a clean start
    print(f"⏳ Setting up vector database at '{VECTOR_DB_PATH}'...")
    if os.path.exists(VECTOR_DB_PATH):
        print(f"   - Found existing database. Deleting '{VECTOR_DB_PATH}' for a fresh build.")
        shutil.rmtree(VECTOR_DB_PATH)

    # Create the client with the same settings as in main.py
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    
    # Create collection with the same name as in main.py
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"✅ Vector database collection '{COLLECTION_NAME}' is ready.")

    # 4. Generate embeddings and add to the database in batches
    print(f"⏳ Generating embeddings and adding {len(data)} documents to the database...")
    
    batch_embeddings = model.encode(
        [create_document_text(entry) for entry in data],
        batch_size=32, # A good default for encoding
        show_progress_bar=True
    )
    
    # Add to ChromaDB in larger batches for efficiency
    with tqdm(total=len(data), desc="Adding to DB") as pbar:
        for i in range(0, len(data), BATCH_SIZE):
            batch_end = i + BATCH_SIZE
            batch_data = data[i:batch_end]
            
            batch_docs = [create_document_text(entry) for entry in batch_data]
            batch_metas = [{k: (v if v is not None else "") for k, v in entry.items()} for entry in batch_data]
            batch_ids = [f"doc_{j}" for j in range(i, i + len(batch_data))]
            
            collection.add(
                embeddings=batch_embeddings[i:batch_end].tolist(),
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            pbar.update(len(batch_data))

    # Verify the data was added
    doc_count = collection.count()
    print(f"\n✅ Verification: Collection now contains {doc_count} documents.")
    
    if doc_count == 0:
        print("❌ ERROR: No documents were added to the collection. Please check the ingestion process.")
        return
    
    print("\n🎉 --- Data Ingestion Complete ---")
    print(f"Successfully added {len(data)} documents to the '{COLLECTION_NAME}' collection.")

if __name__ == '__main__':
    main()