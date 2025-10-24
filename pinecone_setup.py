import os
import pinecone
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define index name (matching your index name)
INDEX_NAME = "care"

def init_pinecone(api_key):
    """Initialize Pinecone with API key"""
    pc = pinecone.Pinecone(api_key=api_key)
    return pc

def create_pinecone_index(pc):
    """Create Pinecone index if it doesn't exist"""
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,  # BAAI/bge-m3 embedding dimension
            metric="cosine"
        )
    return pc.Index(INDEX_NAME)

def ingest_data_to_pinecone(data_file='mayo_keywords_final.json'):
    """Ingest data to Pinecone"""
    # Load the embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('BAAI/bge-m3')
    
    # Load the data
    print(f"Loading data from {data_file}...")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize Pinecone
    api_key = os.environ.get("PINECONE_API_KEY")
    pc = init_pinecone(api_key)
    
    # Connect to Pinecone
    print("Connecting to Pinecone...")
    index = create_pinecone_index(pc)
    
    # Process and upsert data
    batch_size = 50  # Adjust based on your needs
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches", total=total_batches):
        batch = data[i:i+batch_size]
        ids = []
        embeddings = []
        metadatas = []
        
        for j, item in enumerate(batch):
            # Create a combined text for embedding
            text = f"Disease: {item.get('disease', '')}. "
            text += f"Symptoms: {item.get('symptoms_causes', '')}. "
            text += f"Diagnosis: {item.get('diagnosis', '')}. "
            text += f"Tests: {item.get('tests', '')}. "
            text += f"Treatment: {item.get('treatment', '')}. "
            text += f"Prevention: {item.get('prevention', '')}."
            
            # Generate embedding
            embedding = model.encode(text)
            
            # Prepare data
            ids.append(f"doc_{i+j}")
            embeddings.append(embedding.tolist())
            metadatas.append({
                "disease": item.get('disease', ''),
                "symptoms_causes": item.get('symptoms_causes', ''),
                "diagnosis": item.get('diagnosis', ''),
                "tests": item.get('tests', ''),
                "treatment": item.get('treatment', ''),
                "prevention": item.get('prevention', ''),
                "source_url": item.get('source_url', '')
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=list(zip(ids, embeddings, metadatas)))
        print(f"Batch {i//batch_size + 1}/{total_batches} completed")
    
    print(f"Successfully ingested {len(data)} documents to Pinecone")
    return index

def main():
    """Main function to set up Pinecone and ingest data"""
    # Get API key from environment variables
    api_key = os.environ.get("PINECONE_API_KEY")
    
    print(f"API Key: {api_key[:10]}..." if api_key else "API Key not found")
    
    if not api_key:
        print("Please set PINECONE_API_KEY environment variable")
        return
    
    # Ingest data
    index = ingest_data_to_pinecone()
    
    # Print index stats
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")

if __name__ == "__main__":
    main()