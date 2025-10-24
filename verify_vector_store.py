# verify_vector_store.py
import chromadb

# Connect to the vector store
client = chromadb.PersistentClient(path="./clinical_db")
collection = client.get_or_create_collection(name="diseases")

# Check the document count
doc_count = collection.count()
print(f"Document count in collection: {doc_count}")

if doc_count > 0:
    # Get a sample of documents
    sample = collection.get(limit=3, include=["metadatas", "documents"])
    
    print("\nSample documents:")
    for i in range(len(sample["ids"])):
        print(f"\nDocument ID: {sample['ids'][i]}")
        print(f"Metadata: {sample['metadatas'][i]}")
        print(f"Content preview: {sample['documents'][i][:200]}...")
else:
    print("No documents found in the collection.")