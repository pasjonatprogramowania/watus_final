import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.vectordb import initialize_vector_db, search_vector_db
import json

def test_vector_search():
    # Initialize the vector database
    _, collection = initialize_vector_db()
    
    if not collection:
        print("Failed to initialize vector database")
        return
    
    # Get collection count to verify data
    collection_count = collection.count()
    print(f"Total documents in collection: {collection_count}")
    
    if collection_count == 0:
        print("Warning: Vector database is empty. Run vectordb.main() first to populate.")
        return
    
    # Test queries
    test_queries = [
        "Michalina informatyka",  # Based on the text in the sample file
        "rozmowa o studiach",
        "przedstawienie się"
    ]
    
    for query in test_queries:
        print(f"\nVector Search Query: '{query}'")
        results = search_vector_db(collection, query, n_results=3)
        
        if results:
            print("Search Results:")
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]), 1):
                print(f"\nResult {i}:")
                try:
                    # Try to parse and pretty print the document
                    parsed_doc = json.loads(doc)
                    print(f"Document: {json.dumps(parsed_doc, indent=2, ensure_ascii=False)}")
                except json.JSONDecodeError:
                    print(f"Document: {doc}")
                
                print(f"Metadata: {metadata}")
                print(f"Distance: {distance}")
        else:
            print("No results found")

if __name__ == "__main__":
    test_vector_search()
