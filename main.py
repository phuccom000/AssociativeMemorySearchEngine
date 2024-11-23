from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class AssociativeMemoryStore:
    def __init__(self, embedding_dim):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model for embeddings
        self.index = faiss.IndexFlatL2(embedding_dim)  # FAISS index for similarity search
        self.data_store = []  # List to hold content and metadata

    def add_item(self, content, metadata=None):
        # Generate an embedding for the content
        embedding = self.model.encode(content)
        embedding = np.array([embedding], dtype=np.float32)  # Convert to NumPy array
        
        # Add to FAISS index and data store
        self.index.add(embedding)
        self.data_store.append({
            "content": content,
            "metadata": metadata,
            "embedding": embedding.tolist()
        })

    def suggest(self, prefix):
        # Find items starting with the given prefix
        suggestions = []
        for item in self.data_store:
            if item["content"].lower().startswith(prefix.lower()):
                suggestions.append(item["content"])
        return suggestions

    def search(self, query, top_k=5):
        # Generate embedding for the query
        query_embedding = self.model.encode(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Perform similarity search
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.data_store):  # Ensure valid index
                result = self.data_store[idx]
                result['distance'] = dist  # Add distance to result
                results.append(result)
        return results


# Initialize the store
store = AssociativeMemoryStore(embedding_dim=384)

# Add items to the store
store.add_item("A bright red apple is on the table.", metadata={"category": "fruit", "tags": ["apple", "red"]})
store.add_item("The quick brown fox jumps over the lazy dog.", metadata={"category": "animal", "tags": ["fox", "dog"]})
store.add_item("A beautiful sunset over the ocean.", metadata={"category": "nature", "tags": ["sunset", "ocean"]})

# Simulate a search bar
while True:
    query = input("\nEnter your query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    
    # Suggest completions if the input is incomplete
    suggestions = store.suggest(query)
    if suggestions:
        print("\nDid you mean:")
        for suggestion in suggestions:
            print(f" - {suggestion}")
    
    # Perform semantic search for the query
    results = store.search(query, top_k=3)
    print("\nSearch Results:")
    for result in results:
        print(f"Content: {result['content']}, Distance: {result['distance']:.4f}, Metadata: {result['metadata']}")
