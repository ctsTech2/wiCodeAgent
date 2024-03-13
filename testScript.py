# Import the function
from app import initialize_components

# Run the function
embeddings, index, vectorstore, retriever = initialize_components()

# Check the types or properties of the returned objects
print("Embeddings:", type(embeddings))
print("Pinecone Index:", type(index))
print("Vector Store:", type(vectorstore))
print("Retriever:", type(retriever))

# Optionally, perform some basic operations to ensure they work as expected
# For example, you might try retrieving a small sample using the retriever
# or checking the embeddings of a short text
