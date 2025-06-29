from pinecone import Pinecone, ServerlessSpec 
from pinecone.exceptions import PineconeApiException 
from config.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
import time 
from typing import Any

# PineconeClient class to handle Pinecone operations like creating an index, upserting vectors, querying, and deleting vectors.
# This class abstracts the complexity of interacting with Pinecone's API, providing a simple interface for vector operations.
class PineconeClient:
    def __init__(self, index_name=PINECONE_INDEX_NAME, dimension=768): 
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = index_name
        self.dimension = dimension
        self.index: Any = self._get_or_create_index()
    
    # This method checks if the index exists, creates it if not, and returns the index object.
    # It handles potential errors during index creation and readiness checks, providing informative messages.
    # If the index already exists, it checks if it's ready and returns the index object.
    # If the index creation fails, it raises an exception with a helpful message.
    # The method also handles API exceptions and provides feedback on the index status.
    # It uses the ServerlessSpec to specify the cloud and region for the index.
    # The dimension parameter specifies the dimensionality of the vectors to be stored in the index.
    # The index_name parameter allows customization of the index name, defaulting to PINECONE_INDEX_NAME from the config.
    # The method returns an Index object that can be used for further operations like upserting and querying vectors.
    def _get_or_create_index(self):
        try:
            existing_indexes = self.pc.list_indexes().names()
        except PineconeApiException as e:
            print(f"Error listing Pinecone indexes: {e}")
            print("Please check your Pinecone API key and network connection.")
            raise 

        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine", 
                    spec=ServerlessSpec(cloud="aws", region="us-east-1") 
                )
                print(f"Index '{self.index_name}' created. Waiting for it to be ready...")
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1) 
                print(f"Index '{self.index_name}' is ready.")
            except PineconeApiException as e:
                print(f"Error creating Pinecone index '{self.index_name}': {e}")
                print("This might be due to free-tier limits, an invalid API key, or a region issue.")
                raise 
        else:
            print(f"Pinecone index '{self.index_name}' already exists.")
            try:
                if not self.pc.describe_index(self.index_name).status['ready']:
                    print(f"Index '{self.index_name}' is not yet ready. Waiting...")
                    while not self.pc.describe_index(self.index_name).status['ready']:
                        time.sleep(1)
                    print(f"Index '{self.index_name}' is now ready.")
            except PineconeApiException as e:
                print(f"Error describing Pinecone index '{self.index_name}': {e}")
                print("Please check your Pinecone index status in the console.")
                raise 

        return self.pc.Index(self.index_name)
 
    # This method upserts a list of vectors to the Pinecone index.
    # It takes a list of vectors, where each vector is expected to be a dictionary with 'id', 'values', and optionally 'metadata'.
    # If the list is empty, it prints a message and returns without performing any operation.
    # It handles potential API exceptions during the upsert operation, providing informative error messages.
    # If the upsert is successful, it prints the number of vectors upserted.
    # The vectors should be in the format expected by Pinecone, typically a list of dictionaries with 'id' and 'values'.
    # The 'values' should be a list of floats representing the vector, and 'id' should be a unique identifier for each vector.
    # The method returns nothing, but it prints the result of the upsert operation.
    def upsert_vectors(self, vectors: list):
        """Upserts vectors to Pinecone."""
        if not vectors:
            print("No vectors to upsert.")
            return
        try:
            self.index.upsert(vectors=vectors)
            print(f"Upserted {len(vectors)} vectors to Pinecone index '{self.index_name}'.")
        except PineconeApiException as e:
            print(f"Error upserting vectors to Pinecone: {e}")
            print("Please ensure the index is ready and vector dimensions/format are correct.")
        except Exception as e:
            print(f"An unexpected error occurred during upsert: {e}")

    # This method queries the Pinecone index for vectors similar to the provided query_embedding.
    # It takes a query_embedding, which is a list of floats representing the vector to search for.
    # The top_k parameter specifies how many similar vectors to return, defaulting to 3.
    # It handles potential API exceptions during the query operation, providing informative error messages.
    # If the query is successful, it returns a list of matches, each containing the vector ID, score, and metadata.
    # The method returns an empty list if no matches are found or if an error occurs.
    # The query_embedding should match the dimensionality of the vectors stored in the index.

    def query_vectors(self, query_embedding: list, top_k: int = 3) -> list:
        """Queries Pinecone for similar vectors."""
        try:
            results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
            return results.matches
        except PineconeApiException as e:
            print(f"Error querying Pinecone: {e}")
            print("Please check your Pinecone API key, index status, and query parameters.")
            return []
        except Exception as e:
            print(f"An unexpected error occurred during query: {e}")
            return []

    # This method deletes all vectors from the Pinecone index.
    # It handles potential API exceptions during the delete operation, providing informative error messages.
    # If the delete operation is successful, it prints a confirmation message.

    def delete_all_vectors(self):
        """Deletes all vectors from the index."""
        try:
            self.index.delete(delete_all=True)
            print(f"All vectors deleted from index: {self.index_name}")
        except PineconeApiException as e:
            print(f"Error deleting all vectors from Pinecone: {e}")
            print("This can happen if the index is not found or other API issues.")
        except Exception as e:
            print(f"An unexpected error occurred during full index delete: {e}")