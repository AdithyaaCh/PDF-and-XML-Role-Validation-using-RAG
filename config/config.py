import os
from dotenv import load_dotenv

load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "role-comparison-index") 
PDF_CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", 1000))
PDF_CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP", 100))
ROLE_EXTRACTION_PROMPT = os.getenv(
    "ROLE_EXTRACTION_PROMPT",
    "List all the roles mentioned in the following document. "
    "Provide a comma-separated list of unique roles. If no roles are found, respond with 'None'."
)
FUZZY_MATCH_THRESHOLD = int(os.getenv("FUZZY_MATCH_THRESHOLD", 80))

