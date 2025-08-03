
import fitz  
from src.utils import chunk_text
from src.gemini_client import GeminiClient
from src.pinecone_client import PineconeClient
from config.config import PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP, ROLE_EXTRACTION_PROMPT
import uuid 
from pinecone.exceptions import NotFoundException


#This class handles the extraction of text and tables from PDF files, processes them into chunks, generates embeddings using Google's Gemini model, 
#and interacts with Pinecone for vector storage and retrieval.

class RAGPDFExtractor:
    def __init__(self):
        self.gemini_client = GeminiClient()
        self.pinecone_client = PineconeClient()

    def _extract_text_and_tables_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts text content from a PDF file, including tables.
        Handles text and table extraction. For images, OCR might be needed
        as a pre-processing step if they contain relevant text.
        """
        full_text = []
        try:
            pdf_document = fitz.open(pdf_path)
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text_blocks = page.get_text("blocks")
                for block in text_blocks:
                    # block[4] is the text content
                    full_text.append(block[4].strip())

                # Extract tables which an conrain roles.
                tables = page.find_tables()
                for table in tables:
                    table_rows = []
                    for row_data in table.extract():
                        table_rows.append(" | ".join([cell if cell is not None else "" for cell in row_data]))
                    table_str = "\n".join(table_rows)
                    full_text.append(f"\n--- DATA TABLE WITH ROLES AND COUNTS ---\n{table_str.strip()}\n--- END OF TABLE DATA ---")
            pdf_document.close()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")  
        full_document_text = "\n\n".join(full_text)
        print(f"\n--- DEBUG: Full Extracted PDF Text (including tables) ---\n{full_document_text}\n---------------------------------------------------\n")
        return full_document_text


    def process_pdf(self, pdf_path: str, pdf_id: str):
        """Processes the PDF: extracts text, chunks, embeds, and upserts to Pinecone."""
        text = self._extract_text_and_tables_from_pdf(pdf_path)
        if not text.strip():
            print(f"No content extracted from {pdf_path}. Skipping indexing.")
            return

        chunks = chunk_text(text, PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP)
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            embedding = self.gemini_client.embed_text(chunk)
            if embedding:
                vector_id = f"{pdf_id}-{uuid.uuid4().hex}"
                vectors_to_upsert.append((vector_id, embedding, {"pdf_id": pdf_id, "chunk_index": i, "content": chunk}))
        
        if vectors_to_upsert:
            self.pinecone_client.upsert_vectors(vectors=vectors_to_upsert)
            print(f"Processed and indexed {len(chunks)} chunks from {pdf_path}")
        else:
            print(f"No embeddings generated for {pdf_path}. Skipping indexing.")

    def extract_roles_from_pdf(self, pdf_path: str) -> list:
        """Extracts roles from the PDF using Gemini LLM."""
        extracted_text = self._extract_text_and_tables_from_pdf(pdf_path)
        if not extracted_text.strip():
            print(f"No content extracted from {pdf_path} for role extraction.")
            return []

        prompt = f"{ROLE_EXTRACTION_PROMPT}\n\nDocument Content:\n{extracted_text}"
        print("Sending prompt to Gemini for role extraction...")
        raw_roles_str = self.gemini_client.generate_text(prompt)

        roles = []
        if raw_roles_str and raw_roles_str.lower() != 'none':
            roles = [role.strip() for role in raw_roles_str.split(',') if role.strip()]
            return list(set(roles))
        elif raw_roles_str.lower() == 'none':
            print("Gemini reported no roles found in the document.")
        else:
            print("Gemini returned empty or unparseable response for roles.")
        return []

    def clear_pdf_data(self, pdf_id: str):
        """Deletes all vectors associated with a specific PDF ID from Pinecone."""
        try:
            self.pinecone_client.index.delete(filter={"pdf_id": {"$eq": pdf_id}})
            print(f"Deleted data for PDF ID: {pdf_id} from Pinecone.")
        except NotFoundException:
            print(f"No existing data found for PDF ID: {pdf_id} in Pinecone. Skipping delete.")
        except Exception as e:
            print(f"An unexpected error occurred while deleting data for PDF ID {pdf_id}: {e}")

    def query_pdf_for_roles_from_pinecone(self, pdf_path: str, query: str) -> str:
        """
        Queries the processed PDF in Pinecone for `query`. Returns LLM answer.
        """
        # 1. Embed the user query
        query_embedding = self.gemini_client.embed_text(query)
        if not query_embedding:
            return "Could not generate query embedding."

        # 2. Run the vector search
        raw_results = self.pinecone_client.query_vectors(query_embedding, top_k=20)

        # 3. Normalize to a list of matches
        if hasattr(raw_results, "matches"):
            matches = raw_results.matches
        elif isinstance(raw_results, list):
            matches = raw_results
        else:
            return "Unexpected response format from Pinecone."

        print(f"\n--- DEBUG: Raw Pinecone Query Results (top {len(matches)} matches) ---")
        for m in matches:
            content_snip = m.metadata.get("content", "")[:100]
            print(f"  ID: {m.id}, Score: {m.score:.4f}, Content: {content_snip}...")
        print("---------------------------------------------------\n")

        if not matches:
            return "No relevant information found in PDF."

        # 4. Pull out the text to send to the LLM
        contexts = []
        for m in matches:
            txt = m.metadata.get("content")
            if txt:
                contexts.append(txt)
            else:
                print(f"Warning: no content for vector ID {m.id}")

        if not contexts:
            return "No retrievable content in the matched chunks."

        full_context = "\n\n".join(contexts)
        print("\n--- DEBUG: Context sent to LLM for general query ---")
        print(full_context)
        print("---------------------------------------------------\n")

        # 5. Build a prompt that tries to lean on tables if the question is about counts
        if any(k in query.lower() for k in ("table", "count", "number of", "how many")):
            prompt = (
                f"Based on the following document excerpts, specifically focus on any tables or structured lists "
                f"to answer: '{query}'. If exact numbers are provided, use them. If no relevant table is found, say so.\n\n"
                f"Document Excerpts:\n{full_context}\n\nAnswer:"
            )
        else:
            prompt = (
                f"Based on the following document excerpts, answer: '{query}'.\n\n"
                f"Document Excerpts:\n{full_context}\n\nAnswer:"
            )

        # 6. Let Gemini answer
        return self.gemini_client.generate_text(prompt)
