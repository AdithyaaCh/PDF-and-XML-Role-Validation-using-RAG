import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from config.config import GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

#Let us make a GeminiClient class that can generate text and embeddings using Google's Gemini model.
class GeminiClient:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model = genai.GenerativeModel(
            model_name,
            safety_settings=[
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            ]
        )

    # Generates text using the Gemini model based on a given prompt.
    # Returns the generated text or an empty string if an error occurs.
    def generate_text(self, prompt: str) -> str:
        """Generates text using the Gemini model."""
        try:
            response = self.model.generate_content(prompt)
            if response.candidates:
                return response.text
            else:
                print("Gemini API returned no candidates.")
                return ""
        except Exception as e:
            print(f"Error generating text with Gemini: {e}")
            return ""


    def embed_text(self, text: str) -> list:
        """Generates embeddings for the given text using Google's embedding model."""
        try:
            response = genai.embed_content(model="models/embedding-001", content=[text])
            if response and 'embedding' in response:
                embedding = response['embedding']
                if isinstance(embedding, list) and all(isinstance(x, (float, int)) for x in embedding):
                    return embedding
                elif isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], list):
                    return embedding[0]
                else:
                    print(f"Unexpected embedding format: {type(embedding)} - {embedding}")
                    return []
            else:
                print("Gemini Embedding API returned no embedding.")
                return []
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    # It handles different formats of the embedding response and returns a list of floats.
    # For instance , if the response is a list of lists, it flattens it to return a single list.
    # If the response is a single list, it returns that list directly.
    # If the response is empty or malformed, it returns an empty list.
    # If an error occurs during the API call, it prints the error message and returns an empty list.
    