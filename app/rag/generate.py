import os
import logging
from typing import List, Dict, Any
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a helpful assistant. 
Answer the user's question using ONLY the provided text context. 
If the answer isn't in the context, say "I don't know".
"""

def generate_answer(query: str, retrieved_chunks: List[Dict[str, Any]], model: str = "gemini-2.5-flash") -> Dict[str, Any]:
    """
    Takes text chunks from Pinecone, packages them up, and uses 
    Google Gemini to generate a factual, source-grounded response.
    """
    logger.info(f"Generating answer for query: '{query}' using Gemini model: {model}")
    
    # 1. Glue retrieved chunks into one background note block
    context_text = ""
    for chunk in retrieved_chunks:
        context_text += f"\n[Source: {chunk['source']}, Page: {chunk['page_number']}]\n"
        context_text += f"Text: {chunk['text']}\n"

    # 2. Package user question and context notes together
    user_payload = f"Context Material:\n{context_text}\n\nQuestion:\n{query}"
    
    try:
        # 3. Initialize the Gemini developer client
        # It automatically finds GEMINI_API_KEY inside your environment variables
        client = genai.Client()
        
        # 4. Call the Gemini API model
        response = client.models.generate_content(
            model=model,
            contents=user_payload,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2  # Keeps it strict and factual
            )
        )
        
        # 5. Build clean citation mapping for output
        sources = [{"source": c["source"], "page_number": c["page_number"]} for c in retrieved_chunks]
        
        return {
            "answer": response.text,
            "citations": sources
        }
        
    except Exception as e:
        logger.error(f"Gemini API call failed: {str(e)}")
        raise e