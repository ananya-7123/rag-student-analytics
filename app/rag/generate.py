import os
import logging
from typing import List, Dict, Any
from groq import Groq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a helpful assistant. 
Answer the user's question using ONLY the provided text context. 
If the answer isn't in the context, say "I don't know".
"""

def generate_answer(query: str, retrieved_chunks: List[Dict[str, Any]], model: str = "llama-3.3-70b-versatile") -> Dict[str, Any]:
    """
    Takes text chunks from Pinecone, packages them up, and uses 
    Groq to generate a factual, source-grounded response.
    """
    logger.info(f"Generating answer for query: '{query}' using Groq model: {model}")
    
    # 1. Glue retrieved chunks into one background note block
    context_text = ""
    for chunk in retrieved_chunks:
        context_text += f"\n[Source: {chunk['source']}, Page: {chunk['page_number']}]\n"
        context_text += f"Text: {chunk['text']}\n"

    # 2. Package user question and context notes together
    user_payload = f"Context Material:\n{context_text}\n\nQuestion:\n{query}"
    
    try:
        # 3. Initialize the Groq client
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # 4. Call the Groq chat completions API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload}
            ],
            temperature=0.2
        )
        
        # 5. Build clean citation mapping for output
        sources = [{"source": c["source"], "page_number": c["page_number"]} for c in retrieved_chunks]
        
        return {
            "answer": response.choices[0].message.content,
            "citations": sources
        }
        
    except Exception as e:
        logger.error(f"Groq API call failed: {str(e)}")
        raise e