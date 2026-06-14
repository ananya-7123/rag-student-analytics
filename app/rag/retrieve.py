import os
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

logger = logging.getLogger(__name__)

# Load the embedding model globally (Exactly the same one used in transform.py)
logger.info("Initializing embedding model for real-time retrieval (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_context(query: str, top_k: int = 5, index_name: str = "pdf-rag-etl") -> List[Dict[str, Any]]:
    """
    Takes a plain text question, converts it to a vector fingerprint, 
    and fetches the most relevant matching text chunks from the cloud.
    """
    logger.info(f"Retrieving top {top_k} context chunks for query: '{query}'")
    
    # 1. Verify the Pinecone API key is present
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        logger.error("PINECONE_API_KEY is missing from environment variables.")
        raise ValueError("Please set your PINECONE_API_KEY before running retrieval.")

    try:
        # 2. Turn the user's text question into an array of 384 numbers
        # This is semantic translation—matching meaning, not just keywords
        query_vector = embedding_model.encode(query).tolist()
        
        # 3. Connect to the Pinecone service and your specific index
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        
        # 4. Perform the mathematical similarity search in the cloud
        search_results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True  
        )
        
        # 5. Extract the raw text and source data out of the messy database output
        retrieved_chunks = []
        for match in search_results.get("matches", []):
            metadata = match.get("metadata", {})
            
            retrieved_chunks.append({
                "chunk_id": match.get("id"),
                "score": match.get("score"),  # The similarity confidence rating
                "text": metadata.get("text", ""),
                "source": metadata.get("source", "unknown"),
                "page_number": metadata.get("page_number", -1)
            })
            
        logger.info(f"Successfully retrieved {len(retrieved_chunks)} relevant matches from Pinecone.")
        return retrieved_chunks

    except Exception as e:
        logger.exception(f"Error during context retrieval stage: {str(e)}")
        raise e