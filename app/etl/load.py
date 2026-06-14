import os
import re
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from app.utils.logger import setup_custom_logger

logger = setup_custom_logger(__name__)

def sanitize_index_name(name: str, fallback: str = "pdf-rag-etl") -> str:
    """Ensures the index name matches Pinecone requirements: lowercase alphanumeric and hyphens."""
    if not name or not name.strip():
        return fallback
    cleaned = re.sub(r'[^a-z0-9-]', '', name.lower().strip())
    if not cleaned:
        return fallback
    return cleaned

def load_to_pinecone(transformed_chunks: List[Dict[str, Any]], metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("Starting load process to Pinecone...")
    
    # Robust index name fallback
    raw_index_name = os.environ.get("PINECONE_INDEX_NAME", "")
    index_name = sanitize_index_name(raw_index_name)
    logger.info(f"Targeting Pinecone index: '{index_name}'")
    
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        logger.error("PINECONE_API_KEY environment variable is missing.")
        raise ValueError("Please set your PINECONE_API_KEY environment variable before running.")

    try:
        pc = Pinecone(api_key=api_key)
        dimension = 384 

        if not pc.has_index(index_name):
            logger.info(f"Index '{index_name}' not found. Creating a new serverless index...")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",  
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info(f"Index '{index_name}' created successfully.")
        else:
            logger.info(f"Index '{index_name}' already exists. Connecting...")

        index = pc.Index(index_name)

        vectors_to_upsert = []
        for i, chunk in enumerate(transformed_chunks):
            # Merge base chunk metadata with the explicitly provided metadata_list
            base_meta = chunk.get("metadata", {})
            extra_meta = metadata_list[i] if i < len(metadata_list) else {}
            
            merged_meta = {**base_meta, **extra_meta}
            merged_meta["text"] = chunk.get("text", "")

            vectors_to_upsert.append((
                chunk["chunk_id"],       
                chunk["embedding"],      
                merged_meta                 
            ))

        batch_size = 100
        logger.info(f"Starting upsert of {len(vectors_to_upsert)} vectors in batches of {batch_size}...")
        
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i : i + batch_size]
            index.upsert(vectors=batch)
            logger.info(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")

        logger.info("All data successfully loaded into Pinecone!")
        
        stats = index.describe_index_stats()
        return {
            "status": "success",
            "total_vectors_now_in_database": stats.get("total_vector_count", 0),
            "vectors_inserted_this_run": len(vectors_to_upsert)
        }

    except Exception as e:
        logger.exception(f"Error during Pinecone load stage: {str(e)}")
        raise e