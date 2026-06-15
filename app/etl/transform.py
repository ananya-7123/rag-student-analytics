import logging
import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, before_sleep_log
from google.genai import errors

logger = logging.getLogger(__name__)

client = genai.Client()

def should_retry_embedding(exception):
    if isinstance(exception, errors.ServerError):
        return True
    err_str = str(exception)
    if "503" in err_str or "504" in err_str or "UNAVAILABLE" in err_str:
        return True
    return False

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception(should_retry_embedding),
    before_sleep=before_sleep_log(logger, logging.INFO),
    reraise=True
)
def get_embedding(text: str):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={"output_dimensionality": 384}
    )
    return response.embeddings[0].values

def clean_text(text: str) -> str:
    """Removes excessive whitespace and PDF newline artifacts."""
    text = re.sub(r'\n+', ' ', text)  
    text = re.sub(r'\s+', ' ', text)  
    return text.strip()

def transform_data(extracted_pages: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 500) -> List[Dict[str, Any]]:
    """
    Cleans, chunks, and embeds the extracted PDF text in optimized batches.
    Defaults adjusted to 1000/500 for better context retention in large documents.
    """
    logger.info("Starting transformation: Cleaning, Chunking, and Embedding...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    all_text_chunks = []
    all_chunk_metadata = []

    try:
        # Step A: Chunking Phase
        for page in extracted_pages:
            raw_text = page.get("text", "")
            base_metadata = page.get("metadata", {})
            
            cleaned_text = clean_text(raw_text)
            if not cleaned_text:
                continue

            chunks = text_splitter.split_text(cleaned_text)

            for chunk in chunks:
                all_text_chunks.append(chunk)
                all_chunk_metadata.append({
                    "source": base_metadata.get("source", "unknown"),
                    "page_number": base_metadata.get("page_number", -1)
                })

        logger.info(f"Generated {len(all_text_chunks)} text chunks. Starting batch embedding...")

        # Step B: Batch Embedding Phase 
        embeddings = [get_embedding(chunk) for chunk in all_text_chunks]

        # Step C: Stitch it all together into the final payload
        transformed_chunks = []
        for i, (chunk, metadata, embedding) in enumerate(zip(all_text_chunks, all_chunk_metadata, embeddings)):
            chunk_payload = {
                "chunk_id": f"{metadata['source']}_page_{metadata['page_number']}_chunk_{i}",
                "text": chunk,
                "embedding": embedding,
                "metadata": metadata
            }
            transformed_chunks.append(chunk_payload)

        logger.info(f"Transformation complete. Returning {len(transformed_chunks)} embedded chunks.")
        return transformed_chunks

    except Exception as e:
        logger.exception(f"Error during transformation: {str(e)}")
        raise e