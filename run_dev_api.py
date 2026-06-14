# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()  # Loads PINECONE_API_KEY and OPENAI_API_KEY from .env 

import os
import shutil
import logging
import uuid  

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, HTTPException

# pyrefly: ignore [missing-import]
import uvicorn
from pydantic import BaseModel

from app.api.endpoints import router as api_router

# ETL Components
from app.config.logging_config import setup_logging
from app.etl.extract import extract_text_from_pdf
from app.etl.transform import transform_data
from app.etl.load import load_to_pinecone

# RAG Components
from app.rag.retrieve import retrieve_context
from app.rag.generate import generate_answer

# Initialize centralized logging configuration
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Student Success Analytics Platform",
    description="Swagger interface to test ETL ingestion and run live RAG document queries.",
    version="1.0.0"
)

# Attach our fresh modular endpoint routes
app.include_router(api_router, prefix="/api/v1")

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


# Schema for inbound user text questions
class QueryRequest(BaseModel):
    query: str
    model: str = "gemini-2.5-flash"  

    

#          ETL INGESTION ENDPOINTS ~~~~~~~~~~~~~~~~~~~


@app.post("/etl/extract", tags=["ETL Stages"])
async def test_extraction_endpoint(file: UploadFile = File(...)):
    """Upload a PDF to test text and structured table extraction."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are supported.")

    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_file_path = os.path.join(TEMP_DIR, unique_filename)
    
    try:
        logger.info(f"Receiving file for extraction via API: {file.filename}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_data = extract_text_from_pdf(temp_file_path)
        return {"status": "success", "file_processed": file.filename, "total_pages": len(extracted_data), "data": extracted_data}
    except Exception as e:
        logger.error(f"API Extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/etl/transform", tags=["ETL Stages"])
async def test_transform_endpoint(file: UploadFile = File(...)):
    """Upload a PDF to test extraction and text chunk recursive splitting + embeddings."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are supported.")

    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_file_path = os.path.join(TEMP_DIR, unique_filename)
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_data = extract_text_from_pdf(temp_file_path)
        transformed_data = transform_data(extracted_data)
        return {"status": "success", "total_chunks_generated": len(transformed_data), "sample_chunk": transformed_data[0] if transformed_data else None}
    except Exception as e:
        logger.error(f"API Transform failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transform error: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/etl/run-full-pipeline", tags=["Complete Ingestion Pipeline"])
async def run_full_etl_pipeline(file: UploadFile = File(...)):
    """Extracts, transforms, and loads an entire document into Pinecone vector cloud storage."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are supported.")

    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_file_path = os.path.join(TEMP_DIR, unique_filename)
    
    try:
        logger.info(f"--- STARTING FULL PIPELINE FOR: {file.filename} ---")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_data = extract_text_from_pdf(temp_file_path)
        transformed_data = transform_data(extracted_data)
        
        # Pass an empty metadata dictionary for each chunk since this test route doesn't gather subject tags
        metadata_list = [{}] * len(transformed_data)
        load_results = load_to_pinecone(transformed_data, metadata_list)
        
        logger.info(f"--- PIPELINE SUCCESS FOR: {file.filename} ---")
        return {
            "status": "success",
            "pipeline_summary": {
                "file_processed": file.filename,
                "pages_extracted": len(extracted_data),
                "chunks_embedded": len(transformed_data),
                "database_stats": load_results
            }
        }
    except Exception as e:
        logger.error(f"Full pipeline execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)



#             RAG QUERY LAYER


@app.post("/rag/query", tags=["RAG Query Layer"])
def query_document_system(request: QueryRequest):
    """
    Accepts a natural language question, retrieves matching text vectors 
    from Pinecone, and synthesizes a grounded answer using OpenAI.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
        
    try:
        logger.info(f"--- INBOUND RAG QUERY: '{request.query}' ---")
        
        # 1. Step 1: Retrieve context chunks
        retrieved_chunks = retrieve_context(query=request.query, top_k=5, index_name="pdf-rag-etl")
        
        if not retrieved_chunks:
            return {
                "answer": "I could not find any relevant information in the database context.",
                "citations": []
            }
            
        # 2. Step 2: Synthesize answer with OpenAI
        result = generate_answer(
            query=request.query, 
            retrieved_chunks=retrieved_chunks, 
            model=request.model
        )
        
        logger.info("--- RAG GENERATION CYCLE COMPLETE ---")
        return result
        
    except Exception as e:
        logger.error(f"RAG query layer failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG Pipeline Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("run_dev_api:app", host="127.0.0.1", port=8000, reload=True)