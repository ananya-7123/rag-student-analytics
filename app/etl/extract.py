import fitz  # PyMuPDF
import logging
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)

def convert_to_markdown_table(table_data: List[List[str]]) -> str:
    """
    Converts a 2D list (rows and columns) extracted from a PDF into a clean Markdown table.
    LLMs understand Markdown tables perfectly.
    """
    if not table_data or not table_data[0]:
        return ""

    markdown_lines = []
    
    # 1. Process Header Row
    headers = [str(cell).strip() if cell else "" for cell in table_data[0]]
    markdown_lines.append("| " + " | ".join(headers) + " |")
    
    # 2. Process Separator Row (e.g., | --- | --- |)
    separators = ["---"] * len(headers)
    markdown_lines.append("| " + " | ".join(separators) + " |")
    
    # 3. Process Data Rows
    for row in table_data[1:]:
        # Ensure row lengths match header lengths to prevent broken grids
        cleaned_row = [str(cell).strip() if cell else "" for cell in row]
        if len(cleaned_row) < len(headers):
            cleaned_row += [""] * (len(headers) - len(cleaned_row))
        elif len(cleaned_row) > len(headers):
            cleaned_row = cleaned_row[:len(headers)]
            
        markdown_lines.append("| " + " | ".join(cleaned_row) + " |")
        
    return "\n".join(markdown_lines) + "\n"


def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Opens a PDF and extracts both standard text and structured tables page-by-page.
    """
    logger.info(f"Opening PDF for advanced extraction: {file_path}")
    extracted_data = []

    try:
        doc = fitz.open(file_path)
        source_name = os.path.basename(file_path)

        for page_num, page in enumerate(doc, start=1):
            # Step A: Extract standard page text
            page_text = page.get_text("text") or ""
            
            # Step B: Identify and extract structured tables
            table_markdown_blocks = []
            try:
                tables = page.find_tables()
                if tables:
                    logger.info(f"Found {len(tables.tables)} table(s) on Page {page_num}")
                    for table in tables:
                        raw_table_data = table.extract()
                        if raw_table_data:
                            md_table = convert_to_markdown_table(raw_table_data)
                            table_markdown_blocks.append(md_table)
            except Exception as table_err:
                # If table parsing fails for a weirdly formatted page, log it but don't crash the pipeline
                logger.warning(f"Skipped table parsing on Page {page_num} due to: {str(table_err)}")

            # Step C: Combine text and markdown tables cleanly
            final_combined_text = page_text
            if table_markdown_blocks:
                final_combined_text += "\n\n### Extracted Structural Data:\n" + "\n".join(table_markdown_blocks)

            # Only append if the page actually contains information
            if final_combined_text.strip():
                extracted_data.append({
                    "text": final_combined_text,
                    "metadata": {
                        "source": source_name,
                        "page_number": page_num
                    }
                })

        logger.info(f"Successfully extracted data from {len(extracted_data)} pages.")
        return extracted_data

    except Exception as e:
        logger.exception(f"Critical error during PDF extraction: {str(e)}")
        raise e
    

    finally:
        if 'doc' in locals():
            doc.close()