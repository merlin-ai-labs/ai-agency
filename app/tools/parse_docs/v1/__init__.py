"""Parse Documents tool v1.

Extracts text and structured data from uploaded documents (PDF, Word, etc.).

TODO:
- Implement PDF parsing (PyPDF2, pdfplumber, or Vertex AI Document AI)
- Implement Word parsing (python-docx)
- Extract entities, tables, images
- Handle multi-page documents
- Return structured output
"""

from typing import Any, Dict, List

import structlog

logger = structlog.get_logger()


async def parse_documents(document_urls: list[str]) -> dict[str, Any]:
    """
    Parse documents and extract structured data.

    Args:
        document_urls: List of GCS URLs or local paths to documents

    Returns:
        Dict with extracted text, entities, tables, etc.

    TODO:
    - Download documents from GCS
    - Detect document type
    - Parse with appropriate library
    - Extract structured data
    - Return normalized output
    """
    logger.info("parse_documents.v1", urls=document_urls)

    # Stub implementation
    result = {
        "documents": [],
        "total_pages": 0,
        "extracted_text": "",
        "entities": [],
        "tables": [],
    }

    return result
