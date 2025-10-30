"""FastAPI endpoints for Invoice Manager Agent.

Provides endpoints for:
- POST /api/v1/invoice-manager/run - Process invoice or query
- GET /api/v1/invoice-manager/capabilities - Get agent capabilities
- POST /api/v1/invoice-manager/search - Search invoices
- POST /api/v1/invoice-manager/upload - Upload invoice file
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import File, HTTPException, UploadFile, Query
from pydantic import BaseModel

from sqlmodel import Session

from app.core.decorators import log_execution, timeout
from app.core.langgraph_checkpointer import create_checkpointer_config
from app.db.base import get_session
from app.db.repositories.conversation_repository import ConversationRepository
from app.flows.invoice_manager.graph import create_invoice_manager_graph
from app.tools.sharepoint.v1 import search_invoices

logger = logging.getLogger(__name__)


# Request/Response models
class InvoiceManagerRunRequest(BaseModel):
    """Request to run invoice manager agent."""

    message: str
    tenant_id: str
    conversation_id: str | None = None
    invoice_file_path: str | None = None  # Path to uploaded invoice file
    provider: str | None = None  # LLM provider ("openai", "vertex", "mistral"). If None, uses settings default
    model: str | None = None  # LLM model name. If None, uses provider default


class InvoiceManagerRunResponse(BaseModel):
    """Response from invoice manager agent."""

    response: str
    conversation_id: str
    extracted_data: dict[str, Any] | None = None
    renamed_filename: str | None = None
    target_folder: str | None = None
    duplicate_check: dict[str, Any] | None = None
    upload_result: dict[str, Any] | None = None
    search_results: list[dict[str, Any]] | None = None
    error: str | None = None


class InvoiceSearchRequest(BaseModel):
    """Request to search invoices."""

    vendor_name: str | None = None
    date_from: str | None = None  # YYYY-MM-DD
    date_to: str | None = None  # YYYY-MM-DD
    amount_min: float | None = None
    amount_max: float | None = None
    invoice_number: str | None = None
    tenant_id: str
    limit: int = 50


class InvoiceSearchResponse(BaseModel):
    """Response from invoice search."""

    invoices: list[dict[str, Any]]
    total_count: int


class CapabilitiesResponse(BaseModel):
    """Response with agent capabilities."""

    capabilities: list[str]
    description: str


# Store uploaded files temporarily (in production, use proper storage)
_uploaded_files: dict[str, str] = {}

# File size limit: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024


async def upload_invoice_file(
    file: UploadFile = File(...),
    tenant_id: str = Query(..., description="Tenant identifier"),
) -> dict[str, str]:
    """Upload invoice file for processing.

    Args:
        file: Invoice file (PDF or JPEG, max 10MB)
        tenant_id: Tenant identifier

    Returns:
        {
            "file_id": str,
            "file_path": str,
            "message": str
        }
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".pdf", ".jpg", ".jpeg", ".png"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: .pdf, .jpg, .jpeg, .png",
        )

    # Read file content and validate size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {len(content)} bytes. Maximum: {MAX_FILE_SIZE} bytes (10MB)",
        )

    # Save file temporarily with path validation
    file_id = str(uuid4())
    temp_dir = Path(tempfile.gettempdir()).resolve()
    file_path = (temp_dir / f"invoice_{file_id}{file_ext}").resolve()

    # Security: Ensure file is within temp directory (prevent path traversal)
    if not str(file_path).startswith(str(temp_dir)):
        raise HTTPException(status_code=400, detail="Invalid file path")

    try:
        with open(file_path, "wb") as f:
            f.write(content)

        _uploaded_files[file_id] = str(file_path)

        logger.info(
            "Invoice file uploaded",
            extra={
                "file_id": file_id,
                "filename": file.filename,
                "size": len(content),
                "tenant_id": tenant_id,
            },
        )

        return {
            "file_id": file_id,
            "file_path": str(file_path),
            "message": f"File uploaded successfully: {file_id}",
        }
    except Exception as e:
        logger.exception("File upload failed", extra={"tenant_id": tenant_id})
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}") from e


@log_execution
@timeout(seconds=120.0)
async def run_invoice_manager(
    request: InvoiceManagerRunRequest,
) -> InvoiceManagerRunResponse:
    """Run invoice manager agent.

    Processes user message through the invoice manager agent flow.
    Can handle capability queries, invoice processing, or invoice searches.

    Args:
        request: Request with message and tenant_id

    Returns:
        Response with agent output and any processing results
    """
    tenant_id = request.tenant_id
    message = request.message
    conversation_id = request.conversation_id or str(uuid4())

    # Resolve file path if file_id provided
    invoice_file_path = request.invoice_file_path

    try:
        # Create graph instance with provider-agnostic adapter
        # Provider can be specified in request or will use default from settings
        graph = create_invoice_manager_graph(
            tenant_id=tenant_id,
            provider=request.provider,  # Use adapter pattern - supports any provider
            model=request.model,  # Use provider default if not specified
        )

        # Prepare initial state
        from langchain_core.messages import HumanMessage

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "invoice_file_path": invoice_file_path,
            "tenant_id": tenant_id,
            "provider": request.provider,
            "model": request.model,
            "extracted_data": None,
            "renamed_filename": None,
            "target_folder": None,
            "duplicate_check_result": None,
            "upload_result": None,
            "search_results": None,
            "next_action": None,
            "error": None,
        }

        # Create checkpointer config
        config = create_checkpointer_config(
            tenant_id=tenant_id,
            thread_id=conversation_id,
        )

        # Run graph
        result = await graph.ainvoke(initial_state, config=config)

        # Extract response from messages
        messages = result.get("messages", [])
        response_text = ""
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                response_text = last_message.content
            elif isinstance(last_message, dict):
                response_text = last_message.get("content", "")

        # Save conversation and messages to database
        # Fixed: Use correct Session pattern with engine, remove silent error catching
        engine = get_session()
        with Session(engine) as session:
            repo = ConversationRepository(session)

            # Create or get conversation (with tenant validation for security)
            if not repo.conversation_exists_for_tenant(conversation_id, tenant_id):
                repo.create_conversation(
                    tenant_id=tenant_id,
                    flow_type="invoice_manager",
                    conversation_id=conversation_id,
                    flow_metadata={
                        "has_extracted_data": result.get("extracted_data") is not None,
                        "has_upload_result": result.get("upload_result") is not None,
                    },
                )
                logger.info(
                    f"Created invoice_manager conversation {conversation_id}",
                    extra={"tenant_id": tenant_id, "conversation_id": conversation_id},
                )

            # Save user message
            repo.save_message(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                flow_type="invoice_manager",
                role="user",
                content=message,
                message_metadata={
                    "has_invoice_file": invoice_file_path is not None,
                    "provider": request.provider or "default",
                    "model": request.model or "default",
                },
            )

            # Save assistant response
            repo.save_message(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                flow_type="invoice_manager",
                role="assistant",
                content=response_text,
                message_metadata={
                    "extracted_data": result.get("extracted_data"),
                    "renamed_filename": result.get("renamed_filename"),
                    "target_folder": result.get("target_folder"),
                    "duplicate_check": result.get("duplicate_check_result"),
                    "upload_result": result.get("upload_result"),
                    "search_results": result.get("search_results"),
                },
            )

            # Save tool execution results if any tools were used
            # Extract tool calls from state if available
            if result.get("extracted_data"):
                # Tool: extract_invoice_content
                repo.save_message(
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    flow_type="invoice_manager",
                    role="tool",
                    content=str(result.get("extracted_data")),
                    message_metadata={
                        "tool_name": "extract_invoice_content",
                        "tool_result": result.get("extracted_data"),
                    },
                )

            if result.get("renamed_filename"):
                # Tool: rename_invoice
                repo.save_message(
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    flow_type="invoice_manager",
                    role="tool",
                    content=f"Renamed to: {result.get('renamed_filename')}",
                    message_metadata={
                        "tool_name": "rename_invoice",
                        "result": {"filename": result.get("renamed_filename")},
                    },
                )

            if result.get("duplicate_check_result"):
                # Tool: detect_duplicate_invoice
                repo.save_message(
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    flow_type="invoice_manager",
                    role="tool",
                    content=str(result.get("duplicate_check_result")),
                    message_metadata={
                        "tool_name": "detect_duplicate_invoice",
                        "tool_result": result.get("duplicate_check_result"),
                    },
                )

        logger.info(
            "Invoice manager run completed",
            extra={
                "tenant_id": tenant_id,
                "conversation_id": conversation_id,
                "has_extracted_data": result.get("extracted_data") is not None,
            },
        )

        return InvoiceManagerRunResponse(
            response=response_text,
            conversation_id=conversation_id,
            extracted_data=result.get("extracted_data"),
            renamed_filename=result.get("renamed_filename"),
            target_folder=result.get("target_folder"),
            duplicate_check=result.get("duplicate_check_result"),
            upload_result=result.get("upload_result"),
            search_results=result.get("search_results"),
            error=result.get("error"),
        )

    except Exception as e:
        logger.exception(
            "Invoice manager run failed",
            extra={
                "tenant_id": tenant_id,
                "conversation_id": conversation_id,
                "message_preview": message[:100] if message else "",
                "has_file": invoice_file_path is not None,
            },
        )
        return InvoiceManagerRunResponse(
            response="",
            conversation_id=conversation_id,
            error=str(e),
        )
    finally:
        # Clean up uploaded file after processing
        if invoice_file_path and os.path.exists(invoice_file_path):
            try:
                os.remove(invoice_file_path)
                logger.debug(
                    f"Cleaned up temp file: {invoice_file_path}",
                    extra={"tenant_id": tenant_id, "file_path": invoice_file_path},
                )
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to clean up temp file: {cleanup_error}",
                    extra={"tenant_id": tenant_id, "file_path": invoice_file_path},
                )


async def get_capabilities() -> CapabilitiesResponse:
    """Get invoice manager agent capabilities.

    Returns:
        List of capabilities and description
    """
    return CapabilitiesResponse(
        capabilities=[
            "Process Invoices: Upload PDF or JPEG invoices, extract content, and automatically rename them",
            "Smart Storage: Scan SharePoint folders and find the right location for invoices",
            "Duplicate Detection: Check if an invoice already exists before uploading",
            "Upload Management: Upload renamed invoices to the correct SharePoint folders",
            "Invoice Search: Search and retrieve invoices by vendor, date, amount, or invoice number",
        ],
        description=(
            "I'm an Invoice Manager Agent that can help you process, organize, "
            "and search invoices automatically. I extract data from invoices, "
            "rename them intelligently, check for duplicates, and upload them "
            "to the right SharePoint folders."
        ),
    )


async def search_invoices_endpoint(
    request: InvoiceSearchRequest,
) -> InvoiceSearchResponse:
    """Search for invoices in SharePoint.

    Args:
        request: Search parameters

    Returns:
        Matching invoices with file links
    """
    try:
        result = await search_invoices(
            vendor_name=request.vendor_name,
            date_from=request.date_from,
            date_to=request.date_to,
            amount_min=request.amount_min,
            amount_max=request.amount_max,
            invoice_number=request.invoice_number,
            tenant_id=request.tenant_id,
            limit=request.limit,
        )

        if result.get("success"):
            invoices = result.get("result", {}).get("invoices", [])
            return InvoiceSearchResponse(
                invoices=invoices,
                total_count=len(invoices),
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Search failed"),
            )

    except Exception as e:
        logger.exception("Invoice search failed", extra={"tenant_id": request.tenant_id})
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e
