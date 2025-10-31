"""Invoice Manager Agent - LangGraph Flow.

This flow orchestrates invoice processing, SharePoint operations, and user queries.
"""

import logging
from typing import Any, Annotated, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.adapters.langgraph_adapter import LangGraphLLMAdapter
from app.adapters.langgraph_tools import create_langgraph_tool
from app.adapters.llm_factory import get_llm_adapter
from app.core.langgraph_checkpointer import (
    create_checkpointer_config,
    get_langgraph_checkpointer,
)
from app.tools.invoice.v1 import (
    DetectDuplicateInvoiceTool,
    ExtractInvoiceContentTool,
    RenameInvoiceTool,
)
from app.tools.sharepoint.v1 import (
    find_target_folder,
    search_invoices,
    upload_to_sharepoint,
)

logger = logging.getLogger(__name__)


# State schema for invoice manager flow
class InvoiceManagerState(TypedDict):
    """State schema for invoice manager agent."""

    messages: Annotated[list, add_messages]
    """Conversation messages."""

    # Invoice processing state
    invoice_file_path: str | None
    """Path to uploaded invoice file."""

    extracted_data: dict | None
    """Extracted invoice data (date, vendor, amount, etc.)."""

    renamed_filename: str | None
    """Generated filename for invoice."""

    target_folder: str | None
    """SharePoint folder path for storing invoice."""

    duplicate_check_result: dict | None
    """Duplicate detection results."""

    upload_result: dict | None
    """Upload result with file URL."""

    # Search state
    search_results: list[dict] | None
    """Search results for invoice queries."""

    # Flow control
    next_action: (
        Literal["capability_query", "process_invoice", "search_invoices", "upload", "end"] | None
    )
    """Next action to take."""

    tenant_id: str
    """Tenant identifier."""

    # LLM configuration
    provider: str | None
    """LLM provider (openai, mistral, vertex)."""

    model: str | None
    """LLM model name."""

    error: str | None
    """Error message if any."""


# Initialize tools
extract_tool = ExtractInvoiceContentTool()
rename_tool = RenameInvoiceTool()
duplicate_tool = DetectDuplicateInvoiceTool()


def create_invoice_manager_graph(
    tenant_id: str,
    provider: str | None = None,
    model: str | None = None,
) -> Any:  # Returns compiled LangGraph instance
    """Create invoice manager LangGraph instance.

    Args:
        tenant_id: Tenant identifier
        provider: LLM provider (optional)
        model: LLM model (optional)

    Returns:
        Compiled LangGraph instance
    """
    # Initialize LLM adapter
    llm_adapter = get_llm_adapter(provider=provider, model=model)
    langgraph_llm = LangGraphLLMAdapter(llm_adapter)

    # Create LangGraph tools
    langgraph_tools = [
        create_langgraph_tool(extract_tool, tenant_id=tenant_id),
        create_langgraph_tool(rename_tool, tenant_id=tenant_id),
        create_langgraph_tool(duplicate_tool, tenant_id=tenant_id),
    ]

    # Bind tools to LLM
    # Note: LangGraph handles tools differently - tools are passed to nodes, not bound to LLM
    # We'll use tool calls within nodes instead

    # Create graph
    workflow = StateGraph(InvoiceManagerState)

    # Add nodes
    workflow.add_node("route", route_message)
    workflow.add_node("capability_query", handle_capability_query)
    workflow.add_node("process_invoice", process_invoice_node)
    workflow.add_node("scan_folders", scan_folders_node)
    workflow.add_node("check_duplicates", check_duplicates_node)
    workflow.add_node("upload_invoice", upload_invoice_node)
    workflow.add_node("search_invoices", search_invoices_node)
    workflow.add_node("agent", create_agent_node(langgraph_llm, langgraph_tools))

    # Set entry point
    workflow.set_entry_point("route")

    # Add edges
    workflow.add_conditional_edges(
        "route",
        route_to_action,
        {
            "capability_query": "capability_query",
            "process_invoice": "process_invoice",
            "search_invoices": "search_invoices",
            "agent": "agent",
        },
    )

    workflow.add_edge("capability_query", END)
    workflow.add_edge("process_invoice", "scan_folders")
    workflow.add_edge("scan_folders", "check_duplicates")
    workflow.add_edge("check_duplicates", "upload_invoice")
    workflow.add_edge("upload_invoice", END)
    workflow.add_edge("search_invoices", END)
    workflow.add_edge("agent", END)

    # Compile with checkpointer
    # Note: Checkpointing gracefully degrades if database is unavailable
    # Programming errors are not caught to fail fast during development
    try:
        checkpointer = get_langgraph_checkpointer(tenant_id=tenant_id)
        graph = workflow.compile(checkpointer=checkpointer)
    except (ConnectionError, TimeoutError, OSError) as e:
        # Only catch infrastructure failures - database unavailable, network issues
        logger.warning(
            f"Database unavailable, running without checkpointing: {e}",
            extra={"tenant_id": tenant_id, "error": str(e)},
        )
        # Compile without checkpointer - graph will still work
        graph = workflow.compile()
    except Exception as e:
        # Programming errors should fail fast to surface bugs quickly
        logger.exception(
            f"Failed to compile graph: {e}",
            extra={"tenant_id": tenant_id},
        )
        raise

    return graph


def route_message(state: InvoiceManagerState) -> InvoiceManagerState:
    """Route message to appropriate handler based on user intent."""
    messages = state.get("messages", [])
    if not messages:
        return state

    last_message = messages[-1]
    # Handle both dict and LangChain message objects
    if hasattr(last_message, "content"):
        content = last_message.content.lower() if last_message.content else ""
    elif isinstance(last_message, dict):
        content = last_message.get("content", "").lower()
    else:
        content = str(last_message).lower()

    # Detect intent
    if any(phrase in content for phrase in ["what can", "capabilities", "what do you", "help"]):
        state["next_action"] = "capability_query"
    elif any(phrase in content for phrase in ["upload", "process", "invoice", "file"]):
        state["next_action"] = "process_invoice"
    elif any(phrase in content for phrase in ["search", "find", "retrieve", "list"]):
        state["next_action"] = "search_invoices"
    else:
        state["next_action"] = "agent"  # Let LLM handle it

    return state


def route_to_action(state: InvoiceManagerState) -> str:
    """Determine next action based on state."""
    next_action = state.get("next_action")
    if next_action:
        return next_action
    return "agent"


async def handle_capability_query(state: InvoiceManagerState) -> InvoiceManagerState:
    """Handle capability query - tell user what the agent can do."""
    capabilities = """I'm an Invoice Manager Agent. I can help you with:

1. **Process Invoices**: Upload PDF or JPEG invoices, extract content, and automatically rename them
2. **Smart Storage**: Scan SharePoint folders and find the right location for invoices
3. **Duplicate Detection**: Check if an invoice already exists before uploading
4. **Upload Management**: Upload renamed invoices to the correct SharePoint folders
5. **Invoice Search**: Search and retrieve invoices by vendor, date, amount, or invoice number

To get started:
- Upload an invoice: "Please process this invoice [file]"
- Search invoices: "Find all invoices from Acme Corp in 2024"
- Ask questions: "What can you do?" or "How do I upload an invoice?"

I'll handle the entire workflow automatically, including extracting data, renaming files, checking for duplicates, and organizing them in SharePoint."""

    from langchain_core.messages import AIMessage

    messages = state.get("messages", [])
    messages.append(AIMessage(content=capabilities))
    state["messages"] = messages

    return state


async def process_invoice_node(state: InvoiceManagerState) -> InvoiceManagerState:
    """Process invoice: extract content and rename."""
    tenant_id = state.get("tenant_id", "")
    provider = state.get("provider")
    model = state.get("model")

    # Extract invoice file path from messages or state
    invoice_file_path = state.get("invoice_file_path")
    if not invoice_file_path:
        # Try to extract from messages
        messages = state.get("messages", [])
        for msg in reversed(messages):
            # Look for file reference in message
            # TODO: Extract file path from message attachments
            pass

    if not invoice_file_path:
        state["error"] = "No invoice file provided"
        return state

    # Extract content with specified provider/model
    extract_result = await extract_tool.execute(
        file_path=invoice_file_path,
        tenant_id=tenant_id,
        provider=provider,
        model=model,
    )

    if not extract_result.get("success"):
        state["error"] = extract_result.get("error", "Extraction failed")
        return state

    extracted_data = extract_result.get("result", {})
    state["extracted_data"] = extracted_data

    # Rename invoice
    rename_result = await rename_tool.execute(
        date=extracted_data.get("date"),
        vendor_name=extracted_data.get("vendor_name"),
        amount=extracted_data.get("amount"),
        currency=extracted_data.get("currency"),
        original_filename=invoice_file_path.split("/")[-1],
        tenant_id=tenant_id,
    )

    if not rename_result.get("success"):
        state["error"] = rename_result.get("error", "Renaming failed")
        return state

    state["renamed_filename"] = rename_result.get("result", {}).get("filename")

    return state


async def scan_folders_node(state: InvoiceManagerState) -> InvoiceManagerState:
    """Scan SharePoint to find target folder."""
    tenant_id = state.get("tenant_id", "")
    extracted_data = state.get("extracted_data", {})

    # Find target folder
    folder_result = await find_target_folder(
        vendor_name=extracted_data.get("vendor_name"),
        date=extracted_data.get("date"),
        tenant_id=tenant_id,
    )

    if folder_result.get("success"):
        state["target_folder"] = folder_result.get("result", {}).get("folder_path")
    else:
        state["error"] = folder_result.get("error", "Folder scan failed")

    return state


async def check_duplicates_node(state: InvoiceManagerState) -> InvoiceManagerState:
    """Check for duplicate invoices."""
    tenant_id = state.get("tenant_id", "")
    filename = state.get("renamed_filename")
    extracted_data = state.get("extracted_data", {})
    target_folder = state.get("target_folder", "/")

    if not filename:
        state["error"] = "No filename to check"
        return state

    # Check duplicates
    duplicate_result = await duplicate_tool.execute(
        filename=filename,
        invoice_data=extracted_data,
        sharepoint_folder_path=target_folder,
        tenant_id=tenant_id,
    )

    if duplicate_result.get("success"):
        state["duplicate_check_result"] = duplicate_result.get("result", {})
    else:
        state["error"] = duplicate_result.get("error", "Duplicate check failed")

    return state


async def upload_invoice_node(state: InvoiceManagerState) -> InvoiceManagerState:
    """Upload invoice to SharePoint."""
    tenant_id = state.get("tenant_id", "")
    invoice_file_path = state.get("invoice_file_path")
    renamed_filename = state.get("renamed_filename")
    target_folder = state.get("target_folder", "/")

    if not invoice_file_path or not renamed_filename:
        state["error"] = "Missing file path or filename"
        return state

    # Upload to SharePoint
    upload_result = await upload_to_sharepoint(
        file_path=invoice_file_path,
        file_name=renamed_filename,
        target_folder_path=target_folder,
        tenant_id=tenant_id,
        overwrite=False,
    )

    if upload_result.get("success"):
        state["upload_result"] = upload_result.get("result", {})
    else:
        state["error"] = upload_result.get("error", "Upload failed")

    return state


async def search_invoices_node(state: InvoiceManagerState) -> InvoiceManagerState:
    """Search for invoices in SharePoint."""
    tenant_id = state.get("tenant_id", "")
    messages = state.get("messages", [])

    # Extract search parameters from last message
    # TODO: Use LLM to extract search parameters from natural language
    # For now, placeholder

    search_result = await search_invoices(
        tenant_id=tenant_id,
    )

    if search_result.get("success"):
        state["search_results"] = search_result.get("result", {}).get("invoices", [])
    else:
        state["error"] = search_result.get("error", "Search failed")

    return state


def create_agent_node(llm, tools):
    """Create agent node factory."""

    async def agent_node(state: InvoiceManagerState) -> InvoiceManagerState:
        """Agent node that processes messages with LLM."""
        from langchain_core.messages import HumanMessage

        messages = state.get("messages", [])
        if not messages:
            return state

        # Convert messages to LangGraph format
        langgraph_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    langgraph_messages.append(HumanMessage(content=content))
                # Add other message types as needed

        # Call LLM
        response = await llm.ainvoke(langgraph_messages)

        # Add response to messages
        messages.append({"role": "assistant", "content": response.content})
        state["messages"] = messages

        return state

    return agent_node
