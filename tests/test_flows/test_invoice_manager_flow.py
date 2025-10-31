"""Tests for Invoice Manager LangGraph flow."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from langchain_core.messages import HumanMessage, AIMessage

from app.flows.invoice_manager.graph import (
    InvoiceManagerState,
    route_message,
    route_to_action,
    handle_capability_query,
    process_invoice_node,
    scan_folders_node,
    check_duplicates_node,
    upload_invoice_node,
    search_invoices_node,
)


@pytest.fixture
def base_state():
    """Create base state for testing."""
    return {
        "messages": [],
        "invoice_file_path": None,
        "extracted_data": None,
        "renamed_filename": None,
        "target_folder": None,
        "duplicate_check_result": None,
        "upload_result": None,
        "search_results": None,
        "next_action": None,
        "tenant_id": "test_tenant",
        "error": None,
    }


class TestRouteMessage:
    """Test message routing logic."""

    def test_route_capability_query(self, base_state):
        """Test routing to capability query."""
        state = base_state.copy()
        state["messages"] = [{"role": "user", "content": "What can you do?"}]

        result = route_message(state)

        assert result["next_action"] == "capability_query"

    def test_route_process_invoice(self, base_state):
        """Test routing to invoice processing."""
        state = base_state.copy()
        state["messages"] = [{"role": "user", "content": "Please process this invoice"}]

        result = route_message(state)

        assert result["next_action"] == "process_invoice"

    def test_route_search(self, base_state):
        """Test routing to search."""
        state = base_state.copy()
        state["messages"] = [{"role": "user", "content": "Find invoices from Acme Corp"}]

        result = route_message(state)

        assert result["next_action"] == "search_invoices"

    def test_route_agent_fallback(self, base_state):
        """Test routing to agent as fallback."""
        state = base_state.copy()
        state["messages"] = [{"role": "user", "content": "Hello"}]

        result = route_message(state)

        assert result["next_action"] == "agent"


class TestCapabilityQuery:
    """Test capability query handler."""

    @pytest.mark.asyncio
    async def test_handle_capability_query(self, base_state):
        """Test capability query response."""
        state = base_state.copy()
        state["messages"] = [HumanMessage(content="What can you do?")]

        result = await handle_capability_query(state)

        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][-1], AIMessage)
        assert "Invoice Manager Agent" in result["messages"][-1].content


class TestProcessInvoiceNode:
    """Test invoice processing node."""

    @pytest.mark.asyncio
    @patch("app.flows.invoice_manager.graph.extract_tool")
    @patch("app.flows.invoice_manager.graph.rename_tool")
    async def test_process_invoice_success(self, mock_rename, mock_extract, base_state):
        """Test successful invoice processing."""
        # Mock extract tool
        mock_extract.execute = AsyncMock(
            return_value={
                "success": True,
                "result": {
                    "date": "2024-01-15",
                    "vendor_name": "Acme Corp",
                    "amount": 1234.56,
                    "currency": "USD",
                    "invoice_number": "INV-001",
                },
            }
        )

        # Mock rename tool
        mock_rename.execute = AsyncMock(
            return_value={
                "success": True,
                "result": {
                    "filename": "20240115_AcmeCorp_1234.56USD.pdf",
                },
            }
        )

        state = base_state.copy()
        state["invoice_file_path"] = "/tmp/test_invoice.pdf"

        result = await process_invoice_node(state)

        assert result["extracted_data"] is not None
        assert result["renamed_filename"] == "20240115_AcmeCorp_1234.56USD.pdf"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_process_invoice_missing_file(self, base_state):
        """Test processing with missing file."""
        state = base_state.copy()
        state["invoice_file_path"] = None

        result = await process_invoice_node(state)

        assert result["error"] is not None
        assert "file" in result["error"].lower()


class TestScanFoldersNode:
    """Test folder scanning node."""

    @pytest.mark.asyncio
    @patch("app.flows.invoice_manager.graph.find_target_folder")
    async def test_scan_folders_success(self, mock_find_folder, base_state):
        """Test successful folder scanning."""
        mock_find_folder.return_value = {
            "success": True,
            "result": {
                "folder_path": "/Invoices/2024/AcmeCorp",
                "exists": True,
            },
        }

        state = base_state.copy()
        state["extracted_data"] = {
            "vendor_name": "Acme Corp",
            "date": "2024-01-15",
        }

        result = await scan_folders_node(state)

        assert result["target_folder"] == "/Invoices/2024/AcmeCorp"
        assert result["error"] is None


class TestCheckDuplicatesNode:
    """Test duplicate checking node."""

    @pytest.mark.asyncio
    @patch("app.flows.invoice_manager.graph.duplicate_tool")
    async def test_check_duplicates_no_duplicate(self, mock_duplicate, base_state):
        """Test duplicate check with no duplicates."""
        mock_duplicate.execute = AsyncMock(
            return_value={
                "success": True,
                "result": {
                    "is_duplicate": False,
                    "exact_match": False,
                    "similar_matches": [],
                },
            }
        )

        state = base_state.copy()
        state["renamed_filename"] = "20240115_AcmeCorp_1234.56USD.pdf"
        state["extracted_data"] = {"vendor_name": "Acme Corp"}
        state["target_folder"] = "/Invoices/2024"

        result = await check_duplicates_node(state)

        assert result["duplicate_check_result"]["is_duplicate"] is False
        assert result["error"] is None


class TestUploadInvoiceNode:
    """Test invoice upload node."""

    @pytest.mark.asyncio
    @patch("app.flows.invoice_manager.graph.upload_to_sharepoint")
    async def test_upload_invoice_success(self, mock_upload, base_state):
        """Test successful invoice upload."""
        mock_upload.return_value = {
            "success": True,
            "result": {
                "file_url": "https://sharepoint.com/invoice.pdf",
                "file_path": "/Invoices/2024/invoice.pdf",
            },
        }

        state = base_state.copy()
        state["invoice_file_path"] = "/tmp/test_invoice.pdf"
        state["renamed_filename"] = "20240115_AcmeCorp_1234.56USD.pdf"
        state["target_folder"] = "/Invoices/2024"

        result = await upload_invoice_node(state)

        assert result["upload_result"] is not None
        assert result["upload_result"]["file_url"] == "https://sharepoint.com/invoice.pdf"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_upload_invoice_missing_file(self, base_state):
        """Test upload with missing file."""
        state = base_state.copy()
        state["invoice_file_path"] = None

        result = await upload_invoice_node(state)

        assert result["error"] is not None
        assert "file" in result["error"].lower()


class TestSearchInvoicesNode:
    """Test invoice search node."""

    @pytest.mark.asyncio
    @patch("app.flows.invoice_manager.graph.search_invoices")
    async def test_search_invoices_success(self, mock_search, base_state):
        """Test successful invoice search."""
        mock_search.return_value = {
            "success": True,
            "result": {
                "invoices": [
                    {
                        "file_name": "20240115_AcmeCorp_1234.56USD.pdf",
                        "file_url": "https://sharepoint.com/invoice.pdf",
                    }
                ],
            },
        }

        state = base_state.copy()
        state["messages"] = [HumanMessage(content="Find invoices")]

        result = await search_invoices_node(state)

        assert result["search_results"] is not None
        assert len(result["search_results"]) == 1
        assert result["error"] is None
