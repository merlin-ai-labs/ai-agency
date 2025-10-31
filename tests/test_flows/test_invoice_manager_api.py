"""Tests for Invoice Manager API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_invoice_file():
    """Create a mock invoice file."""
    from io import BytesIO

    return ("test_invoice.pdf", BytesIO(b"fake pdf content"), "application/pdf")


class TestInvoiceManagerCapabilities:
    """Test capabilities endpoint."""

    def test_get_capabilities(self, client):
        """Test getting agent capabilities."""
        response = client.get("/api/v1/invoice-manager/capabilities")

        assert response.status_code == 200
        data = response.json()
        assert "capabilities" in data
        assert "description" in data
        assert isinstance(data["capabilities"], list)
        assert len(data["capabilities"]) > 0


class TestInvoiceManagerUpload:
    """Test file upload endpoint."""

    def test_upload_pdf_file(self, client, mock_invoice_file):
        """Test uploading a PDF file."""
        filename, file_content, content_type = mock_invoice_file

        response = client.post(
            "/api/v1/invoice-manager/upload?tenant_id=test_tenant",
            files={"file": (filename, file_content, content_type)},
        )

        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert "file_path" in data
        assert "message" in data

    def test_upload_jpeg_file(self, client):
        """Test uploading a JPEG file."""
        from io import BytesIO

        response = client.post(
            "/api/v1/invoice-manager/upload?tenant_id=test_tenant",
            files={"file": ("test_invoice.jpg", BytesIO(b"fake jpeg content"), "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data

    def test_upload_invalid_file_type(self, client):
        """Test uploading an invalid file type."""
        from io import BytesIO

        response = client.post(
            "/api/v1/invoice-manager/upload?tenant_id=test_tenant",
            files={"file": ("test.txt", BytesIO(b"text content"), "text/plain")},
        )

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_missing_filename(self, client):
        """Test uploading file without filename.

        Note: FastAPI returns 422 (Unprocessable Entity) for validation errors,
        not 400 (Bad Request). This is standard FastAPI behavior.
        """
        from io import BytesIO

        response = client.post(
            "/api/v1/invoice-manager/upload?tenant_id=test_tenant",
            files={"file": ("", BytesIO(b"content"), "application/pdf")},
        )

        # FastAPI validation returns 422 for invalid input
        assert response.status_code == 422
        assert "detail" in response.json()


class TestInvoiceManagerRun:
    """Test run endpoint."""

    @pytest.mark.asyncio
    @patch("app.flows.invoice_manager.api.create_invoice_manager_graph")
    async def test_run_capability_query(self, mock_create_graph, client):
        """Test capability query through run endpoint."""
        # Mock graph
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [{"role": "assistant", "content": "I can help with invoices..."}],
            "extracted_data": None,
            "renamed_filename": None,
            "target_folder": None,
            "duplicate_check_result": None,
            "upload_result": None,
            "search_results": None,
            "error": None,
        }
        mock_create_graph.return_value = mock_graph

        response = client.post(
            "/api/v1/invoice-manager/run",
            json={
                "message": "What can you do?",
                "tenant_id": "test_tenant",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "conversation_id" in data
        assert "error" in data

    @pytest.mark.asyncio
    @patch("app.flows.invoice_manager.api.create_invoice_manager_graph")
    async def test_run_with_invoice_file(self, mock_create_graph, client):
        """Test processing invoice through run endpoint."""
        # Mock graph
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [{"role": "assistant", "content": "Invoice processed successfully"}],
            "extracted_data": {
                "date": "2024-01-15",
                "vendor_name": "Acme Corp",
                "amount": 1234.56,
                "currency": "USD",
            },
            "renamed_filename": "20240115_AcmeCorp_1234.56USD.pdf",
            "target_folder": "/Invoices/2024/AcmeCorp",
            "duplicate_check_result": {"is_duplicate": False},
            "upload_result": {"file_url": "https://sharepoint.com/invoice.pdf"},
            "search_results": None,
            "error": None,
        }
        mock_create_graph.return_value = mock_graph

        response = client.post(
            "/api/v1/invoice-manager/run",
            json={
                "message": "Please process this invoice",
                "tenant_id": "test_tenant",
                "invoice_file_path": "/tmp/test_invoice.pdf",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data.get("extracted_data") is not None
        assert data.get("renamed_filename") is not None
        assert data.get("target_folder") is not None

    @pytest.mark.asyncio
    @patch("app.flows.invoice_manager.api.create_invoice_manager_graph")
    async def test_run_error_handling(self, mock_create_graph, client):
        """Test error handling in run endpoint."""
        # Mock graph to raise exception
        mock_graph = AsyncMock()
        mock_graph.ainvoke.side_effect = Exception("Processing failed")
        mock_create_graph.return_value = mock_graph

        response = client.post(
            "/api/v1/invoice-manager/run",
            json={
                "message": "Process invoice",
                "tenant_id": "test_tenant",
            },
        )

        assert response.status_code == 200  # Returns 200 with error in response
        data = response.json()
        assert "error" in data
        assert data["error"] is not None


class TestInvoiceManagerSearch:
    """Test search endpoint."""

    @pytest.mark.asyncio
    @patch("app.flows.invoice_manager.api.search_invoices")
    async def test_search_invoices(self, mock_search, client):
        """Test searching invoices."""
        mock_search.return_value = {
            "success": True,
            "result": {
                "invoices": [
                    {
                        "file_name": "20240115_AcmeCorp_1234.56USD.pdf",
                        "file_url": "https://sharepoint.com/invoice.pdf",
                        "vendor": "Acme Corp",
                        "amount": 1234.56,
                        "date": "2024-01-15",
                    }
                ],
                "total_count": 1,
            },
        }

        response = client.post(
            "/api/v1/invoice-manager/search",
            json={
                "tenant_id": "test_tenant",
                "vendor_name": "Acme Corp",
                "date_from": "2024-01-01",
                "date_to": "2024-12-31",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "invoices" in data
        assert "total_count" in data
        assert len(data["invoices"]) == 1

    @pytest.mark.asyncio
    @patch("app.flows.invoice_manager.api.search_invoices")
    async def test_search_error_handling(self, mock_search, client):
        """Test search error handling."""
        mock_search.return_value = {
            "success": False,
            "error": "Search failed",
        }

        response = client.post(
            "/api/v1/invoice-manager/search",
            json={
                "tenant_id": "test_tenant",
            },
        )

        assert response.status_code == 500
        assert "Search failed" in response.json()["detail"]
