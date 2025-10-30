"""Tests for Invoice Manager tools."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from app.tools.invoice.v1.extract_content import ExtractInvoiceContentTool
from app.tools.invoice.v1.rename_invoice import RenameInvoiceTool
from app.tools.invoice.v1.detect_duplicate import DetectDuplicateInvoiceTool


@pytest.fixture
def extract_tool():
    """Create ExtractInvoiceContentTool instance."""
    return ExtractInvoiceContentTool()


@pytest.fixture
def rename_tool():
    """Create RenameInvoiceTool instance."""
    return RenameInvoiceTool()


@pytest.fixture
def duplicate_tool():
    """Create DetectDuplicateInvoiceTool instance."""
    return DetectDuplicateInvoiceTool()


class TestExtractInvoiceContentTool:
    """Test ExtractInvoiceContentTool."""

    def test_init(self, extract_tool):
        """Test tool initialization."""
        assert extract_tool.name == "extract_invoice_content"
        assert extract_tool.version == "1.0.0"

    def test_validate_input_valid(self, extract_tool):
        """Test input validation with valid input."""
        assert extract_tool.validate_input(
            file_path="/path/to/invoice.pdf",
            tenant_id="test_tenant",
        )

    def test_validate_input_missing_file(self, extract_tool):
        """Test input validation with missing file."""
        assert not extract_tool.validate_input(tenant_id="test_tenant")

    def test_validate_input_missing_tenant(self, extract_tool):
        """Test input validation with missing tenant."""
        assert not extract_tool.validate_input(file_path="/path/to/invoice.pdf")

    @pytest.mark.asyncio
    @patch("app.tools.invoice.v1.extract_content.ExtractInvoiceContentTool._extract_pdf_text")
    @patch("app.tools.invoice.v1.extract_content.ExtractInvoiceContentTool._parse_with_llm")
    async def test_execute_pdf_success(self, mock_parse, mock_extract, extract_tool):
        """Test successful PDF extraction."""
        mock_extract.return_value = "Invoice Date: 2024-01-15\nVendor: Acme Corp\nAmount: $1234.56"
        mock_parse.return_value = {
            "date": "2024-01-15",
            "vendor_name": "Acme Corp",
            "amount": 1234.56,
            "currency": "USD",
            "invoice_number": "INV-001",
        }

        result = await extract_tool.execute(
            file_path="/tmp/test_invoice.pdf",
            tenant_id="test_tenant",
        )

        assert result["success"] is True
        assert result["result"]["vendor_name"] == "Acme Corp"
        assert result["result"]["amount"] == 1234.56

    @pytest.mark.asyncio
    async def test_execute_missing_file(self, extract_tool):
        """Test execution with missing file."""
        result = await extract_tool.execute(tenant_id="test_tenant")

        assert result["success"] is False
        assert "file_path" in result["error"] or "file_content" in result["error"]

    @pytest.mark.asyncio
    @patch("app.tools.invoice.v1.extract_content.ExtractInvoiceContentTool._extract_pdf_text")
    async def test_execute_extraction_error(self, mock_extract, extract_tool):
        """Test handling of extraction errors."""
        mock_extract.side_effect = Exception("PDF extraction failed")

        result = await extract_tool.execute(
            file_path="/tmp/test_invoice.pdf",
            tenant_id="test_tenant",
        )

        assert result["success"] is False
        assert "extraction failed" in result["error"].lower()


class TestRenameInvoiceTool:
    """Test RenameInvoiceTool."""

    def test_init(self, rename_tool):
        """Test tool initialization."""
        assert rename_tool.name == "rename_invoice"
        assert rename_tool.version == "1.0.0"

    def test_validate_input_valid(self, rename_tool):
        """Test input validation with valid input."""
        assert rename_tool.validate_input(
            vendor_name="Acme Corp",
            amount=1234.56,
        )

    def test_validate_input_missing_vendor(self, rename_tool):
        """Test input validation with missing vendor."""
        assert not rename_tool.validate_input(amount=1234.56)

    def test_validate_input_missing_amount(self, rename_tool):
        """Test input validation with missing amount."""
        assert not rename_tool.validate_input(vendor_name="Acme Corp")

    @pytest.mark.asyncio
    async def test_execute_success(self, rename_tool):
        """Test successful filename generation."""
        result = await rename_tool.execute(
            date="2024-01-15",
            vendor_name="Acme Corp",
            amount=1234.56,
            currency="USD",
            original_filename="invoice.pdf",
            tenant_id="test_tenant",
        )

        assert result["success"] is True
        assert "filename" in result["result"]
        filename = result["result"]["filename"]
        assert filename.startswith("20240115_")
        assert "AcmeCorp" in filename
        assert "1234.56USD" in filename
        assert filename.endswith(".pdf")

    @pytest.mark.asyncio
    async def test_execute_date_formats(self, rename_tool):
        """Test handling different date formats."""
        test_cases = [
            ("2024-01-15", "20240115"),
            ("2024/01/15", "20240115"),
            ("15/01/2024", "20240115"),
        ]

        for date_input, expected_prefix in test_cases:
            result = await rename_tool.execute(
                date=date_input,
                vendor_name="Test",
                amount=100,
                currency="USD",
                original_filename="test.pdf",
                tenant_id="test",
            )
            assert result["success"]
            assert result["result"]["filename"].startswith(expected_prefix)

    @pytest.mark.asyncio
    async def test_execute_vendor_sanitization(self, rename_tool):
        """Test vendor name sanitization."""
        result = await rename_tool.execute(
            date="2024-01-15",
            vendor_name="Acme Corp & Co.",
            amount=100,
            currency="USD",
            original_filename="test.pdf",
            tenant_id="test",
        )

        assert result["success"]
        filename = result["result"]["filename"]
        # Should not contain special characters
        assert "&" not in filename
        assert "." not in filename.split(".")[0]  # Except extension

    @pytest.mark.asyncio
    async def test_execute_amount_formatting(self, rename_tool):
        """Test amount formatting."""
        result = await rename_tool.execute(
            date="2024-01-15",
            vendor_name="Test",
            amount=1234.56789,
            currency="EUR",
            original_filename="test.pdf",
            tenant_id="test",
        )

        assert result["success"]
        filename = result["result"]["filename"]
        assert "1234.56EUR" in filename
        assert "1234.56789" not in filename  # Should be rounded

    def test_format_date_datetime_object(self, rename_tool):
        """Test date formatting with datetime object."""
        date_obj = datetime(2024, 1, 15)
        formatted = rename_tool._format_date(date_obj)
        assert formatted == "20240115"

    def test_format_date_none(self, rename_tool):
        """Test date formatting with None (uses current date)."""
        formatted = rename_tool._format_date(None)
        assert len(formatted) == 8
        assert formatted.isdigit()

    def test_sanitize_vendor_name_empty(self, rename_tool):
        """Test vendor name sanitization with empty string."""
        sanitized = rename_tool._sanitize_vendor_name("")
        assert sanitized == "UnknownVendor"

    def test_sanitize_vendor_name_long(self, rename_tool):
        """Test vendor name sanitization with very long name."""
        long_name = "A" * 100
        sanitized = rename_tool._sanitize_vendor_name(long_name)
        assert len(sanitized) <= 50


class TestDetectDuplicateInvoiceTool:
    """Test DetectDuplicateInvoiceTool."""

    def test_init(self, duplicate_tool):
        """Test tool initialization."""
        assert duplicate_tool.name == "detect_duplicate_invoice"
        assert duplicate_tool.version == "1.0.0"

    def test_validate_input_valid(self, duplicate_tool):
        """Test input validation with valid input."""
        assert duplicate_tool.validate_input(
            filename="test.pdf",
            invoice_data={"vendor": "Acme", "amount": 100},
            tenant_id="test_tenant",
        )

    def test_validate_input_missing_filename(self, duplicate_tool):
        """Test input validation with missing filename."""
        assert not duplicate_tool.validate_input(
            invoice_data={},
            tenant_id="test_tenant",
        )

    @pytest.mark.asyncio
    @patch("app.tools.invoice.v1.detect_duplicate.scan_sharepoint_folders")
    async def test_execute_no_duplicate(self, mock_scan, duplicate_tool):
        """Test duplicate detection with no duplicates found."""
        mock_scan.return_value = {
            "success": True,
            "result": {
                "files": [],  # No matching files
            },
        }

        result = await duplicate_tool.execute(
            filename="20240115_AcmeCorp_1234.56USD.pdf",
            invoice_data={
                "vendor_name": "Acme Corp",
                "amount": 1234.56,
                "date": "2024-01-15",
            },
            sharepoint_folder_path="/Invoices/2024",
            tenant_id="test_tenant",
        )

        assert result["success"] is True
        assert result["result"]["is_duplicate"] is False

    @pytest.mark.asyncio
    @patch("app.tools.invoice.v1.detect_duplicate.scan_sharepoint_folders")
    async def test_execute_exact_duplicate(self, mock_scan, duplicate_tool):
        """Test duplicate detection with exact filename match."""
        mock_scan.return_value = {
            "success": True,
            "result": {
                "files": [
                    {
                        "name": "20240115_AcmeCorp_1234.56USD.pdf",
                        "path": "/Invoices/2024/20240115_AcmeCorp_1234.56USD.pdf",
                    }
                ],
            },
        }

        result = await duplicate_tool.execute(
            filename="20240115_AcmeCorp_1234.56USD.pdf",
            invoice_data={
                "vendor_name": "Acme Corp",
                "amount": 1234.56,
            },
            sharepoint_folder_path="/Invoices/2024",
            tenant_id="test_tenant",
        )

        assert result["success"] is True
        assert result["result"]["is_duplicate"] is True
        assert result["result"]["exact_match"] is True

    @pytest.mark.asyncio
    async def test_execute_missing_tenant(self, duplicate_tool):
        """Test execution with missing tenant_id."""
        result = await duplicate_tool.execute(
            filename="test.pdf",
            invoice_data={},
            sharepoint_folder_path="/",
        )

        assert result["success"] is False
        assert "tenant_id" in result["error"]

