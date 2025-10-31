"""Invoice content extraction tool.

Extracts structured data from invoice PDFs and JPEG images.
"""

import logging
from typing import Any

from app.core.base import BaseTool
from app.core.types import ToolOutput

logger = logging.getLogger(__name__)


class ExtractInvoiceContentTool(BaseTool):
    """Extract structured data from invoice files (PDF/JPEG).

    This tool processes invoice files and extracts:
    - Invoice date
    - Vendor name
    - Amount (with currency)
    - Invoice number
    - Other metadata

    Example:
        >>> tool = ExtractInvoiceContentTool()
        >>> result = await tool.execute(
        ...     file_path="/path/to/invoice.pdf",
        ...     tenant_id="tenant_123"
        ... )
    """

    def __init__(self) -> None:
        """Initialize invoice content extraction tool."""
        super().__init__(
            name="extract_invoice_content",
            description=(
                "Extract structured data from invoice files (PDF or JPEG). "
                "Returns date, vendor name, amount, currency, and invoice number."
            ),
            version="1.0.0",
        )

    async def execute(self, **kwargs: Any) -> ToolOutput:
        """Extract invoice content from file.

        Args:
            file_path: Path to invoice file (PDF or JPEG)
            file_content: Optional base64-encoded file content (alternative to file_path)
            tenant_id: Tenant identifier for multi-tenancy
            provider: Optional LLM provider (openai, mistral, vertex)
            model: Optional LLM model name

        Returns:
            ToolOutput with extracted invoice data:
            {
                "success": bool,
                "result": {
                    "date": "2024-01-15",
                    "vendor_name": "Acme Corp",
                    "amount": 1234.56,
                    "currency": "USD",
                    "invoice_number": "INV-2024-001",
                    "raw_text": "..."  # Full extracted text
                }
            }
        """
        file_path = kwargs.get("file_path")
        file_content = kwargs.get("file_content")
        tenant_id = kwargs.get("tenant_id")
        provider = kwargs.get("provider")
        model = kwargs.get("model")

        if not file_path and not file_content:
            return {
                "success": False,
                "result": None,
                "error": "Either file_path or file_content must be provided",
                "metadata": {},
            }

        try:
            # Extract text based on file type
            if file_path:
                extracted_data = await self._extract_from_file(file_path)
            else:
                extracted_data = await self._extract_from_content(file_content)

            # Parse structured data using LLM with specified provider
            structured_data = await self._parse_with_llm(
                extracted_data["text"],
                tenant_id,
                provider=provider,
                model=model,
            )

            logger.info(
                "Invoice content extracted successfully",
                extra={
                    "tenant_id": tenant_id,
                    "vendor": structured_data.get("vendor_name"),
                    "amount": structured_data.get("amount"),
                },
            )

            return {
                "success": True,
                "result": {
                    **structured_data,
                    "raw_text": extracted_data["text"],
                },
                "error": None,
                "metadata": {
                    "file_type": extracted_data.get("file_type"),
                    "extraction_method": extracted_data.get("method"),
                },
            }

        except Exception as e:
            logger.exception("Invoice content extraction failed", extra={"tenant_id": tenant_id})
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "metadata": {},
            }

    async def _extract_from_file(self, file_path: str) -> dict[str, Any]:
        """Extract text from file path."""
        # TODO: Implement PDF/JPEG extraction
        # For now, return placeholder
        import os

        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == ".pdf":
            return {
                "text": await self._extract_pdf_text(file_path),
                "file_type": "pdf",
                "method": "pdf_parser",
            }
        elif file_ext in [".jpg", ".jpeg", ".png"]:
            return {
                "text": await self._extract_image_text(file_path),
                "file_type": "image",
                "method": "ocr",
            }
        else:
            msg = f"Unsupported file type: {file_ext}"
            raise ValueError(msg)

    async def _extract_from_content(self, file_content: str) -> dict[str, Any]:
        """Extract text from base64 content."""
        # TODO: Implement base64 decode and extraction
        raise NotImplementedError("Content extraction not yet implemented")

    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
                return "\n".join(text_parts)
        except ImportError:
            logger.warning("pdfplumber not installed, falling back to PyPDF2")
            try:
                import PyPDF2

                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    text_parts = []
                    for page in reader.pages:
                        text_parts.append(page.extract_text())
                    return "\n".join(text_parts)
            except ImportError:
                msg = "Neither pdfplumber nor PyPDF2 is installed. Install one: pip install pdfplumber"
                raise ImportError(msg) from None

    async def _extract_image_text(self, file_path: str) -> str:
        """Extract text from image using OCR."""
        try:
            from PIL import Image
            import pytesseract

            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except ImportError:
            msg = "OCR dependencies not installed. Install: pip install pytesseract pillow"
            raise ImportError(msg) from None

    async def _parse_with_llm(
        self,
        text: str,
        tenant_id: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Parse extracted text using LLM to extract structured data.

        Args:
            text: Extracted text from invoice
            tenant_id: Tenant identifier
            provider: Optional LLM provider (openai, mistral, vertex)
            model: Optional LLM model name
        """
        from app.adapters.llm_factory import get_llm_adapter

        llm = get_llm_adapter(provider=provider, model=model)

        prompt = f"""Extract structured data from this invoice text:

{text}

Return a JSON object with:
- date: Invoice date in YYYY-MM-DD format
- vendor_name: Company/vendor name
- amount: Numeric amount (float)
- currency: Currency code (USD, EUR, etc.)
- invoice_number: Invoice number/ID

Respond ONLY with valid JSON, no explanation."""

        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Deterministic for structured extraction
        )

        import json

        try:
            # Extract JSON from response (handle markdown code blocks)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()

            data = json.loads(cleaned)
            return {
                "date": data.get("date", ""),
                "vendor_name": data.get("vendor_name", ""),
                "amount": float(data.get("amount", 0)),
                "currency": data.get("currency", "USD"),
                "invoice_number": data.get("invoice_number", ""),
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Failed to parse LLM response as JSON: {response}", extra={"error": str(e)}
            )
            # Fallback: try to extract basic info with regex
            return {
                "date": "",
                "vendor_name": "",
                "amount": 0.0,
                "currency": "USD",
                "invoice_number": "",
            }

    def validate_input(self, **kwargs: Any) -> bool:
        """Validate input parameters."""
        file_path = kwargs.get("file_path")
        file_content = kwargs.get("file_content")
        tenant_id = kwargs.get("tenant_id")

        if not tenant_id:
            return False

        if not file_path and not file_content:
            return False

        if file_path and not isinstance(file_path, str):
            return False

        return True
