"""Invoice renaming tool.

Generates standardized filenames for invoices based on extracted data.
"""

import logging
import re
from datetime import datetime
from typing import Any

from app.core.base import BaseTool
from app.core.types import ToolOutput

logger = logging.getLogger(__name__)


class RenameInvoiceTool(BaseTool):
    """Generate standardized filename for invoice.

    Creates filename in format: {date_yyyymmdd}_{vendor}_{amount}.{ext}

    Example:
        >>> tool = RenameInvoiceTool()
        >>> result = await tool.execute(
        ...     date="2024-01-15",
        ...     vendor_name="Acme Corp",
        ...     amount=1234.56,
        ...     currency="USD",
        ...     original_filename="invoice.pdf"
        ... )
        >>> print(result["result"])
        "20240115_AcmeCorp_1234.56USD.pdf"
    """

    def __init__(self) -> None:
        """Initialize invoice renaming tool."""
        super().__init__(
            name="rename_invoice",
            description=(
                "Generate standardized filename for invoice: "
                "{date_yyyymmdd}_{vendor}_{amount}.{ext}"
            ),
            version="1.0.0",
        )

    async def execute(self, **kwargs: Any) -> ToolOutput:
        """Generate standardized filename.

        Args:
            date: Invoice date (YYYY-MM-DD format or datetime object)
            vendor_name: Vendor/company name
            amount: Invoice amount (float or string)
            currency: Currency code (USD, EUR, etc.)
            original_filename: Original filename to preserve extension

        Returns:
            ToolOutput with new filename:
            {
                "success": bool,
                "result": {
                    "filename": "20240115_AcmeCorp_1234.56USD.pdf",
                    "date_formatted": "20240115"
                }
            }
        """
        date = kwargs.get("date")
        vendor_name = kwargs.get("vendor_name", "")
        amount = kwargs.get("amount", 0)
        currency = kwargs.get("currency", "USD")
        original_filename = kwargs.get("original_filename", "invoice.pdf")

        try:
            # Format date
            date_str = self._format_date(date)

            # Sanitize vendor name
            vendor_sanitized = self._sanitize_vendor_name(vendor_name)

            # Format amount
            amount_str = self._format_amount(amount, currency)

            # Get extension
            extension = self._get_extension(original_filename)

            # Build filename
            filename = f"{date_str}_{vendor_sanitized}_{amount_str}.{extension}"

            logger.info(
                "Invoice filename generated",
                extra={
                    "original": original_filename,
                    "new": filename,
                },
            )

            return {
                "success": True,
                "result": {
                    "filename": filename,
                    "date_formatted": date_str,
                    "vendor_sanitized": vendor_sanitized,
                    "amount_formatted": amount_str,
                },
                "error": None,
                "metadata": {
                    "original_filename": original_filename,
                    "currency": currency,
                },
            }

        except Exception as e:
            logger.exception("Invoice renaming failed")
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "metadata": {},
            }

    def _format_date(self, date: str | datetime | None) -> str:
        """Format date to YYYYMMDD."""
        if date is None:
            return datetime.now().strftime("%Y%m%d")

        if isinstance(date, datetime):
            return date.strftime("%Y%m%d")

        if isinstance(date, str):
            # Try parsing various formats
            formats = ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"]
            for fmt in formats:
                try:
                    parsed = datetime.strptime(date, fmt)
                    return parsed.strftime("%Y%m%d")
                except ValueError:
                    continue

            # If no format matches, try to extract date parts
            match = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", date)
            if match:
                return f"{match.group(1)}{match.group(2)}{match.group(3)}"

        # Fallback to current date
        return datetime.now().strftime("%Y%m%d")

    def _sanitize_vendor_name(self, vendor_name: str) -> str:
        """Sanitize vendor name for filename."""
        if not vendor_name:
            return "UnknownVendor"

        # Remove special characters, keep only alphanumeric and spaces
        cleaned = re.sub(r"[^a-zA-Z0-9\s-]", "", vendor_name)

        # Replace spaces and hyphens with nothing (camelCase style)
        cleaned = re.sub(r"[\s-]+", "", cleaned)

        # Limit length
        if len(cleaned) > 50:
            cleaned = cleaned[:50]

        # Ensure it starts with a letter
        if cleaned and not cleaned[0].isalpha():
            cleaned = "Vendor" + cleaned

        return cleaned if cleaned else "UnknownVendor"

    def _format_amount(self, amount: float | str | int, currency: str) -> str:
        """Format amount with currency."""
        try:
            if isinstance(amount, str):
                # Remove currency symbols and whitespace
                cleaned = re.sub(r"[^\d.]", "", amount)
                amount_float = float(cleaned)
            else:
                amount_float = float(amount)

            # Format with 2 decimal places, no thousand separators
            formatted = f"{amount_float:.2f}"

            # Append currency
            currency_upper = currency.upper() if currency else "USD"
            return f"{formatted}{currency_upper}"
        except (ValueError, TypeError):
            return "0.00USD"

    def _get_extension(self, filename: str) -> str:
        """Extract file extension."""
        if "." in filename:
            return filename.rsplit(".", 1)[1].lower()
        return "pdf"  # Default extension

    def validate_input(self, **kwargs: Any) -> bool:
        """Validate input parameters."""
        vendor_name = kwargs.get("vendor_name")
        amount = kwargs.get("amount")

        if not vendor_name:
            return False

        if amount is None:
            return False

        return True
