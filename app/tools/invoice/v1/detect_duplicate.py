"""Invoice duplicate detection tool.

Checks for duplicate invoices by filename or content similarity.
"""

import logging
from typing import Any

from app.core.base import BaseTool
from app.core.types import ToolOutput

logger = logging.getLogger(__name__)


class DetectDuplicateInvoiceTool(BaseTool):
    """Detect duplicate invoices in SharePoint.

    Checks for:
    - Exact filename matches
    - Similar content (using LLM comparison)

    Example:
        >>> tool = DetectDuplicateInvoiceTool()
        >>> result = await tool.execute(
        ...     filename="20240115_AcmeCorp_1234.56USD.pdf",
        ...     invoice_data={"vendor": "Acme Corp", "date": "2024-01-15", "amount": 1234.56},
        ...     sharepoint_folder_path="/Invoices/2024",
        ...     tenant_id="tenant_123"
        ... )
    """

    def __init__(self) -> None:
        """Initialize duplicate detection tool."""
        super().__init__(
            name="detect_duplicate_invoice",
            description=(
                "Check if an invoice already exists in SharePoint. "
                "Checks for exact filename matches and similar content."
            ),
            version="1.0.0",
        )

    async def execute(self, **kwargs: Any) -> ToolOutput:
        """Detect duplicate invoices.

        Args:
            filename: Proposed filename for the invoice
            invoice_data: Extracted invoice data (vendor, date, amount)
            sharepoint_folder_path: SharePoint folder path to search
            tenant_id: Tenant identifier

        Returns:
            ToolOutput with duplicate detection results:
            {
                "success": bool,
                "result": {
                    "is_duplicate": bool,
                    "exact_match": bool,  # Exact filename match
                    "similar_matches": [],  # List of similar invoices
                    "confidence": float  # Similarity confidence (0-1)
                }
            }
        """
        filename = kwargs.get("filename")
        invoice_data = kwargs.get("invoice_data", {})
        sharepoint_folder_path = kwargs.get("sharepoint_folder_path", "/")
        tenant_id = kwargs.get("tenant_id")

        if not tenant_id:
            return {
                "success": False,
                "result": None,
                "error": "tenant_id is required",
                "metadata": {},
            }

        try:
            # Check for exact filename match
            exact_match = await self._check_exact_filename(
                filename, sharepoint_folder_path, tenant_id
            )

            # Check for similar content
            similar_matches = await self._check_similar_content(
                invoice_data, sharepoint_folder_path, tenant_id
            )

            is_duplicate = exact_match["found"] or len(similar_matches) > 0

            logger.info(
                "Duplicate detection completed",
                extra={
                    "tenant_id": tenant_id,
                    "filename": filename,
                    "is_duplicate": is_duplicate,
                    "exact_match": exact_match["found"],
                    "similar_count": len(similar_matches),
                },
            )

            return {
                "success": True,
                "result": {
                    "is_duplicate": is_duplicate,
                    "exact_match": exact_match["found"],
                    "exact_match_file": exact_match.get("file_path"),
                    "similar_matches": similar_matches,
                    "confidence": 1.0
                    if exact_match["found"]
                    else (similar_matches[0]["confidence"] if similar_matches else 0.0),
                },
                "error": None,
                "metadata": {
                    "search_path": sharepoint_folder_path,
                },
            }

        except Exception as e:
            logger.exception("Duplicate detection failed", extra={"tenant_id": tenant_id})
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "metadata": {},
            }

    async def _check_exact_filename(
        self, filename: str, folder_path: str, tenant_id: str
    ) -> dict[str, Any]:
        """Check for exact filename match in SharePoint."""
        # TODO: Implement SharePoint file search
        # For now, return placeholder
        from app.tools.sharepoint.v1.scan_folders import scan_sharepoint_folders

        try:
            # Get files in folder
            folder_data = await scan_sharepoint_folders(
                folder_path=folder_path,
                tenant_id=tenant_id,
                include_files=True,
            )

            if folder_data.get("success"):
                files = folder_data.get("result", {}).get("files", [])
                for file in files:
                    if file.get("name") == filename:
                        return {
                            "found": True,
                            "file_path": file.get("path"),
                        }

            return {"found": False}
        except Exception as e:
            logger.warning(f"Error checking exact filename: {e}")
            return {"found": False}

    async def _check_similar_content(
        self, invoice_data: dict[str, Any], folder_path: str, tenant_id: str
    ) -> list[dict[str, Any]]:
        """Check for similar invoice content using LLM comparison."""
        # TODO: Implement content similarity check
        # For now, return empty list
        # This would:
        # 1. Get recent invoices from SharePoint
        # 2. Extract their content
        # 3. Use LLM to compare similarity
        # 4. Return matches with confidence scores

        vendor = invoice_data.get("vendor_name", "")
        amount = invoice_data.get("amount", 0)
        date = invoice_data.get("date", "")

        if not vendor or not amount:
            return []

        # Placeholder: would implement full similarity check
        return []

    def validate_input(self, **kwargs: Any) -> bool:
        """Validate input parameters."""
        filename = kwargs.get("filename")
        invoice_data = kwargs.get("invoice_data")
        tenant_id = kwargs.get("tenant_id")

        if not filename:
            return False

        if not invoice_data:
            return False

        if not tenant_id:
            return False

        return True
