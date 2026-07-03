import time
import fitz

from app.core.exceptions import ValidationException
from app.core.logging import get_logger

logger = get_logger(__name__)


class PDFExtractionService:
    """Service to handle raw text extraction from PDF documents using PyMuPDF."""

    def extract_text(self, pdf_bytes: bytes) -> str:
        """Extract text from all pages in the PDF document.

        Args:
            pdf_bytes (bytes): The raw PDF document bytes.

        Returns:
            str: The extracted raw text contents.

        Raises:
            ValidationException: If the PDF is empty, malformed, or cannot be parsed.
        """
        if not pdf_bytes:
            raise ValidationException("PDF bytes cannot be empty.")

        start_time = time.perf_counter()
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise ValidationException(f"Invalid PDF file: {str(e)}")

        try:
            page_count = doc.page_count
            if page_count == 0:
                raise ValidationException("PDF contains no pages.")

            text_pages = []
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                text_pages.append(page.get_text())
        except Exception as e:
            if isinstance(e, ValidationException):
                raise e
            raise ValidationException(f"Failed to extract text from PDF: {str(e)}")
        finally:
            doc.close()

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            event="pdf_extraction_completed",
            duration_ms=round(duration_ms, 2),
            page_count=page_count,
        )

        return "\n".join(text_pages)
