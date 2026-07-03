import time
import re

from app.core.logging import get_logger

logger = get_logger(__name__)


class TextNormalizationService:
    """Service to normalize extracted text by removing excessive whitespace and duplicates."""

    def normalize_text(self, text: str) -> str:
        """Normalize whitespace, line endings, and consecutive blank lines.

        Args:
            text (str): Raw extracted text content.

        Returns:
            str: The cleaned, normalized text.
        """
        if not text:
            return ""

        start_time = time.perf_counter()

        # 1. Normalize line endings to \n
        normalized_endings = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Split into lines
        lines = normalized_endings.split("\n")

        # 3. Trim each line and collapse spaces/tabs to a single space
        cleaned_lines = []
        for line in lines:
            # Collapse multiple spaces and tabs to a single space, and strip the line
            cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
            cleaned_lines.append(cleaned_line)

        # 4. Remove duplicate blank lines (consecutive empty lines reduced to a single empty line)
        final_lines = []
        for line in cleaned_lines:
            if not line:
                # If this is a blank line, only add it if the previous line was NOT blank
                # also ensure we don't start with a blank line
                if final_lines and final_lines[-1] != "":
                    final_lines.append("")
            else:
                final_lines.append(line)

        # 5. Join lines and trim overall document whitespace
        result = "\n".join(final_lines).strip()

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            event="text_normalization_completed",
            duration_ms=round(duration_ms, 2),
            original_length=len(text),
            normalized_length=len(result),
        )

        return result
