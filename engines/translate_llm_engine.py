from __future__ import annotations

from typing import List, Optional, Tuple

from google.cloud import translate_v3 as translate
from dotenv import load_dotenv
import google.auth
import os


load_dotenv()


class CloudTranslateEngine:
    """Translate batches of lines using Cloud Translation API v3."""

    def __init__(self, location: str = "global") -> None:
        self.location = location
        # Build client with credentials that explicitly use the PROJECT_ID as quota project
        creds, _ = google.auth.default()
        quota_proj = (
            os.environ.get("PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
        )
        if quota_proj and hasattr(creds, "with_quota_project"):
            try:
                creds = creds.with_quota_project(quota_proj)
            except Exception:
                pass
        self.client = translate.TranslationServiceClient(credentials=creds)

    def translate_lines(
        self,
        lines: List[str],
        target_language: str,
        source_language: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[str]:
        if not lines:
            return []

        project_id = (
            os.environ.get("PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
        )
        if not project_id:
            raise ValueError("PROJECT_ID is not set. Configure environment or .env.")

        parent = f"projects/{project_id}/locations/{self.location}"

        # Split into safe chunks to avoid request size limits
        chunks = _chunk_by_chars(lines, max_chars=80000, max_items=256)
        out: List[str] = []
        for chunk in chunks:
            # Cloud Translation rejects empty contents; filter them but keep indices to rebuild order.
            non_empty_indices = [i for i, s in enumerate(chunk) if (s or "").strip() != ""]
            if not non_empty_indices:
                # Entire chunk is empty/whitespace; pass through unchanged
                out.extend([s for s in chunk])
                continue

            filtered_contents = [chunk[i] for i in non_empty_indices]

            request = {
                "parent": parent,
                "contents": filtered_contents,
                "target_language_code": target_language,
                "mime_type": "text/plain",
            }
            if source_language:
                request["source_language_code"] = source_language
            if model:
                request["model"] = model  # e.g., "general/base"

            response = self.client.translate_text(request=request)
            translations = [t.translated_text for t in response.translations]
            # Reconstruct full chunk with translated non-empty items and original empties
            rebuilt = list(chunk)
            for idx, translated_text in zip(non_empty_indices, translations):
                rebuilt[idx] = translated_text
            out.extend(rebuilt)
        return out


def _chunk_by_chars(items: List[str], max_chars: int = 80000, max_items: int = 256) -> List[List[str]]:
    chunks: List[List[str]] = []
    cur: List[str] = []
    cur_len = 0
    for it in items:
        it_len = len(it or "")
        if cur and (cur_len + it_len > max_chars or len(cur) >= max_items):
            chunks.append(cur)
            cur = []
            cur_len = 0
        cur.append(it or "")
        cur_len += it_len
    if cur:
        chunks.append(cur)
    return chunks
