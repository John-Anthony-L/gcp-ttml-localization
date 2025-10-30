from __future__ import annotations

from typing import List, Optional

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

    def translate_lines(self, lines: List[str], target_language: str, model: Optional[str] = None) -> List[str]:
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

        # API supports multiple contents preserving order
        request = {
            "parent": parent,
            "contents": lines,
            "target_language_code": target_language,
            "mime_type": "text/plain",
        }
        if model:
            request["model"] = model  # e.g., "general/base"

        response = self.client.translate_text(request=request)
        # Order matches
        return [t.translated_text for t in response.translations]
