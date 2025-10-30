from __future__ import annotations

import json
import os
from typing import List, Optional

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv


load_dotenv()


class GeminiTranslator:
    """Translate lists of short lines using Gemini with strong structure guarantees."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self.model_name = model_name
        # Init Vertex AI once
        project = (
            os.environ.get("PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
        )
        location = os.environ.get("GCP_REGION", "us-central1")
        if project:
            try:
                vertexai.init(project=project, location=location)
            except Exception:
                # Allow lazy init if env is not fully configured at import time
                pass

        self.model = GenerativeModel(self.model_name)
        self.safety = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

    def translate_lines(self, lines: List[str], target_language: str) -> List[str]:
        if not lines:
            return []

        # Single-call, whole-script translation for maximum context preservation.
        # Uses divide-and-conquer fallback if the response shape is invalid or request is too large.

        # Prompt ensures one-to-one mapping, natural phrasing, and minimal expansion for reading speed.
        rules = (
            "You are a professional subtitle translator. Translate each input line to "
            f"{target_language} using the most natural phrasing for TV/film dialogue.\n\n"
            "Return ONLY a JSON array of strings, exactly the same length and order as the input array.\n"
            "Do not add or remove lines, do not merge or split.\n"
            "Preserve speaker intent, tone, and register. Keep punctuation natural.\n"
            "Keep length close to source for reading speed (aim within ±15% characters per line when possible)."
        )

        response_schema = {
            "type": "array",
            "items": {"type": "string"},
        }

        gen_cfg = GenerationConfig(
            temperature=0.25,
            top_p=0.4,
            max_output_tokens=8192,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        def translate_chunk(chunk: List[str]) -> List[str]:
            # Try one request for this chunk.
            try:
                resp = self.model.generate_content(
                    [rules, json.dumps(chunk, ensure_ascii=False)],
                    generation_config=gen_cfg,
                    safety_settings=self.safety,
                    stream=False,
                )
                text = resp.text or "[]"
                arr = json.loads(text)
                if isinstance(arr, list) and len(arr) == len(chunk):
                    return [str(x) if x is not None else "" for x in arr]
            except Exception:
                pass

            # If we reach here, try to split the chunk to reduce size or recover from drift.
            if len(chunk) == 1:
                # Last resort, per-line fallback
                return self._fallback_per_line(chunk, target_language)
            mid = len(chunk) // 2
            left = translate_chunk(chunk[:mid])
            right = translate_chunk(chunk[mid:])
            return left + right

        return translate_chunk(lines)

    def _fallback_per_line(self, lines: List[str], target_language: str) -> List[str]:
        gen_cfg = GenerationConfig(
            temperature=0.2,
            top_p=0.4,
            max_output_tokens=1024,
            response_mime_type="text/plain",
        )
        out: List[str] = []
        for ln in lines:
            prompt = (
                "Translate the following subtitle line to "
                f"{target_language}. Return only the translation.\n\n"
                f"Line: {ln}"
            )
            try:
                resp = self.model.generate_content([prompt], generation_config=gen_cfg, safety_settings=self.safety)
                out.append((resp.text or "").strip())
            except Exception:
                out.append(ln)
        return out
