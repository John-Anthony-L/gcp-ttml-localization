import json
import os
import vertexai
import google.auth
from vertexai import generative_models
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
    HarmCategory,
    HarmBlockThreshold,
)

from google.cloud import storage
from google.api_core.exceptions import Conflict
from google.oauth2 import service_account
from dotenv import load_dotenv
# from google.cloud import texttospeech

load_dotenv()

def _resolve_project_id() -> str | None:
    """Resolve active GCP project id with precedence:
    1) PROJECT_ID env, 2) GOOGLE_CLOUD_PROJECT, 3) GCLOUD_PROJECT, 4) ADC default()
    """
    pid = (
        os.environ.get("PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
    )
    if pid:
        return pid
    try:
        creds, detected = google.auth.default()
        return detected
    except Exception:
        return None

def _ensure_vertexai_init():
    pid = _resolve_project_id()
    loc = os.environ.get("GCP_REGION", "us-central1")
    if pid and loc:
        try:
            vertexai.init(project=pid, location=loc)
        except Exception:
            pass
    else:
        print("Warning: Could not resolve PROJECT_ID/GCP_REGION for Vertex AI initialization")


# -------------------------------
# Internal GCS helper utilities
# -------------------------------
def _get_or_create_bucket(project_id: str | None, preferred_bucket: str | None = None) -> storage.Bucket:
    """Get or create a GCS bucket for Gemini assets.

    Bucket name resolution order:
      1) preferred_bucket (env GEMINI_ASSETS_BUCKET)
      2) f"{project_id}-gemini-assets"
    """
    if not project_id:
        raise ValueError("Could not resolve PROJECT_ID. Set PROJECT_ID in .env or run 'gcloud auth application-default login' and ensure a default project is set.")
    client = storage.Client(project=project_id)
    # Respect BUCKET_NAME from .env if provided
    env_bucket = os.environ.get("BUCKET_NAME")
    bucket_name = preferred_bucket or env_bucket or os.environ.get("GEMINI_ASSETS_BUCKET") or f"{project_id}-gemini-assets"
    bucket = client.bucket(bucket_name)
    try:
        bucket.reload()
        return bucket
    except Exception as e:
        # If user explicitly specified BUCKET_NAME and it doesn't exist or is inaccessible,
        # attempt creation; if name is globally taken, raise a clear error.
        try:
            location = os.environ.get("GCP_REGION", "us-central1")
            bucket = client.create_bucket(bucket_name, location=location)
            return bucket
        except Conflict:
            raise ValueError(
                "BUCKET_NAME is set to a globally unavailable bucket name. "
                f"Tried to create '{bucket_name}' but it already exists in another project. "
                "Please set BUCKET_NAME in .env to a unique name (e.g., '<project>-assets-<random>') "
                "and update INPUT_FOLDER/OUTPUT_FOLDER accordingly, or use an existing bucket you control."
            ) from None


def _upload_file_to_bucket(local_path: str, bucket: storage.Bucket, object_name: str | None = None) -> str:
    from pathlib import Path
    # Use INPUT_FOLDER as a prefix if provided. Accept either:
    # - "gs://bucket/prefix" (bucket part ignored; BUCKET_NAME is authoritative)
    # - "prefix" (folder name under the selected bucket)
    input_folder = os.environ.get("INPUT_FOLDER")
    prefix = None
    if input_folder:
        if input_folder.startswith("gs://"):
            # Parse gs://bucket/prefix
            try:
                _, path = input_folder.split("gs://", 1)
                parts = path.split("/", 1)
                in_prefix = parts[1] if len(parts) > 1 else ""
                prefix = in_prefix.strip("/") or None
            except Exception:
                prefix = None
        else:
            prefix = input_folder.strip("/") or None

    base_name = object_name or Path(local_path).name
    obj_name = f"{prefix}/{base_name}" if prefix else base_name
    # Ensure the GCS "folder" exists (optional, for UI convenience)
    if prefix:
        _ensure_prefix_exists(bucket, prefix)
    blob = bucket.blob(obj_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket.name}/{obj_name}"


def _ensure_prefix_exists(bucket: storage.Bucket, prefix: str) -> None:
    """Create a zero-byte placeholder object to mimic a folder in GCS."""
    placeholder = f"{prefix.rstrip('/')}/"
    blob = bucket.blob(placeholder)
    if not blob.exists():
        try:
            blob.upload_from_string("")
        except Exception:
            pass

def generate(parts, response_schema=None):
    """Wrapper around model.generate_content with sane defaults.

        # Ensure the "folder" exists by writing a placeholder if needed
        if prefix:
            _ensure_prefix_exists(bucket, prefix)
    - Coerces any plain-text parts into Part.from_text to avoid proto parsing issues.
    - Leaves binary/URI parts as-is.
    """
    _ensure_vertexai_init()
    model = GenerativeModel("gemini-2.5-flash")
    def _ensure_prefix_exists(bucket: storage.Bucket, prefix: str) -> None:
        """Create a zero-byte placeholder for a GCS prefix to mimic folder existence."""
        # GCS doesn't need folders, but some UIs prefer a placeholder object ending with '/'
        placeholder = f"{prefix.rstrip('/')}/"
        blob = bucket.blob(placeholder)
        # Only create if it doesn't exist
        if not blob.exists():
            try:
                blob.upload_from_string("")
            except Exception:
                # Non-fatal if we can't create the placeholder
                pass

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
    
    if response_schema == None:
        generation_config = GenerationConfig(
            max_output_tokens=65535,
            temperature=1.2,
            top_p=0.7,
            response_mime_type="application/json",
        )
    else:
        generation_config = GenerationConfig(
            temperature=1.2,
            top_p=0.7,
            max_output_tokens=65535,  # Increase token limit
            response_mime_type="application/json",
            response_schema=response_schema
        )
        
    # Normalize parts: ensure strings are wrapped as text parts
    normalized_parts = [Part.from_text(p) if isinstance(p, str) else p for p in parts]

    response = model.generate_content(
        normalized_parts,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False,
    )
    
    return response.text


def extract_scene_metadata_diarized_transcript(video_path: str):
    """
    Extract diarized transcript metadata from a video scene.

    Accepts a local file path; the video will be uploaded to the configured
    GCS bucket/prefix and referenced by URI for the Gemini call (to avoid
    large inline blobs).
    """
    try:
        # Upload local video to GCS and use URI for Gemini
        bucket = _get_or_create_bucket(_resolve_project_id())
        gs_uri = _upload_file_to_bucket(video_path, bucket)
        video_part = Part.from_uri(uri=gs_uri, mime_type="video/mp4")

        prompt = """
        Analyze the provided video scene with meticulous attention to detail and extract the following metadata in a strict JSON format adhering to the specified schema. Your analysis should be comprehensive, accurate, and suitable for production use.

        **Video Scene Analysis and Diarized Transcript Extraction:**

        1.  **Diarized Transcript (`diarized_transcript`):**
            * Generate a detailed diarized transcript of the video scene.
            * For each speaker, identify:
                * **Person:** If possible, identify the speaker by name. If not, use generic identifiers like "Speaker 1," "Speaker 2," etc.
                * **Script:** Transcribe the speaker's spoken words verbatim.
            * Structure the transcript as a JSON list, with each element being a dictionary containing the speaker's name or identifier and their spoken words.
            * Example:
            ```json
            {{
                "diarized_transcript": [
                    {{"Speaker 1": "I'm very excited about this project."}},
                    {{"Speaker 2": "However, there are some potential risks."}},
                    {{"Speaker 1": "I understand your concerns."}}
                ]
            }}
            ```

        **Strict JSON Response Schema:**

        Your response must be a valid JSON object conforming to the following schema:
        """
        response_schema = {
            "type": "object",
            "properties": {
            "diarized_transcript": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "Person": {"type":"string"},
                        "Script": {"type":"string"}
                    }
                }
            }
            },
            "required": [
            "diarized_transcript"
            ]
        }

        response = generate([prompt, video_part], response_schema)

        try:
            metadata = json.loads(response)
            return json.dumps(metadata, indent=2, ensure_ascii=False)

        except json.JSONDecodeError:
            return f"Error decoding JSON response: {response}"

    except Exception as e:
        return f"Error processing video: {e}"


########################################################
# Guided diarized transcript using ASR tokens
########################################################
def extract_scene_guided_diarized_transcript(
    video_path: str,
    transcription,
    max_guidance_words: int = 8000,
):
    """
    Generate a diarized transcript using Gemini, guided by the ASR transcription.

    Inputs:
      - video_path: local path to the scene video; will be uploaded to GCS and referenced by URI
      - transcription: either the Python dict produced by transcribe_video() or a path to that JSON file
      - max_guidance_words: safety cap on how many ASR words to include in the guidance prompt

    Output:
      - JSON string containing an object with 'diarized_transcript': list of utterances.
        Each utterance includes:
          - Person: string
          - Script: string (Gemini's best text)
          - TokenStart: integer (start index into ASR word sequence; -1 if unknown)
          - TokenEnd: integer (end index into ASR word sequence; -1 if unknown)
          - start_time: float seconds (mapped from ASR based on TokenStart)
          - end_time: float seconds (mapped from ASR based on TokenEnd)
    """

    def _normalize_text(s: str) -> str:
        import re as _re
        s = s.lower()
        s = _re.sub(r"[^a-z0-9\s]", " ", s)
        s = _re.sub(r"\s+", " ", s).strip()
        return s

    def _flatten_words(trans):
        # Accept a dict or a path to JSON
        import json as _json
        from pathlib import Path as _Path
        if isinstance(trans, str):
            try:
                trans = _json.loads(_Path(trans).read_text(encoding="utf-8"))
            except Exception:
                # If string but not a path or fails to load, assume it's already JSON text
                trans = _json.loads(trans)
        words = []
        for entry in trans.get("transcriptions", []):
            if entry.get("alternative") != 1:
                continue
            for w in entry.get("words", []):
                text = str(w.get("word", ""))
                norm = _normalize_text(text)
                if not norm:
                    continue
                words.append({
                    "word": text,
                    "norm": norm,
                    "start": float(w.get("start_time", 0.0)),
                    "end": float(w.get("end_time", 0.0)),
                })
        # Ensure chronological order
        words.sort(key=lambda x: (x["start"], x["end"]))
        return words

    try:
        # Upload video and build guidance
        bucket = _get_or_create_bucket(_resolve_project_id())
        gs_uri = _upload_file_to_bucket(video_path, bucket)
        video_part = Part.from_uri(uri=gs_uri, mime_type="video/mp4")

        asr_words = _flatten_words(transcription)
        total = len(asr_words)
        limit = min(total, max_guidance_words)
        # Build a compact guidance text with indexes and normalized tokens to constrain alignment
        lines = [
            "ASR_GUIDANCE: This is the canonical ASR word sequence. Use contiguous spans of these token indexes to anchor each utterance.",
            "For each utterance, return TokenStart and TokenEnd as inclusive indexes into this sequence.",
            "If you cannot confidently align, set both indexes to -1.",
            f"Token count provided: {limit} of {total} (0-based indexes).",
            "Format: i=INDEX; w=WORD",
        ]
        for i in range(limit):
            lines.append(f"i={i}; w={asr_words[i]['norm']}")
        asr_guidance_text = "\n".join(lines)

        prompt = (
            "You are given a video scene and an ASR word sequence from an automatic transcript. "
            "Your job is to produce a diarized transcript with the best possible verbatim Script, "
            "but anchor each utterance to a contiguous span in the ASR sequence using TokenStart/TokenEnd (inclusive). "
            "Respect the chronological order: each subsequent utterance should occur later in the sequence than the previous one. "
            "Choose the span that best covers the words in the Script (allowing minor ASR differences). "
            "Do not invent indexes outside the provided range. If unsure, use -1 for both. "
            "Return only valid JSON conforming to the schema."
        )

        response_schema = {
            "type": "object",
            "properties": {
                "diarized_transcript": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Person": {"type": "string"},
                            "Script": {"type": "string"},
                            "TokenStart": {"type": "integer", "minimum": -1},
                            "TokenEnd": {"type": "integer", "minimum": -1}
                        },
                        "required": ["Person", "Script", "TokenStart", "TokenEnd"]
                    }
                }
            },
            "required": ["diarized_transcript"]
        }

        response = generate([prompt, video_part, asr_guidance_text], response_schema)

        # Parse and post-process into times
        try:
            obj = json.loads(response)
        except json.JSONDecodeError as e:
            tail = response[-400:] if isinstance(response, str) else ""
            return (
                f"Error decoding JSON response: {e}.\n"
                f"Response (tail): {tail}"
            )

        diar = obj.get("diarized_transcript")
        if not isinstance(diar, list):
            return f"Error: Expected 'diarized_transcript' list, got {type(diar)}"

        # Map token spans to start/end times; clamp and handle -1
        results = []
        for i, item in enumerate(diar):
            if not isinstance(item, dict):
                continue
            person = str(item.get("Person", "")).strip() or "Unknown"
            script = str(item.get("Script", "")).strip()
            ts = item.get("TokenStart", -1)
            te = item.get("TokenEnd", -1)

            start_time = None
            end_time = None
            if isinstance(ts, int) and isinstance(te, int) and ts >= 0 and te >= ts:
                ts_clamp = max(0, min(ts, limit - 1)) if limit > 0 else -1
                te_clamp = max(0, min(te, limit - 1)) if limit > 0 else -1
                if ts_clamp >= 0 and te_clamp >= ts_clamp and limit > 0:
                    start_time = float(asr_words[ts_clamp]["start"]) if ts_clamp < len(asr_words) else None
                    end_time = float(asr_words[te_clamp]["end"]) if te_clamp < len(asr_words) else None

            results.append({
                "Person": person,
                "Script": script,
                "TokenStart": ts if isinstance(ts, int) else -1,
                "TokenEnd": te if isinstance(te, int) else -1,
                "start_time": start_time,
                "end_time": end_time,
            })

        final_obj = {"diarized_transcript": results, "asr_token_count_used": limit, "asr_token_count_total": total}
        return json.dumps(final_obj, indent=2, ensure_ascii=False)

    except Exception as e:
        return f"Error processing video: {e}"


########################################################
# Translate WebVTT subtitles to another language (e.g., Spanish)
########################################################
def translate_webvtt_to_language(
    vtt_path: str,
    target_language: str = "es",
    keep_speaker_prefix: bool = True,
    model_name: str = "gemini-2.5-flash",
) -> str:
    """
    Translate a WebVTT file's subtitle text to the target language while preserving
    WebVTT structure, header, cue indices, timestamps, and settings.

    Args:
        vtt_path: Path to the .vtt file to translate.
        target_language: BCP-47 language code (e.g., 'es', 'es-ES').
        keep_speaker_prefix: If a line contains a speaker prefix like 'Name: text',
                                keep the prefix before the first colon unchanged and
                                translate only the text after the colon.
        model_name: Vertex AI model to use.

    Returns:
        The translated WebVTT content as a UTF-8 string (complete .vtt text).

    Notes:
        - This function returns text/plain VTT, not JSON.
        - Timecodes and cue ordering are preserved.
        - Only subtitle text is translated.
    """
    from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold
    from pathlib import Path as _Path

    original_vtt = _Path(vtt_path).read_text(encoding="utf-8")

    # Clear, constrained prompt to preserve structure.
    rules = f"""
You are given a WebVTT subtitle file. Translate ONLY the spoken subtitle text to {target_language}.

Strict rules:
- Preserve the exact WebVTT structure:
    - Keep the 'WEBVTT' header line if present.
    - Preserve cue numbers (indexes), timestamps, settings (e.g., position, align), notes, and empty lines.
    - Do not change timestamps or add/remove cues.
- If a text line contains a speaker prefix in the form 'PREFIX: content':
    - {'Keep PREFIX unchanged and translate only content.' if keep_speaker_prefix else 'Translate the entire line, including PREFIX.'}
- Preserve any simple inline markup or formatting; translate the text content only.
- Return ONLY the fully-formed .vtt text; no explanations, no JSON, no additional commentary.
"""

    model = GenerativeModel(model_name)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
    gen_cfg = GenerationConfig(
        temperature=0.2,  # low temperature to preserve structure
        top_p=0.3,
        max_output_tokens=65535,
        response_mime_type="text/plain",
    )

    response = model.generate_content(
        [rules, original_vtt],
        generation_config=gen_cfg,
        safety_settings=safety_settings,
        stream=False,
    )

    # Return raw text; caller may write to disk
    return response.text


def translate_vtt_file_to_spanish(vtt_path: str, out_path: str | None = None) -> str:
    """
    Convenience wrapper to translate a .vtt file to Spanish and write the result.

    Args:
        vtt_path: Source .vtt file.
        out_path: Optional output path; defaults to '<stem>_es.vtt' next to input.

    Returns:
        The output file path written.
    """
    from pathlib import Path as _Path

    dest = _Path(out_path) if out_path else _Path(vtt_path).with_name(_Path(vtt_path).stem + "_es.vtt")
    translated = translate_webvtt_to_language(vtt_path, target_language="es")
    dest.write_text(translated, encoding="utf-8")
    return str(dest)