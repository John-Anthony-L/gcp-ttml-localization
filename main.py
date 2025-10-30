from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from ttml_translate import translate_ttml, write_ttml
from utils.gcs_utils import ensure_bucket, upload_file, resolve_project_id, expand_env
from engines.gemini_engine import GeminiTranslator
from engines.translate_llm_engine import CloudTranslateEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translate TTML subtitles using Gemini or Cloud Translation.")
    p.add_argument("-f", "--file", required=True, help="Path to input .ttml file")
    p.add_argument(
        "-engine",
        default="gemini",
        choices=["gemini", "translate", "translateLLM"],
        help="Translation engine: 'gemini' (Vertex AI Gemini) or 'translate'/'translateLLM' (Cloud Translation API)",
    )
    p.add_argument(
        "-lang",
        required=True,
        help="Comma-separated list of target language codes (e.g., 'es,fr,de')",
    )
    return p.parse_args()


def get_langs(spec: str) -> List[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def main() -> None:
    # Load .env and force the project to the .env PROJECT_ID for all downstream clients
    load_dotenv()
    project_id = os.environ.get("PROJECT_ID")
    if not project_id:
        raise SystemExit("PROJECT_ID must be set in .env or environment to run main.py")
    # Force all Google client libraries to resolve this project
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GCLOUD_PROJECT"] = project_id
    # Optional: also set Cloud SDK project var to avoid surprises
    os.environ["CLOUDSDK_CORE_PROJECT"] = project_id
    # Critical: override quota/billing project used by google-auth (x-goog-user-project)
    os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = project_id
    args = parse_args()

    input_path = args.file
    if not os.path.exists(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    langs = get_langs(args.lang)
    if not langs:
        raise SystemExit("No target languages provided")

    # Prepare engines and derive output labeling/paths
    translate_fn_factory = None
    engine_label = "gemini" if args.engine == "gemini" else "translateLLM"
    if args.engine == "gemini":
        gem = GeminiTranslator()

        def translate_fn(lines, lang):
            return gem.translate_lines(lines, lang)

        translate_fn_factory = translate_fn
    else:
        ct = CloudTranslateEngine()

        def translate_fn(lines, lang):
            return ct.translate_lines(lines, lang)

        translate_fn_factory = translate_fn

    # Local output directory (engine-specific)
    local_out_dir = Path(f"translated_outputs_{engine_label}")
    local_out_dir.mkdir(parents=True, exist_ok=True)

    # GCS bucket and prefix
    bucket_name = expand_env(os.environ.get("BUCKET_NAME")) or os.environ.get("BUCKET_NAME")
    if not bucket_name:
        raise SystemExit("BUCKET_NAME must be set in .env or environment")
    output_prefix = os.environ.get("OUTPUT_FOLDER", "output").strip("/")

    # Ensure bucket exists (explicitly pass forced project_id)
    gcs_bucket = ensure_bucket(bucket_name, location=os.environ.get("GCP_REGION"), project_id=project_id)

    in_stem = Path(input_path).stem
    results: List[str] = []
    gcs_results: List[str] = []

    for lang in langs:
        tree, count = translate_ttml(input_path, translate_fn_factory, lang)
        # Include engine label in the filename for traceability
        out_name = f"{in_stem}_{lang}_{engine_label}.ttml"
        out_path = local_out_dir / out_name
        write_ttml(tree, str(out_path))
        results.append(str(out_path))

        # Upload to GCS
        gcs_uri = upload_file(str(out_path), gcs_bucket, object_name=out_name, prefix=output_prefix)
        gcs_results.append(gcs_uri)
        print(f"Translated {count} lines -> {out_path} | Uploaded: {gcs_uri}")

    print("\nDone. Outputs:")
    for p, u in zip(results, gcs_results):
        print(f"- {p}  |  {u}")


if __name__ == "__main__":
    main()
