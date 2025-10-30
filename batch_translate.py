from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv

from ttml_translate import translate_ttml, write_ttml
from utils.gcs_utils import ensure_bucket, upload_file, resolve_project_id, expand_env
from engines.gemini_engine import GeminiTranslator
from engines.translate_llm_engine import CloudTranslateEngine


DEFAULT_LANGS = "en,de,fr-fr,pt-br,es-419,es-es,tr"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch translate all TTML files in a folder using Gemini or Cloud Translation."
    )
    p.add_argument(
        "-d",
        "--dir",
        default="English-Non-English OV Templates",
        help="Folder containing TTML files (default: 'English-Non-English OV Templates')",
    )
    p.add_argument(
        "-engine",
        default="translateLLM",
        choices=["gemini", "translate", "translateLLM"],
        help="Translation engine: 'gemini' (Vertex AI) or 'translate'/'translateLLM' (Cloud Translation)",
    )
    p.add_argument(
        "-lang",
        default=DEFAULT_LANGS,
        help=f"Comma-separated list of target languages (default: {DEFAULT_LANGS})",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories and process all *.ttml files",
    )
    p.add_argument(
        "--pattern",
        default="*.ttml",
        help="Glob pattern to match files (default: *.ttml)",
    )
    return p.parse_args()


def get_langs(spec: str) -> List[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def iter_files(root: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from root.rglob(pattern)
    else:
        yield from root.glob(pattern)


def main() -> None:
    # Load env and enforce project id for all downstream clients
    load_dotenv()
    project_id = os.environ.get("PROJECT_ID")
    if not project_id:
        raise SystemExit("PROJECT_ID must be set in .env or environment")
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GCLOUD_PROJECT"] = project_id
    os.environ["CLOUDSDK_CORE_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = project_id

    args = parse_args()
    src_dir = Path(args.dir)
    if not src_dir.exists() or not src_dir.is_dir():
        raise SystemExit(f"Input directory not found: {src_dir}")

    langs = get_langs(args.lang)
    if not langs:
        raise SystemExit("No target languages provided")

    # Prepare engine and output dirs
    engine_label = "gemini" if args.engine == "gemini" else "translateLLM"
    if args.engine == "gemini":
        engine = GeminiTranslator()

        def translate_fn(lines, lang):
            return engine.translate_lines(lines, lang)

    else:
        engine = CloudTranslateEngine()

        def translate_fn(lines, lang):
            return engine.translate_lines(lines, lang)

    local_out_root = Path(f"translated_outputs_{engine_label}")
    local_out_root.mkdir(parents=True, exist_ok=True)

    # GCS bucket/prefix
    bucket_name = expand_env(os.environ.get("BUCKET_NAME")) or os.environ.get("BUCKET_NAME")
    if not bucket_name:
        raise SystemExit("BUCKET_NAME must be set in .env or environment")
    output_prefix = os.environ.get("OUTPUT_FOLDER", "output").strip("/")
    gcs_bucket = ensure_bucket(bucket_name, location=os.environ.get("GCP_REGION"), project_id=project_id)

    count_files = 0
    for path in iter_files(src_dir, args.pattern, args.recursive):
        if not path.is_file():
            continue
        in_stem = path.stem
        # Translate once per target lang
        for lang in langs:
            try:
                tree, line_count = translate_ttml(str(path), translate_fn, lang)
                out_name = f"{in_stem}_{lang}_{engine_label}.ttml"
                out_path = local_out_root / out_name
                write_ttml(tree, str(out_path))
                gcs_uri = upload_file(str(out_path), gcs_bucket, object_name=out_name, prefix=output_prefix)
                print(f"[{engine_label}] {path.name} -> {out_path.name} ({line_count} lines) | {gcs_uri}")
            except Exception as e:
                print(f"[ERROR] Failed {path.name} lang={lang}: {e}")
        count_files += 1

    if count_files == 0:
        print("No files matched. Check --pattern or the directory contents.")
    else:
        print(f"\nDone. Processed {count_files} file(s) from: {src_dir}")


if __name__ == "__main__":
    main()
