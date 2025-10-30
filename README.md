# TTML Translation Framework

Translate TTML subtitle files with two engines and upload outputs to GCS, while preserving TTML structure and timing.

- Engines:
  - `gemini`: Vertex AI Gemini (GenerativeModel) for natural, context-friendly translation.
  - `translate` / `translateLLM`: Cloud Translation API v3 for fast, reliable MT.
- Preserves TTML structure (header, styles, regions, timestamps). Only subtitle text inside `<p>/<span>` is translated.
- Writes translated files to `translated_outputs/` and uploads each to your configured GCS bucket/prefix.

## Pipelines

Common pipeline for both engines:

1) Read input `.ttml` file.
2) Parse TTML and collect text from `<span>` children under each `<p>` (keeps styles/regions/timestamps intact).
3) Translate those lines with the chosen engine, one-to-one, preserving order.
4) Replace the original `<span>` text with translated text.
5) Write output to `translated_outputs/<basename>_<lang>.ttml`.
6) Ensure the GCS bucket exists (create if missing), ensure the `OUTPUT_FOLDER` prefix exists, and upload the file.

Gemini route (`-engine gemini`):

- Uses Vertex AI GenerativeModel with a structure-preserving prompt that returns a JSON array the same length as the input lines.
- If the model returns malformed JSON or the wrong count, it falls back to per-line translations.
- Aims for the most natural, fluent subtitle phrasing suitable for TV/film dialogue.

Cloud Translation route (`-engine translate` or `-engine translateLLM`):

- Uses Cloud Translation API v3 with batched input to guarantee order and stable output.
- Optimized for speed and reliability; strong for literal-to-semi-natural translations.

## Engines compared (when to use what)

- Naturalness (idioms, tone):
  - Gemini: Stronger for conversational nuance and idiomatic phrasing.
  - Translation API: More literal; can be very good but prioritizes reliability and speed.
- Determinism:
  - Gemini: Slight variability (temperature kept low).
  - Translation API: Highly consistent across runs.
- Latency & throughput:
  - Gemini: Good; processed in small batches to maintain structure.
  - Translation API: Very fast and scalable for large volumes.
- Cost & quotas:
  - Both use your `.env` project for billing/quota; see “Project selection & quotas.”

## Requirements

- Python 3.9+
- Google Cloud project with APIs enabled:
  - Vertex AI API (for Gemini)
  - Cloud Translation API (for `translate/translateLLM`)
  - Cloud Storage API (for uploads)
- Authenticated GCP credentials (ADC):
  - `gcloud auth application-default login` and set active project
- `.env` file with at least:

```
PROJECT_ID=your-project-id
GCP_REGION=us-central1
BUCKET_NAME=subtitle-localization-bucket-${PROJECT_ID}
OUTPUT_FOLDER=output
```

Install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: enable APIs and set up venv automatically:

```bash
bash setup.sh -p your-project-id
```

## How to run

Translate with Gemini:

```bash
python main.py -f LimitlessWithChrisHemsworth_F005-Memory_eng_67365b5ec8c96d3a7d33572d.ttml -engine gemini -lang es,fr
```

Translate with Cloud Translation:

```bash
python main.py -f LimitlessWithChrisHemsworth_F005-Memory_eng_67365b5ec8c96d3a7d33572d.ttml -engine translateLLM -lang es,fr
```

Multiple languages: pass them comma-separated via `-lang` (e.g., `es,fr,de`).

Outputs:
- Local: `translated_outputs/<basename>_<lang>.ttml`
- GCS: `gs://$BUCKET_NAME/$OUTPUT_FOLDER/<basename>_<lang>.ttml`

## Project selection & quotas (important)

This project forces the `.env` `PROJECT_ID` at runtime so requests don’t accidentally use another project from your ADC config.

In `main.py`, the following environment variables are set before any clients are created:

- `GOOGLE_CLOUD_PROJECT`
- `GCLOUD_PROJECT`
- `CLOUDSDK_CORE_PROJECT`
- `GOOGLE_CLOUD_QUOTA_PROJECT`

Additionally, the Cloud Translation and Cloud Storage clients are constructed with credentials that carry the `.env` project as the quota/billing project. This ensures API calls and uploads are authorized and billed to your specified project.

Tip: you can still run `gcloud config set project <id>` to keep your CLI aligned, but it’s not required for this script.

## How it preserves TTML

- Parses XML and only modifies text nodes within `<span>` elements under `<p>`.
- Keeps all timing, style attributes (e.g., `tts:fontStyle="italic"`), regions, and namespaces unchanged.
- Writes with XML declaration and proper namespaces so it remains compatible with TTML/IMSC players.

## Troubleshooting

- “PermissionDenied/Service disabled”: Ensure the API is enabled on the `.env` project and that your ADC has permission.
- Bucket already exists and is owned by another project: pick a unique `BUCKET_NAME` (GCS names are global).
- BUCKET_NAME using `${PROJECT_ID}`: the script expands this automatically.
- If an engine fails mid-batch, the framework falls back safely to avoid corrupting alignment.
