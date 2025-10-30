import os
from typing import Optional

from google.cloud import storage
import google.auth
from dotenv import load_dotenv


load_dotenv()


def expand_env(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    # Expand ${VAR} style using current environment
    return os.path.expandvars(value)


def resolve_project_id() -> Optional[str]:
    pid = (
        os.environ.get("PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
    )
    return pid


def ensure_bucket(bucket_name: str, location: Optional[str] = None, project_id: Optional[str] = None) -> storage.Bucket:
    """Get or create a GCS bucket.

    - bucket_name may contain env placeholders; it will be expanded.
    - location defaults to 'US' if not provided.
    - project_id is required to create the bucket if it doesn't exist.
    """
    bucket_name = expand_env(bucket_name) or bucket_name
    project_id = project_id or resolve_project_id()
    if not project_id:
        raise ValueError("PROJECT_ID is not set. Ensure .env is loaded or environment is configured.")

    # Use credentials with quota project override when available
    creds, _ = google.auth.default()
    if hasattr(creds, "with_quota_project"):
        try:
            creds = creds.with_quota_project(project_id)
        except Exception:
            pass
    client = storage.Client(project=project_id, credentials=creds)
    bucket = client.bucket(bucket_name)
    try:
        bucket.reload()
        return bucket
    except Exception:
        # Try creating the bucket
        loc = (location or os.environ.get("GCP_REGION") or "US")
        bucket.storage_class = "STANDARD"
        created = client.create_bucket(bucket, location=loc)
        return created


def ensure_prefix(bucket: storage.Bucket, prefix: str) -> None:
    """Ensure a 'folder-like' prefix exists by uploading a zero-byte placeholder.

    This is optional in GCS but helps with console UIs.
    """
    if not prefix:
        return
    placeholder = f"{prefix.rstrip('/')}/"
    blob = bucket.blob(placeholder)
    if not blob.exists():
        blob.upload_from_string(b"")


def upload_file(
    local_path: str,
    bucket: storage.Bucket,
    object_name: Optional[str] = None,
    prefix: Optional[str] = None,
) -> str:
    base = os.path.basename(local_path)
    key = f"{prefix.rstrip('/')}/{object_name or base}" if prefix else (object_name or base)
    if prefix:
        ensure_prefix(bucket, prefix)
    blob = bucket.blob(key)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket.name}/{key}"
