"""
Google Drive upload service using a service account.

The full service account JSON is stored as the GOOGLE_SERVICE_ACCOUNT_JSON env var.
"""

import io
import json
import logging

from app.config import settings

logger = logging.getLogger(__name__)

_drive_service = None


def _get_drive_service():
    global _drive_service
    if _drive_service is not None:
        return _drive_service

    if not settings.google_service_account_json:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not set in environment variables")

    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    info = json.loads(settings.google_service_account_json)
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    _drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
    return _drive_service


def test_connection(folder_id: str | None = None) -> dict:
    """
    Test the Drive connection. Raises with a descriptive message on failure.
    Returns {"status": "ok", "folder_id": ..., "account": ...} on success.
    """
    service = _get_drive_service()
    folder = folder_id or settings.google_drive_folder_id

    # Try to get metadata for the target folder to confirm access
    if folder:
        file_meta = service.files().get(fileId=folder, fields="id, name").execute()
        folder_name = file_meta.get("name", folder)
    else:
        folder_name = "(no folder configured)"

    # Parse service account email from the JSON for display
    info = json.loads(settings.google_service_account_json)
    account_email = info.get("client_email", "unknown")

    return {
        "status": "ok",
        "folder_id": folder,
        "folder_name": folder_name,
        "service_account": account_email,
    }


def upload_bytes(
    content: bytes,
    filename: str,
    mimetype: str,
    folder_id: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Upload bytes to Google Drive.
    Returns (shareable_url, error_message).
    url is None and error is set if upload fails.
    """
    try:
        service = _get_drive_service()
    except Exception as exc:
        err = str(exc)
        logger.error("Drive client init failed for '%s': %s", filename, err)
        return None, err

    folder = folder_id or settings.google_drive_folder_id
    metadata = {"name": filename}
    if folder:
        metadata["parents"] = [folder]

    try:
        from googleapiclient.http import MediaIoBaseUpload

        media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mimetype, resumable=False)
        file = service.files().create(
            body=metadata, media_body=media, fields="id, webViewLink"
        ).execute()

        # Make the file readable by anyone with the link
        service.permissions().create(
            fileId=file["id"],
            body={"role": "reader", "type": "anyone"},
        ).execute()

        return file.get("webViewLink"), None
    except Exception as exc:
        err = str(exc)
        logger.error("Drive upload failed for '%s': %s", filename, err)
        return None, err
