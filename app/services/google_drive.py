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
        logger.warning("GOOGLE_SERVICE_ACCOUNT_JSON not set — Drive upload disabled")
        return None

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        info = json.loads(settings.google_service_account_json)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive"]
        )
        _drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return _drive_service
    except Exception as exc:
        logger.error("Failed to initialise Google Drive client: %s", exc)
        return None


def upload_bytes(
    content: bytes,
    filename: str,
    mimetype: str,
    folder_id: str | None = None,
) -> str | None:
    """
    Upload bytes to Google Drive.
    Returns a shareable URL or None if upload fails / Drive not configured.
    """
    service = _get_drive_service()
    if service is None:
        return None

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

        return file.get("webViewLink")
    except Exception as exc:
        logger.error("Drive upload failed for '%s': %s", filename, exc)
        return None
