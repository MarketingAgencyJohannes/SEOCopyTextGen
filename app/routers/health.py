from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/drive/test")
def test_drive_connection():
    """
    Test the Google Drive connection.
    Returns {"status": "ok", ...} or {"status": "error", "detail": "..."} with the real error.
    """
    try:
        from app.services.google_drive import test_connection
        result = test_connection()
        return result
    except Exception as exc:
        return JSONResponse(
            status_code=200,  # 200 so the UI can always read the body
            content={"status": "error", "detail": str(exc)},
        )
