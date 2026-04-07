from fastapi import APIRouter, BackgroundTasks

from app.models.agent3 import TranscribeRequest
from app.models.jobs import JobCreated
from app.services.job_store import create_job, update_job
from app.agents.agent3_transcriber import run_transcribe

router = APIRouter()


def _background_transcribe(job_id: str, req: TranscribeRequest):
    update_job(job_id, "running")
    try:
        result = run_transcribe(req)
        update_job(job_id, "completed", result=result.model_dump())
    except Exception as exc:
        update_job(job_id, "failed", error=str(exc))


@router.post("/transcribe", response_model=JobCreated)
def transcribe_videos(req: TranscribeRequest, background_tasks: BackgroundTasks):
    """
    Transcribe 1-20 YouTube videos.

    - Tries youtube-transcript-api first (free, instant).
    - Falls back to Whisper if no caption track is available.
    - Poll `GET /jobs/{job_id}` for status and results.
    - Pass the returned `job_id` to Agent 4's `transcript_job_ids` field.
    """
    job_id = create_job("agent3")
    background_tasks.add_task(_background_transcribe, job_id, req)
    return JobCreated(job_id=job_id)
