from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import PlainTextResponse

from app.models.agent3 import TranscribeRequest
from app.models.jobs import JobCreated
from app.services.job_store import create_job, update_job, get_job
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
    Poll GET /jobs/{job_id} for status.
    Download all transcripts as a .txt file via GET /agent3/download/{job_id}.
    """
    job_id = create_job("agent3")
    background_tasks.add_task(_background_transcribe, job_id, req)
    return JobCreated(job_id=job_id)


@router.get("/download/{job_id}")
def download_transcripts(job_id: str):
    """Download all transcripts from this job as a single .txt file."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed yet (status: {job['status']})")

    result = job.get("result", {})
    transcripts = result.get("transcripts", [])

    lines = []
    for t in transcripts:
        lines.append(f"{'='*60}")
        lines.append(f"Title:    {t.get('title') or t.get('video_id', '')}")
        lines.append(f"URL:      https://www.youtube.com/watch?v={t.get('video_id', '')}")
        lines.append(f"Language: {t.get('language', 'unknown')}")
        lines.append(f"Source:   {t.get('method', 'unknown')}")
        lines.append(f"{'='*60}")
        lines.append("")
        lines.append(t.get("text") or f"[Unavailable: {t.get('error', 'no transcript')}]")
        lines.append("")

    # Prepend UTF-8 BOM so Windows/Notepad opens with correct encoding
    content = "\ufeff" + "\n".join(lines)
    return PlainTextResponse(
        content=content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=transcripts_{job_id[:8]}.txt"},
    )
