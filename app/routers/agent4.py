from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import PlainTextResponse

from app.models.agent4 import SEOCopyRequest
from app.models.jobs import JobCreated
from app.services.job_store import create_job, update_job, get_job
from app.agents.agent4_seo_generator import run_seo_generation

router = APIRouter()


def _background_generate(job_id: str, req: SEOCopyRequest, transcripts: list[str]):
    update_job(job_id, "running")
    try:
        result = run_seo_generation(req, transcripts)
        update_job(job_id, "completed", result=result.model_dump())
    except Exception as exc:
        update_job(job_id, "failed", error=str(exc))


@router.post("/generate", response_model=JobCreated)
def generate_seo_copy(req: SEOCopyRequest, background_tasks: BackgroundTasks):
    """
    Start SEO text generation.

    - Provide transcripts via `transcript_texts` (raw text) or `transcript_job_ids`
      (references to completed Agent 3 jobs — transcripts are resolved server-side).
    - Poll `GET /jobs/{job_id}` for status.
    """
    # Resolve transcripts from Agent 3 job IDs
    transcripts = list(req.transcript_texts)
    for jid in req.transcript_job_ids:
        job = get_job(jid)
        if job and job.get("status") == "completed" and job.get("result"):
            for t in job["result"].get("transcripts", []):
                if t.get("text"):
                    transcripts.append(t["text"])

    if len(transcripts) > 5:
        transcripts = transcripts[:5]

    job_id = create_job("agent4")
    background_tasks.add_task(_background_generate, job_id, req, transcripts)
    return JobCreated(job_id=job_id)


@router.get("/download/{job_id}")
def download_seo_copy(job_id: str):
    """Download the generated SEO copy as a plain .txt file."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed yet (status: {job['status']})")

    text = job.get("result", {}).get("text", "")
    slug = job_id[:8]
    return PlainTextResponse(
        content=text,
        headers={"Content-Disposition": f"attachment; filename=seo_copy_{slug}.txt"},
    )
