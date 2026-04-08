import io
import csv

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from app.models.agent1 import ChannelScrapeRequest
from app.models.jobs import JobCreated
from app.services.job_store import create_job, update_job, get_job
from app.agents.agent1_youtube_scraper import run_channel_scrape

router = APIRouter()


def _background_scrape(job_id: str, req: ChannelScrapeRequest):
    update_job(job_id, "running")
    try:
        result = run_channel_scrape(req)
        result_dict = result.model_dump()
        # Store ALL videos in the job for direct download
        update_job(job_id, "completed", result=result_dict)
    except Exception as exc:
        update_job(job_id, "failed", error=str(exc))


@router.post("/scrape", response_model=JobCreated)
def scrape_channel(req: ChannelScrapeRequest, background_tasks: BackgroundTasks):
    """
    Crawl all public videos from a YouTube channel.

    - Supports both `@handle` and `/channel/ID` URL formats.
    - Outputs CSV + Excel to Google Drive.
    - Also available for direct download via GET /agent1/download/{job_id}.
    - Poll `GET /jobs/{job_id}` for status.
    """
    job_id = create_job("agent1")
    background_tasks.add_task(_background_scrape, job_id, req)
    return JobCreated(job_id=job_id)


@router.get("/download/{job_id}")
def download_csv(job_id: str):
    """Download the scraped video list as a CSV file directly."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed yet (status: {job['status']})")

    result = job.get("result", {})
    videos = result.get("videos", [])
    channel = result.get("channel_title", "channel").replace(" ", "_")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["position", "title", "url", "video_id", "description", "published_at", "view_count", "duration"])
    writer.writeheader()
    for v in videos:
        writer.writerow({k: v.get(k, "") for k in writer.fieldnames})

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{channel}_videos.csv"'},
    )
