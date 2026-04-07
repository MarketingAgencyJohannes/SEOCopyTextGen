from fastapi import APIRouter, BackgroundTasks

from app.models.agent1 import ChannelScrapeRequest
from app.models.jobs import JobCreated
from app.services.job_store import create_job, update_job
from app.agents.agent1_youtube_scraper import run_channel_scrape

router = APIRouter()


def _background_scrape(job_id: str, req: ChannelScrapeRequest):
    update_job(job_id, "running")
    try:
        result = run_channel_scrape(req)
        # Only store metadata in the job; full video list can be large
        result_dict = result.model_dump()
        result_dict["videos"] = result_dict["videos"][:50]  # cap stored rows
        result_dict["note"] = "Full dataset available in Google Drive links above"
        update_job(job_id, "completed", result=result_dict)
    except Exception as exc:
        update_job(job_id, "failed", error=str(exc))


@router.post("/scrape", response_model=JobCreated)
def scrape_channel(req: ChannelScrapeRequest, background_tasks: BackgroundTasks):
    """
    Crawl all public videos from a YouTube channel.

    - Supports both `@handle` and `/channel/ID` URL formats.
    - Outputs CSV + Excel to Google Drive.
    - Poll `GET /jobs/{job_id}` for status and Drive links.
    """
    job_id = create_job("agent1")
    background_tasks.add_task(_background_scrape, job_id, req)
    return JobCreated(job_id=job_id)
