from fastapi import APIRouter, BackgroundTasks

from app.models.agent2 import SerpAnalysisRequest
from app.models.jobs import JobCreated
from app.services.job_store import create_job, update_job
from app.agents.agent2_serp_analyzer import run_serp_analysis

router = APIRouter()


def _background_analyze(job_id: str, req: SerpAnalysisRequest):
    update_job(job_id, "running")
    try:
        result = run_serp_analysis(req)
        # Trim page_summaries body text before storing (it's large)
        result_dict = result.model_dump()
        update_job(job_id, "completed", result=result_dict)
    except Exception as exc:
        update_job(job_id, "failed", error=str(exc))


@router.post("/analyze", response_model=JobCreated)
def analyze_serp(req: SerpAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze Google SERP for a keyword and identify content gaps.

    - Fetches top 30 organic results via Serper.dev (2,500 free searches/month).
    - Crawls each page with Playwright + BeautifulSoup.
    - Uses Claude to identify saturated, underserved, and missing topics.
    - Uploads a human-readable gap report to Google Drive.
    - Poll `GET /jobs/{job_id}` for status. Typical duration: 2-5 minutes.
    """
    job_id = create_job("agent2")
    background_tasks.add_task(_background_analyze, job_id, req)
    return JobCreated(job_id=job_id)
