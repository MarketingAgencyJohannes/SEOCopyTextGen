from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import PlainTextResponse

from app.models.agent2 import SerpAnalysisRequest
from app.models.jobs import JobCreated
from app.services.job_store import create_job, update_job, get_job
from app.agents.agent2_serp_analyzer import run_serp_analysis

router = APIRouter()


def _background_analyze(job_id: str, req: SerpAnalysisRequest):
    update_job(job_id, "running")
    try:
        result = run_serp_analysis(req)
        update_job(job_id, "completed", result=result.model_dump())
    except Exception as exc:
        update_job(job_id, "failed", error=str(exc))


@router.post("/analyze", response_model=JobCreated)
def analyze_serp(req: SerpAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze Google SERP for a keyword and identify content gaps.
    Poll GET /jobs/{job_id} for status. Typical duration: 2-5 minutes.
    Download the full report via GET /agent2/download/{job_id}.
    """
    job_id = create_job("agent2")
    background_tasks.add_task(_background_analyze, job_id, req)
    return JobCreated(job_id=job_id)


@router.get("/download/{job_id}")
def download_report(job_id: str):
    """Download the full content gap report as a .txt file."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed yet (status: {job['status']})")

    result = job.get("result", {})
    keyword = result.get("keyword", "report").replace(" ", "_")[:50]

    # Rebuild the plain-text report from stored result data
    gaps = result.get("content_gaps", [])
    saturated = result.get("saturated_topics", [])
    underserved = result.get("underserved_topics", [])

    page_summaries = result.get("page_summaries", [])

    lines = [
        f"CONTENT GAP REPORT — {result.get('keyword', '')}",
        "=" * 60,
        "",
        f"Pages analyzed: {result.get('pages_analyzed', 0)}",
        f"Pages failed:   {result.get('pages_failed', 0)}",
        f"Content gaps:   {len(gaps)}",
        "",
        "ANALYZED ARTICLES (Google Rankings):",
    ]
    for s in page_summaries:
        status_tag = "[ok]  " if s.get("crawl_status") == "ok" else "[FAIL]"
        title = s.get("title") or "No title"
        domain = s.get("domain") or ""
        pos = s.get("position", 0)
        lines.append(f"  {pos:2d}. {status_tag} {s.get('url', '')}  — {title} ({domain})")
    lines += [
        "",
        "SATURATED TOPICS (already well covered — avoid duplication):",
        *([f"  • {t}" for t in saturated] or ["  (none identified)"]),
        "",
        "UNDERSERVED TOPICS (low competition — worth targeting):",
        *([f"  • {t}" for t in underserved] or ["  (none identified)"]),
        "",
        "CONTENT GAP RECOMMENDATIONS (prioritised):",
    ]
    for i, gap in enumerate(gaps, 1):
        lines += [
            f"\n{i}. {gap.get('topic', '')}",
            f"   Suggested title:   {gap.get('suggested_title', '')}",
            f"   Competition level: {gap.get('competition_level', '')}",
            f"   Content type:      {gap.get('content_type', '')}",
            f"   Reasoning:         {gap.get('reasoning', '')}",
        ]

    # Prepend UTF-8 BOM so Windows/Notepad/Word opens with correct encoding
    report_text = "\ufeff" + "\n".join(lines)
    return PlainTextResponse(
        content=report_text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=gap_report_{keyword}.txt"},
    )
