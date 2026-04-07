"""
Agent 2 — Google SERP Content Gap Analyzer

Pipeline:
  1. Serper.dev API — fetch top N organic results
  2. Playwright + BeautifulSoup — crawl each page (async, semaphore=5)
  3. Claude — summarize pages and identify content gaps
  4. Output: gap report uploaded to Google Drive
"""

import asyncio
import json
import logging
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from app.config import settings
from app.models.agent2 import (
    ContentGap,
    PageSummary,
    SerpAnalysisRequest,
    SerpAnalysisResult,
)
from app.services.claude_client import complete
from app.services.google_drive import upload_bytes

logger = logging.getLogger(__name__)

CRAWL_TIMEOUT_SEC = 10
CRAWL_SEMAPHORE = 5
MAX_PAGE_CHARS = 3000  # characters sent to Claude per page


# ---------------------------------------------------------------------------
# Step 1 — Serper.dev search
# ---------------------------------------------------------------------------

async def _serper_search(keyword: str, num: int, language: str) -> list[dict]:
    if not settings.serper_api_key:
        raise ValueError("SERPER_API_KEY is not set")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": settings.serper_api_key, "Content-Type": "application/json"},
            json={"q": keyword, "num": num, "hl": language},
        )
        resp.raise_for_status()
        data = resp.json()

    organic = data.get("organic", [])
    return [
        {
            "position": item.get("position", i + 1),
            "url": item.get("link", ""),
            "title": item.get("title", ""),
            "domain": urlparse(item.get("link", "")).netloc,
        }
        for i, item in enumerate(organic)
    ]


# ---------------------------------------------------------------------------
# Step 2 — Playwright page crawl
# ---------------------------------------------------------------------------

async def _crawl_page(page, url: str, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        try:
            await page.goto(url, timeout=CRAWL_TIMEOUT_SEC * 1000, wait_until="domcontentloaded")
            html = await page.content()
            return {"url": url, "html": html, "status": "ok"}
        except Exception as exc:
            logger.warning("Crawl failed for %s: %s", url, exc)
            return {"url": url, "html": None, "status": "failed"}


async def _crawl_all(urls: list[str]) -> list[dict]:
    semaphore = asyncio.Semaphore(CRAWL_SEMAPHORE)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        pages = [await context.new_page() for _ in urls]
        tasks = [_crawl_page(pages[i], url, semaphore) for i, url in enumerate(urls)]
        results = await asyncio.gather(*tasks)
        await browser.close()
    return results


def _parse_html(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    # Remove nav, footer, ads
    for tag in soup(["nav", "footer", "header", "aside", "script", "style"]):
        tag.decompose()

    h2_headings = [h.get_text(strip=True) for h in soup.find_all("h2")][:10]
    h3_headings = [h.get_text(strip=True) for h in soup.find_all("h3")][:10]

    body = soup.get_text(separator=" ", strip=True)
    word_count = len(body.split())

    return {
        "body_text": body[:MAX_PAGE_CHARS],
        "h2_headings": h2_headings,
        "h3_headings": h3_headings,
        "word_count": word_count,
    }


# ---------------------------------------------------------------------------
# Step 3 — Claude analysis
# ---------------------------------------------------------------------------

def _summarize_pages(serp_items: list[dict], crawl_results: list[dict]) -> list[PageSummary]:
    summaries: list[PageSummary] = []

    for serp, crawl in zip(serp_items, crawl_results):
        if crawl["status"] == "failed" or not crawl["html"]:
            summaries.append(PageSummary(
                position=serp["position"],
                url=serp["url"],
                title=serp.get("title"),
                domain=serp.get("domain"),
                crawl_status="failed",
            ))
            continue

        parsed = _parse_html(crawl["html"])
        heading_text = "; ".join(parsed["h2_headings"][:5])
        content_type = _classify_content(serp["url"], parsed["body_text"])

        summaries.append(PageSummary(
            position=serp["position"],
            url=serp["url"],
            title=serp.get("title"),
            domain=serp.get("domain"),
            word_count=parsed["word_count"],
            h2_headings=parsed["h2_headings"],
            content_type=content_type,
            crawl_status="ok",
        ))

    return summaries


def _classify_content(url: str, body: str) -> str:
    body_lower = body.lower()
    if any(w in body_lower for w in ["kaufen", "preis", "angebot", "buy", "price", "shop"]):
        return "commercial"
    if any(w in url.lower() or w in body_lower for w in ["stadtplan", "freiburg", "berlin", "münchen"]):
        return "local"
    if any(w in body_lower for w in ["was ist", "wie funktioniert", "what is", "how to"]):
        return "informational"
    return "mixed"


def _identify_gaps(keyword: str, summaries: list[PageSummary]) -> dict:
    all_headings = []
    for s in summaries:
        if s.crawl_status == "ok":
            all_headings.extend(s.h2_headings)

    headings_text = "\n".join(f"- {h}" for h in all_headings[:150])
    page_titles = "\n".join(
        f"{s.position}. {s.title} ({s.domain})" for s in summaries[:30]
    )

    system = (
        "You are an SEO content strategist. Analyse the provided SERP data "
        "and return a JSON object with these keys:\n"
        '  "saturated_topics": list of str (topics covered by 3+ pages)\n'
        '  "underserved_topics": list of str (topics covered by 1-2 pages)\n'
        '  "content_gaps": list of objects with keys:\n'
        '    "topic": str\n'
        '    "suggested_title": str (H1 for a new page)\n'
        '    "competition_level": "Low" | "Medium" | "High"\n'
        '    "content_type": str\n'
        '    "reasoning": str (1-2 sentences)\n'
        "Return ONLY valid JSON."
    )
    user = (
        f"Keyword: {keyword}\n\n"
        f"Top ranking pages:\n{page_titles}\n\n"
        f"H2 headings found across all pages:\n{headings_text}\n\n"
        "Identify what topics/angles are missing or underserved in these rankings."
    )
    raw = complete(system, user, max_tokens=1500)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Failed to parse Claude gap analysis JSON: %s", raw[:200])
        return {"saturated_topics": [], "underserved_topics": [], "content_gaps": []}


def _build_report(keyword: str, result: SerpAnalysisResult) -> bytes:
    lines = [
        f"CONTENT GAP REPORT — {keyword}",
        "=" * 60,
        "",
        f"Pages analyzed: {result.pages_analyzed}",
        f"Pages failed: {result.pages_failed}",
        "",
        "SATURATED TOPICS (already well covered):",
        *[f"  • {t}" for t in result.saturated_topics],
        "",
        "UNDERSERVED TOPICS:",
        *[f"  • {t}" for t in result.underserved_topics],
        "",
        "CONTENT GAP RECOMMENDATIONS:",
    ]
    for i, gap in enumerate(result.content_gaps, 1):
        lines += [
            f"\n{i}. {gap.topic}",
            f"   Suggested title: {gap.suggested_title}",
            f"   Competition: {gap.competition_level} | Type: {gap.content_type}",
            f"   Why: {gap.reasoning}",
        ]
    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_serp_analysis(req: SerpAnalysisRequest) -> SerpAnalysisResult:
    full_keyword = f"{req.keyword} {req.location}".strip() if req.location else req.keyword

    # Step 1: search
    serp_items = asyncio.run(_serper_search(full_keyword, req.num_results, req.language))

    # Step 2: crawl
    urls = [item["url"] for item in serp_items]
    crawl_results = asyncio.run(_crawl_all(urls))

    # Step 3: parse + summarise
    summaries = _summarize_pages(serp_items, crawl_results)
    pages_ok = sum(1 for s in summaries if s.crawl_status == "ok")

    # Step 4: gap analysis
    gap_data = _identify_gaps(full_keyword, summaries)

    content_gaps = [ContentGap(**g) for g in gap_data.get("content_gaps", [])]

    result = SerpAnalysisResult(
        keyword=full_keyword,
        pages_analyzed=pages_ok,
        pages_failed=len(summaries) - pages_ok,
        saturated_topics=gap_data.get("saturated_topics", []),
        underserved_topics=gap_data.get("underserved_topics", []),
        content_gaps=content_gaps,
        page_summaries=summaries,
    )

    # Upload report to Drive
    report_bytes = _build_report(full_keyword, result)
    slug = re.sub(r"[^\w]", "_", full_keyword)[:50]
    drive_url = upload_bytes(report_bytes, f"gap_report_{slug}.txt", "text/plain")
    result.report_drive_url = drive_url

    return result
