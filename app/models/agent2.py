from pydantic import BaseModel, Field


class SerpAnalysisRequest(BaseModel):
    keyword: str = Field(..., description="Primary keyword to analyze")
    location: str | None = Field(default=None, description="Location modifier (e.g. 'Freiburg')")
    language: str = Field(default="de", description="Search language")
    num_results: int = Field(default=30, ge=5, le=30)


class PageSummary(BaseModel):
    position: int
    url: str
    title: str | None = None
    domain: str | None = None
    word_count: int | None = None
    h2_headings: list[str] = []
    summary: str | None = None
    content_type: str | None = None
    crawl_status: str  # "ok" | "failed"


class ContentGap(BaseModel):
    topic: str
    suggested_title: str
    competition_level: str  # "Low" | "Medium" | "High"
    content_type: str
    reasoning: str


class SerpAnalysisResult(BaseModel):
    keyword: str
    pages_analyzed: int
    pages_failed: int
    saturated_topics: list[str]
    underserved_topics: list[str]
    content_gaps: list[ContentGap]
    page_summaries: list[PageSummary]
    report_drive_url: str | None = None
    drive_error: str | None = None
