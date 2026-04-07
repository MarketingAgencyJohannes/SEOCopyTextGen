from pydantic import BaseModel, Field


class ChannelScrapeRequest(BaseModel):
    channel_url: str = Field(
        ...,
        description="YouTube channel URL (@handle or /channel/ID format)",
    )
    max_videos: int = Field(default=500, ge=1, le=5000)
    drive_folder_id: str | None = Field(
        default=None,
        description="Google Drive folder ID to upload results (overrides env default)",
    )


class VideoEntry(BaseModel):
    position: int
    title: str
    url: str
    video_id: str
    description: str | None = None
    published_at: str | None = None
    view_count: int | None = None
    duration: str | None = None


class ChannelScrapeResult(BaseModel):
    channel_title: str
    channel_id: str
    video_count: int
    videos: list[VideoEntry]
    csv_drive_url: str | None = None
    xlsx_drive_url: str | None = None
