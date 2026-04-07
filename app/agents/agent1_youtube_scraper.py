"""
Agent 1 — YouTube Channel Content Extractor

Uses YouTube Data API v3 to crawl all public videos from a channel.
Quota usage: ~20 API units per 500-video channel (well within 10k/day free limit).

Output: CSV + Excel uploaded to Google Drive.
"""

import io
import re
import logging

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.config import settings
from app.models.agent1 import ChannelScrapeRequest, ChannelScrapeResult, VideoEntry
from app.services.google_drive import upload_bytes

logger = logging.getLogger(__name__)


def _get_youtube_client():
    if not settings.youtube_api_key:
        raise ValueError("YOUTUBE_API_KEY is not set")
    return build("youtube", "v3", developerKey=settings.youtube_api_key, cache_discovery=False)


def _resolve_channel_id(youtube, channel_url: str) -> tuple[str, str]:
    """Return (channel_id, channel_title) from any channel URL format."""
    # Extract handle or channel ID from URL
    handle_match = re.search(r"@([\w.-]+)", channel_url)
    id_match = re.search(r"/channel/(UC[\w-]+)", channel_url)

    if handle_match:
        handle = handle_match.group(1)
        resp = youtube.search().list(
            part="snippet", q=f"@{handle}", type="channel", maxResults=1
        ).execute()
        items = resp.get("items", [])
        if not items:
            raise ValueError(f"Channel not found for handle @{handle}")
        channel_id = items[0]["snippet"]["channelId"]
        title = items[0]["snippet"]["channelTitle"]
        return channel_id, title

    if id_match:
        channel_id = id_match.group(1)
        resp = youtube.channels().list(part="snippet", id=channel_id).execute()
        items = resp.get("items", [])
        if not items:
            raise ValueError(f"Channel not found for ID {channel_id}")
        return channel_id, items[0]["snippet"]["title"]

    raise ValueError(f"Cannot parse channel URL: {channel_url}")


def _get_uploads_playlist_id(youtube, channel_id: str) -> str:
    resp = youtube.channels().list(
        part="contentDetails", id=channel_id
    ).execute()
    items = resp.get("items", [])
    if not items:
        raise ValueError(f"Could not get contentDetails for channel {channel_id}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def _fetch_playlist_videos(youtube, playlist_id: str, max_videos: int) -> list[dict]:
    """Paginate through playlistItems and return raw video items."""
    videos = []
    next_page_token = None

    while len(videos) < max_videos:
        params = {
            "part": "snippet,contentDetails",
            "playlistId": playlist_id,
            "maxResults": min(50, max_videos - len(videos)),
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        resp = youtube.playlistItems().list(**params).execute()
        videos.extend(resp.get("items", []))
        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break

    return videos


def _enrich_with_statistics(youtube, video_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch statistics for up to 50 videos at a time."""
    stats: dict[str, dict] = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        resp = youtube.videos().list(
            part="statistics,contentDetails",
            id=",".join(batch),
        ).execute()
        for item in resp.get("items", []):
            stats[item["id"]] = {
                "view_count": int(item["statistics"].get("viewCount", 0)),
                "duration": item["contentDetails"].get("duration", ""),
            }
    return stats


def _format_duration(iso_duration: str) -> str:
    """Convert ISO 8601 duration (PT4M13S) to mm:ss."""
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration)
    if not match:
        return iso_duration
    hours, minutes, seconds = (int(x or 0) for x in match.groups())
    if hours:
        return f"{hours}:{minutes:02}:{seconds:02}"
    return f"{minutes}:{seconds:02}"


def run_channel_scrape(req: ChannelScrapeRequest) -> ChannelScrapeResult:
    youtube = _get_youtube_client()
    channel_id, channel_title = _resolve_channel_id(youtube, req.channel_url)

    uploads_playlist_id = _get_uploads_playlist_id(youtube, channel_id)
    raw_items = _fetch_playlist_videos(youtube, uploads_playlist_id, req.max_videos)

    video_ids = [
        item["contentDetails"]["videoId"]
        for item in raw_items
        if "videoId" in item.get("contentDetails", {})
    ]
    stats = _enrich_with_statistics(youtube, video_ids)

    entries: list[VideoEntry] = []
    for i, item in enumerate(raw_items, start=1):
        snippet = item.get("snippet", {})
        vid_id = item.get("contentDetails", {}).get("videoId", "")
        vid_stats = stats.get(vid_id, {})
        entries.append(VideoEntry(
            position=i,
            title=snippet.get("title", ""),
            url=f"https://www.youtube.com/watch?v={vid_id}",
            video_id=vid_id,
            description=snippet.get("description", "")[:300] or None,
            published_at=snippet.get("publishedAt"),
            view_count=vid_stats.get("view_count"),
            duration=_format_duration(vid_stats.get("duration", "")),
        ))

    # Build DataFrame → CSV + Excel
    df = pd.DataFrame([e.model_dump() for e in entries])
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)
    df["position"] = range(1, len(df) + 1)

    folder_id = req.drive_folder_id or settings.google_drive_folder_id
    slug = re.sub(r"[^\w]", "_", channel_title)[:40]

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    csv_url = upload_bytes(csv_bytes, f"{slug}_videos.csv", "text/csv", folder_id)

    xlsx_buffer = io.BytesIO()
    df.to_excel(xlsx_buffer, index=False)
    xlsx_url = upload_bytes(
        xlsx_buffer.getvalue(),
        f"{slug}_videos.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        folder_id,
    )

    return ChannelScrapeResult(
        channel_title=channel_title,
        channel_id=channel_id,
        video_count=len(entries),
        videos=entries,
        csv_drive_url=csv_url,
        xlsx_drive_url=xlsx_url,
    )
