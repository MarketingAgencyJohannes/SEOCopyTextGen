"""
Agent 3 — YouTube Video Transcriber

Priority order per video:
  1. youtube-transcript-api (free, instant, no API key)
  2. yt-dlp audio → Whisper transcription (fallback, CPU-heavy)
  3. Flag as unavailable
"""

import re
import os
import tempfile
import logging

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from app.config import settings
from app.models.agent3 import TranscribeRequest, TranscribeResult, TranscriptEntry

logger = logging.getLogger(__name__)

# Whisper model loaded once at module level to avoid per-request cold start
_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None and settings.whisper_enabled:
        import whisper
        logger.info("Loading Whisper model '%s'...", settings.whisper_model_size)
        _whisper_model = whisper.load_model(settings.whisper_model_size)
        logger.info("Whisper model loaded.")
    return _whisper_model


def _extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
        r"(?:shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _clean_transcript_text(raw: str) -> str:
    """Remove filler tokens and normalise whitespace."""
    text = re.sub(r"\[.*?\]", "", raw)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _fetch_via_transcript_api(
    video_id: str, language: str, language_fallback: str
) -> tuple[str, str] | None:
    """Returns (text, language) or None on failure."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try preferred language first
        for lang in [language, language_fallback]:
            try:
                transcript = transcript_list.find_transcript([lang])
                entries = transcript.fetch()
                text = " ".join(e["text"] for e in entries)
                return _clean_transcript_text(text), lang
            except Exception:
                continue
        # Try any available transcript (auto-generated)
        transcript = transcript_list.find_generated_transcript(
            transcript_list._generated_transcripts.keys()
        )
        entries = transcript.fetch()
        text = " ".join(e["text"] for e in entries)
        return _clean_transcript_text(text), transcript.language_code
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as exc:
        logger.warning("transcript_api failed for %s: %s", video_id, exc)
        return None


def _fetch_via_whisper(video_id: str, video_url: str) -> tuple[str, str] | None:
    """Download audio with yt-dlp, transcribe with Whisper. Returns (text, language)."""
    model = _get_whisper_model()
    if model is None:
        return None

    try:
        import yt_dlp  # noqa: PLC0415
    except ImportError:
        logger.warning("yt-dlp not installed; skipping Whisper fallback")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, f"{video_id}.mp3")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as exc:
            logger.warning("yt-dlp download failed for %s: %s", video_id, exc)
            return None

        # yt-dlp appends .mp3 to the outtmpl
        actual_path = audio_path if os.path.exists(audio_path) else audio_path + ".mp3"
        if not os.path.exists(actual_path):
            logger.warning("Audio file not found after yt-dlp for %s", video_id)
            return None

        try:
            result = model.transcribe(actual_path)
            text = _clean_transcript_text(result["text"])
            lang = result.get("language", "unknown")
            return text, lang
        except Exception as exc:
            logger.warning("Whisper transcription failed for %s: %s", video_id, exc)
            return None


def _get_video_title(video_id: str) -> str | None:
    """Best-effort title fetch via yt-dlp (no download)."""
    try:
        import yt_dlp  # noqa: PLC0415
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False
            )
            return info.get("title")
    except Exception:
        return None


def run_transcribe(req: TranscribeRequest) -> TranscribeResult:
    entries: list[TranscriptEntry] = []

    for url in req.video_urls:
        video_id = _extract_video_id(url)
        if not video_id:
            entries.append(TranscriptEntry(
                url=url, video_id="", method="unavailable",
                error="Could not extract video ID from URL",
            ))
            continue

        title = _get_video_title(video_id)

        # Attempt 1: youtube-transcript-api
        result = _fetch_via_transcript_api(video_id, req.language, req.language_fallback)
        if result:
            text, lang = result
            entries.append(TranscriptEntry(
                url=url, video_id=video_id, title=title,
                language=lang, method="transcript_api", text=text,
            ))
            continue

        # Attempt 2: Whisper fallback
        if req.use_whisper_fallback and settings.whisper_enabled:
            result = _fetch_via_whisper(video_id, url)
            if result:
                text, lang = result
                entries.append(TranscriptEntry(
                    url=url, video_id=video_id, title=title,
                    language=lang, method="whisper", text=text,
                ))
                continue

        # Failed
        entries.append(TranscriptEntry(
            url=url, video_id=video_id, title=title,
            method="unavailable",
            error="No transcript available and Whisper fallback did not succeed",
        ))

    successful = sum(1 for e in entries if e.text)
    return TranscribeResult(
        transcripts=entries,
        total=len(entries),
        successful=successful,
        failed=len(entries) - successful,
    )
