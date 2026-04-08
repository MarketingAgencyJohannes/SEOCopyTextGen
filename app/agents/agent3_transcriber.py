"""
Agent 3 — YouTube Video Transcriber

Priority order per video:
  1. youtube-transcript-api (free, instant)
  2. Invidious public API (fallback when YouTube rate-limits the server IP)
  3. yt-dlp audio → faster-whisper transcription (CPU-heavy, last resort)
  4. Flag as unavailable
"""

import re
import os
import tempfile
import logging

import httpx
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from app.config import settings
from app.models.agent3 import TranscribeRequest, TranscribeResult, TranscriptEntry

logger = logging.getLogger(__name__)

TRANSCRIPT_API_TIMEOUT = 15   # seconds per video for transcript API
YTDLP_SOCKET_TIMEOUT = 30     # seconds for yt-dlp network operations

# Invidious public instances — tried in order, first success wins
INVIDIOUS_INSTANCES = [
    "https://inv.nadeko.net",
    "https://invidious.slipfox.xyz",
    "https://invidious.nerdvpn.de",
    "https://invidious.io.lol",
]

# Whisper model loaded once at module level to avoid per-request cold start
_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None and settings.whisper_enabled:
        from faster_whisper import WhisperModel
        logger.info("Loading faster-whisper model '%s'...", settings.whisper_model_size)
        _whisper_model = WhisperModel(
            settings.whisper_model_size,
            device="cpu",
            compute_type="int8",
        )
        logger.info("faster-whisper model loaded.")
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


def _snippet_text(entry) -> str:
    """Handle both dict (old API) and object (new API) transcript entries."""
    if isinstance(entry, dict):
        return entry.get("text", "")
    return getattr(entry, "text", "") or ""


def _fetch_via_transcript_api(
    video_id: str, language: str, language_fallback: str
) -> tuple[tuple[str, str], None] | tuple[None, str]:
    """Returns ((text, language), None) on success or (None, error_reason) on failure."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try preferred language first, then fallback
        for lang in [language, language_fallback]:
            try:
                transcript = transcript_list.find_transcript([lang])
                entries = transcript.fetch()
                text = " ".join(_snippet_text(e) for e in entries)
                return (_clean_transcript_text(text), lang), None
            except Exception:
                continue

        # Try any available transcript (manual first, then generated)
        all_transcripts = list(transcript_list)
        manual = [t for t in all_transcripts if not getattr(t, "is_generated", False)]
        generated = [t for t in all_transcripts if getattr(t, "is_generated", True)]
        for t in (manual + generated):
            try:
                entries = t.fetch()
                text = " ".join(_snippet_text(e) for e in entries)
                return (_clean_transcript_text(text), t.language_code), None
            except Exception:
                continue

        return None, "No usable transcript found (all languages failed to fetch)"
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video"
    except NoTranscriptFound:
        return None, "No captions available for this video"
    except Exception as exc:
        logger.warning("transcript_api failed for %s: %s", video_id, exc)
        return None, f"Transcript API error: {type(exc).__name__}: {exc}"


def _fetch_title_from_transcript_api(video_id: str) -> str | None:
    """Fast title fetch using transcript API metadata (no yt-dlp needed)."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # The video title isn't directly in the transcript API, return None
        # and use video_id as fallback — avoids blocking yt-dlp title call
        return None
    except Exception:
        return None


def _parse_vtt(vtt_content: str) -> str:
    """Extract plain text from WebVTT, stripping timestamps and duplicate lines."""
    seen: set[str] = set()
    parts: list[str] = []
    for line in vtt_content.split("\n"):
        line = line.strip()
        if not line or "-->" in line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        clean = re.sub(r"<[^>]+>", "", line).strip()
        if clean and clean not in seen:
            seen.add(clean)
            parts.append(clean)
    return " ".join(parts)


def _fetch_via_invidious(
    video_id: str, language: str
) -> tuple[tuple[str, str], None] | tuple[None, str]:
    """Fetch captions from a public Invidious instance (bypasses Railway IP blocks).
    Returns ((text, lang_code), None) on success or (None, error_reason) on failure.
    """
    for base in INVIDIOUS_INSTANCES:
        try:
            r = httpx.get(f"{base}/api/v1/captions/{video_id}", timeout=10)
            if r.status_code != 200:
                logger.debug("Invidious %s returned %s for %s", base, r.status_code, video_id)
                continue

            captions = r.json().get("captions", [])
            if not captions:
                return None, "No captions listed on Invidious"

            # Prefer requested language, otherwise take the first available
            target = next(
                (c for c in captions if c.get("languageCode", "").startswith(language)),
                captions[0],
            )
            cap_url = target.get("url", "")
            if not cap_url:
                continue

            full_url = f"{base}{cap_url}" if cap_url.startswith("/") else cap_url
            r2 = httpx.get(full_url, timeout=15)
            if r2.status_code != 200:
                continue

            text = _clean_transcript_text(_parse_vtt(r2.text))
            if not text:
                continue

            lang_code = target.get("languageCode", language)
            logger.info("Invidious transcript OK for %s via %s (%s)", video_id, base, lang_code)
            return (text, lang_code), None

        except Exception as exc:
            logger.warning("Invidious %s failed for %s: %s", base, video_id, exc)
            continue

    return None, "All Invidious instances failed"


def _fetch_via_whisper(video_id: str, video_url: str) -> tuple[str, str] | None:
    """Download audio with yt-dlp, transcribe with faster-whisper. Returns (text, language)."""
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
            "socket_timeout": YTDLP_SOCKET_TIMEOUT,
            "retries": 2,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as exc:
            logger.warning("yt-dlp download failed for %s: %s", video_id, exc)
            return None

        # yt-dlp may append .mp3 to the outtmpl
        actual_path = audio_path if os.path.exists(audio_path) else audio_path + ".mp3"
        if not os.path.exists(actual_path):
            logger.warning("Audio file not found after yt-dlp for %s", video_id)
            return None

        try:
            segments, info = model.transcribe(actual_path)
            text = _clean_transcript_text(" ".join(seg.text for seg in segments))
            lang = info.language if hasattr(info, "language") else "unknown"
            return text, lang
        except Exception as exc:
            logger.warning("Whisper transcription failed for %s: %s", video_id, exc)
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

        # Attempt 1: youtube-transcript-api (fast, no extra requests)
        result, api_error = _fetch_via_transcript_api(video_id, req.language, req.language_fallback)
        if result:
            text, lang = result
            entries.append(TranscriptEntry(
                url=url, video_id=video_id,
                title=f"Video {video_id}",
                language=lang, method="transcript_api", text=text,
            ))
            continue

        # Attempt 2: Invidious (bypasses Railway/GCP IP blocks from YouTube rate limiting)
        is_rate_limited = api_error and ("429" in api_error or "YouTubeRequestFailed" in api_error)
        if is_rate_limited or api_error:
            inv_result, inv_error = _fetch_via_invidious(video_id, req.language)
            if inv_result:
                text, lang = inv_result
                entries.append(TranscriptEntry(
                    url=url, video_id=video_id,
                    title=f"Video {video_id}",
                    language=lang, method="invidious", text=text,
                ))
                continue
            # Surface the Invidious error too
            api_error = f"{api_error} | Invidious: {inv_error}"

        # Attempt 3: Whisper (CPU-heavy audio transcription, last resort)
        if req.use_whisper_fallback and settings.whisper_enabled:
            whisper_result = _fetch_via_whisper(video_id, url)
            if whisper_result:
                text, lang = whisper_result
                entries.append(TranscriptEntry(
                    url=url, video_id=video_id,
                    title=f"Video {video_id}",
                    language=lang, method="whisper", text=text,
                ))
                continue

        # All attempts failed
        entries.append(TranscriptEntry(
            url=url, video_id=video_id,
            title=f"Video {video_id}",
            method="unavailable",
            error=api_error or "No transcript available",
        ))

    successful = sum(1 for e in entries if e.text)
    return TranscribeResult(
        transcripts=entries,
        total=len(entries),
        successful=successful,
        failed=len(entries) - successful,
    )
