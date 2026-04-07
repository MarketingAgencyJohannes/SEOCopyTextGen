from pydantic import BaseModel, Field


class TranscribeRequest(BaseModel):
    video_urls: list[str] = Field(..., min_length=1, max_length=20,
                                  description="1-20 YouTube video URLs")
    language: str = Field(default="de", description="Preferred transcript language (ISO 639-1)")
    language_fallback: str = Field(default="en", description="Fallback language")
    use_whisper_fallback: bool = Field(default=True,
                                       description="Use Whisper when no caption track available")
    strip_timestamps: bool = Field(default=True)


class TranscriptEntry(BaseModel):
    url: str
    video_id: str
    title: str | None = None
    language: str | None = None
    method: str  # "transcript_api" | "whisper" | "unavailable"
    text: str | None = None
    error: str | None = None


class TranscribeResult(BaseModel):
    transcripts: list[TranscriptEntry]
    total: int
    successful: int
    failed: int
