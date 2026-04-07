from pydantic import BaseModel, Field


class BusinessParams(BaseModel):
    company_name: str = Field(..., description="Name of the business")
    expert_name: str = Field(..., description="Name of the practitioner")
    location: str = Field(..., description="City or region")
    target_audience: str = Field(..., description="Who the service is for")
    usp: str = Field(..., description="Unique selling proposition, 1-3 sentences")


class SEOCopyRequest(BaseModel):
    keyword: str = Field(..., description="Primary keyword, always includes location")
    content_topic: str = Field(..., description="Content angle from SERP gap analysis")
    business: BusinessParams
    cta: str = Field(..., description="Call to action instruction, 1-2 sentences")
    transcript_texts: list[str] = Field(
        default_factory=list,
        description="Raw transcript texts for tonality extraction (1-5)",
    )
    transcript_job_ids: list[str] = Field(
        default_factory=list,
        description="Job IDs from Agent 3 whose transcripts to use",
    )


class TonalityProfile(BaseModel):
    avg_sentence_length: float
    short_sentence_share: float
    medium_sentence_share: float
    long_sentence_share: float
    dominant_opener: str
    register: str
    characteristic_connectors: list[str]
    emotional_tone: str
    burstiness: str


class ValidationResult(BaseModel):
    passed: bool
    word_count: int
    keyword_occurrences: int
    keyword_density: float
    h1_count: int
    h2_count: int
    bold_count: int
    banned_words_found: list[str]
    errors: list[str]


class SEOCopyResult(BaseModel):
    text: str
    tonality_profile: TonalityProfile
    validation: ValidationResult
    generation_attempts: int
    tonality_attempts: int
