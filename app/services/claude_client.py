import logging
import time
import anthropic
from app.config import settings

_client: anthropic.Anthropic | None = None
logger = logging.getLogger(__name__)

FALLBACK_MODEL = "claude-haiku-4-5-20251001"


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def complete(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    model: str = "claude-sonnet-4-6",
    retries: int = 3,
) -> str:
    """Call Claude. On 529/503 overload, retries the primary model once then
    immediately falls back to Haiku — no long waits."""
    client = get_client()
    last_error: Exception | None = None
    current_model = model

    for attempt in range(retries):
        try:
            message = client.messages.create(
                model=current_model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            if current_model != model:
                logger.info("Completed with fallback model %s", current_model)
            return message.content[0].text

        except anthropic.RateLimitError as e:
            last_error = e
            wait = min(2 ** attempt, 30)
            logger.warning("Claude rate limit on %s (attempt %d/%d), retrying in %ds",
                           current_model, attempt + 1, retries, wait)
            time.sleep(wait)

        except anthropic.APIStatusError as e:
            last_error = e
            if e.status_code in (529, 500, 503):
                if current_model != FALLBACK_MODEL:
                    # Switch to Haiku immediately instead of waiting
                    logger.warning(
                        "Claude %s overloaded (%d) — switching to fallback model %s",
                        current_model, e.status_code, FALLBACK_MODEL,
                    )
                    current_model = FALLBACK_MODEL
                    time.sleep(2)  # brief pause before fallback attempt
                else:
                    # Fallback also overloaded — short wait and retry
                    wait = min(15 * (attempt + 1), 60)
                    logger.warning("Fallback model also overloaded, retrying in %ds", wait)
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Claude API error {e.status_code}: {last_error}") from e

        except anthropic.APIError as e:
            last_error = e
            if attempt == retries - 1:
                break
            time.sleep(2)

    raise RuntimeError(f"Claude API failed after {retries} attempts: {last_error}")
