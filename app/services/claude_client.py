import logging
import time
import anthropic
from app.config import settings

_client: anthropic.Anthropic | None = None
logger = logging.getLogger(__name__)


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
    retries: int = 6,
) -> str:
    client = get_client()
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return message.content[0].text
        except anthropic.RateLimitError as e:
            last_error = e
            wait = min(2 ** attempt, 60)
            logger.warning("Claude rate limit (attempt %d/%d), retrying in %ds", attempt + 1, retries, wait)
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            last_error = e
            # 529 = overloaded, 500 = server error — both are transient, retry with backoff
            if e.status_code in (529, 500, 503):
                wait = min(10 * (attempt + 1), 60)
                logger.warning("Claude API %d overloaded (attempt %d/%d), retrying in %ds", e.status_code, attempt + 1, retries, wait)
                time.sleep(wait)
            else:
                # Non-transient error (400, 401, etc.) — don't retry
                raise RuntimeError(f"Claude API error {e.status_code}: {last_error}") from e
        except anthropic.APIError as e:
            last_error = e
            if attempt == retries - 1:
                break
            time.sleep(2)

    raise RuntimeError(f"Claude API failed after {retries} attempts: {last_error}")
