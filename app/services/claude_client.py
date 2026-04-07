import time
import anthropic
from app.config import settings

_client: anthropic.Anthropic | None = None


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
            wait = 2 ** attempt
            time.sleep(wait)
        except anthropic.APIError as e:
            last_error = e
            if attempt == retries - 1:
                break
            time.sleep(1)

    raise RuntimeError(f"Claude API failed after {retries} attempts: {last_error}")
