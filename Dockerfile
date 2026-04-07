FROM python:3.11-slim

# System deps: ffmpeg (Whisper), Chromium (Playwright/Agent 2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright's bundled Chromium (used by Agent 2)
RUN playwright install chromium --with-deps

# Download NLTK punkt tokenizer data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

COPY . .

# Create DB tables at build time (also runs at startup via on_event)
RUN python scripts/init_db.py

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
