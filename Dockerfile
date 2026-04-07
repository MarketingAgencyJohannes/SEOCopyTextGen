# Official Playwright image — Python 3.11 + Chromium pre-installed on Ubuntu Jammy
FROM mcr.microsoft.com/playwright/python:v1.51.0-jammy

# ffmpeg needed for faster-whisper audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK tokenizer data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

COPY . .

# Create SQLite tables at build time
RUN python scripts/init_db.py

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
