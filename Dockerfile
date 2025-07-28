# ---------- Stage 1: Build dependencies ---------- #
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies only for building
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    libmagic-dev \
    tesseract-ocr \
    poppler-utils \
    && apt-get clean

COPY requirements.txt .

# Install requirements locally to /root/.local
RUN pip install --user --no-cache-dir -r requirements.txt

# ---------- Stage 2: Runtime-only minimal ---------- #
FROM python:3.11-slim

WORKDIR /app

# Add the installed pip binaries to PATH
ENV PATH="/root/.local/bin:$PATH"

# Re-install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libpoppler-cpp-dev \
    libmagic-dev \
    tesseract-ocr \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy app code and pip-installed packages
COPY --from=builder /root/.local /root/.local
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
