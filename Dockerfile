# ---- Builder Stage ----
# Use a more complete image that has build tools, and name this stage "builder"
FROM python:3.11 as builder

# Set the working directory
WORKDIR /app

# Install system dependencies required for building Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements file to leverage Docker's layer caching
COPY requirements.txt .

# =================================================================================
# CRITICAL CHANGE HERE:
# Install PyTorch with the --index-url flag to get the much smaller CPU version.
# Then install the rest of your requirements. Pip will see torch is already
# installed and won't try to download the huge default version.
# =================================================================================
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt


# ---- Final Stage ----
# Use the slim image for a smaller final image size
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install only the RUNTIME system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy your application code
COPY . .

# Activate the virtual environment in the final image
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port your application will run on
EXPOSE 8000

# Command to run your application
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
