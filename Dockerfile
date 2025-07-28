# ---- Builder Stage ----
# Use a more complete image that has build tools, and name this stage "builder"
FROM python:3.11 as builder

# Set the working directory
WORKDIR /app

# =================================================================================
# Install a comprehensive set of common system dependencies to prevent
# "cannot open shared object file" errors for libraries like OpenCV and Poppler.
# =================================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    libmagic-dev \
    # --- Dependencies for OpenCV, graphics, and rendering ---
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # --- Font and other common utilities ---
    libfontconfig1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements file to leverage Docker's layer caching
COPY requirements.txt .

# Install the smaller CPU-only version of PyTorch first to prevent downloading
# the massive default version with GPU support.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt


# ---- Final Stage ----
# Use the slim image for a smaller final image size
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install the same comprehensive set of RUNTIME system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libmagic-dev \
    # --- Dependencies for OpenCV, graphics, and rendering ---
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # --- Font and other common utilities ---
    libfontconfig1 \
    libx11-6 \
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