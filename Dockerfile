FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy only the new backend requirements file
COPY backend-requirements.txt requirements.txt

# --- CRITICAL MEMORY FIX (v2) ---
# Install the small, CPU-only version of torch FIRST
# --no-deps stops it from pulling in other libraries
RUN pip install \
    --no-cache-dir \
    torch \
    --no-deps \
    --index-url https://download.pytorch.org/whl/cpu

# Install all other backend dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a directory for usage tracking
RUN mkdir -p usage_tracking

# Expose port
EXPOSE 8000

# Start command
CMD uvicorn main:app --host 0.0.0.0 --port $PORT