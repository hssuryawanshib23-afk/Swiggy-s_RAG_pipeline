# Use official Python slim image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (so Docker caches this layer)
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Build the FAISS vector index from the extracted JSON data
# This runs ONCE at build time so the index is baked into the image
RUN python embed_data.py data/full_pdf_extracted.json --output_dir vectorstore

# Hugging Face Spaces REQUIRES port 7860
EXPOSE 7860

# Start the Flask app via gunicorn on port 7860
# --workers 1  → keeps memory low on free tier
# --timeout 120 → prevents timeout during first model load
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "main:app"]
