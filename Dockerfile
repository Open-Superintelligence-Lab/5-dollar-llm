# Use the specified PyTorch base image with CUDA support
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV TOKENIZERS_PARALLELISM=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/logs /app/checkpoints

# Expose any necessary ports (if you have a dashboard, etc.)
EXPOSE 8000

# Default command to run training
CMD ["python", "train_llm.py", "--help"]