# omniASR Streaming Server
# Supports: CUDA (NVIDIA GPU), CPU
# Note: MPS (Apple Silicon) not available in Docker

# ============================================
# Base image with CUDA support
# ============================================
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies + uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    cmake \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# ============================================
# Python dependencies
# ============================================
COPY requirements.txt .

# Install Python packages with uv (much faster)
RUN uv pip install --system --no-cache -r requirements.txt


# ============================================
# Application code
# ============================================
COPY . .

# ============================================
# Configuration
# ============================================
# Server settings
ENV HOST=0.0.0.0
ENV PORT=8000

# Model settings (can be overridden)
ENV MODEL_CARD=omniASR_CTC_300M_v2
ENV DEFAULT_LANG=eng_Latn

# Device auto-detection (cuda if available, else cpu)
# Can override with: ENV DEVICE=cuda or ENV DEVICE=cpu

# ============================================
# Health check
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ============================================
# Expose port and run
# ============================================
EXPOSE ${PORT}

# Run the server
CMD ["python", "server.py"]
