FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy source
COPY . .

# Create required directories
RUN mkdir -p data/raw data/processed data/graphs \
             models/detection models/prediction models/rl_agents \
             logs results

# Expose ports
EXPOSE 8000 8501

# Default: start API + dashboard
CMD ["python", "main.py", "--services", "api", "dashboard", "--init-db"]
