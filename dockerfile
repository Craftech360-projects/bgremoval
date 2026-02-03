# CRITICAL CHANGE: Use NVIDIA CUDA 11.8 + cuDNN 8 Base Image
# This provides libcudnn.so.8 and libcublasLt.so.11 required by onnxruntime-gpu
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV U2NET_HOME=/root/.u2net

WORKDIR /app

# 1. Install Python 3.10 and System Dependencies (OpenCV + Request)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Alias 'python' to 'python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# 3. Upgrade pip to ensure clean installs
RUN pip install --upgrade pip

# 4. Install Python Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Run Builder (Downloads the 1GB Model)
COPY builder.py .
RUN python builder.py

# 6. Start the Worker
COPY handler.py .
CMD ["python", "-u", "handler.py"]