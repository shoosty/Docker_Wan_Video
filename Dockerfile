FROM --platform=linux/amd64 runpod/base:0.6.2-cuda12.1.0

# Wan2.1 Video Generation — v1
# Supports TASK env var: "t2v" (text-to-video) or "i2v" (image-to-video)
# Requires env vars:
#   SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
#   WAN_BUCKET (optional) — default "wan-outputs"
#
# CRITICAL: Models must be pre-downloaded to /runpod-volume/models/Wan2.1
# NOT /workspace. RunPod serverless mounts at /runpod-volume only.

RUN apt-get update && apt-get install -y python3.11 python3.11-dev git curl ffmpeg && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    apt-get clean

RUN pip3.11 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3.11 install diffusers==0.31.0 transformers==4.50.0 accelerate \
        huggingface_hub imageio imageio-ffmpeg \
        opencv-python-headless Pillow \
        runpod supabase && \
    pip3.11 install "click>=8.0"

RUN mkdir -p /app
COPY handler.py /app/handler.py

CMD ["python3.11", "/app/handler.py"]
