# Wan2.1 14B T2V + I2V — RunPod serverless handler image.
#
# Base: PyTorch 2.5 + CUDA 12.4 on Ubuntu 22.04. Matches the family
# Wan2.1's reference repo tests against (PyTorch 2.4+, CUDA 12.1+).
#
# Model weights are NOT baked into the image — they download from
# HuggingFace at startup into the HF cache. First cold-start on a
# brand-new RunPod worker downloads ~28GB per model (T2V + I2V are
# loaded lazily, only the mode requested on the first call is
# pulled). Subsequent warm calls skip the download.
#
# ffmpeg is installed for the final mux step (handler writes h264
# MP4 from the per-frame tensors).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stephen 2026-06-20: API verification (huggingface.co/docs/diffusers/
# main/en/api/pipelines/wan) confirms Wan2.1 ships in diffusers itself
# via `from diffusers import WanPipeline, WanImageToVideoPipeline`.
# No git clone of the reference repo needed.
COPY handler.py .

CMD ["python", "-u", "handler.py"]
