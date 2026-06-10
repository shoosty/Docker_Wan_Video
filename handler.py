"""Wan2.1 Video Generation — v1 — RunPod Serverless handler.

Generates video from text or image using Wan2.1.
Uploads to Supabase and returns a public URL.

Required env vars:
  SUPABASE_URL              — https://<project>.supabase.co
  SUPABASE_SERVICE_ROLE_KEY — service-role JWT
  WAN_BUCKET (optional)     — default "wan-outputs"
  TASK (optional)           — "t2v" or "i2v" (default: "t2v")

CRITICAL: RunPod serverless mounts network volumes at /runpod-volume,
NOT /workspace. Every path here uses /runpod-volume/.
"""
import runpod
import os
import uuid
import time
import traceback
import tempfile
import urllib.request

import torch
from diffusers import WanPipeline, WanImageToVideoPipeline
from PIL import Image

# ── Model paths ───────────────────────────────────────────────────
TASK = os.environ.get("TASK", "t2v").lower()

if TASK == "i2v":
    MODEL_PATH = "/runpod-volume/models/Wan2.1-I2V-14B-480P"
else:
    MODEL_PATH = "/runpod-volume/models/Wan2.1-T2V-14B"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Models not found at {MODEL_PATH} - check volume mount!")

print(f"Models found at {MODEL_PATH} (TASK={TASK})")

print(f"Loading Wan2.1 pipeline (task={TASK})...")
if TASK == "i2v":
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )
else:
    pipe = WanPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )
pipe.to("cuda")
print("Pipeline loaded!")

# ── Supabase client ────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
WAN_BUCKET = os.environ.get("WAN_BUCKET", "wan-outputs")

_supabase_client = None
def supabase_client():
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set."
        )
    from supabase import create_client
    _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _supabase_client

def upload_to_supabase(local_path: str, storage_path: str, content_type: str) -> str:
    sb = supabase_client()
    with open(local_path, "rb") as f:
        data = f.read()
    sb.storage.from_(WAN_BUCKET).upload(
        path=storage_path,
        file=data,
        file_options={"content-type": content_type, "upsert": "true"},
    )
    res = sb.storage.from_(WAN_BUCKET).get_public_url(storage_path)
    if isinstance(res, dict):
        return res.get("publicUrl") or res.get("publicURL") or res.get("public_url")
    return res

def download_to_temp(url: str, suffix: str = ".png") -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        local_path = f.name
    urllib.request.urlretrieve(url, local_path)
    return local_path

def handler(job):
    """RunPod serverless entrypoint.

    Inputs:
      prompt (str)              — text prompt. Required.
      negative_prompt (str)     — what to avoid. Optional.
      num_frames (int)          — number of frames. Default 81.
      fps (int)                 — frames per second. Default 16.
      width (int)               — video width. Default 832.
      height (int)              — video height. Default 480.
      num_inference_steps (int) — Default 50.
      guidance_scale (float)    — Default 5.0.
      seed (int)                — for reproducibility. Optional.
      image_url (str)           — for i2v mode: source image URL. Required if TASK=i2v.
      storage_path (str)        — explicit object path. Optional.

    Output (success):
      {
        "video_url":    "<public URL>",
        "storage_path": "<path>",
        "num_frames":   <int>,
        "fps":          <int>,
        "seed":         <int>
      }

    Output (failure):
      { "error": "<message>", "traceback": "<full trace>" }
    """
    img_temp = None
    try:
        inp = job.get("input", {}) or {}

        prompt = inp.get("prompt")
        if not prompt:
            return {"error": "prompt is required"}

        negative_prompt = inp.get("negative_prompt", "")
        num_frames      = int(inp.get("num_frames", 81))
        fps             = int(inp.get("fps", 16))
        width           = int(inp.get("width", 832))
        height          = int(inp.get("height", 480))
        steps           = int(inp.get("num_inference_steps", 50))
        guidance        = float(inp.get("guidance_scale", 5.0))

        seed = inp.get("seed")
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))
        else:
            seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device="cuda").manual_seed(seed)

        ts       = int(time.time())
        short_id = uuid.uuid4().hex[:12]
        storage_path = inp.get("storage_path") or f"wan-runs/{ts}-{short_id}.mp4"

        kwargs = {
            "prompt":              prompt,
            "negative_prompt":     negative_prompt,
            "num_frames":          num_frames,
            "width":               width,
            "height":              height,
            "num_inference_steps": steps,
            "guidance_scale":      guidance,
            "generator":           generator,
        }

        if TASK == "i2v":
            image_url = inp.get("image_url")
            if not image_url:
                return {"error": "image_url is required for i2v task"}
            img_temp = download_to_temp(image_url, suffix=".png")
            image = Image.open(img_temp).convert("RGB")
            kwargs["image"] = image

        output = pipe(**kwargs)
        frames = output.frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name

        try:
            import imageio
            writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264", quality=8)
            for frame in frames:
                writer.append_data(frame)
            writer.close()

            video_url = upload_to_supabase(tmp_path, storage_path, "video/mp4")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return {
            "video_url":    video_url,
            "storage_path": storage_path,
            "num_frames":   num_frames,
            "fps":          fps,
            "seed":         seed,
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
    finally:
        if img_temp and os.path.exists(img_temp):
            os.unlink(img_temp)

runpod.serverless.start({"handler": handler})
