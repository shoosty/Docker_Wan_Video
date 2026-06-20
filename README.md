# Docker_Wan — Wan2.1 14B T2V + I2V RunPod handler

Self-hosted video generation for shoosty-studio. Same shape as
[Docker_Ace_Step](../Docker_Ace_Step) — a Dockerfile + RunPod
serverless handler that wraps an open-source model.

## What it generates

| Mode | Input | Output |
|------|-------|--------|
| `t2v` | text prompt | 5-10s MP4 at 720p, 16 fps |
| `i2v` | image URL or b64 + motion prompt | 5-10s MP4 at 720p, 16 fps animating the input image |

Both modes share one Docker image. The `mode` field in the request
selects which Wan2.1 checkpoint to load (cached after first run).

## Hardware

- Wan2.1-14B needs **~24GB VRAM** at fp16, 720p, 16 fps, 5s.
- Targets RTX 4090 (24GB) or A100 (40-80GB) on RunPod.
- Cold-start: ~3-5 min for first run (model download from HuggingFace
  cache), ~10-15s for subsequent invocations on the same warm worker.

## Request shape

```jsonc
{
  "input": {
    "mode": "t2v",                       // or "i2v"
    "prompt": "cinematic close-up of a peony unfurling, golden hour",

    // i2v only:
    "image_url": "https://....png",      // or
    "image_b64": "iVBORw0KG...",

    // optional:
    "duration_sec": 5,                   // 1-10, default 5
    "width": 1280,
    "height": 720,
    "fps": 16,
    "guidance_scale": 6.0,
    "num_inference_steps": 50,
    "seed": 42,                          // null = random

    // upload destination (handler writes mp4 to Supabase if set):
    "storage_path": "song-videos/<song_id>/<gen_id>.mp4",

    // RunPod task id for telemetry rows:
    "job_id": "<runpod-task-id>"
  }
}
```

## Response shape

```jsonc
{
  "ok": true,
  "mp4_url": "https://...supabase.../song-videos/<song>/<gen>.mp4",
  "mp4_b64": null,                       // populated only when storage_path absent
  "duration_sec": 5.0,
  "width": 1280,
  "height": 720,
  "fps": 16,
  "model_load_sec": 8.2,
  "inference_sec": 142.1,
  "upload_sec": 3.4,
  "vram_peak_gb": 22.8
}
```

## Environment

| Var | Required | Purpose |
|-----|----------|---------|
| `SUPABASE_URL` | yes (for upload path) | shoosty-studio Supabase URL |
| `SUPABASE_SECRET_KEY` or `SUPABASE_SERVICE_ROLE_KEY` | yes (for upload path) | write key |
| `SUPABASE_VIDEO_BUCKET` | no (default `song-videos`) | target bucket |
| `HF_HOME` | no (default `/root/.cache/huggingface`) | where weights cache |

## Telemetry

Per-stage rows written to `wan_worker_telemetry` (Supabase). Same shape
as `ace_worker_telemetry`:

```
job_id | worker_id | stage | status | vram_gb | message | created_at
```

Stages: `params → load_model → encode_prompt → diffusion → decode_video
→ mux → upload → complete`.

## Build & deploy

```bash
docker build -t shoosty1/wan:v1 .
docker push shoosty1/wan:v1
# Create RunPod serverless endpoint pointing at shoosty1/wan:v1
# GPU: 24GB+, min instances 0, max workers 1 for first pass
```

## Versions

- **v1** (2026-06-20) — first build. T2V + I2V, weights downloaded at
  startup, ace-step-style telemetry, sweet-spot defaults TBD.
