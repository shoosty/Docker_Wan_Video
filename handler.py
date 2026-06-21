"""Wan2.1 14B T2V + I2V — RunPod serverless handler.

Modeled on Docker_Ace_Step/handler.py (v61) so the operational
patterns match across the self-hosted fleet:

  - log() / log_env() / run_cmd() helpers
  - Per-stage telemetry rows to Supabase (wan_worker_telemetry)
  - Worker-id stamping so concurrent workers stay separable
  - VRAM logged at each stage so OOMs are diagnosable
  - Supabase upload of the MP4 when storage_path is provided
  - b64 fallback for lab-mode (no storage_path) so /test-wan can play
    the result inline without round-tripping through storage

Two model checkpoints live behind the same handler:
  - Wan-AI/Wan2.1-T2V-14B   (text → video)
  - Wan-AI/Wan2.1-I2V-14B-720P  (image + text → video)

They're lazily instantiated on the first call that needs them and
cached on a module-level dict. A warm worker that sees only t2v
calls will never load the i2v weights, keeping VRAM tight.

v1 (2026-06-20): first build. Sweet-spot defaults are educated
guesses; the lab pass at /test-wan will dial them in the way
/test-music did for ACE.
"""

import os
import sys
import time
import base64
import secrets
import tempfile
import traceback
import subprocess
from pathlib import Path

import requests
import runpod
from supabase import create_client


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def log(msg: str, level: str = "INFO") -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def log_env() -> None:
    log("── Environment ──────────────────────────────")
    for key, val in sorted(os.environ.items()):
        if any(s in key.upper() for s in ("KEY", "SECRET", "TOKEN", "PASSWORD")):
            display = f"{val[:6]}…(redacted)" if val else "(empty)"
        else:
            display = val
        log(f"  {key}={display}")
    log("─────────────────────────────────────────────")


def run_cmd(cmd: list, label: str) -> subprocess.CompletedProcess:
    log(f"▶ {label}")
    log(f"  CMD: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    log(f"  exit={result.returncode}  time={elapsed:.1f}s")
    if result.stdout.strip():
        log(f"  STDOUT:\n{result.stdout.strip()}")
    if result.stderr.strip():
        log(f"  STDERR (tail 2000):\n{result.stderr[-2000:]}")
    if result.returncode != 0:
        raise RuntimeError(
            f"[{label}] failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
        )
    log(f"✓ {label} ({elapsed:.1f}s)")
    return result


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────

log("━━━ shoosty-wan starting ━━━")
log_env()

WORKER_ID = secrets.token_hex(6)
log(f"WORKER_ID = {WORKER_ID}")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = (
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    or os.environ.get("SUPABASE_SECRET_KEY")
)
SUPABASE_BUCKET = os.environ.get("SUPABASE_VIDEO_BUCKET", "song-videos")

if SUPABASE_URL and SUPABASE_KEY:
    log(f"SUPABASE_URL    = {SUPABASE_URL}")
    log(f"SUPABASE_BUCKET = {SUPABASE_BUCKET}")
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    # v7 (Stephen 2026-06-21): "do we even know if the bucket
    # collection is working?" Smoke-test once at boot — write a
    # 1 KB probe file, list it, delete it. Tells us if RLS, key
    # format, and bucket existence are all live BEFORE the first
    # real job. Failures here don't kill the worker; they just log.
    try:
        log("[smoke] supabase storage probe — start")
        probe_path = f"wan-smoke/{WORKER_ID}-{int(time.time())}.txt"
        probe_body = f"hello from worker {WORKER_ID}".encode()
        supa.storage.from_(SUPABASE_BUCKET).upload(
            path=probe_path,
            file=probe_body,
            file_options={"content-type": "text/plain", "upsert": "true"},
        )
        url = supa.storage.from_(SUPABASE_BUCKET).get_public_url(probe_path)
        log(f"[smoke] uploaded ok → {url}")
        try:
            supa.storage.from_(SUPABASE_BUCKET).remove([probe_path])
            log("[smoke] delete ok")
        except Exception as e:
            log(f"[smoke] delete failed (non-fatal): {e}", "WARN")
    except Exception as e:
        log(f"[smoke] STORAGE PROBE FAILED — uploads will fail too: "
            f"{type(e).__name__}: {e}", "ERROR")
else:
    log("SUPABASE_URL / KEY unset — running in lab mode (no upload). "
        "Handler will return mp4_b64 instead of mp4_url.", "WARN")
    supa = None

try:
    r = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True, check=True
    )
    log(f"ffmpeg: {r.stdout.splitlines()[0]}")
except Exception as e:
    log(f"FATAL — ffmpeg not found: {e}", "ERROR")
    sys.exit(1)

# Lazy GPU import so the file is still importable from a CPU host
# (e.g. for dry-run linting). Real handler code below is gated on
# torch.cuda.is_available().
import torch  # noqa: E402

log(f"torch       = {torch.__version__}")
log(f"cuda avail  = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"cuda device = {torch.cuda.get_device_name(0)}")
    log(f"cuda total  = {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

log("━━━ startup OK ━━━")


# ─────────────────────────────────────────────
# Telemetry helpers
# ─────────────────────────────────────────────

def stamp(job_id: str, stage: str, status: str, message: str = "") -> None:
    """Write one row to wan_worker_telemetry. Best-effort — never
    raises. If Supabase isn't configured we just print."""
    vram_gb = None
    if torch.cuda.is_available():
        try:
            vram_gb = round(
                torch.cuda.memory_allocated(0) / (1024 ** 3), 2
            )
        except Exception:
            vram_gb = None
    log(f"[stamp] {stage:18s} {status:9s} vram={vram_gb} {message[:120]}")
    if supa is None:
        return
    try:
        supa.table("wan_worker_telemetry").insert(
            {
                "job_id": job_id or None,
                "worker_id": WORKER_ID,
                "stage": stage,
                "status": status,
                "vram_gb": vram_gb,
                "message": message[:1000] if message else None,
            }
        ).execute()
    except Exception as e:
        log(f"[stamp] insert failed (swallowed): {e}", "WARN")


# ─────────────────────────────────────────────
# Model cache
# ─────────────────────────────────────────────

# Wan reference repo provides a pipeline class; we import lazily so
# the failure surface is at first call, not at boot (helps RunPod's
# "warming up" status make sense).
_MODELS: dict = {}


def get_pipeline(mode: str, job_id: str):
    """Lazy-load + cache the Wan pipeline for the given mode.

    API verified 2026-06-20 against
    huggingface.co/docs/diffusers/main/en/api/pipelines/wan:
      - T2V: WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers")
      - I2V: WanImageToVideoPipeline with an explicit
             AutoencoderKLWan VAE in float32 (the VAE is the
             precision-sensitive piece; transformer + text encoder
             stay in bfloat16).
    """
    if mode in _MODELS:
        return _MODELS[mode]

    stamp(job_id, "load_model", "started", f"mode={mode}")
    t0 = time.time()

    # v4 (Stephen 2026-06-20): Wan2.1-14B at bfloat16 is ~22.5 GB,
    # which OOMs an RTX 4090 (24 GB) at the eager .to("cuda") step
    # — last 50 MiB of overhead crashes the load. Switching to
    # enable_model_cpu_offload(): components live in CPU RAM and
    # are swapped to GPU per forward pass. ~10-20% slower per job
    # but fits comfortably in 24 GB VRAM.
    if mode == "t2v":
        from diffusers import WanPipeline
        pipe = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            torch_dtype=torch.bfloat16,
        )
        # v7 (Stephen 2026-06-21): on a 48GB+ pod (A6000/A100/H100),
        # 14B fits natively. Native .to("cuda") is way faster than
        # any offload pattern (no swap overhead per step). The
        # endpoint's GPU allow-list excludes 24 GB cards so this
        # path is the only one that runs.
        pipe = pipe.to("cuda")
    elif mode == "i2v":
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch.bfloat16,
        )
        # v7 (Stephen 2026-06-21): see T2V comment above.
        pipe = pipe.to("cuda")
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected 't2v' or 'i2v')")

    # v7 (Stephen 2026-06-21): debug-on — keep diffusers' per-step
    # progress bar enabled so the log shows real-time progress and
    # we can spot a stall vs. just slow. Was disabled in v1-v6.
    try:
        pipe.set_progress_bar_config(disable=False, leave=True)
    except Exception:
        pass
    _MODELS[mode] = pipe
    elapsed = time.time() - t0
    stamp(job_id, "load_model", "completed", f"mode={mode} took={elapsed:.1f}s")
    return pipe


# ─────────────────────────────────────────────
# Image fetch (i2v)
# ─────────────────────────────────────────────

def load_input_image(inp: dict, job_id: str):
    """Resolve the i2v input image into a PIL.Image."""
    from PIL import Image
    from io import BytesIO

    url = inp.get("image_url")
    b64 = inp.get("image_b64")
    if url:
        stamp(job_id, "fetch_image", "started", f"url={url[:80]}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        stamp(job_id, "fetch_image", "completed", f"size={img.size}")
        return img
    if b64:
        stamp(job_id, "fetch_image", "started", f"b64 len={len(b64)}")
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        stamp(job_id, "fetch_image", "completed", f"size={img.size}")
        return img
    raise ValueError("i2v requires image_url or image_b64")


# ─────────────────────────────────────────────
# MP4 mux
# ─────────────────────────────────────────────

def frames_to_mp4(frames, out_path: Path, fps: int) -> None:
    """Write a list of PIL Images to an h264 MP4. Uses diffusers'
    export_to_video helper which wraps imageio-ffmpeg. Output is
    yuv420p so it plays in every browser."""
    from diffusers.utils import export_to_video

    export_to_video(frames, str(out_path), fps=fps)


# ─────────────────────────────────────────────
# Supabase upload
# ─────────────────────────────────────────────

def upload_to_supabase(local_path: Path, storage_path: str, job_id: str) -> str:
    if supa is None:
        raise RuntimeError(
            "Supabase not configured — cannot upload. "
            "Set SUPABASE_URL + SUPABASE_SECRET_KEY or omit storage_path."
        )
    stamp(job_id, "upload", "started", f"path={storage_path}")
    t0 = time.time()
    with open(local_path, "rb") as f:
        data = f.read()
    # Strip any leading bucket name from storage_path: the storage
    # API takes the path RELATIVE to the bucket.
    rel = storage_path
    if rel.startswith(f"{SUPABASE_BUCKET}/"):
        rel = rel[len(SUPABASE_BUCKET) + 1 :]
    # Upsert so a retry on the same storage_path replaces silently.
    supa.storage.from_(SUPABASE_BUCKET).upload(
        path=rel,
        file=data,
        file_options={"content-type": "video/mp4", "upsert": "true"},
    )
    public_url = supa.storage.from_(SUPABASE_BUCKET).get_public_url(rel)
    elapsed = time.time() - t0
    stamp(job_id, "upload", "completed", f"took={elapsed:.1f}s url={public_url}")
    return public_url


# ─────────────────────────────────────────────
# Main handler
# ─────────────────────────────────────────────

def handler(event: dict) -> dict:
    inp = event.get("input") or {}
    job_id = inp.get("job_id") or event.get("id") or "unknown"

    try:
        stamp(job_id, "params", "started", "")

        mode = inp.get("mode", "t2v").lower()
        if mode not in ("t2v", "i2v"):
            raise ValueError(f"mode must be 't2v' or 'i2v', got {mode!r}")

        prompt = (inp.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is required")

        # v7 (Stephen 2026-06-21): defaults trimmed to fast lab
        # values. Wan2.1 was trained primarily on 832×480 16:9 —
        # native aspect, fastest defaults that look honest. Bump
        # the request explicitly to 1280×720 when you want full HD.
        # 3s clip / 16fps / 30 steps is the "prove it works" recipe;
        # bump steps to 50 for production renders.
        duration_sec = float(inp.get("duration_sec", 3.0))
        width = int(inp.get("width", 832))
        height = int(inp.get("height", 480))
        fps = int(inp.get("fps", 16))
        guidance = float(inp.get("guidance_scale", 6.0))
        steps = int(inp.get("num_inference_steps", 30))
        seed = inp.get("seed")
        if seed is None:
            seed = secrets.randbits(31)
        else:
            seed = int(seed)
        storage_path = inp.get("storage_path")
        negative_prompt = (inp.get("negative_prompt") or "").strip() or None

        num_frames = max(1, int(duration_sec * fps))

        stamp(
            job_id,
            "params",
            "completed",
            f"mode={mode} {width}x{height}@{fps}fps "
            f"frames={num_frames} steps={steps} cfg={guidance} seed={seed}",
        )

        # ── load model ───────────────────────────────────────────
        pipe = get_pipeline(mode, job_id)

        # ── i2v: fetch image ─────────────────────────────────────
        image = None
        if mode == "i2v":
            image = load_input_image(inp, job_id)

        # ── diffusion ────────────────────────────────────────────
        stamp(job_id, "diffusion", "started", f"steps={steps}")
        t_diff = time.time()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        if mode == "i2v":
            kwargs["image"] = image
        out = pipe(**kwargs)
        # diffusers' WanPipeline + WanImageToVideoPipeline return an
        # object with .frames as a list-of-batches; first (only)
        # batch is our PIL Image sequence.
        frames = out.frames[0]
        diff_sec = time.time() - t_diff
        stamp(
            job_id,
            "diffusion",
            "completed",
            f"took={diff_sec:.1f}s frames={len(frames)}",
        )

        # ── mux to mp4 ───────────────────────────────────────────
        stamp(job_id, "mux", "started", "")
        t_mux = time.time()
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as tf:
            mp4_path = Path(tf.name)
        frames_to_mp4(frames, mp4_path, fps=fps)
        mux_sec = time.time() - t_mux
        size_mb = round(mp4_path.stat().st_size / (1024 ** 2), 2)
        stamp(
            job_id,
            "mux",
            "completed",
            f"took={mux_sec:.1f}s size={size_mb}MB",
        )

        # ── upload OR return b64 ─────────────────────────────────
        mp4_url = None
        mp4_b64 = None
        upload_sec = 0.0
        if storage_path:
            t_up = time.time()
            mp4_url = upload_to_supabase(mp4_path, storage_path, job_id)
            upload_sec = time.time() - t_up
        else:
            with open(mp4_path, "rb") as f:
                mp4_b64 = base64.b64encode(f.read()).decode("ascii")
            stamp(job_id, "upload", "skipped", "lab mode — returning b64")

        try:
            mp4_path.unlink()
        except Exception:
            pass

        vram_peak = None
        if torch.cuda.is_available():
            try:
                vram_peak = round(
                    torch.cuda.max_memory_allocated(0) / (1024 ** 3), 2
                )
                torch.cuda.reset_peak_memory_stats(0)
            except Exception:
                pass

        stamp(job_id, "complete", "completed", f"vram_peak={vram_peak}")

        return {
            "ok": True,
            "mp4_url": mp4_url,
            "mp4_b64": mp4_b64,
            "duration_sec": duration_sec,
            "width": width,
            "height": height,
            "fps": fps,
            "diffusion_sec": round(diff_sec, 2),
            "mux_sec": round(mux_sec, 2),
            "upload_sec": round(upload_sec, 2),
            "vram_peak_gb": vram_peak,
            "seed": seed,
            "mode": mode,
            "worker_id": WORKER_ID,
        }

    except Exception as e:
        tb = traceback.format_exc()
        log(f"FATAL: {e}\n{tb}", "ERROR")
        stamp(job_id, "complete", "failed", f"{e}")
        return {
            "ok": False,
            "error": str(e),
            "traceback": tb[-4000:],
            "worker_id": WORKER_ID,
        }


log("━━━ handler ready ━━━")

runpod.serverless.start({"handler": handler})
