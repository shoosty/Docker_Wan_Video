# Wan2.1 Setup — getting v1 to first job

## 1. Build + push the image

```bash
cd /Users/shoosty/Code/Docker_Wan
docker build -t shoosty1/wan:v1 .
docker push shoosty1/wan:v1
```

First build pulls the PyTorch base (~5GB) + clones Wan2.1 reference
repo + installs Python deps. Expect 10-20 min on a fast connection.

## 2. Supabase table for telemetry

Run in the shoosty-studio Supabase SQL editor:

```sql
create table if not exists wan_worker_telemetry (
  id           bigserial primary key,
  job_id       text,
  worker_id    text,
  stage        text not null,
  status       text not null,
  vram_gb      numeric,
  message      text,
  created_at   timestamptz not null default now()
);
create index if not exists wan_worker_telemetry_job_idx
  on wan_worker_telemetry (job_id, created_at desc);
create index if not exists wan_worker_telemetry_worker_idx
  on wan_worker_telemetry (worker_id, created_at desc);
```

RLS off (admin-only table). The handler writes with the service-role
key.

## 3. Create the RunPod endpoint

In the RunPod serverless dashboard:

- **Container image:** `shoosty1/wan:v1`
- **GPU:** RTX 4090 (24GB) or A100 (40-80GB). 4090 is fine for 720p.
- **Container disk:** at least **80 GB** — Wan T2V-14B + I2V-14B = ~56GB
  cached at /root/.cache/huggingface, plus PyTorch + Python overhead.
- **Min workers:** 0
- **Max workers:** 1 (raise after v1 proves out)
- **Idle timeout:** 5 seconds (kills cold workers fast so a stuck one
  doesn't burn cash)
- **Execution timeout:** 600 seconds (10 min, room for the first
  download)
- **Environment:**
  - `SUPABASE_URL`
  - `SUPABASE_SECRET_KEY` (the new sb_secret_ key format — needs
    `apikey` header too, see shoosty-supabase-storage-apikey memory)
  - `SUPABASE_VIDEO_BUCKET=song-videos`
  - `HF_HUB_ENABLE_HF_TRANSFER=1`

Copy the endpoint ID. Looks like `xxxxxxxxxxxxxx` (14 chars).

## 4. Wire into shoosty-studio

Add to Vercel env + `.env.local`:

```
RUNPOD_WAN_ENDPOINT_ID=<the-id-from-step-3>
```

The `/test-wan` admin page (separate PR in shoosty-studio repo) will
post to `https://api.runpod.ai/v2/<endpoint-id>/run` with the request
shape documented in `README.md`.

## 5. First job — lab mode

Lab mode: omit `storage_path` and the handler returns `mp4_b64` so
you can play it inline without Supabase upload risk on the first
fire. Post via curl from your machine while watching RunPod logs:

```bash
curl -X POST https://api.runpod.ai/v2/$WAN_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{
        "mode":"t2v",
        "prompt":"cinematic close-up of a peony unfurling, golden hour, slow motion",
        "duration_sec":3,
        "seed":42
      }}'
```

First run on a cold worker: ~3-5 min model download + ~2 min
inference. Subsequent warm runs: ~2 min inference only.

## 6. First job — Supabase upload

Once lab mode works, add `storage_path`:

```jsonc
{"input": {
   "mode": "t2v",
   "prompt": "...",
   "storage_path": "test/wan-smoke-test.mp4"
}}
```

Verify the file lands in the `song-videos` bucket and is publicly
readable.

## Versions

| Tag | Notes |
|-----|-------|
| v1  | first build, T2V + I2V, weights downloaded at startup |
