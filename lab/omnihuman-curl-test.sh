#!/usr/bin/env bash
# Shoosty Avatar smoke test — direct REST against AceData Omnihuman 1.5
#
# Fires a lip-sync video job with a portrait + audio. Polls until
# done, downloads the result MP4, plays it via `open` on macOS.
#
# Requires:
#   $ACEDATA_API_KEY   — your AceData API key (set in env or .env.local)
#
# Usage:
#   ./omnihuman-curl-test.sh
#   ./omnihuman-curl-test.sh <portrait_url> <audio_url>
#
# Cost: ~$0.69 per fire.
#
# Stephen — the exact endpoint URL is a placeholder. After you check
# the AceData docs (login required to see API specs), update the
# variables marked # TODO: below.

set -e

# ─── Config (FILL IN AFTER CHECKING ACEDATA DOCS) ────────────────────
ACEDATA_BASE="${ACEDATA_BASE:-https://api.acedata.cloud}"   # TODO: confirm
ACEDATA_OMNIHUMAN_PATH="${ACEDATA_OMNIHUMAN_PATH:-/video/dreamina}"  # TODO: confirm exact path
ACEDATA_MODEL="omnihuman-1.5"                                # TODO: confirm exact model slug

if [ -z "${ACEDATA_API_KEY:-}" ]; then
  echo "ERROR: ACEDATA_API_KEY not set."
  echo "  Either export it:    export ACEDATA_API_KEY=your_key"
  echo "  Or source from env:  source /path/to/.env.local && ./omnihuman-curl-test.sh"
  exit 1
fi

# ─── Inputs ──────────────────────────────────────────────────────────
PORTRAIT_URL="${1:-https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=512}"
AUDIO_URL="${2:-https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3}"

echo "── Shoosty Avatar smoke test ──────────────────────"
echo "  portrait:  $PORTRAIT_URL"
echo "  audio:     $AUDIO_URL"
echo "  model:     $ACEDATA_MODEL"
echo "  endpoint:  $ACEDATA_BASE$ACEDATA_OMNIHUMAN_PATH"
echo "  cost est:  ~\$0.69"
echo "─────────────────────────────────────────────────────"

# ─── Submit ──────────────────────────────────────────────────────────
echo ""
echo "[1/3] Submitting job..."
SUBMIT_RESPONSE=$(curl -sS -X POST "$ACEDATA_BASE$ACEDATA_OMNIHUMAN_PATH" \
  -H "Authorization: Bearer $ACEDATA_API_KEY" \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "$ACEDATA_MODEL",
  "input": {
    "image_url": "$PORTRAIT_URL",
    "audio_url": "$AUDIO_URL"
  }
}
EOF
)

echo "  submit response:"
echo "$SUBMIT_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$SUBMIT_RESPONSE"

# Extract job id — exact JSON path depends on AceData's response shape.
# Try a few common paths.
JOB_ID=$(echo "$SUBMIT_RESPONSE" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for k in ['id', 'task_id', 'job_id', 'request_id']:
        if k in d:
            print(d[k]); sys.exit(0)
    if 'data' in d and isinstance(d['data'], dict):
        for k in ['id', 'task_id', 'job_id']:
            if k in d['data']:
                print(d['data'][k]); sys.exit(0)
except Exception as e:
    print(f'PARSE_ERR: {e}', file=sys.stderr)
" 2>/dev/null)

if [ -z "$JOB_ID" ]; then
  echo "ERROR: couldn't extract job id from submit response."
  echo "Check the response shape above and update the JOB_ID parsing in this script."
  exit 1
fi
echo "  job_id: $JOB_ID"

# ─── Poll ────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Polling for completion (expect 30-90s)..."
START=$(date +%s)
for i in {1..30}; do
  sleep 4
  STATUS_RESPONSE=$(curl -sS -X GET "$ACEDATA_BASE$ACEDATA_OMNIHUMAN_PATH?id=$JOB_ID" \
    -H "Authorization: Bearer $ACEDATA_API_KEY")
  STATUS=$(echo "$STATUS_RESPONSE" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    s = d.get('status') or d.get('state') or (d.get('data') or {}).get('status') or '?'
    print(s)
except: print('?')
" 2>/dev/null)
  ELAPSED=$(( $(date +%s) - START ))
  echo "  [$i] t=${ELAPSED}s  status=$STATUS"

  if [[ "$STATUS" == "success" || "$STATUS" == "completed" || "$STATUS" == "done" ]]; then
    break
  fi
  if [[ "$STATUS" == "failed" || "$STATUS" == "error" || "$STATUS" == "cancelled" ]]; then
    echo "  TERMINAL FAILURE — response:"
    echo "$STATUS_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$STATUS_RESPONSE"
    exit 1
  fi
done

# ─── Download ────────────────────────────────────────────────────────
echo ""
echo "[3/3] Extracting video URL..."
VIDEO_URL=$(echo "$STATUS_RESPONSE" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for k in ['video_url', 'output_url', 'result_url', 'url']:
        if k in d: print(d[k]); sys.exit(0)
    if 'data' in d:
        dd = d['data']
        for k in ['video_url', 'output_url', 'result_url', 'url', 'output']:
            v = dd.get(k) if isinstance(dd, dict) else None
            if isinstance(v, str): print(v); sys.exit(0)
            if isinstance(v, list) and v: print(v[0]); sys.exit(0)
except Exception as e:
    print(f'PARSE_ERR: {e}', file=sys.stderr)
" 2>/dev/null)

if [ -z "$VIDEO_URL" ]; then
  echo "ERROR: couldn't extract video URL from status response. Full response:"
  echo "$STATUS_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$STATUS_RESPONSE"
  exit 1
fi

OUT="/tmp/shoosty-avatar-smoke-$(date +%s).mp4"
echo "  video URL: $VIDEO_URL"
echo "  downloading to: $OUT"
curl -sSL "$VIDEO_URL" -o "$OUT"

echo ""
echo "── DONE ────────────────────────────────────────────"
echo "  saved: $OUT"
echo "  size:  $(ls -lh "$OUT" | awk '{print $5}')"
echo "  open:  open $OUT"
echo "─────────────────────────────────────────────────────"
open "$OUT" 2>/dev/null || true
