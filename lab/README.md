# lab/ — Omnihuman smoke tests

Quick standalone tests for the Shoosty Avatar feature (powered by Dreamina
Omnihuman 1.5 via AceData). Stephen's lab pattern: run a single
command, get a result file you can play, sanity-check that the magic
actually works before building UI around it.

## Why this matters

Omnihuman 1.5 is the **Shoosty Avatar** feature — feed it a portrait
photo + audio, get a video of the portrait lip-syncing the audio.

For the JAX silk-art show: silk artwork photo → song → Shoosty Avatar
animates the silk piece "singing" the song → QR code → public listen
page with the lip-sync video embedded.

Cost per test: **~$0.69**.

## Test paths

### Path A — MCP probe (recommended)

Paste `mcp-omnihuman-probe.md` into your **main Claude session** (the
one with AceData MCP wired in). That session:

1. Introspects the actual AceData MCP tool name + parameters for
   Omnihuman
2. Fires the call with your test portrait + audio
3. Polls for completion
4. Returns the video URL you can open in a browser

Why this path first: the MCP tool is self-describing, so even if the
underlying AceData API changes, the prompt still works. Also: the
main session already has your AceData credentials configured.

### Path B — Direct REST (fallback)

`omnihuman-curl-test.sh` — for when you want to fire from any
terminal without a Claude session. You provide the API key via env
var and the test sends a request to AceData's REST endpoint.

**Note:** the exact endpoint URL is a placeholder pending docs
verification. After you confirm the real URL in the AceData docs
(behind your account login), update `omnihuman-curl-test.sh` lines
marked `# TODO:`.

## Test inputs

Default test uses:
- **Portrait:** any 512×512+ headshot or stylized portrait
- **Audio:** a short 5-10 sec WAV/MP3 vocal track (a melody hum or
  song clip works)

Replace `tests/portrait.png` and `tests/audio.mp3` with real files
before running.

## What "success" looks like

- Job submits without auth error
- Returns within ~30-90 seconds with a video URL
- Video plays a person/figure whose mouth moves in sync with the audio
- Result file is an MP4 you can play in any browser

If the lip-sync is convincing on a silk-art photo, **that's the
demo for the JAX show.** If not, we learn what the model expects
(real human face? stylized OK? exact aspect ratio?) and adjust.

## After the smoke test

Once Omnihuman works:
1. Ship the user-facing "Shoosty Avatar" picker in the writer page
   (main Claude session is doing this UI work)
2. Wire it into the JAX flow: song completes → user picks an image →
   Shoosty Avatar animates it → embed in listen page
3. Add the rest of the AceData video catalog as additional Shoosty
   brand tiers (Shoosty Sketch, Shoosty Premiere, etc.)

## File index

- `README.md` — this file
- `mcp-omnihuman-probe.md` — paste-into-main-Claude prompt (Path A)
- `omnihuman-curl-test.sh` — standalone curl test (Path B)
- `tests/` — sample portrait + audio you can use to fire the test
