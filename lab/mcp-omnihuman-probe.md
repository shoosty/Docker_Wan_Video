# Shoosty Avatar smoke test — paste into the MAIN Claude session

> Paste the block below into your **main** Claude session (the one
> with AceData MCP wired in for Suno + access to shoosty-studio's
> Supabase). It introspects the AceData MCP tool surface, finds the
> Omnihuman / lip-sync tool, and fires three test jobs — one per
> Horse Adjutant character portrait — against the "Oh Smile at Me"
> audio.

---

```
Smoke test for "Shoosty Avatar" — Dreamina Omnihuman 1.5 via AceData
MCP. Total budget: ~$2.07 (~$0.69 × 3 portraits). Real content this
time, not a stock image.

THE AUDIO

Song: "Oh Smile at Me" — Stephen Shooster, from The Horse Adjutant
musical. It's one of the 17 completed songs already in the Supabase
catalog. Look it up:

  SELECT s.id, s.title, g.audio_url_1, g.id as gen_id
  FROM songs s
  JOIN generations g ON g.song_id = s.id
  WHERE s.title ILIKE '%Oh Smile at Me%'
    AND g.audio_url_1 IS NOT NULL
  ORDER BY g.completed_at DESC
  LIMIT 1;

Use the audio_url_1 from that row as the audio input.

THE PORTRAITS

Stephen will provide three reference photos for the Horse Adjutant
characters:
  1. Max Blauner
  2. Girlfriend
  3. Jan Kipura

If they're already uploaded to the gallery, find them:

  SELECT id, name, image_url
  FROM gallery_images
  WHERE name ILIKE ANY (ARRAY['%Max Blauner%', '%Girlfriend%', '%Jan Kipura%'])
  ORDER BY name;

If not uploaded yet, ask Stephen to upload them via /preview/gallery
and re-run this prompt. Don't substitute stock images — the whole
point is to see whether Omnihuman handles HIS content.

STEPS

1. List every AceData MCP video tool. Identify the Omnihuman 1.5
   tool. Quote its parameter schema.

2. Fire 3 jobs sequentially (don't parallelize — easier to debug if
   one fails):

   Job A: portrait = Max Blauner photo,    audio = Oh Smile at Me
   Job B: portrait = Girlfriend photo,     audio = Oh Smile at Me
   Job C: portrait = Jan Kipura photo,     audio = Oh Smile at Me

   For each: report job ID, elapsed time, final video URL.

3. After all three land, give me a verdict table:

   | Character    | Job ID | Elapsed | Video URL | Convincing? |
   |--------------|--------|---------|-----------|-------------|

   For "convincing" — peek at the video, judge whether the mouth
   movement tracks the audio. Don't be polite, be honest. The
   answer determines whether this becomes Shoosty Avatar's
   shipping feature or stays as a parked experiment.

4. Stamp the total cost (sum of per-call cost from AceData's
   response, or estimate at ~$0.69 each if not returned).

5. If any job fails, capture the error verbatim and report it. Don't
   retry automatically — Stephen wants to see failures.

WHY THIS MATTERS

This is the JAX silk-art show MVP test. If three character portraits
sing "Oh Smile at Me" in convincing lip-sync, that's the feature we
ship. If they don't, we learn what Omnihuman expects (real photos
vs stylized? specific aspect ratio? portrait orientation?) and
re-strategize before building the dropdown UI.
```

---

## What to expect back

A response from the main Claude with:
- AceData MCP tool list (focus on video tools)
- The specific Omnihuman tool name + schema
- 3 job IDs + 3 result URLs
- A verdict table per character
- Total billed cost

## If it doesn't work

If Omnihuman fails on stylized art or specific portrait shapes,
useful follow-ups:
- Try with cropped face-only versions of the photos
- Try with realistic photos as a control to verify Omnihuman works at all
- Check if AceData has a "Dreamina Omnihuman" variant tuned for art
- Fall back to a different lip-sync model in the catalog (none other
  visible right now, but worth checking the MCP list)
