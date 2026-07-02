# Genre-stratified out-of-envelope benchmark — results

The in-envelope benchmark in the main README (n=29) is pre-screened by an
audio-feature operating envelope (median Demucs drums-stem RMS ≥ 0.018), which
selects for material Demucs already separates cleanly — in practice, mostly
rock/pop songs with a conventional acoustic or electronic drum kit. This
benchmark instead samples **outside** that envelope on purpose: 25 songs
stratified across five genre buckets, chosen from a local Clone Hero library
without any RMS pre-screen, to measure how stock STRUM performs on material a
kit-trained separation+detection pipeline was never tuned for (EDM/electronic,
J-pop/vocaloid, orchestral, and percussion-forward/pit-percussion charts),
with a rock bucket as an in-envelope control.

This directory ships **manifest + tooling + results only** — `genre_manifest.json`
(title/artist/bucket/charter/notes.mid MD5/selection evidence) and
`scripts/build_genre_benchmark.py` (reconstructs the audio mixes from a user's
own Clone Hero library). No audio or `.mid` chart files are distributed here;
see "Reproducing" below.

## Results

Aggregate per-bucket **drums** onset F1 and lane accuracy, evaluated with
STRUM's low-RMS drums-stem fallback (WS2, `STRUM_STEM_FALLBACK`) both off
(stock separation-only behavior) and on (default going forward). n = number
of songs in the bucket; all 25 selected songs were evaluated (0 dropped, 0
pipeline failures).

| Bucket | n | F1 (fallback OFF) | F1 (fallback ON) | Lane acc. (OFF) | Lane acc. (ON) |
|---|---|---|---|---|---|
| rock (control) | 6 | 88.4% | 88.4% | 73.4% | 73.4% |
| edm_electronic | 6 | 87.9% | 87.9% | 60.3% | 60.3% |
| jpop_vocaloid | 6 | 85.3% | 85.3% | 49.4% | 49.4% |
| orchestral | 3 | 80.1% | 80.1% | 65.2% | 65.2% |
| percussion_forward | 4 | 67.1% | 76.0% | 64.0% | 63.7% |

The rock control clears the sanity gate (mean F1 ≥ 75%), confirming stock
STRUM reaches its published in-envelope range on this harness. Every other
bucket sits below rock, in a rough progression EDM ≈ rock > J-pop/vocaloid >
orchestral > percussion-forward — i.e. distance from "conventional kit in a
mixed track" tracks distance from what the pipeline was built for, with
percussion-forward the clear frontier.

The fallback ON/OFF columns are identical for every bucket except
percussion-forward, because the fallback only activates when a song's Demucs
drums-stem RMS falls below the 0.018 floor (WS2). Across all 25 songs that
happened exactly once: **"Believing" (Bang on a Can All-Stars / Julia Wolfe,
percussion_forward)** measured a drums-stem RMS of 0.0013 — two orders of
magnitude below the floor, a Demucs mis-separation on pit-percussion audio,
not merely a quiet mix — which triggers fallback to the full mix for onset
detection. That one song accounts for the entire OFF→ON delta in the table
above: percussion-forward mean F1 rises from 67.1% to 76.0%, and it is the
sole reason that bucket's column differs at all. See `bench_on.log` for the
runtime warning and measured RMS.

## Methodology

- **Selection.** Candidates were drawn from a scan of the local Clone Hero
  library (`/Users/maple/Clone Hero/Songs`, ≈5,072 songs). Hard requirements:
  `notes.mid` present, Expert drums difficulty ≥ 1, and drum notes actually
  present in `PART DRUMS`. Quality requirements (charter reputation):
  preference for `icon=rbn`/`rb1` (official Rock Band/Harmonix-sourced
  charts) or known community charter groups; where the chart is a community
  (non-RBN) chart, at least 2 of {multi-event tempo map, hand-authored
  difficulty tiers with strictly increasing note counts Easy<Medium<Hard<Expert,
  overdrive phrases, drum animations} were required. Per-song evidence for
  each of these signals is recorded in `genre_manifest.json` under
  `selection_evidence`.
- **Buckets.** 4–6 songs per bucket, from Classic Rock/Alternative (rock
  control), Electronic/Dance/Techno (edm_electronic), J-pop/Vocaloid/J-rock
  (jpop_vocaloid), solo-orchestra recordings (orchestral), and
  marching/pit-percussion-forward charts (percussion_forward).
- **Mix construction.** Each song's full mix is rebuilt with
  `scripts/build_genre_benchmark.py`: an `ffmpeg` `amix` (`normalize=0`) of
  every `.ogg` stem in the chart folder, followed by `alimiter=limit=0.97`,
  encoded to 44.1 kHz stereo 16-bit PCM WAV. The script locates each manifest
  song in a user-supplied library by (artist, title) from `song.ini`, and
  verifies the located folder's `notes.mid` MD5 against the manifest before
  using it, so results are only ever reported against the exact ground-truth
  chart the benchmark was defined with.
- **Inference.** Stock `scripts/batch_infer_hybrid.py`
  (`PYTORCH_ENABLE_MPS_FALLBACK=1`), run once with `STRUM_STEM_FALLBACK=0`
  (baseline/OFF column) and once with the default (fallback ON, WS2).
- **Evaluation.** `scripts/eval_benchmark.py --gt-dir <gt> --pred-dir <pred>
  --tolerance-ms 100 --global-offset-search`. Tolerance is ±100 ms;
  `--global-offset-search` sweeps a ±200 ms / 10 ms-step per-song,
  per-instrument global offset before scoring, to neutralize chart-authoring
  sync conventions. Aggregates in the table above are unweighted per-bucket
  means of each song's drums F1 / lane accuracy.
- **Song counts and shortfalls.** rock=6, edm_electronic=6, jpop_vocaloid=6,
  orchestral=3, percussion_forward=4 (25 total). orchestral and
  percussion_forward fall short of the 4–6 target range's top end because the
  local library genuinely does not contain more charts meeting the selection
  bar in those genres: the orchestral bucket in particular draws all 3 songs
  from a single charter/performer (Paul Henry Smith & The Fauxharmonic
  Orchestra) because no other solo-orchestra recordings with drums charts
  passed the quality bar. `percussion_forward` includes one entry flagged
  `weak_evidence: true` in the manifest — "Epic Symphony in A Flat Minor,
  First Movement: Marching Out" (Van Friscia) was selected only because its
  title contains "Marching"; it was not independently verified as
  marching-arts/pit-percussion audio (it is a solo-artist prog track), so its
  bucket assignment should be treated as low-confidence.
- **Dropped songs.** 0. All 25 selected songs produced a prediction and an
  eval row in both the OFF and ON runs; none errored or were silently
  excluded.

## Reproducing

```bash
python scripts/build_genre_benchmark.py \
  --library-dir "/path/to/Clone Hero/Songs" \
  --manifest benchmarks/genre_manifest.json \
  --output-dir bench_audio

PYTORCH_ENABLE_MPS_FALLBACK=1 STRUM_STEM_FALLBACK=0 \
  python scripts/batch_infer_hybrid.py --input-dir bench_audio --output-dir output/bench_off
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python scripts/batch_infer_hybrid.py --input-dir bench_audio --output-dir output/bench_on

python scripts/eval_benchmark.py --gt-dir <your-gt-dir> --pred-dir output/bench_off \
  --tolerance-ms 100 --global-offset-search --out results_off.json
python scripts/eval_benchmark.py --gt-dir <your-gt-dir> --pred-dir output/bench_on \
  --tolerance-ms 100 --global-offset-search --out results_on.json
```

`<your-gt-dir>` needs one folder per song (matching the prediction folder
names `batch_infer_hybrid.py` derives) containing that song's own
`notes.mid`, which you must supply from your own library — this benchmark
does not redistribute charts.

A reproducibility spot-check (delete one song's mix, rebuild it via
`build_genre_benchmark.py`, re-run inference and eval) reproduced
"Kick Back" (Kenshi Yonezu) within 0.5 F1 points of the original run
(84.55% → 84.06%, Δ = −0.49 points); see `validation/ws3/repro_check.txt`
for full command output (not shipped in this directory — local evidence
only).

## Limitations

- **Single-artist orchestral bucket.** All 3 orchestral songs are performed
  by the same artist/charter (Paul Henry Smith & The Fauxharmonic Orchestra).
  The 80.1% F1 for that bucket should be read as "STRUM on this one
  performer's recordings/charting style," not as a generalizable orchestral
  number — the local library did not have enough independently-produced
  orchestral drum charts to diversify it.
- **percussion_forward is small (n=4) and includes one low-confidence bucket
  assignment** (the Van Friscia "Marching Out" entry, flagged
  `weak_evidence` in the manifest — see Methodology above).
- **This benchmark measures stock STRUM at commit `582f7b6`** (the tip of
  `feat/stem-fallback` at the time this benchmark was built, i.e. including
  the WS2 low-RMS fallback feature, toggled here via `STRUM_STEM_FALLBACK`).
  It is not run against upstream `opria123/strum` and is not a comparison to
  any other tool.
- **Small n per bucket (3–6 songs).** These are means over a handful of
  songs each, sampled from one user's Clone Hero library — read them as
  directional signal about where the pipeline is weak (percussion-forward,
  clearly; lane accuracy on EDM/vocaloid, clearly), not as tight statistical
  estimates.
- **Lane accuracy is a harder, more sensitive metric than onset F1** here:
  note the EDM/vocaloid buckets score well on onset F1 (85–88%) but poorly on
  lane accuracy (49–60%) — the pipeline is finding the right *times* far more
  reliably than the right *drum lanes* on this material.
