#!/usr/bin/env python3
"""Stratified E-GMD replay-buffer extraction for the WS5c V14 fine-tune.

The stock V14 detector was trained on kit drums whose exact data is not
available locally (configs/drums_v14.yaml points at a DGX-only path). To guard
the fine-tune against catastrophic forgetting on kit drums we mix in a small,
diverse slice of Google's Expanded Groove-MIDI Dataset (E-GMD) as a replay
buffer (see validation/ws5c/NOTES.md §6.2).

This script, run once by the orchestrator:

  1. Reads the metadata CSV straight out of the (read-only, NAS-mounted) MIDI
     zip's central directory — no full unzip.
  2. Stratifies a *train* slice (~2-3 h audio) across drummers x style-families
     x virtual kits, plus a small held-out *kit-validation* slice (~15-20 min)
     drawn from E-GMD's own ``validation`` split — the anti-forgetting proxy the
     trainer early-stops on. Deterministic given ``--seed``.
  3. Extracts ONLY the selected .wav/.midi member pairs from the 96 GB audio
     zip via random-access ``ZipFile.read`` (confirmed practical over SMB).
  4. Converts each performance's MIDI (General-MIDI drum pitches, native
     velocities) to the V14 ``drums_labels.json`` format.
  5. Emits a data_dir-style manifest scripts/build_v14_finetune_dataset.py
     merges with the synthetic corpus.

``--dry-run`` prints the selection + compressed-size estimate and extracts
nothing. Nothing under the NAS mount is ever written.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── General-MIDI drum-pitch -> (lane, is_cymbal) (NOTES.md §6.2). Consistent
# with src/models/drums_v14_dataset.py LANE_CYMBAL_TO_CLASS. Pitches with no
# clean lane (39 clap, 54 tambourine, 58 vibraslap) are dropped — negligible
# for a replay buffer whose job is onset detection, not perfect class fidelity.
GM_PITCH_TO_LANE: dict[int, tuple[int, bool]] = {
    36: (0, False),                                    # Kick
    35: (0, False),                                    # Acoustic Bass Drum
    37: (1, False), 38: (1, False), 40: (1, False),    # Snare (rim/head/electric)
    42: (2, True), 44: (2, True), 46: (2, True),       # HiHat (closed/pedal/open)
    22: (2, True), 26: (2, True),                      # HiHat (edge/open-edge variants)
    48: (2, False), 47: (2, False), 50: (2, False),    # HighTom (hi-mid/lo-mid/high)
    45: (3, False),                                    # LowTom
    41: (4, False), 43: (4, False),                    # FloorTom (low/high floor)
    51: (3, True), 53: (3, True), 59: (3, True),       # Ride (bow/bell/edge)
    49: (4, True), 52: (4, True), 55: (4, True), 57: (4, True),  # Crash / China / Splash
}

CSV_MEMBER = "e-gmd-v1.0.0/e-gmd-v1.0.0.csv"
ZIP_PREFIX = "e-gmd-v1.0.0/"  # audio/midi member names are this + CSV filename


def _style_family(style: str) -> str:
    return style.split("/", 1)[0] if style else "unknown"


def _clip_id(audio_filename: str) -> str:
    stem = audio_filename.rsplit(".", 1)[0]
    return "egmd_" + stem.replace("/", "_").replace(" ", "_")


def read_metadata(midi_zip: Path) -> list[dict]:
    with zipfile.ZipFile(midi_zip) as z:
        data = z.read(CSV_MEMBER).decode()
    rows = list(csv.DictReader(io.StringIO(data)))
    for r in rows:
        r["duration"] = float(r["duration"])
        r["style_family"] = _style_family(r["style"])
        r["clip_id"] = _clip_id(r["audio_filename"])
    return rows


def _round_robin(strata: dict, keys: list) -> list[dict]:
    """Flatten shuffled substrata into one queue by round-robin over sorted keys."""
    cursor = {k: 0 for k in keys}
    queue: list[dict] = []
    progress = True
    while progress:
        progress = False
        for k in keys:
            i = cursor[k]
            if i < len(strata[k]):
                queue.append(strata[k][i])
                cursor[k] += 1
                progress = True
    return queue


def stratified_select(
    rows: list[dict],
    target_seconds: float,
    seed: int,
) -> list[dict]:
    """Hierarchical round-robin: drummer (outer) -> (style_family, kit_name)
    (inner), accumulating audio duration until the target is reached.

    Rotating the DRUMMER axis fastest guarantees drummer diversity even when
    the target is met after only a few dozen clips (a plain flat round-robin
    over sorted (drummer,style,kit) keys instead concentrates on the first
    drummer alphabetically). Deterministic given ``seed``.
    """
    rng = random.Random(seed)

    # drummer -> (style,kit) substratum -> shuffled clips
    by_drummer: dict[str, dict[tuple, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_drummer[r["drummer"]][(r["style_family"], r["kit_name"])].append(r)

    drummers = sorted(by_drummer)
    drummer_queues: dict[str, list[dict]] = {}
    for di, d in enumerate(drummers):
        sub = by_drummer[d]
        # Sort substrata by (kit, style) and rotate the start point by the
        # drummer index so different drummers begin at different kits — this
        # spreads kit-timbre coverage across the (few) clips actually taken.
        sub_keys = sorted(sub.keys(), key=lambda kt: (kt[1], kt[0]))
        for k in sub_keys:
            rng.shuffle(sub[k])
        if sub_keys:
            off = di % len(sub_keys)
            sub_keys = sub_keys[off:] + sub_keys[:off]
        drummer_queues[d] = _round_robin(sub, sub_keys)

    cursor = {d: 0 for d in drummers}
    selected: list[dict] = []
    total = 0.0
    made_progress = True
    while total < target_seconds and made_progress:
        made_progress = False
        for d in drummers:
            if total >= target_seconds:
                break
            i = cursor[d]
            if i < len(drummer_queues[d]):
                r = drummer_queues[d][i]
                cursor[d] += 1
                selected.append(r)
                total += r["duration"]
                made_progress = True
    return selected


def selection_report(selected: list[dict], label: str) -> str:
    total_s = sum(r["duration"] for r in selected)
    drummers = Counter(r["drummer"] for r in selected)
    styles = Counter(r["style_family"] for r in selected)
    kits = Counter(r["kit_name"] for r in selected)
    lines = [
        f"  [{label}] clips={len(selected)}  audio={total_s / 3600:.2f} h "
        f"({total_s / 60:.1f} min)",
        f"    drummers ({len(drummers)}): "
        + ", ".join(f"{d}={n}" for d, n in sorted(drummers.items())),
        f"    style families ({len(styles)}): "
        + ", ".join(f"{s}={n}" for s, n in sorted(styles.items())),
        f"    kits ({len(kits)}): "
        + ", ".join(f"{k}={n}" for k, n in sorted(kits.items())),
    ]
    return "\n".join(lines)


def estimate_compressed_bytes(audio_zip: Path, selected: list[dict]) -> int:
    """Sum the compressed sizes of the selected .wav + .midi members without
    extracting them (central-directory metadata only)."""
    total = 0
    with zipfile.ZipFile(audio_zip) as z:
        for r in selected:
            for fn in (r["audio_filename"], r["midi_filename"]):
                try:
                    total += z.getinfo(ZIP_PREFIX + fn).compress_size
                except KeyError:
                    print(f"    WARN: member not in audio zip: {fn}", file=sys.stderr)
    return total


def midi_bytes_to_hits(midi_bytes: bytes) -> tuple[list[dict], Counter]:
    import pretty_midi  # local import: heavy, only needed for real extraction

    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    hits: list[dict] = []
    dropped = Counter()
    for inst in pm.instruments:
        for note in inst.notes:
            mapping = GM_PITCH_TO_LANE.get(note.pitch)
            if mapping is None:
                dropped[note.pitch] += 1
                continue
            lane, is_cymbal = mapping
            hits.append(
                {
                    "time_ms": float(note.start) * 1000.0,
                    "lane": lane,
                    "is_cymbal": is_cymbal,
                    "velocity": int(note.velocity),
                }
            )
    hits.sort(key=lambda h: h["time_ms"])
    return hits, dropped


def check_free_gb(path: str = "/System/Volumes/Data") -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1e9


def extract_and_convert(
    audio_zip: Path,
    selected: list[dict],
    split_tag: str,
    source_game: str,
    output_dir: Path,
) -> list[dict]:
    songs: list[dict] = []
    dropped_total: Counter = Counter()
    with zipfile.ZipFile(audio_zip) as z:
        for r in selected:
            clip_id = r["clip_id"]
            clip_dir = output_dir / clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)

            wav_out = clip_dir / "audio.wav"
            with z.open(ZIP_PREFIX + r["audio_filename"]) as src, open(wav_out, "wb") as dst:
                shutil.copyfileobj(src, dst)

            midi_bytes = z.read(ZIP_PREFIX + r["midi_filename"])
            hits, dropped = midi_bytes_to_hits(midi_bytes)
            dropped_total.update(dropped)
            with open(clip_dir / "drums_labels.json", "w") as f:
                json.dump({"hits": hits}, f)

            songs.append(
                {
                    "id": clip_id,
                    "split": split_tag,
                    "charts": {"drums": True},
                    "stems": {"drums": str(wav_out.resolve())},
                    "source_game": source_game,
                    "drummer": r["drummer"],
                    "style": r["style"],
                    "kit_name": r["kit_name"],
                    "bpm": r["bpm"],
                    "duration_sec": r["duration"],
                    "n_label_hits": len(hits),
                }
            )
    if dropped_total:
        top = ", ".join(f"{p}:{n}" for p, n in dropped_total.most_common(6))
        print(f"    [{split_tag}] dropped unmapped GM pitches: {top}")
    return songs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--midi-zip", type=Path, default=Path("/Volumes/Datasets/e-gmd-v1.0.0-midi.zip"))
    p.add_argument("--audio-zip", type=Path, default=Path("/Volumes/Datasets/e-gmd-v1.0.0.zip"))
    p.add_argument("--output-dir", type=Path, default=Path("datasets/egmd_subset"))
    p.add_argument("--hours", type=float, default=3.0, help="target train-slice audio hours")
    p.add_argument("--proxy-minutes", type=float, default=18.0,
                   help="target held-out kit-validation slice minutes (E-GMD validation split)")
    p.add_argument("--seed", type=int, default=20260702)
    p.add_argument("--min-clip-sec", type=float, default=15.0,
                   help="drop clips shorter than this (DrumsV14FullSongDataset skips <10s)")
    p.add_argument("--max-clip-sec", type=float, default=360.0,
                   help="drop clips longer than this (matches max_song_duration_sec)")
    p.add_argument("--min-free-gb", type=float, default=10.0,
                   help="abort real extraction if free disk drops below this")
    p.add_argument("--dry-run", action="store_true",
                   help="print selection + compressed-size estimate, extract nothing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("E-GMD stratified subset extraction (WS5c replay buffer)")
    print("=" * 72)
    print(f"MIDI zip : {args.midi_zip}")
    print(f"Audio zip: {args.audio_zip}")
    print(f"Seed     : {args.seed}   Target train: {args.hours} h   "
          f"Proxy val: {args.proxy_minutes} min")
    print(f"Free disk: {check_free_gb():.1f} GB")

    if not args.midi_zip.exists():
        print(f"FATAL: MIDI zip not found: {args.midi_zip}", file=sys.stderr)
        return 2
    if not args.audio_zip.exists():
        print(f"FATAL: audio zip not found: {args.audio_zip}", file=sys.stderr)
        return 2

    rows = read_metadata(args.midi_zip)
    by_split = Counter(r["split"] for r in rows)
    print(f"\nMetadata: {len(rows)} clips   splits={dict(by_split)}")

    def eligible(r):
        return args.min_clip_sec <= r["duration"] <= args.max_clip_sec

    train_pool = [r for r in rows if r["split"] == "train" and eligible(r)]
    val_pool = [r for r in rows if r["split"] == "validation" and eligible(r)]
    print(f"Eligible ({args.min_clip_sec:.0f}-{args.max_clip_sec:.0f}s): "
          f"{len(train_pool)} train / {len(val_pool)} validation clips")

    train_sel = stratified_select(train_pool, args.hours * 3600.0, args.seed)
    val_sel = stratified_select(val_pool, args.proxy_minutes * 60.0, args.seed + 1)

    print("\nSelection:")
    print(selection_report(train_sel, "train"))
    print(selection_report(val_sel, "egmd_val"))

    print("\nEstimating extracted size (compressed central-directory sizes)...")
    est_bytes = estimate_compressed_bytes(args.audio_zip, train_sel + val_sel)
    print(f"  Estimated extracted size: {est_bytes / 1e9:.2f} GB "
          f"({len(train_sel) + len(val_sel)} clips x 2 members)")

    if args.dry_run:
        print("\n[DRY RUN] No members extracted, no files written.")
        return 0

    free_gb = check_free_gb()
    need_gb = est_bytes / 1e9
    if free_gb - need_gb < args.min_free_gb:
        print(f"FATAL: extraction (~{need_gb:.2f} GB) would leave < "
              f"{args.min_free_gb} GB free (have {free_gb:.1f} GB). Aborting.",
              file=sys.stderr)
        return 3

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExtracting {len(train_sel)} train + {len(val_sel)} val clips "
          f"to {args.output_dir} ...")
    songs = extract_and_convert(args.audio_zip, train_sel, "train", "e_gmd_replay", args.output_dir)
    songs += extract_and_convert(args.audio_zip, val_sel, "egmd_val", "e_gmd_replay_proxy", args.output_dir)

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_midi_zip": str(args.midi_zip),
        "source_audio_zip": str(args.audio_zip),
        "seed": args.seed,
        "target_train_hours": args.hours,
        "target_proxy_minutes": args.proxy_minutes,
        "gm_pitch_to_lane": {str(k): list(v) for k, v in GM_PITCH_TO_LANE.items()},
        "n_train": len(train_sel),
        "n_egmd_val": len(val_sel),
        "songs": songs,
    }
    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)

    print(f"\nWrote manifest: {manifest_path}  ({len(songs)} songs)")
    print(f"Free disk after extraction: {check_free_gb():.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
