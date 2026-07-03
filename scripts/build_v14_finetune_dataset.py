#!/usr/bin/env python3
"""Build the combined WS5c V14 fine-tune data_dir + manifest.

Merges two label sources into a single ``DrumsV14FullSongDataset``-compatible
data_dir (labels at ``<data_dir>/<id>/drums_labels.json``, stems referenced by
absolute path so NO audio is copied):

  (a) the synthetic front-ensemble corpus (ws5_assets/corpus), re-using the
      authoritative 239/27 train/test split from datasets/ws4b_percussion.
      Unlike WS4b's own drums_labels.json (which omits ``velocity`` and would
      raise KeyError in DrumsV14FullSongDataset._load_labels), this RE-DERIVES
      each hit's velocity from the corpus MIDI (labels.json ``events[].vel``),
      closing the compat gap without touching src/models/drums_v14_dataset.py
      (NOTES.md §1/§5.2).

  (b) an optional E-GMD replay buffer produced by
      scripts/extract_egmd_subset.py (--egmd-manifest). Its train slice merges
      into the training pool; its held-out ``egmd_val`` proxy slice stays a
      separate split for the trainer's anti-forgetting tripwire.

Splits in the combined manifest:
  train    : synthetic-train (minus carved val) + E-GMD replay train slice
  val      : synthetic-val carved from synthetic-train (early-stop primary;
             kept OUT of the pristine 27-piece test set so the ≥85 F1 gate
             stays untainted)
  test     : the 27 synthetic held-out pieces (identical membership to WS4b)
  egmd_val : E-GMD kit-drum proxy (anti-forgetting guard); empty if no --egmd-manifest

Per-source ``source_game`` tags: synthetic_ensemble / e_gmd_replay /
e_gmd_replay_proxy. A ``mixing`` block records the configurable synthetic/replay
mix ratio the trainer turns into WeightedRandomSampler weights.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Reuse the settled Option-C instrument->lane mapping verbatim (do not redecide).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_synth_percussion_dataset import INSTRUMENT_TO_LANE  # noqa: E402


def load_split_map(split_manifest: Path) -> dict[str, str]:
    with open(split_manifest) as f:
        m = json.load(f)
    return {s["id"]: s["split"] for s in m["songs"]}


def corpus_events_to_hits(events: list[dict]) -> tuple[list[dict], list[str]]:
    """Map corpus labels.json events -> V14 hits WITH velocity.

    Mirrors build_synth_percussion_dataset.events_to_hits (same INSTRUMENT_TO_LANE,
    same 1:1 event order, unmapped instruments dropped) but additionally carries
    ``event['vel']`` through as ``hits[].velocity`` — the one field WS4b stripped.
    """
    hits: list[dict] = []
    unmapped: list[str] = []
    for ev in events:
        mapping = INSTRUMENT_TO_LANE.get(ev["class"])
        if mapping is None:
            unmapped.append(ev["class"])
            continue
        lane, is_cymbal = mapping
        hits.append(
            {
                "time_ms": ev["t"] * 1000.0,
                "lane": lane,
                "is_cymbal": is_cymbal,
                "velocity": int(ev["vel"]),
            }
        )
    return hits, unmapped


def build_synthetic(
    corpus_dir: Path,
    split_map: dict[str, str],
    output_dir: Path,
    synth_val_frac: float,
    seed: int,
) -> list[dict]:
    # Deterministically carve a val slice out of the synthetic-train ids.
    train_ids = sorted(i for i, s in split_map.items() if s == "train")
    rng = random.Random(seed)
    rng.shuffle(train_ids)
    n_val = max(1, round(len(train_ids) * synth_val_frac))
    val_ids = set(train_ids[:n_val])

    songs: list[dict] = []
    unmapped_total: Counter = Counter()
    n_missing = 0
    for song_id, orig_split in sorted(split_map.items()):
        piece_dir = corpus_dir / song_id
        labels_path = piece_dir / "labels.json"
        mix_path = piece_dir / "mix.wav"
        if not labels_path.exists() or not mix_path.exists():
            n_missing += 1
            continue

        with open(labels_path) as f:
            labels = json.load(f)
        hits, unmapped = corpus_events_to_hits(labels["events"])
        unmapped_total.update(unmapped)

        if orig_split == "test":
            split = "test"
        elif song_id in val_ids:
            split = "val"
        else:
            split = "train"

        out_song_dir = output_dir / song_id
        out_song_dir.mkdir(parents=True, exist_ok=True)
        with open(out_song_dir / "drums_labels.json", "w") as f:
            json.dump({"hits": hits}, f)

        songs.append(
            {
                "id": song_id,
                "split": split,
                "charts": {"drums": True},
                "stems": {"drums": str(mix_path.resolve())},
                "source_game": "synthetic_ensemble",
                "preset": labels.get("preset"),
                "seed": labels.get("seed"),
                "n_label_hits": len(hits),
            }
        )

    if unmapped_total:
        print(f"  WARN: {sum(unmapped_total.values())} unmapped synthetic events "
              f"dropped: {dict(unmapped_total)}")
    if n_missing:
        print(f"  Skipped {n_missing} corpus ids missing labels.json/mix.wav")
    return songs


def merge_egmd(egmd_manifest: Path, output_dir: Path) -> list[dict]:
    with open(egmd_manifest) as f:
        m = json.load(f)
    egmd_dir = egmd_manifest.parent
    songs: list[dict] = []
    for s in m["songs"]:
        song_id = s["id"]
        # Locate the E-GMD clip's label json (data_dir-style: <dir>/<id>/drums_labels.json)
        src_labels = egmd_dir / song_id / "drums_labels.json"
        if not src_labels.exists():
            print(f"  WARN: E-GMD labels missing, skipping {song_id}")
            continue
        out_song_dir = output_dir / song_id
        out_song_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_labels, out_song_dir / "drums_labels.json")
        # Copy the manifest entry, keeping its absolute stem path (no audio copy).
        songs.append(dict(s))
    return songs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", type=Path, default=Path("ws5_assets/corpus"))
    p.add_argument("--split-manifest", type=Path, default=Path("datasets/ws4b_percussion/manifest.json"))
    p.add_argument("--egmd-manifest", type=Path, default=None,
                   help="datasets/egmd_subset/manifest.json (from extract_egmd_subset.py); "
                        "omit for a synthetic-only fine-tune")
    p.add_argument("--output-dir", type=Path, default=Path("datasets/ws5c_finetune"))
    p.add_argument("--synth-val-frac", type=float, default=0.10,
                   help="fraction of synthetic-train carved into the early-stop val split")
    p.add_argument("--mix-ratio", type=float, default=0.75,
                   help="target expected fraction of SYNTHETIC songs per epoch (rest = E-GMD replay)")
    p.add_argument("--seed", type=int, default=20260702)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 72)
    print("Build combined WS5c V14 fine-tune dataset")
    print("=" * 72)

    if not args.split_manifest.exists():
        print(f"FATAL: split manifest not found: {args.split_manifest}", file=sys.stderr)
        return 2
    split_map = load_split_map(args.split_manifest)
    print(f"Split source: {args.split_manifest}  ({len(split_map)} ids, "
          f"{sum(v == 'train' for v in split_map.values())} train / "
          f"{sum(v == 'test' for v in split_map.values())} test)")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nSynthetic corpus (re-deriving velocity from MIDI)...")
    songs = build_synthetic(args.corpus_dir, split_map, args.output_dir,
                            args.synth_val_frac, args.seed)

    egmd_songs: list[dict] = []
    if args.egmd_manifest:
        if not args.egmd_manifest.exists():
            print(f"FATAL: --egmd-manifest not found: {args.egmd_manifest}", file=sys.stderr)
            return 2
        print(f"\nMerging E-GMD replay buffer: {args.egmd_manifest}")
        egmd_songs = merge_egmd(args.egmd_manifest, args.output_dir)
        songs += egmd_songs
    else:
        print("\nNo --egmd-manifest given: synthetic-only manifest "
              "(trainer falls back to uniform sampling, no forgetting guard).")

    split_counts = Counter(s["split"] for s in songs)
    source_counts = Counter(s["source_game"] for s in songs)

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "corpus_dir": str(args.corpus_dir),
        "split_manifest": str(args.split_manifest),
        "egmd_manifest": str(args.egmd_manifest) if args.egmd_manifest else None,
        "seed": args.seed,
        "synth_val_frac": args.synth_val_frac,
        "mixing": {
            "synthetic_frac": args.mix_ratio,
            "sources": ["synthetic_ensemble", "e_gmd_replay"],
        },
        "split_counts": dict(split_counts),
        "source_counts": dict(source_counts),
        "songs": songs,
    }
    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)

    print("\nWrote", manifest_path)
    print(f"  splits : {dict(split_counts)}")
    print(f"  sources: {dict(source_counts)}")
    print(f"  total  : {len(songs)} songs (no audio copied; stems are absolute paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
