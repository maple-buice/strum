#!/usr/bin/env python3
"""
Add a CQT memmap ({split}_cqt.npy) to an onset-window cache built by
scripts/preprocess_onset_windows.py.

Reimplements the missing script referenced by
configs/onset_classifier_v12_clean.yaml ("Requires: CQT added to existing
cache via scripts/add_cqt_to_cache.py"). Without it, V12c-style models
(use_lowfreq_branch: true) silently train on the 30-2000 Hz mel while
inference feeds them CQT — a train/inference feature skew.

CQT features match inference (scripts/batch_infer_hybrid.py) EXACTLY:
  - librosa.cqt over the FULL song: hop 512, fmin 30 Hz, 24 bins/octave,
    144 bins; log(|C| + 1e-8); trimmed to the first 128 bins
    (batch_infer_hybrid.py CQT_* constants + extract_onset_windows)
  - per onset, the window is a frame slice of the full-song CQT:
    start_sample = int(time_ms/1000*44100) - 4410 (100 ms pre-onset);
    cqt_start = start_sample // 512; take 44 frames; zero-pad past the end
    (batch_infer_hybrid.py extract_onset_windows CQT slicing)

Row alignment with the existing cache is guaranteed by replicating
preprocess_onset_windows.py's iteration exactly (its phase1_count/parse_onsets
are imported, and phase 2's audio loading + skip conditions are mirrored), and
then PROVEN per-row: each written row's multi-hot label recomputed from
drums_labels.json must equal the cached {split}_labels.npy row, or the run
aborts and removes the partial file.

Output: {split}_cqt.npy — (N, 128, 44) float16, N == index total_onsets —
and {split}_index.json gains files.cqt (which the cached dataset prefers over
mel_lowfreq).

Usage:
    add_cqt_to_cache.py --manifest <data_dir>/manifest.json \
        --cache-dir <output_dir>/cache [--split both]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from preprocess_onset_windows import (  # noqa: E402
    SAMPLE_RATE,
    WINDOW_AFTER_SAMP,
    WINDOW_BEFORE_SAMP,
    phase1_count,
)

# CQT params — MUST match scripts/batch_infer_hybrid.py (CQT_FMIN, CQT_BINS_PER_OCT,
# CQT_N_BINS, CQT_TRIM_BINS, CQT_HOP, CQT_FRAMES)
CQT_FMIN = 30.0
CQT_BINS_PER_OCT = 24
CQT_N_BINS = 144
CQT_TRIM_BINS = 128
CQT_HOP = 512
CQT_FRAMES = (WINDOW_BEFORE_SAMP + WINDOW_AFTER_SAMP) // CQT_HOP + 1  # 44


def full_song_cqt(audio: np.ndarray) -> np.ndarray:
    """Full-song log-CQT, trimmed to 128 bins — mirrors
    batch_infer_hybrid.extract_onset_windows lines computing cqt_full."""
    C = np.abs(librosa.cqt(
        y=audio, sr=SAMPLE_RATE, hop_length=CQT_HOP,
        fmin=CQT_FMIN, n_bins=CQT_N_BINS, bins_per_octave=CQT_BINS_PER_OCT,
    ))
    cqt = np.log(C + 1e-8).astype(np.float32)
    return cqt[:CQT_TRIM_BINS, :]


def add_cqt_split(manifest_path: Path, split: str, cache_dir: Path) -> None:
    index_path = cache_dir / f"{split}_index.json"
    if not index_path.exists():
        print(f"FATAL: {index_path} not found — run preprocess_onset_windows.py "
              f"for split '{split}' first", file=sys.stderr)
        sys.exit(1)
    with open(index_path) as f:
        index = json.load(f)
    total = index["total_onsets"]

    cqt_path = cache_dir / f"{split}_cqt.npy"
    if cqt_path.exists():
        print(f"FATAL: {cqt_path} already exists — remove it to regenerate",
              file=sys.stderr)
        sys.exit(1)

    # Cached labels: ground truth for row alignment verification
    labels_mm = np.load(str(cache_dir / index["files"]["labels"]), mmap_mode="r")
    if labels_mm.shape[0] != total:
        print(f"FATAL: labels file rows {labels_mm.shape[0]} != index total "
              f"{total}", file=sys.stderr)
        sys.exit(1)

    # Same song list + ordering as preprocess_onset_windows.extract_split
    with open(manifest_path) as f:
        manifest = json.load(f)
    data_dir = manifest_path.parent
    songs = [s for s in manifest["songs"]
             if s["split"] == split and s["charts"].get("drums")]
    print(f"[{split}] {len(songs)} songs with drum charts; cache rows: {total}")

    # phase1_count reproduces preprocess's song filtering AND per-song onset
    # lists (parse_onsets: 5 ms binning, class mapping) in identical order.
    total_expected, song_info = phase1_count(songs, data_dir)
    if total_expected < total:
        print(f"FATAL: phase-1 recount ({total_expected}) < cache rows "
              f"({total}) — manifest/audio changed since the cache was built",
              file=sys.stderr)
        sys.exit(1)

    mm_cqt = np.lib.format.open_memmap(
        str(cqt_path), mode="w+", dtype=np.float16,
        shape=(total, CQT_TRIM_BINS, CQT_FRAMES))

    offset = 0
    failed_songs = 0
    try:
        for si, sinfo in enumerate(song_info):
            if si % 20 == 0:
                print(f"  [{si}/{len(song_info)}] {offset}/{total} rows written...")
            song = sinfo["song"]
            onset_list = sinfo["onset_list"]
            audio_path = data_dir / song["stems"]["drums"]

            # Audio loading mirrors phase2_extract (sf.read float32, channel
            # mean, torchaudio resample) so skip conditions match row-for-row.
            try:
                audio, sr = sf.read(str(audio_path), dtype="float32")
            except Exception:
                failed_songs += 1
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if len(audio) == 0:
                failed_songs += 1
                continue
            if sr != SAMPLE_RATE:
                audio_t = torch.from_numpy(audio).float().unsqueeze(0)
                audio_t = torchaudio.functional.resample(audio_t, sr, SAMPLE_RATE)
                audio = audio_t.squeeze(0).numpy()

            cqt_full = full_song_cqt(audio)

            for time_ms, classes in onset_list:
                center = int(time_ms / 1000 * SAMPLE_RATE)
                start = center - WINDOW_BEFORE_SAMP
                if start < 0 or center + WINDOW_AFTER_SAMP > len(audio):
                    continue
                if offset >= total:
                    break

                # Row alignment proof: recomputed label must equal cached label
                label = np.zeros(8, dtype=np.uint8)
                for c in classes:
                    label[c] = 1
                if not np.array_equal(label, labels_mm[offset]):
                    raise RuntimeError(
                        f"Row alignment mismatch at row {offset} "
                        f"(song '{song['id']}', t={time_ms:.1f} ms): "
                        f"recomputed label {label.tolist()} != cached "
                        f"{labels_mm[offset].tolist()}. Cache was not built "
                        f"from this manifest/audio — aborting.")

                # Window slice — mirrors batch_infer_hybrid CQT windowing
                cqt_start = start // CQT_HOP
                cqt_end = cqt_start + CQT_FRAMES
                win = np.zeros((CQT_TRIM_BINS, CQT_FRAMES), dtype=np.float32)
                if cqt_end <= cqt_full.shape[1]:
                    win[:] = cqt_full[:, cqt_start:cqt_end]
                else:
                    avail = max(0, cqt_full.shape[1] - cqt_start)
                    if avail > 0:
                        win[:, :avail] = cqt_full[:, cqt_start:cqt_start + avail]
                mm_cqt[offset] = win.astype(np.float16)
                offset += 1

        if offset != total:
            raise RuntimeError(
                f"Wrote {offset} rows but cache has {total} "
                f"(failed songs: {failed_songs}). Iteration did not replicate "
                f"the original preprocessing — aborting.")
    except Exception:
        del mm_cqt
        cqt_path.unlink(missing_ok=True)
        raise

    del mm_cqt  # flush

    index["files"]["cqt"] = f"{split}_cqt.npy"
    index["cqt_params"] = {
        "fmin": CQT_FMIN, "bins_per_octave": CQT_BINS_PER_OCT,
        "n_bins": CQT_N_BINS, "trim_bins": CQT_TRIM_BINS,
        "hop": CQT_HOP, "frames": CQT_FRAMES,
        "windowing": "full-song librosa CQT, frame slice at "
                     "(center_sample - 4410) // 512, 44 frames, zero-padded",
    }
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    size_gb = cqt_path.stat().st_size / 1e9
    print(f"[{split}] OK: {cqt_path} — shape ({total}, {CQT_TRIM_BINS}, "
          f"{CQT_FRAMES}), {size_gb:.2f} GB; all {total} rows label-verified; "
          f"index updated (files.cqt)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--manifest", type=Path, required=True,
                    help="data_dir/manifest.json used to build the cache")
    ap.add_argument("--cache-dir", type=Path, required=True,
                    help="cache dir containing {split}_index.json etc.")
    ap.add_argument("--split", default="both", choices=["train", "test", "both"])
    args = ap.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        t0 = time.time()
        add_cqt_split(args.manifest, split, args.cache_dir)
        print(f"[{split}] took {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
