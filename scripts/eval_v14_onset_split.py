#!/usr/bin/env python3
"""Direct frame-level onset-F1 eval of a V14 checkpoint on a manifest split.

Implements RUNBOOK step (f): the held-out synthetic test gate (onset F1 >= 85%).
Loads a V14 checkpoint (stock or WS5c fine-tuned), runs each song in the chosen
split through the model, peak-picks onsets with the SAME logic as inference
(scripts/batch_infer_hybrid.detect_onsets_v14: find_peaks(height=onset_threshold,
distance=min_distance_ms)), and scores onset-only precision/recall/F1 at
±tolerance-ms against each song's drums_labels.json hits — ignoring lane/class
entirely (detector-only: "the detector needs ALL onsets regardless of class").

Pieces are short (<=~93 s) so each song is one full-song forward pass. The
log-mel is computed by DrumsV14FullSongDataset exactly as in training/inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.drums_v13 import TwoStageDrumsCRNN  # noqa: E402
from src.models.drums_v14_dataset import DrumsV14FullSongDataset  # noqa: E402


def pick_device() -> torch.device:
    import os
    env = os.environ.get("STRUM_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: Path, device: torch.device) -> TwoStageDrumsCRNN:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}).get("model", {})
    model = TwoStageDrumsCRNN(
        n_mels=cfg.get("n_mels", 128),
        conv_channels=cfg.get("conv_channels", [64, 128, 256, 512]),
        freq_subbands=cfg.get("freq_subbands", [32, 64, 96, 128]),
        subband_proj_dim=cfg.get("subband_proj_dim", 256),
        lstm_hidden=cfg.get("lstm_hidden", 640),
        lstm_layers=cfg.get("lstm_layers", 3),
        attention_heads=cfg.get("attention_heads", 10),
        attention_type=cfg.get("attention_type", "flash"),
        attention_window=cfg.get("attention_window", 512),
        dropout=0.0,
        onset_detector_hidden=cfg.get("onset_detector_hidden", 320),
        classifier_hidden=cfg.get("classifier_hidden", 640),
        num_classes=cfg.get("num_classes", 8),
        predict_velocity=cfg.get("predict_velocity", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded {checkpoint}  (epoch={ckpt.get('epoch')} best_f1={ckpt.get('best_f1')})")
    return model


def match_times(pred_ms: list[float], gt_ms: list[float], tol_ms: float) -> tuple[int, int, int]:
    pred = sorted(pred_ms)
    gt = sorted(gt_ms)
    used = [False] * len(pred)
    tp = 0
    for g in gt:
        best, best_d = -1, tol_ms + 1
        for k, p in enumerate(pred):
            if used[k]:
                continue
            d = abs(p - g)
            if d <= tol_ms and d < best_d:
                best, best_d = k, d
        if best >= 0:
            used[best] = True
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tolerance-ms", type=float, default=100.0)
    p.add_argument("--onset-threshold", type=float, default=0.5)
    p.add_argument("--min-distance-ms", type=float, default=20.0)
    p.add_argument("--gate-f1", type=float, default=85.0)
    p.add_argument("--dedupe-gt-ms", type=float, default=0.0,
                   help="Collapse GT hits closer than this into one onset "
                        "event before matching (0 = off). A frame-level "
                        "detector emits at most one onset per frame, so "
                        "simultaneous chord notes (51%% of the WS5b test "
                        "split's hits) are otherwise unmatchable FNs — raw "
                        "per-hit GT caps recall at 49%%.")
    p.add_argument("--chunk-seconds", type=float, default=0.0,
                   help="Run the forward pass in fixed chunks of this many "
                        "seconds (0 = full song). Matches the WS5c trainer's "
                        "chunked training regime for train/eval-consistent "
                        "measurement.")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = pick_device()
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    sr, hop, n_mels, n_fft = 44100, 512, 128, 2048
    ds = DrumsV14FullSongDataset(
        manifest_path=args.manifest, split=args.split,
        sample_rate=sr, n_mels=n_mels, hop_length=hop, n_fft=n_fft,
        augment=False, max_duration_sec=600.0,
    )
    print(f"Split '{args.split}': {len(ds)} songs")

    min_dist = max(1, int(args.min_distance_ms / 1000.0 * sr / hop))
    frame_ms = hop / sr * 1000.0

    per_song = []
    tot_tp = tot_fp = tot_fn = 0
    for i in range(len(ds)):
        song = ds.songs[i]
        mel, _ = ds[i]
        chunk_frames = int(args.chunk_seconds * sr / hop) if args.chunk_seconds > 0 else 0
        T = mel.shape[-1]
        starts = list(range(0, T, chunk_frames)) if 0 < chunk_frames < T else [0]
        parts = []
        with torch.no_grad():
            for s0 in starts:
                s1 = min(s0 + chunk_frames, T) if chunk_frames > 0 else T
                mel_c = mel[..., s0:s1]
                if chunk_frames > 0 and mel_c.shape[-1] < chunk_frames:
                    pad = chunk_frames - mel_c.shape[-1]
                    mel_c = torch.nn.functional.pad(mel_c, (0, pad), value=float(mel_c.min()))
                out = model(mel_c.unsqueeze(0).to(device))
                parts.append(out["onset_probs"].squeeze(0).squeeze(-1).float().cpu()[: s1 - s0])
        probs = torch.cat(parts).numpy()
        peaks, _ = find_peaks(probs, height=args.onset_threshold, distance=min_dist)
        pred_ms = [float(f) * frame_ms for f in peaks]

        labels_path = ds.data_dir / song["id"] / "drums_labels.json"
        with open(labels_path) as f:
            gt_ms = [h["time_ms"] for h in json.load(f)["hits"]]
        if args.dedupe_gt_ms > 0:
            deduped = []
            for t in sorted(gt_ms):
                if not deduped or t - deduped[-1] >= args.dedupe_gt_ms:
                    deduped.append(t)
            gt_ms = deduped

        tp, fp, fn = match_times(pred_ms, gt_ms, args.tolerance_ms)
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_song.append({"id": song["id"], "precision": prec * 100, "recall": rec * 100,
                         "f1": f1 * 100, "tp": tp, "fp": fp, "fn": fn,
                         "n_pred": len(pred_ms), "n_gt": len(gt_ms)})

    prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0.0
    rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    macro_f1 = float(np.mean([s["f1"] for s in per_song])) if per_song else 0.0
    passed = (f1 * 100) >= args.gate_f1

    summary = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "split": args.split,
        "n_songs": len(ds),
        "tolerance_ms": args.tolerance_ms,
        "onset_threshold": args.onset_threshold,
        "micro": {"precision": prec * 100, "recall": rec * 100, "f1": f1 * 100,
                  "tp": tot_tp, "fp": tot_fp, "fn": tot_fn},
        "macro_f1": macro_f1,
        "gate_f1": args.gate_f1,
        "gate_pass": passed,
        "per_song": per_song,
    }
    print(f"\nMicro onset F1 = {f1 * 100:.2f}  (P={prec * 100:.2f} R={rec * 100:.2f}) "
          f"| macro F1 = {macro_f1:.2f}")
    print(f"GATE (>= {args.gate_f1}): {'PASS' if passed else 'FAIL'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"Wrote {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
