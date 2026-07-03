#!/usr/bin/env python3
"""From-scratch training loop for the V14 two-stage drum onset detector.

The repo ships the V14 *dataset* (src/models/drums_v14_dataset.py), *model* and
*loss* (src/models/drums_v13.py: TwoStageDrumsCRNN / TwoStageLoss) and a
self-describing config (configs/drums_v14.yaml), but the training loop itself
lived only on off-repo DGX hardware (NOTES.md §1). This is that missing loop,
written for the WS5c front-ensemble fine-tune but usable for any V14 run.

Design (see validation/ws5c/NOTES.md / RUNBOOK.md):
  * Consumes configs/drums_v14_finetune.yaml (architecture verbatim from
    drums_v14.yaml). Log-mel is computed on the fly by DrumsV14FullSongDataset
    (44.1k / n_fft 2048 / hop 512 / 128 mels); single-frame binary onset
    targets; TwoStageLoss (focal BCE onset + focal BCE classify + MSE velocity).
  * Warm-starts model_state_dict (strict=False) from checkpoints/drums_v14/best.pt
    into a FRESH AdamW (no optimizer-state resume). Prints that checkpoint's
    embedded training/loss config so the fine-tune mirrors what's relevant.
  * Device: STRUM_DEVICE override -> cuda -> mps -> cpu.
  * Multi-source mixing via WeightedRandomSampler over each song's source_game
    (synthetic vs E-GMD replay), ratio from config.mixing.synthetic_frac.
  * Gradient accumulation (songs are whole, variable-length; batch_size=1).
  * Early stopping on a COMPOSITE: primary = synthetic-val onset frame-F1;
    guard = E-GMD-val frame-F1 must not fall > egmd_guard_drop below its
    epoch-0 warm-start baseline (anti-forgetting tripwire) — abort if tripped
    egmd_guard_trip_limit times consecutively.
  * Frame-F1 at ±frame_tolerance frames with find_peaks(height=onset_threshold,
    distance=min_distance_ms) peak-picking, mirroring inference (detect_onsets_v14).
  * Checkpoints to config.paths.checkpoint_dir (defaults to
    checkpoints/drums_v14_ensemble_ft) — NEVER the stock checkpoints/drums_v14.
  * Optional W&B offline logging (opt-in via WANDB_MODE), same pattern as
    scripts/train_onset_classifier.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.drums_v13 import TwoStageDrumsCRNN, TwoStageLoss  # noqa: E402
from src.models.drums_v14_dataset import DrumsV14FullSongDataset  # noqa: E402

# ── Optional Weights & Biases (opt-in via WANDB_MODE; mirrors train_onset_classifier) ──
_WANDB = None


def _maybe_init_wandb(config) -> None:
    global _WANDB
    if not os.environ.get("WANDB_MODE"):
        return
    try:
        import wandb  # noqa: PLC0415

        cfg = OmegaConf.to_container(config, resolve=True)
        wandb_cfg = config.get("wandb", {}) if hasattr(config, "get") else {}
        _WANDB = wandb.init(
            project=os.environ.get("WANDB_PROJECT")
            or wandb_cfg.get("project", "strum-v14-frontensemble"),
            name=os.environ.get("WANDB_RUN_NAME") or Path(sys.argv[1]).stem,
            config=cfg,
            tags=list(wandb_cfg.get("tags", []) or []),
        )
        print(f"  W&B logging enabled (mode={os.environ['WANDB_MODE']})")
    except Exception as e:  # pragma: no cover - defensive
        _WANDB = None
        print(f"  W&B logging unavailable ({e}); continuing without it")


def _wandb_log(metrics: dict, step: int | None = None) -> None:
    if _WANDB is None:
        return
    try:
        _WANDB.log(metrics, step=step)
    except Exception:
        pass


def _wandb_finish() -> None:
    global _WANDB
    if _WANDB is None:
        return
    try:
        _WANDB.finish()
    except Exception:
        pass
    _WANDB = None


class _EmptyDataset:
    """Stand-in for an absent split. DrumsV14FullSongDataset.__init__ raises on
    an empty song list (durs.max() on a zero-size array), so we avoid building
    it at all when a split has no drums songs (e.g. no E-GMD replay merged in).
    """

    songs: list = []

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx):  # pragma: no cover - never iterated
        raise IndexError


def split_has_songs(manifest_path: Path, split: str, source_filter: set | None = None) -> bool:
    with open(manifest_path) as f:
        m = json.load(f)
    for s in m["songs"]:
        if s.get("split") != split or not s.get("charts", {}).get("drums"):
            continue
        if source_filter and s.get("source_game") not in source_filter:
            continue
        return True
    return False


# ── Device ──
def pick_device() -> torch.device:
    env = os.environ.get("STRUM_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Onset frame-F1 (peak-picked, ±tol frames) ──
def _match_frames(pred: list[int], gt: list[int], tol: int) -> tuple[int, int, int]:
    """Greedy nearest-neighbour matching within ±tol frames."""
    pred = sorted(int(p) for p in pred)
    gt = sorted(int(g) for g in gt)
    used = [False] * len(pred)
    tp = 0
    for g in gt:
        best, best_d = -1, tol + 1
        for k, p in enumerate(pred):
            if used[k]:
                continue
            d = abs(p - g)
            if d <= tol and d < best_d:
                best, best_d = k, d
        if best >= 0:
            used[best] = True
            tp += 1
    fn = len(gt) - tp
    fp = len(pred) - tp
    return tp, fp, fn


@torch.no_grad()
def evaluate_frame_f1(
    model: TwoStageDrumsCRNN,
    dataset: DrumsV14FullSongDataset,
    device: torch.device,
    threshold: float,
    min_dist_frames: int,
    tol_frames: int,
    max_songs: int | None = None,
    chunk_frames: int = 0,
) -> dict | None:
    if len(dataset) == 0:
        return None
    was_training = model.training
    model.eval()
    tp = fp = fn = 0
    n = len(dataset) if max_songs is None else min(len(dataset), max_songs)
    for i in range(n):
        mel, targets = dataset[i]
        # Fixed-shape chunked forward: MPS compiles and caches a kernel per
        # unique tensor shape, so variable full-song lengths bloat the shared
        # pool across epochs ("other allocations" OOM, 2026-07-03 run). Padding
        # every chunk to exactly chunk_frames bounds the shape set to one.
        T = mel.shape[-1]
        starts = list(range(0, T, chunk_frames)) if 0 < chunk_frames < T else [0]
        parts = []
        for s0 in starts:
            s1 = min(s0 + chunk_frames, T) if chunk_frames > 0 else T
            mel_c = mel[..., s0:s1]
            if chunk_frames > 0 and mel_c.shape[-1] < chunk_frames:
                pad = chunk_frames - mel_c.shape[-1]
                mel_c = F.pad(mel_c, (0, pad), value=float(mel_c.min()))
            out = model(mel_c.unsqueeze(0).to(device))
            p = out["onset_probs"].squeeze(0).squeeze(-1).float().cpu()
            parts.append(p[: s1 - s0])
        probs = torch.cat(parts).numpy()
        if device.type == "mps":
            torch.mps.empty_cache()
        peaks, _ = find_peaks(probs, height=threshold, distance=min_dist_frames)
        gt = np.nonzero(targets["onset"].squeeze(-1).numpy() > 0.5)[0].tolist()
        s_tp, s_fp, s_fn = _match_frames(peaks.tolist(), gt, tol_frames)
        tp += s_tp
        fp += s_fp
        fn += s_fn
    if was_training:
        model.train()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "f1": f1 * 100.0,
        "precision": prec * 100.0,
        "recall": rec * 100.0,
        "tp": tp, "fp": fp, "fn": fn, "n_songs": n,
    }


# ── Sampler ──
def build_train_sampler(dataset: DrumsV14FullSongDataset, synthetic_frac: float):
    sources = [s.get("source_game", "synthetic_ensemble") for s in dataset.songs]
    counts = Counter(sources)
    replay = {src for src in counts if src != "synthetic_ensemble"}
    if not replay or "synthetic_ensemble" not in counts:
        return None  # single-source -> plain shuffle
    n_replay_sources = len(replay)
    weights = []
    for src in sources:
        if src == "synthetic_ensemble":
            target = synthetic_frac
        else:
            target = (1.0 - synthetic_frac) / n_replay_sources
        weights.append(target / counts[src])
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def lr_at_epoch(epoch: int, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    if warmup > 0 and epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    denom = max(1, total - warmup)
    prog = min(1.0, (epoch - warmup) / denom)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


def save_checkpoint(path: Path, epoch, model, optimizer, val_loss, best_f1, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_f1": best_f1,
            "config": OmegaConf.to_container(config, resolve=True),
            "architecture": "TwoStageDrumsCRNN",
            "version": "v14_ensemble_ft",
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


def build_model(mcfg) -> TwoStageDrumsCRNN:
    return TwoStageDrumsCRNN(
        n_mels=mcfg.n_mels,
        conv_channels=list(mcfg.conv_channels),
        freq_subbands=list(mcfg.freq_subbands),
        subband_proj_dim=mcfg.subband_proj_dim,
        lstm_hidden=mcfg.lstm_hidden,
        lstm_layers=mcfg.lstm_layers,
        attention_heads=mcfg.attention_heads,
        attention_type=mcfg.attention_type,
        attention_window=mcfg.attention_window,
        dropout=mcfg.dropout,
        onset_detector_hidden=mcfg.onset_detector_hidden,
        classifier_hidden=mcfg.classifier_hidden,
        num_classes=mcfg.num_classes,
        predict_velocity=mcfg.predict_velocity,
    )


def warm_start(model: TwoStageDrumsCRNN, warm_path: str, device) -> None:
    print(f"\nWarm-start: {warm_path}")
    if not Path(warm_path).exists():
        print(f"  WARN: warm-start checkpoint not found — training from random init.")
        return
    ckpt = torch.load(warm_path, map_location="cpu", weights_only=False)
    # Print the embedded config so the fine-tune mirrors what's relevant.
    emb = ckpt.get("config", {})
    print(f"  embedded: epoch={ckpt.get('epoch')} best_f1={ckpt.get('best_f1')} "
          f"version={ckpt.get('version')} arch={ckpt.get('architecture')}")
    if isinstance(emb, dict):
        for section in ("training", "loss"):
            if section in emb:
                print(f"  embedded config.{section}: {emb[section]}")
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    print(f"  loaded state_dict (strict=False): "
          f"{len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing:
        print(f"    missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"    unexpected (first 5): {unexpected[:5]}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path, help="fine-tune YAML (e.g. configs/drums_v14_finetune.yaml)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--accumulation-steps", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--synthetic-frac", type=float, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--onset-threshold", type=float, default=None)
    p.add_argument("--max-train-songs", type=int, default=None)
    p.add_argument("--max-val-songs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = OmegaConf.load(args.config)

    # CLI overrides
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.accumulation_steps is not None:
        config.training.accumulation_steps = args.accumulation_steps
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.synthetic_frac is not None:
        config.mixing.synthetic_frac = args.synthetic_frac
    if args.checkpoint_dir is not None:
        config.paths.checkpoint_dir = str(args.checkpoint_dir)
    if args.onset_threshold is not None:
        config.eval.onset_threshold = args.onset_threshold

    device = pick_device()
    print("=" * 72)
    print("Train Drums V14 (two-stage onset detector) — WS5c fine-tune")
    print("=" * 72)
    print(f"Config : {args.config}")
    print(f"Device : {device}")

    ckpt_dir = Path(config.paths.checkpoint_dir)
    stock = Path("checkpoints/drums_v14")
    if ckpt_dir.resolve() == stock.resolve():
        print("FATAL: checkpoint_dir must NOT be the stock checkpoints/drums_v14.",
              file=sys.stderr)
        return 2
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)

    _maybe_init_wandb(config)

    manifest_path = Path(config.paths.data_dir) / "manifest.json"
    mcfg = config.model
    aug_cfg = OmegaConf.to_container(config.get("augmentation", {}), resolve=True) or {}
    aug_enabled = bool(aug_cfg.get("enabled", False))
    max_dur = float(config.training.get("max_song_duration_sec", 600.0))
    chunk_sec = float(config.training.get("chunk_seconds", 0.0))
    chunk_frames = (
        int(chunk_sec * mcfg.sample_rate / mcfg.hop_length) if chunk_sec > 0 else 0
    )

    common = dict(
        manifest_path=manifest_path,
        sample_rate=mcfg.sample_rate,
        n_mels=mcfg.n_mels,
        hop_length=mcfg.hop_length,
        n_fft=mcfg.n_fft,
        max_duration_sec=max_dur,
    )
    train_ds = DrumsV14FullSongDataset(
        split="train", augment=aug_enabled, augment_config=aug_cfg, **common
    )
    synth_val_ds = (
        DrumsV14FullSongDataset(split="val", augment=False,
                                source_filter={"synthetic_ensemble"}, **common)
        if split_has_songs(manifest_path, "val", {"synthetic_ensemble"})
        else _EmptyDataset()
    )
    egmd_val_ds = (
        DrumsV14FullSongDataset(split="egmd_val", augment=False, **common)
        if split_has_songs(manifest_path, "egmd_val")
        else _EmptyDataset()
    )
    print(f"\nDatasets: train={len(train_ds)}  synth_val={len(synth_val_ds)}  "
          f"egmd_val={len(egmd_val_ds)}")
    src_counts = Counter(s.get("source_game", "synthetic_ensemble") for s in train_ds.songs)
    print(f"  train sources: {dict(src_counts)}")
    if len(train_ds) == 0:
        print("FATAL: empty training set.", file=sys.stderr)
        return 2

    # Model + warm-start
    model = build_model(mcfg).to(device)
    warm_start(model, config.training.warm_start, device)

    # Loss
    lcfg = config.loss
    loss_fn = TwoStageLoss(
        onset_pos_weight=lcfg.onset_pos_weight,
        onset_focal_gamma=lcfg.onset_focal_gamma,
        classify_class_weight=list(lcfg.classify_class_weight),
        classify_focal_gamma=lcfg.classify_focal_gamma,
        classify_label_smoothing=lcfg.classify_label_smoothing,
        onset_loss_weight=lcfg.onset_loss_weight,
        classify_loss_weight=lcfg.classify_loss_weight,
        velocity_loss_weight=lcfg.velocity_loss_weight,
    )

    base_lr = float(config.training.learning_rate)
    min_lr = float(config.training.get("min_lr", base_lr))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr,
        weight_decay=float(config.training.get("weight_decay", 0.0)),
    )

    sampler = build_train_sampler(train_ds, float(config.mixing.synthetic_frac))
    num_workers = int(config.training.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds, batch_size=1, sampler=sampler, shuffle=(sampler is None),
        num_workers=num_workers, pin_memory=bool(config.training.get("pin_memory", False)),
    )
    print(f"  sampler: {'WeightedRandomSampler (mix)' if sampler else 'shuffle (single-source)'}"
          f"  workers={num_workers}")

    # Eval params
    ecfg = config.eval
    threshold = float(ecfg.onset_threshold)
    min_dist_frames = max(1, int(float(ecfg.min_distance_ms) / 1000.0 * mcfg.sample_rate / mcfg.hop_length))
    tol_frames = int(ecfg.frame_tolerance)

    epochs = int(config.training.epochs)
    accum = int(config.training.accumulation_steps)
    grad_clip = float(config.training.get("gradient_clip", 0.0))
    warmup = int(config.training.get("warmup_epochs", 0))
    eval_interval = int(config.training.get("f1_eval_interval", 1))
    patience = int(config.training.get("early_stop_patience", 999))
    guard_drop = float(config.training.get("egmd_guard_drop", 2.0))
    trip_limit = int(config.training.get("egmd_guard_trip_limit", 2))

    # Epoch-0 warm-start baselines (BEFORE any update)
    print("\nComputing epoch-0 warm-start baselines...")
    egmd_baseline = None
    if len(egmd_val_ds) > 0:
        r0 = evaluate_frame_f1(model, egmd_val_ds, device, threshold, min_dist_frames,
                               tol_frames, args.max_val_songs, chunk_frames)
        egmd_baseline = r0["f1"]
        print(f"  E-GMD-val (anti-forgetting) baseline F1 = {egmd_baseline:.2f} "
              f"(guard: abort if drops > {guard_drop} for {trip_limit} evals in a row)")
    else:
        print("  No egmd_val split present -> anti-forgetting guard disabled.")
    if len(synth_val_ds) > 0:
        s0 = evaluate_frame_f1(model, synth_val_ds, device, threshold, min_dist_frames,
                               tol_frames, args.max_val_songs, chunk_frames)
        print(f"  synthetic-val warm-start F1 = {s0['f1']:.2f}")
    _wandb_log({"baseline/egmd_val_f1": egmd_baseline or 0.0})

    best_f1 = -1.0
    consec_trips = 0
    patience_ctr = 0
    global_step = 0

    for epoch in range(epochs):
        lr = lr_at_epoch(epoch, warmup, epochs, base_lr, min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        model.train()
        optimizer.zero_grad()
        t0 = time.time()
        running = 0.0
        comp = defaultdict(float)
        nb = 0
        nsongs = 0
        for i, (mel, targets) in enumerate(train_loader):
            onset_full = targets["onset"]
            class_full = targets["classes"]
            vel_full = targets["velocities"]

            # Backward through a full song's attention/LSTM graph exceeds MPS
            # memory (2026-07-02 kernel panic); train on <=chunk_frames windows
            # instead, moving only the active chunk to the device. Each chunk
            # counts as one accumulation unit.
            T = mel.shape[-1]
            starts = list(range(0, T, chunk_frames)) if 0 < chunk_frames < T else [0]
            for s0 in starts:
                s1 = min(s0 + chunk_frames, T) if chunk_frames > 0 else T
                if s0 != 0 and (s1 - s0) < 256:
                    continue  # degenerate <3 s tail; the rest of the song trained
                mel_c = mel[..., s0:s1]
                onset_c = onset_full[:, s0:s1]
                class_c = class_full[:, s0:s1]
                vel_c = vel_full[:, s0:s1]
                # Pad every chunk to exactly chunk_frames: MPS caches a
                # compiled kernel per unique shape, and variable tail lengths
                # bloat the shared pool across epochs (2026-07-03 OOM at the
                # epoch-2 eval). Padding is silence (mel min) with all-zero
                # targets — the model just learns "no onset in silence" there.
                if chunk_frames > 0 and mel_c.shape[-1] < chunk_frames:
                    pad = chunk_frames - mel_c.shape[-1]
                    mel_c = F.pad(mel_c, (0, pad), value=float(mel_c.min()))
                    onset_c = F.pad(onset_c, (0, 0, 0, pad))
                    class_c = F.pad(class_c, (0, 0, 0, pad))
                    vel_c = F.pad(vel_c, (0, 0, 0, pad))
                mel_c = mel_c.to(device)
                onset_t = onset_c.to(device)
                class_t = class_c.to(device)
                vel_t = vel_c.to(device)

                out = model(mel_c)
                loss, ld = loss_fn(out, onset_t, class_t, vel_t)
                (loss / accum).backward()

                running += loss.item()
                comp["onset"] += ld.get("onset_det", 0.0)
                comp["classify"] += ld.get("classify", 0.0)
                comp["velocity"] += ld.get("velocity", 0.0)
                nb += 1

                if nb % accum == 0:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    _wandb_log({"train/loss": loss.item(), "lr": lr}, step=global_step)

            nsongs += 1
            if args.max_train_songs and nsongs >= args.max_train_songs:
                break

        # Flush trailing partial accumulation window
        if nb % accum != 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        if device.type == "mps":
            torch.mps.empty_cache()

        train_loss = running / max(1, nb)
        dt = time.time() - t0
        print(f"\n[epoch {epoch + 1}/{epochs}] lr={lr:.2e} songs={nsongs} chunks={nb} "
              f"train_loss={train_loss:.4f} "
              f"(onset={comp['onset'] / max(1, nb):.4f} "
              f"classify={comp['classify'] / max(1, nb):.4f} "
              f"vel={comp['velocity'] / max(1, nb):.4f}) "
              f"{dt:.1f}s ({dt / max(1, nsongs):.1f}s/song)")
        _wandb_log({"epoch": epoch + 1, "train/epoch_loss": train_loss}, step=global_step)

        if (epoch % eval_interval != 0) and (epoch != epochs - 1):
            continue

        synth = evaluate_frame_f1(model, synth_val_ds, device, threshold,
                                  min_dist_frames, tol_frames, args.max_val_songs,
                                  chunk_frames)
        egmd = (evaluate_frame_f1(model, egmd_val_ds, device, threshold,
                                  min_dist_frames, tol_frames, args.max_val_songs,
                                  chunk_frames)
                if egmd_baseline is not None else None)
        synth_f1 = synth["f1"] if synth else 0.0
        msg = f"  eval: synth_val F1={synth_f1:.2f}"
        if synth:
            msg += f" (P={synth['precision']:.1f} R={synth['recall']:.1f})"
        if egmd is not None:
            msg += f" | egmd_val F1={egmd['f1']:.2f} (baseline {egmd_baseline:.2f})"
        print(msg)
        _wandb_log({
            "val/synth_f1": synth_f1,
            "val/egmd_f1": (egmd["f1"] if egmd else 0.0),
        }, step=global_step)

        # Save last-epoch checkpoint every eval
        save_checkpoint(ckpt_dir / "last.pt", epoch + 1, model, optimizer,
                        train_loss, best_f1, config)

        # Anti-forgetting tripwire
        if egmd is not None and egmd["f1"] < egmd_baseline - guard_drop:
            consec_trips += 1
            print(f"  ANTI-FORGETTING GUARD tripped ({consec_trips}/{trip_limit}): "
                  f"egmd_val F1 {egmd['f1']:.2f} < baseline {egmd_baseline:.2f} - {guard_drop}")
            if consec_trips >= trip_limit:
                print(f"  ABORT: E-GMD forgetting guard tripped {trip_limit}x consecutively. "
                      f"Best synth F1 so far = {best_f1:.2f}. Stopping.")
                _wandb_finish()
                return 4
        else:
            consec_trips = 0

        # Early stopping on primary metric
        if synth_f1 > best_f1:
            best_f1 = synth_f1
            save_checkpoint(ckpt_dir / "best.pt", epoch + 1, model, optimizer,
                            train_loss, best_f1, config)
            print(f"  -> new best synth_val F1 {best_f1:.2f}; saved {ckpt_dir / 'best.pt'}")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stop: no synth_val F1 improvement in {patience} evals "
                      f"(best {best_f1:.2f}).")
                break

    print(f"\nDone. Best synthetic-val F1 = {best_f1:.2f}. "
          f"Checkpoints in {ckpt_dir}")
    _wandb_finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
