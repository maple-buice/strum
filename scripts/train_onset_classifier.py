#!/usr/bin/env python3
"""
Train the onset window classifier for 8-class drum hit classification.

This replaces V13/V14's Stage 2 classifier with a purpose-built model
trained on class-balanced onset windows. At inference time, V13's onset
detector finds the onset times, then this classifier identifies the drum.

Key features:
  - Class-balanced sampling (every class ~12.5% of each batch)
  - Dual-resolution mel spectrograms (transient + tonal)
  - Hard negative mining after first evaluation
  - Onset context encoding (surrounding hit pattern)
  - Fast: small windows + big batches = minutes per epoch
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from omegaconf import OmegaConf

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.onset_classifier import OnsetClassifier
from src.models.onset_classifier_cached_dataset import (
    CachedOnsetDataset,
    CLASS_NAMES,
)


# ── Optional Weights & Biases logging (fully opt-in, zero-effect by default) ──
# Active only when BOTH: (a) the `wandb` package imports, and (b) the WANDB_MODE
# environment variable is set (e.g. WANDB_MODE=offline). Any failure is swallowed
# so training behaviour is byte-for-byte identical when W&B is absent/unset.
_WANDB = None  # module-level run handle; None means "disabled"


def _maybe_init_wandb(config) -> None:
    """Start a W&B run iff importable AND WANDB_MODE is set. Never raises."""
    global _WANDB
    if not os.environ.get("WANDB_MODE"):
        return
    try:
        import wandb  # noqa: PLC0415
        cfg = OmegaConf.to_container(config, resolve=True) if config is not None else None
        _WANDB = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "strum-onset-classifier"),
            name=os.environ.get("WANDB_RUN_NAME") or Path(sys.argv[-1]).stem,
            config=cfg,
        )
        print(f"  W&B logging enabled (mode={os.environ['WANDB_MODE']})")
    except Exception as e:  # pragma: no cover - defensive, keeps training running
        _WANDB = None
        print(f"  W&B logging unavailable ({e}); continuing without it")


def _wandb_log(metrics: dict, step: int | None = None) -> None:
    """Log a metrics dict to W&B if active. Never raises."""
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


def onset_collate_fn(batch):
    """Custom collate that handles optional mel_lowfreq and crash_flux."""
    # batch is list of (mel_fine, mel_coarse, context, label, mel_lowfreq_or_None, crash_flux_or_None)
    n_elem = len(batch[0])
    core = [(b[0], b[1], b[2], b[3]) for b in batch]
    collated = list(default_collate(core))

    # 5th element: mel_lowfreq (optional)
    if n_elem > 4 and batch[0][4] is not None:
        collated.append(default_collate([b[4] for b in batch]))
    else:
        collated.append(None)

    # 6th element: crash_flux (optional)
    if n_elem > 5 and batch[0][5] is not None:
        collated.append(default_collate([b[5] for b in batch]))
    else:
        collated.append(None)

    return tuple(collated)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float | list[float] = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Focal Binary Cross Entropy loss with per-class weights.

    Focal loss: -alpha * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples, focuses on hard confusions.
    gamma=0 is standard BCE, gamma=2 is the paper default.

    gamma can be a scalar (applied to all classes) or a list of per-class
    values (e.g., [0, 0, 0, 2, 0, 2, 0, 2] to apply focal only to toms).
    """
    # Optional label smoothing
    if label_smoothing > 0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

    # Standard BCE components
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction='none'
    )

    # p_t = probability of correct class
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)

    # Focal modulation: (1 - p_t)^gamma — supports per-class gamma
    if isinstance(gamma, (list, tuple)):
        gamma_t = torch.tensor(gamma, device=logits.device, dtype=logits.dtype)
        focal_weight = (1 - p_t) ** gamma_t.unsqueeze(0)
    else:
        focal_weight = (1 - p_t) ** gamma

    # Apply class weights and focal weight
    loss = focal_weight * bce * class_weights.unsqueeze(0)
    return loss.mean()


def asymmetric_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma_neg: float | list[float] = 4.0,
    gamma_pos: float | list[float] = 1.0,
    clip: float = 0.05,
) -> torch.Tensor:
    """
    Asymmetric Loss for multi-label classification (Ridnik et al., 2021).

    Addresses extreme class imbalance by treating positive and negative
    labels differently:
      - Positive: L+ = (1-p)^gamma_pos * (-log(p))
      - Negative: L- = p_m^gamma_neg * (-log(1-p_m))
        where p_m = max(p - clip, 0) shifts probabilities before focusing

    For rare classes (HighTom at 0.5%), easy negatives dominate gradient.
    High gamma_neg (4) aggressively suppresses their contribution.
    Low gamma_pos (0-1) preserves gradient for hard positives.
    Probability clipping (clip>0) prevents very small negative probs from
    getting any gradient at all — a hard threshold on "easy enough".

    gamma_neg/gamma_pos can be per-class lists for fine-grained control.
    """
    probs = torch.sigmoid(logits)
    eps = 1e-8

    # Probability clipping for negatives: shift probabilities down
    # This makes low-confidence negatives contribute zero gradient
    probs_neg = (probs - clip).clamp(min=0)

    # Positive loss: -targets * (1-p)^gamma_pos * log(p)
    log_pos = torch.log(probs.clamp(min=eps))
    if isinstance(gamma_pos, (list, tuple)):
        gp = torch.tensor(gamma_pos, device=logits.device, dtype=logits.dtype)
        loss_pos = -targets * (1 - probs) ** gp.unsqueeze(0) * log_pos
    else:
        loss_pos = -targets * (1 - probs) ** gamma_pos * log_pos

    # Negative loss: -(1-targets) * p_m^gamma_neg * log(1-p_m)
    log_neg = torch.log((1 - probs_neg).clamp(min=eps))
    if isinstance(gamma_neg, (list, tuple)):
        gn = torch.tensor(gamma_neg, device=logits.device, dtype=logits.dtype)
        loss_neg = -(1 - targets) * probs_neg ** gn.unsqueeze(0) * log_neg
    else:
        loss_neg = -(1 - targets) * probs_neg ** gamma_neg * log_neg

    loss = (loss_pos + loss_neg) * class_weights.unsqueeze(0)
    return loss.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss for multi-label drum classification.

    Pulls together embeddings of samples sharing at least one class label
    and pushes apart embeddings of samples with no shared labels.
    Critical for disentangling rare-class embeddings (HighTom, LowTom,
    FloorTom) from dominant-class embeddings (Kick, Snare).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) L2-normalized projection embeddings
            labels: (B, C) binary multi-hot class labels (pre-mixup)
        Returns:
            Scalar contrastive loss
        """
        B = features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Cosine similarity matrix (features are already L2-normalized)
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Positive mask: samples sharing at least one class
        mask = (torch.matmul(labels, labels.T) > 0).float()
        self_mask = torch.eye(B, device=features.device)
        mask = mask * (1 - self_mask)  # Remove self-pairs

        # Log-softmax with numerical stability
        logits_max = sim.max(dim=1, keepdim=True).values.detach()
        logits = sim - logits_max
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-8))

        # Mean log-prob over positive pairs
        num_pos = mask.sum(dim=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / num_pos.clamp(min=1)

        # Only count samples that have at least one positive pair
        valid = (num_pos > 0).float()
        loss = -(valid * mean_log_prob).sum() / valid.sum().clamp(min=1)
        return loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train onset window classifier")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def build_model(config) -> OnsetClassifier:
    """Build the onset classifier model from config."""
    model_cfg = config.model
    return OnsetClassifier(
        num_classes=model_cfg.num_classes,
        branch_channels=list(model_cfg.branch_channels),
        context_size=model_cfg.context_size,
        context_hidden=model_cfg.context_hidden,
        classifier_hidden=model_cfg.classifier_hidden,
        spectral_dim=model_cfg.get("spectral_dim", 32),
        dropout=model_cfg.dropout,
        use_freq_attn=model_cfg.get("use_freq_attn", True),
        use_hpss=model_cfg.get("use_hpss", False),
        enhanced_spectral=model_cfg.get("enhanced_spectral", False),
        use_contrastive=model_cfg.get("use_contrastive", False),
        use_aux_head=model_cfg.get("use_aux_head", False),
        projection_dim=model_cfg.get("projection_dim", 128),
        use_dual_head=model_cfg.get("use_dual_head", False),
        tom_head_hidden=model_cfg.get("tom_head_hidden", 256),
        use_lowfreq_branch=model_cfg.get("use_lowfreq_branch", False),
        use_lowfreq_spectral=model_cfg.get("use_lowfreq_spectral", False),
        use_crash_flux=model_cfg.get("use_crash_flux", False),
        crash_flux_dim=model_cfg.get("crash_flux_dim", 32),
    )


def evaluate_streaming(
    model: OnsetClassifier,
    cache_dir: str,
    device: torch.device,
    batch_size: int = 256,
    max_samples: int = 0,
    chunk_size: int = 4096,
) -> dict:
    """
    Evaluate on test set by streaming chunks from disk — NO mmap.

    Reads numpy files in small slices (chunk_size onsets at a time),
    never memory-mapping the full test arrays. This avoids page cache
    pressure that causes CUDA unified memory deadlocks.
    """
    model.eval()
    num_classes = 8
    cache_path = Path(cache_dir)

    with open(cache_path / "test_index.json") as f:
        index = json.load(f)

    total = index["total_onsets"]
    if max_samples:
        total = min(total, max_samples)

    # Open files as mmap READ-ONLY just for slicing — we'll copy each chunk
    mel_fine_mm = np.load(str(cache_path / index["files"]["mel_fine"]), mmap_mode='r')
    mel_coarse_mm = np.load(str(cache_path / index["files"]["mel_coarse"]), mmap_mode='r')
    contexts_mm = np.load(str(cache_path / index["files"]["contexts"]), mmap_mode='r')
    labels_mm = np.load(str(cache_path / index["files"]["labels"]), mmap_mode='r')

    # CQT or lowfreq
    cqt_file = index["files"].get("cqt")
    lowfreq_file = index["files"].get("mel_lowfreq")
    lowfreq_mm = None
    if cqt_file and (cache_path / cqt_file).exists():
        lowfreq_mm = np.load(str(cache_path / cqt_file), mmap_mode='r')
    elif lowfreq_file and (cache_path / lowfreq_file).exists():
        lowfreq_mm = np.load(str(cache_path / lowfreq_file), mmap_mode='r')

    # Crash flux (V19+)
    crash_flux_file = index["files"].get("crash_flux")
    crash_flux_mm = None
    if crash_flux_file and (cache_path / crash_flux_file).exists():
        crash_flux_mm = np.load(str(cache_path / crash_flux_file), mmap_mode='r')

    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    num_samples = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            # Read chunk into RAM (copy from mmap, then process)
            mf = mel_fine_mm[start:end].astype(np.float32)
            mc = mel_coarse_mm[start:end].astype(np.float32)
            ctx = contexts_mm[start:end].astype(np.float32)
            lbl = labels_mm[start:end].astype(np.float32)
            lf = lowfreq_mm[start:end].astype(np.float32) if lowfreq_mm is not None else None
            cf = crash_flux_mm[start:end].astype(np.float32) if crash_flux_mm is not None else None

            # Process in mini-batches within the chunk
            for b_start in range(0, len(mf), batch_size):
                b_end = min(b_start + batch_size, len(mf))

                t_fine = torch.from_numpy(mf[b_start:b_end]).unsqueeze(1).to(device)
                t_coarse = torch.from_numpy(mc[b_start:b_end]).unsqueeze(1).to(device)
                t_ctx = torch.from_numpy(ctx[b_start:b_end]).to(device)
                t_lbl = torch.from_numpy(lbl[b_start:b_end]).to(device)
                t_lf = None
                if lf is not None:
                    t_lf = torch.from_numpy(lf[b_start:b_end]).unsqueeze(1).to(device)
                t_cf = None
                if cf is not None:
                    t_cf = torch.from_numpy(cf[b_start:b_end]).to(device)

                logits = model(t_fine, t_coarse, t_ctx, mel_lowfreq=t_lf, crash_flux=t_cf)
                loss = criterion(logits, t_lbl)
                total_loss += loss.item() * t_lbl.shape[0]
                num_samples += t_lbl.shape[0]

                probs = torch.sigmoid(logits).cpu().numpy()
                targets = t_lbl.cpu().numpy()
                preds = (probs > 0.5).astype(np.float32)

                for i in range(targets.shape[0]):
                    true_classes = set(np.where(targets[i] > 0.5)[0])
                    pred_classes = set(np.where(preds[i] > 0.5)[0])
                    if not pred_classes and probs[i].max() > 0.1:
                        pred_classes = {int(probs[i].argmax())}

                    for c in range(num_classes):
                        if c in true_classes and c in pred_classes:
                            tp[c] += 1
                        elif c in pred_classes and c not in true_classes:
                            fp[c] += 1
                        elif c in true_classes and c not in pred_classes:
                            fn[c] += 1

                    if true_classes and pred_classes:
                        for tc in true_classes:
                            for pc in pred_classes:
                                confusion[tc][pc] += 1

            # Explicitly free chunk arrays to release memory
            del mf, mc, ctx, lbl, lf, cf

    # Close mmaps
    del mel_fine_mm, mel_coarse_mm, contexts_mm, labels_mm, lowfreq_mm, crash_flux_mm

    # Per-class metrics
    results = {}
    f1_scores = []
    for c in range(num_classes):
        p = tp[c] / max(tp[c] + fp[c], 1)
        r = tp[c] / max(tp[c] + fn[c], 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        results[CLASS_NAMES[c]] = {"precision": p, "recall": r, "f1": f1}
        f1_scores.append(f1)

    results["overall_f1"] = np.mean(f1_scores)
    results["val_loss"] = total_loss / max(num_samples, 1)
    results["confusion"] = confusion
    results["num_samples"] = num_samples

    # Confusion pairs for hard negative mining
    confusion_pairs = {}
    for true_c in range(num_classes):
        row_total = confusion[true_c].sum()
        if row_total == 0:
            continue
        for pred_c in range(num_classes):
            if pred_c != true_c and confusion[true_c][pred_c] > 0:
                rate = confusion[true_c][pred_c] / row_total
                if rate > 0.05:
                    confusion_pairs[(true_c, pred_c)] = rate

    results["confusion_pairs"] = confusion_pairs
    model.train()
    return results


def print_results(results: dict, epoch: int) -> None:
    """Pretty-print evaluation results."""
    print(f"\n  {'Class':<12} {'F1':>6} {'Prec':>6} {'Recall':>6}")
    print(f"  {'-'*36}")
    for name in CLASS_NAMES:
        m = results[name]
        marker = " ✓" if m["f1"] >= 0.9 else ""
        print(f"  {name:<12} {m['f1']*100:>5.1f}% {m['precision']*100:>5.1f}% "
              f"{m['recall']*100:>5.1f}%{marker}")
    print(f"  {'OVERALL':<12} {results['overall_f1']*100:>5.1f}%")

    # Print top confusion pairs
    pairs = results.get("confusion_pairs", {})
    if pairs:
        print(f"\n  Top confusions:")
        sorted_pairs = sorted(pairs.items(), key=lambda x: -x[1])[:5]
        for (tc, pc), rate in sorted_pairs:
            print(f"    {CLASS_NAMES[tc]} → {CLASS_NAMES[pc]}: {rate*100:.1f}%")


def save_checkpoint(model, optimizer, scheduler, epoch, best_f1, best_val_loss, config, path):
    """Save training checkpoint."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_f1": best_f1,
        "best_val_loss": best_val_loss,
        "config": OmegaConf.to_container(config),
    }, path)


def train(config):
    """Main training loop."""
    print("=" * 70)
    print("ONSET CLASSIFIER TRAINING")
    print("Purpose-built 8-class drum hit classifier")
    print("Class-balanced sampling + dual-resolution mel + onset context")
    print("=" * 70)
    print(f"Config: {sys.argv[-1]}")

    # STRUM_DEVICE override, else cuda > mps > cpu (same policy as
    # batch_infer_hybrid.pick_device). On MPS keep
    # PYTORCH_ENABLE_MPS_FALLBACK=1 set: a few ops fall back to CPU.
    env_device = os.environ.get("STRUM_DEVICE")
    if env_device:
        device = torch.device(env_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: {device} ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        print(f"Device: {device}")

    _maybe_init_wandb(config)

    # Paths
    data_dir = Path(config.paths.data_dir)
    output_dir = Path(config.paths.output_dir)
    checkpoint_dir = Path(config.paths.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / "manifest.json"

    # Dataset (from pre-computed memmap cache)
    print("\nLoading cached datasets...")
    # V3+ can reuse V2 cache via cache_dir config, or fall back to output_dir/cache
    cache_dir = str(config.get("cache_dir", output_dir / "cache"))
    if not Path(cache_dir).exists():
        # Fall back to the main onset_classifier cache
        cache_dir = str(Path("outputs/onset_classifier/cache"))
    print(f"  Cache: {cache_dir}")
    aug_cfg = config.augmentation
    clean_mask_path = config.get("clean_mask_path", None)
    bleed_cache_dir = config.get("bleed_cache_dir", None)
    bleed_prob = config.get("bleed_prob", 0.5)
    # Skip lowfreq branch mmap for models that don't use it (saves ~33 GB)
    skip_lowfreq = not config.model.get("use_lowfreq_branch", False)
    # Skip crash flux mmap for models that don't use it
    skip_crash_flux = not config.model.get("use_crash_flux", False)
    # Cap training onsets to reduce mmap page pressure on unified memory systems
    # (133 GB mmap vs 128.5 GB RAM → deadlock without cap)
    max_train_onsets = config.get("max_train_onsets", 0)
    train_ds = CachedOnsetDataset(
        cache_dir=cache_dir,
        split="train",
        augment=True,
        noise_std=aug_cfg.get("noise_std", 0.005),
        gain_db_range=tuple(aug_cfg.get("gain_db_range", [-2.0, 2.0])),
        spec_augment=aug_cfg.get("spec_augment", False),
        freq_mask_width=aug_cfg.get("freq_mask_width", 12),
        time_mask_width=aug_cfg.get("time_mask_width", 8),
        clean_mask_path=clean_mask_path,
        bleed_cache_dir=bleed_cache_dir,
        bleed_prob=bleed_prob,
        skip_lowfreq=skip_lowfreq,
        max_onsets=max_train_onsets,
        skip_crash_flux=skip_crash_flux,
        primary_class_strategy=config.get("primary_class_strategy", "argmax"),
    )

    # NOTE: val dataset is created on-demand during eval to avoid permanent
    # mmap pressure (test mmaps add ~20 GiB that pushes total over physical RAM)
    print(f"Train: {len(train_ds)} onsets")
    print(f"Val:   loaded on-demand (cache: {cache_dir})")

    # Sampler (class-balanced, tom-boosted, or crash-boosted)
    tcfg = config.training
    max_train = tcfg.get("max_train_batches", 0)
    sampler_num_samples = min(max_train * tcfg.batch_size, len(train_ds)) if max_train else None
    if sampler_num_samples:
        print(f"\nSampler num_samples: {sampler_num_samples} (vs {len(train_ds)} total)")

    crash_boost = config.get("crash_boost_sampling", {})
    tom_boost = config.get("tom_boost_sampling", {})
    if crash_boost.get("enabled", False):
        crash_fraction = crash_boost.get("crash_fraction", 0.3)
        tom_fraction = tom_boost.get("tom_fraction", 0.0) if tom_boost.get("enabled", False) else 0.0
        train_sampler = train_ds.get_crash_boosted_sampler(
            crash_fraction=crash_fraction,
            tom_fraction=tom_fraction,
            num_samples=sampler_num_samples,
        )
        print(f"\nCrash-boosted sampling enabled (crash={crash_fraction}, tom={tom_fraction}):")
    elif tom_boost.get("enabled", False):
        tom_fraction = tom_boost.get("tom_fraction", 0.5)
        train_sampler = train_ds.get_tom_boosted_sampler(tom_fraction=tom_fraction, num_samples=sampler_num_samples)
        print(f"\nTom-boosted sampling enabled (tom_fraction={tom_fraction}):")
    else:
        train_sampler = train_ds.get_sampler(num_samples=sampler_num_samples)
        print("\nClass-balanced sampling enabled:")
    for i, name in enumerate(CLASS_NAMES):
        count = train_ds.class_counts[i]
        pct = 100 * count / len(train_ds)
        print(f"  {name:<12} {count:>8} ({pct:>5.1f}%)")

    nw = tcfg.get("num_workers", 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg.batch_size,
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=(nw > 0),
        drop_last=True,
        collate_fn=onset_collate_fn,
        prefetch_factor=4 if nw > 0 else None,
    )
    # val evaluation uses evaluate_streaming() — no DataLoader needed

    # Drop page cache before model creation — init scanned all labels for _primary_classes
    train_ds.drop_page_cache()
    gc.collect()

    # Model
    print("\nBuilding model...")
    model = build_model(config).to(device)

    # Warm-start from existing checkpoint (V5: load V2 weights, new heads random)
    warm_cfg = config.get("warm_start", {})
    if warm_cfg.get("enabled", False):
        ckpt_path = warm_cfg.checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Filter out keys with size mismatches (e.g., spectral_dim changed)
        src_state = ckpt["model_state_dict"]
        model_state = model.state_dict()
        filtered = {}
        size_skipped = []
        for k, v in src_state.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            elif k in model_state:
                size_skipped.append(k)
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"  Warm-start from {ckpt_path}")
        print(f"  Loaded {len(filtered)} keys")
        if size_skipped:
            print(f"  Size-mismatched (random init): {len(size_skipped)} keys")
        if missing:
            print(f"  New params (randomly initialized): {len(missing)} keys")
        if unexpected:
            print(f"  Unexpected keys (ignored): {len(unexpected)}")

    # V7b: Freeze backbone — only train classifier head (+ aux/projection if present)
    freeze_backbone = config.get("freeze_backbone", False)
    if freeze_backbone:
        trainable_prefixes = (
            "classifier.", "common_classifier.", "tom_classifier.",
            "projection_head.", "aux_head.",
        )
        frozen = 0
        for name, param in model.named_parameters():
            if not name.startswith(trainable_prefixes):
                param.requires_grad = False
                frozen += 1
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Backbone frozen: {frozen} params frozen, "
              f"{trainable/1e6:.2f}M/{total/1e6:.2f}M trainable")

    # Loss: BCE with per-class weighting to further help rare classes
    class_weights = torch.tensor(
        list(config.loss.class_weights), dtype=torch.float32, device=device
    )

    # V5: Supervised contrastive loss
    use_contrastive = config.model.get("use_contrastive", False)
    use_aux_head = config.model.get("use_aux_head", False)
    supcon_criterion = None
    lambda_supcon = 0.0
    lambda_aux = 0.0
    # Tom class indices for auxiliary head
    TOM_INDICES = [3, 5, 7]  # HighTom, LowTom, FloorTom

    if use_contrastive:
        supcon_temp = config.loss.get("supcon_temperature", 0.07)
        lambda_supcon = config.loss.get("lambda_supcon", 0.5)
        supcon_criterion = SupConLoss(temperature=supcon_temp)
        print(f"  SupCon loss: temp={supcon_temp}, lambda={lambda_supcon}")

    if use_aux_head:
        lambda_aux = config.loss.get("lambda_aux", 0.3)
        print(f"  Aux 'is tom?' head: lambda={lambda_aux}")

    # V7: train_classes — restrict loss to specific class indices only
    train_classes = config.loss.get("train_classes", None)
    if train_classes is not None:
        train_classes = list(train_classes)
        # Create class mask (1 for train classes, 0 for others)
        class_mask = torch.zeros(8, device=device)
        class_mask[train_classes] = 1.0
        # Zero out weights for non-train classes
        class_weights = class_weights * class_mask
        names = [CLASS_NAMES[i] for i in train_classes]
        print(f"  Train-classes mode: loss only on {names}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
    )

    # LR scheduler: cosine decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=tcfg.epochs,
        eta_min=tcfg.min_lr,
    )

    # Training state
    best_f1 = 0.0
    best_val_loss = float("inf")
    confusion_pairs = None  # Updated after first eval
    patience_counter = 0
    start_epoch = 0

    # Resume from checkpoint if configured
    resume_path = tcfg.get("resume_from", None)
    if resume_path:
        resume_path = Path(resume_path)
        if resume_path.exists():
            ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict") and scheduler:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            else:
                # Step scheduler forward to the resumed epoch
                for _ in range(ckpt["epoch"] + 1):
                    scheduler.step()
            start_epoch = ckpt["epoch"] + 1
            best_f1 = ckpt.get("best_f1", 0.0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f"\n  ★ Resumed from {resume_path} (epoch {start_epoch}, best_f1={best_f1*100:.1f}%)")
        else:
            print(f"  Warning: resume_from path not found: {resume_path}")

    print(f"\nTraining plan:")
    print(f"  Epochs: {tcfg.epochs}")
    print(f"  Batch size: {tcfg.batch_size}")
    print(f"  LR: {tcfg.learning_rate} → {tcfg.min_lr}")
    print(f"  Max train batches/epoch: {tcfg.get('max_train_batches', 'all')}")
    print(f"  Eval every: {tcfg.eval_interval} epochs")
    print(f"  Hard negative mining: enabled after first eval")
    print()

    max_val = tcfg.get("max_val_batches", 0)

    for epoch in range(start_epoch, tcfg.epochs):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        num_batches = 0

        mixup_alpha = config.training.get("mixup_alpha", 0.0)
        total_train = min(max_train, len(train_loader)) if max_train else len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tcfg.epochs}", total=total_train)
        for batch_idx, batch in enumerate(pbar):
            if max_train and batch_idx >= max_train:
                break

            mel_fine, mel_coarse, context, labels = batch[0], batch[1], batch[2], batch[3]
            mel_lowfreq = batch[4] if len(batch) > 4 else None
            crash_flux = batch[5] if len(batch) > 5 else None

            mel_fine = mel_fine.to(device)
            mel_coarse = mel_coarse.to(device)
            context = context.to(device)
            labels = labels.to(device)
            if mel_lowfreq is not None:
                mel_lowfreq = mel_lowfreq.to(device)
            if crash_flux is not None:
                crash_flux = crash_flux.to(device)

            # Save original labels for contrastive loss (needs hard labels)
            labels_orig = labels.clone() if use_contrastive else None

            # Mixup augmentation
            if mixup_alpha > 0 and mel_fine.shape[0] > 1:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                lam = max(lam, 1 - lam)  # Ensure lam >= 0.5
                idx = torch.randperm(mel_fine.shape[0], device=device)
                mel_fine = lam * mel_fine + (1 - lam) * mel_fine[idx]
                mel_coarse = lam * mel_coarse + (1 - lam) * mel_coarse[idx]
                if mel_lowfreq is not None:
                    mel_lowfreq = lam * mel_lowfreq + (1 - lam) * mel_lowfreq[idx]
                if crash_flux is not None:
                    crash_flux = lam * crash_flux + (1 - lam) * crash_flux[idx]
                context = lam * context + (1 - lam) * context[idx]
                labels = lam * labels + (1 - lam) * labels[idx]

            # Forward pass — V5 returns extra features for contrastive/aux losses
            need_features = use_contrastive or use_aux_head
            if need_features:
                logits, extras = model(mel_fine, mel_coarse, context, return_features=True, mel_lowfreq=mel_lowfreq, crash_flux=crash_flux)
            else:
                logits = model(mel_fine, mel_coarse, context, mel_lowfreq=mel_lowfreq, crash_flux=crash_flux)

            # Main classification loss (focal BCE)
            # Loss computation: ASL or focal BCE
            use_asl = config.loss.get("use_asl", False)
            if use_asl:
                asl_gamma_neg = config.loss.get("gamma_neg", 4.0)
                asl_gamma_pos = config.loss.get("gamma_pos", 1.0)
                asl_clip = config.loss.get("clip", 0.05)
                if isinstance(asl_gamma_neg, (list, tuple)) or (hasattr(asl_gamma_neg, '__iter__') and not isinstance(asl_gamma_neg, str)):
                    asl_gamma_neg = list(asl_gamma_neg)
                if isinstance(asl_gamma_pos, (list, tuple)) or (hasattr(asl_gamma_pos, '__iter__') and not isinstance(asl_gamma_pos, str)):
                    asl_gamma_pos = list(asl_gamma_pos)
                loss = asymmetric_loss_with_logits(
                    logits, labels, class_weights,
                    gamma_neg=asl_gamma_neg,
                    gamma_pos=asl_gamma_pos,
                    clip=asl_clip,
                )
            else:
                # focal_gamma supports scalar (all classes) or list (per-class)
                focal_gamma = config.loss.get("focal_gamma", 2.0)
                if isinstance(focal_gamma, (list, tuple)):
                    focal_gamma = list(focal_gamma)
                elif hasattr(focal_gamma, '__iter__') and not isinstance(focal_gamma, str):
                    focal_gamma = list(focal_gamma)  # OmegaConf ListConfig → list
                label_smooth = config.loss.get("label_smoothing", 0.0)
                loss = focal_bce_with_logits(
                    logits, labels, class_weights,
                    gamma=focal_gamma,
                    label_smoothing=label_smooth,
                )

            # V5: Supervised contrastive loss on projected embeddings
            if use_contrastive and 'projection' in extras:
                supcon_labels = labels_orig if labels_orig is not None else labels
                scl = supcon_criterion(extras['projection'], supcon_labels)
                loss = loss + lambda_supcon * scl

            # V5: Auxiliary "is tom?" binary loss
            if use_aux_head and 'aux_logits' in extras:
                tom_labels = labels[:, TOM_INDICES].max(dim=1).values
                aux_loss = nn.functional.binary_cross_entropy_with_logits(
                    extras['aux_logits'], tom_labels
                )
                loss = loss + lambda_aux * aux_loss

            if math.isnan(loss.item()):
                logger.warning(f"NaN loss at epoch {epoch+1} batch {batch_idx}, skipping")
                optimizer.zero_grad()
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.gradient_clip)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                pbar.set_postfix(loss=f"{running_loss/num_batches:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

            # Periodically release mmap page cache to prevent CUDA memory starvation
            # (community cache is 133 GB vs 128.5 GB unified memory — frequent drops needed)
            if batch_idx % 50 == 49:
                train_ds.drop_page_cache()

        scheduler.step()
        train_loss = running_loss / max(num_batches, 1)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        # Evaluate
        if (epoch + 1) % tcfg.eval_interval == 0 or epoch == 0:
            # Streaming eval: reads test data in chunks, no persistent mmaps
            results = evaluate_streaming(
                model, cache_dir, device,
                batch_size=tcfg.batch_size, max_samples=max_val * tcfg.batch_size if max_val else 0)
            gc.collect()

            val_loss = results["val_loss"]
            overall_f1 = results["overall_f1"]

            print(f"  Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f} "
                  f"F1={overall_f1*100:.1f}% lr={lr:.1e} ({elapsed:.0f}s)")
            print_results(results, epoch + 1)

            _wandb_log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/overall_f1": overall_f1,
                "lr": lr,
                "epoch": epoch + 1,
            }, step=epoch + 1)

            # Memory cleanup before sampler rebuild
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Update hard negative mining (skip if tom-boost is active — it has its own sampling)
            new_pairs = results.get("confusion_pairs", {})
            if new_pairs and not tom_boost.get("enabled", False):
                confusion_pairs = new_pairs
                # Explicitly free old sampler/loader before building new ones
                del train_sampler, train_loader
                gc.collect()
                # Rebuild sampler with hard negative upweighting
                train_sampler = train_ds.get_hard_negative_sampler(
                    confusion_pairs, num_samples=sampler_num_samples)
                train_loader = DataLoader(
                    train_ds,
                    batch_size=tcfg.batch_size,
                    sampler=train_sampler,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=True,
                    collate_fn=onset_collate_fn,
                )
                print(f"  → Hard negative sampler updated: "
                      f"{len(new_pairs)} confusion pairs")

            # Save best
            if overall_f1 > best_f1:
                best_f1 = overall_f1
                save_checkpoint(model, optimizer, scheduler, epoch, best_f1, best_val_loss, config,
                                checkpoint_dir / "best_f1.pt")
                print(f"  ★ New best F1: {best_f1*100:.1f}%")
                patience_counter = 0

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, best_f1, best_val_loss, config,
                                checkpoint_dir / "best_loss.pt")
                print(f"  ★ New best val_loss: {best_val_loss:.4f}")

            # Early stopping
            if overall_f1 <= best_f1:
                patience_counter += 1
            if patience_counter >= tcfg.early_stop_patience:
                print(f"\n  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {tcfg.early_stop_patience} evals)")
                break
        else:
            print(f"  Epoch {epoch+1}: train={train_loss:.4f} "
                  f"lr={lr:.1e} ({elapsed:.0f}s)")
            _wandb_log({"train/loss": train_loss, "lr": lr, "epoch": epoch + 1},
                       step=epoch + 1)

        # Periodic checkpoint
        if (epoch + 1) % tcfg.checkpoint_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_f1, best_val_loss, config,
                            checkpoint_dir / f"epoch_{epoch+1}.pt")

    # Final evaluation
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — FINAL EVALUATION")
    print("=" * 70)

    # Load best checkpoint
    best_path = checkpoint_dir / "best_f1.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best F1 checkpoint (epoch {ckpt['epoch']+1})")

    # Final eval with streaming (no mmap)
    results = evaluate_streaming(model, cache_dir, device, batch_size=tcfg.batch_size)
    gc.collect()
    print_results(results, -1)

    # V13 comparison
    print(f"\n  V13 Baseline (for reference):")
    print(f"  {'Class':<12} {'V13 F1':>8} {'New F1':>8} {'Delta':>8}")
    print(f"  {'-'*42}")
    v13_f1 = {
        'Kick': 96.0, 'Snare': 92.9, 'HiHat': 84.2, 'HighTom': 47.0,
        'Ride': 79.2, 'LowTom': 43.3, 'Crash': 73.1, 'FloorTom': 54.9,
    }
    for name in CLASS_NAMES:
        old = v13_f1[name]
        new = results[name]["f1"] * 100
        delta = new - old
        marker = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"  {name:<12} {old:>7.1f}% {new:>7.1f}% {delta:>+7.1f}% {marker}")
    old_avg = np.mean(list(v13_f1.values()))
    new_avg = results["overall_f1"] * 100
    print(f"  {'OVERALL':<12} {old_avg:>7.1f}% {new_avg:>7.1f}% {new_avg-old_avg:>+7.1f}%")

    # V2 comparison
    v2_f1 = {
        'Kick': 97.1, 'Snare': 95.8, 'HiHat': 92.0, 'HighTom': 52.4,
        'Ride': 84.4, 'LowTom': 63.5, 'Crash': 86.1, 'FloorTom': 86.6,
    }
    print(f"\n  V2 Onset Classifier (for reference):")
    print(f"  {'Class':<12} {'V2 F1':>8} {'New F1':>8} {'Delta':>8}")
    print(f"  {'-'*42}")
    for name in CLASS_NAMES:
        old = v2_f1[name]
        new = results[name]["f1"] * 100
        delta = new - old
        marker = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"  {name:<12} {old:>7.1f}% {new:>7.1f}% {delta:>+7.1f}% {marker}")
    old_avg = np.mean(list(v2_f1.values()))
    print(f"  {'OVERALL':<12} {old_avg:>7.1f}% {new_avg:>7.1f}% {new_avg-old_avg:>+7.1f}%")

    # Save confusion matrix
    cm = results.get("confusion", None)
    if cm is not None:
        np.save(output_dir / "confusion_matrix.npy", cm)
        print(f"\n  Confusion matrix saved to {output_dir / 'confusion_matrix.npy'}")

    # Save training history
    print(f"\n  Best F1: {best_f1*100:.1f}%")
    print(f"  Checkpoint: {best_path}")

    _wandb_log({"final/overall_f1": results["overall_f1"], "final/best_f1": best_f1})
    _wandb_finish()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    train(config)
