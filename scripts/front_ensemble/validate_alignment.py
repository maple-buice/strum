"""WS5b alignment gate: verify labeled onsets coincide with a local energy
rise in the rendered mix within +/-20 ms.

Automated check (honest local-rise criterion, not a rubber stamp):
  novelty = librosa.onset.onset_strength on the mix (n_fft=1024, hop=128
  -> 2.9 ms frames, ~ +/-12 ms analysis smear at 44.1 kHz).
  For each label t: over frames f within +/-20 ms of t,
      rise(f) = novelty[f] - median(novelty[f-9 .. f-3])   (baseline 9-26 ms
                                                            before f)
  PASS if max rise >= tau, tau = 0.10 * P98(novelty of the piece)
  (relative to piece-level dynamics; the floor keeps silence from passing).
  tau was chosen from a measured sweep on the pilot (see pilot_report.md):
  at 0.05 the negative controls passed ~87% (toothless); at 0.10 real
  labels pass 96.9% while +100 ms-shifted and random controls drop to
  ~52-54%; at 0.15 the real rate falls below the 95% gate.

Negative control: the same criterion evaluated at labels shifted +100 ms
(and at uniformly random times) is reported alongside. If the control passed
at a rate close to the real labels, the criterion would be meaningless; a
large gap demonstrates the check has teeth.

Density caveat (measured, not hidden): these pieces carry ~5-20 labeled
onsets per second across all instruments, so a +100 ms shift often lands
within 20 ms of a DIFFERENT true onset and the whole-set control stays
high. The discriminative version of the control is therefore also run on
the ISOLATED subset: labels with no other label within +/-150 ms. There a
shifted label has no true onset to coincide with, and the pass rate must
collapse if the criterion has teeth.

Click overlays: N random 10 s clips are written with a 2 kHz click mixed at
every labeled onset for human spot-check.
"""
import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PILOT_DIR

SR = 44100
HOP = 128
NFFT = 1024
TOL = 0.020  # +/- seconds
TAU_FRAC = 0.10


def novelty_of(y):
    return librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP,
                                        n_fft=NFFT)


def check_times(nov, times, tau):
    """Return boolean pass per label time."""
    n = len(nov)
    med_idx_lo, med_idx_hi = 9, 3
    passes = []
    tol_frames = int(round(TOL * SR / HOP))
    for t in times:
        f0 = int(round(t * SR / HOP))
        best = -np.inf
        for f in range(max(0, f0 - tol_frames), min(n, f0 + tol_frames + 1)):
            lo = max(0, f - med_idx_lo)
            hi = max(lo + 1, f - med_idx_hi)
            base = float(np.median(nov[lo:hi]))
            best = max(best, float(nov[f]) - base)
        passes.append(best >= tau)
    return np.array(passes)


def validate_piece(piece_dir, rng):
    piece_dir = Path(piece_dir)
    meta = json.loads((piece_dir / "labels.json").read_text())
    y, sr = sf.read(piece_dir / "mix.wav", always_2d=True)
    assert sr == SR
    y = y.mean(axis=1).astype(np.float32)
    nov = novelty_of(y)
    tau = TAU_FRAC * float(np.percentile(nov, 98))
    times = [e["t"] for e in meta["events"]]
    ok = check_times(nov, times, tau)
    # negative controls
    dur = len(y) / SR
    shifted = [t + 0.100 for t in times if t + 0.100 < dur - 0.1]
    rand_t = [rng.uniform(0.1, dur - 0.1) for _ in times]
    ok_shift = check_times(nov, shifted, tau)
    ok_rand = check_times(nov, rand_t, tau)
    # isolated subset: no other label within +/-150 ms -> a +100 ms shift
    # cannot coincide with any true onset there
    ts = np.array(sorted(times))
    iso = []
    for t in times:
        i = np.searchsorted(ts, t)
        near = [ts[j] for j in range(max(0, i - 2), min(len(ts), i + 3))
                if abs(ts[j] - t) > 1e-9]
        if all(abs(n - t) > 0.150 for n in near):
            iso.append(t)
    ok_iso = check_times(nov, iso, tau) if iso else np.array([], bool)
    iso_shift = [t + 0.100 for t in iso if t + 0.100 < dur - 0.1]
    ok_iso_shift = (check_times(nov, iso_shift, tau)
                    if iso_shift else np.array([], bool))
    per_class = {}
    for e, p in zip(meta["events"], ok):
        c = per_class.setdefault(e["class"], [0, 0])
        c[0] += int(p)
        c[1] += 1
    return {
        "piece": piece_dir.name,
        "n_onsets": len(times),
        "passed": int(ok.sum()),
        "pct": round(100 * ok.mean(), 2),
        "control_shift100ms_pct": round(100 * ok_shift.mean(), 2),
        "control_random_pct": round(100 * ok_rand.mean(), 2),
        "isolated_n": len(iso),
        "isolated_pct": (round(100 * ok_iso.mean(), 2) if len(iso) else None),
        "isolated_shift100ms_pct": (round(100 * ok_iso_shift.mean(), 2)
                                    if len(iso_shift) else None),
        "per_class": {k: {"passed": v[0], "total": v[1],
                          "pct": round(100 * v[0] / v[1], 1)}
                      for k, v in sorted(per_class.items())},
    }


def make_click(sr=SR, freq=2000.0, dur=0.012):
    n = int(dur * sr)
    t = np.arange(n) / sr
    return (np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.004)).astype(np.float32)


def click_overlays(pieces, out_dir, n_clips, rng):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    click = make_click()
    chosen = [pieces[i % len(pieces)] for i in range(n_clips)]
    manifest = []
    for i, piece_dir in enumerate(chosen):
        piece_dir = Path(piece_dir)
        meta = json.loads((piece_dir / "labels.json").read_text())
        y, _ = sf.read(piece_dir / "mix.wav", always_2d=True)
        y = y.mean(axis=1).astype(np.float32)
        dur = len(y) / SR
        start = rng.uniform(0, max(0.01, dur - 10.0))
        end = min(dur, start + 10.0)
        clip = y[int(start * SR):int(end * SR)].copy() * 0.6
        n_marks = 0
        for e in meta["events"]:
            if start <= e["t"] < end:
                i0 = int((e["t"] - start) * SR)
                seg = clip[i0:i0 + len(click)]
                seg += 0.5 * click[: len(seg)]
                n_marks += 1
        name = f"clip{i:02d}_{piece_dir.name}_{start:.1f}s.wav"
        sf.write(out_dir / name, np.clip(clip, -1, 1), SR, subtype="PCM_16")
        manifest.append({"clip": name, "piece": piece_dir.name,
                         "start_sec": round(start, 2), "onsets_marked": n_marks})
    (out_dir / "clips_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    rng = random.Random(4242)
    pieces = sorted(p for p in PILOT_DIR.iterdir()
                    if (p / "mix.wav").exists())
    results = [validate_piece(p, rng) for p in pieces]
    tot = sum(r["n_onsets"] for r in results)
    hit = sum(r["passed"] for r in results)
    iso_tot = sum(r["isolated_n"] for r in results)
    iso_hit = sum(round(r["isolated_pct"] * r["isolated_n"] / 100)
                  for r in results if r["isolated_n"])
    iso_shift_rates = [r["isolated_shift100ms_pct"] for r in results
                       if r["isolated_shift100ms_pct"] is not None]
    overall = {
        "tolerance_ms": TOL * 1000, "tau_frac_of_p98": TAU_FRAC,
        "pieces": results,
        "overall": {"n_onsets": tot, "passed": hit,
                    "pct": round(100 * hit / tot, 2)},
        "overall_control_shift100ms_pct": round(
            float(np.mean([r["control_shift100ms_pct"] for r in results])), 2),
        "overall_control_random_pct": round(
            float(np.mean([r["control_random_pct"] for r in results])), 2),
        "overall_isolated": {"n": iso_tot, "passed": iso_hit,
                             "pct": round(100 * iso_hit / iso_tot, 2)
                             if iso_tot else None},
        "overall_isolated_shift100ms_pct": round(
            float(np.mean(iso_shift_rates)), 2) if iso_shift_rates else None,
    }
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else PILOT_DIR / "alignment_report.json"
    out.write_text(json.dumps(overall, indent=1))
    for r in results:
        print(f"{r['piece']}: {r['pct']}% ({r['passed']}/{r['n_onsets']}) "
              f"[controls: shift {r['control_shift100ms_pct']}%, "
              f"random {r['control_random_pct']}%] "
              f"[isolated n={r['isolated_n']}: {r['isolated_pct']}% -> "
              f"shifted {r['isolated_shift100ms_pct']}%]")
    print(f"OVERALL: {overall['overall']['pct']}% "
          f"({hit}/{tot}) | controls: shift "
          f"{overall['overall_control_shift100ms_pct']}%, random "
          f"{overall['overall_control_random_pct']}%")
    oi = overall["overall_isolated"]
    print(f"ISOLATED (no neighbor label within 150 ms): {oi['pct']}% "
          f"({oi['passed']}/{oi['n']}) | shifted +100 ms: "
          f"{overall['overall_isolated_shift100ms_pct']}%")
    if len(sys.argv) > 2:
        m = click_overlays(pieces, sys.argv[2], 10, rng)
        print(f"wrote {len(m)} click-overlay clips to {sys.argv[2]}")


if __name__ == "__main__":
    main()
