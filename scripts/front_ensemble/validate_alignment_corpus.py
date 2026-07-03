"""WS5b corpus alignment gate: run the SAME criterion as
validate_alignment.py (used for the pilot, already passed) against a random
10% sample of the full corpus, not the whole corpus -- full-corpus onset
counts run into the hundreds of thousands and validating all of it would
cost much more wall time than the render itself. Sampling is done at the
PIECE level (not onset level) so per-piece stats stay meaningful, and the
sample is seeded for reproducibility.

Reuses validate_piece()/click_overlays() unmodified from validate_alignment
so the pilot's already-passed gate criterion (tau = 0.10 * P98 novelty,
+/-20ms tolerance, shift/random/isolated negative controls) is exactly what
runs here too -- no re-tuning, no new thresholds.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ASSETS
from validate_alignment import click_overlays, validate_piece

CORPUS_DIR = ASSETS / "corpus"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", type=Path,
                    default=Path("validation/ws5/corpus_alignment_sample.json"))
    ap.add_argument("--clips-dir", type=Path, default=None)
    ap.add_argument("--n-clips", type=int, default=6)
    args = ap.parse_args()

    all_pieces = sorted(p for p in args.corpus_dir.iterdir()
                        if (p / "mix.wav").exists())
    rng = random.Random(args.seed)
    n_sample = max(1, round(len(all_pieces) * args.frac))
    sample = sorted(rng.sample(all_pieces, n_sample), key=lambda p: p.name)

    ctl_rng = random.Random(args.seed + 1)
    results = [validate_piece(p, ctl_rng) for p in sample]
    tot = sum(r["n_onsets"] for r in results)
    hit = sum(r["passed"] for r in results)
    iso_tot = sum(r["isolated_n"] for r in results)
    iso_hit = sum(round(r["isolated_pct"] * r["isolated_n"] / 100)
                  for r in results if r["isolated_n"])
    iso_shift_rates = [r["isolated_shift100ms_pct"] for r in results
                       if r["isolated_shift100ms_pct"] is not None]

    overall = {
        "corpus_dir": str(args.corpus_dir),
        "n_pieces_total": len(all_pieces),
        "n_pieces_sampled": len(sample),
        "sample_frac_requested": args.frac,
        "sample_seed": args.seed,
        "sampled_pieces": [p.name for p in sample],
        "pieces": results,
        "overall": {"n_onsets": tot, "passed": hit,
                    "pct": round(100 * hit / tot, 2) if tot else None},
        "overall_control_shift100ms_pct": round(
            float(sum(r["control_shift100ms_pct"] for r in results) / len(results)), 2),
        "overall_control_random_pct": round(
            float(sum(r["control_random_pct"] for r in results) / len(results)), 2),
        "overall_isolated": {"n": iso_tot, "passed": iso_hit,
                             "pct": round(100 * iso_hit / iso_tot, 2)
                             if iso_tot else None},
        "overall_isolated_shift100ms_pct": round(
            float(sum(iso_shift_rates) / len(iso_shift_rates)), 2)
            if iso_shift_rates else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(overall, indent=1))

    print(f"corpus: {len(all_pieces)} pieces total, sampled "
          f"{len(sample)} ({100*len(sample)/len(all_pieces):.1f}%), seed={args.seed}")
    for r in results:
        print(f"{r['piece']}: {r['pct']}% ({r['passed']}/{r['n_onsets']}) "
              f"[controls: shift {r['control_shift100ms_pct']}%, "
              f"random {r['control_random_pct']}%] "
              f"[isolated n={r['isolated_n']}: {r['isolated_pct']}% -> "
              f"shifted {r['isolated_shift100ms_pct']}%]")
    print(f"OVERALL (sampled {len(sample)}/{len(all_pieces)} pieces): "
          f"{overall['overall']['pct']}% ({hit}/{tot}) | controls: shift "
          f"{overall['overall_control_shift100ms_pct']}%, random "
          f"{overall['overall_control_random_pct']}%")
    oi = overall["overall_isolated"]
    print(f"ISOLATED (no neighbor label within 150 ms): {oi['pct']}% "
          f"({oi['passed']}/{oi['n']}) | shifted +100 ms: "
          f"{overall['overall_isolated_shift100ms_pct']}%")

    if args.clips_dir:
        m = click_overlays(sample, args.clips_dir, args.n_clips,
                           random.Random(args.seed + 2))
        print(f"wrote {len(m)} click-overlay clips to {args.clips_dir}")


if __name__ == "__main__":
    main()
