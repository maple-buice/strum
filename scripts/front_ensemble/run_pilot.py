"""Generate + render the WS5b pilot: ~10 short pieces across 3 presets.

Disk discipline (in code, post-mortem 2026-07-02):
  --max-total-mb (default 2000): cumulative size of the output dir is
    checked after every piece; the run aborts cleanly if exceeded.
  --min-free-gb (default 20): free space on the volume is checked before
    every piece; the run aborts cleanly if below the floor.
  Stems are deleted after each piece's mix unless --keep-stems is given
  (piece MIDI + labels.json are always kept, so any stem can be
  re-rendered deterministically later).
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arrange import generate, write_outputs
from common import PILOT_DIR, SFZ_DIR
from render import render_piece

PLAN = [("concert_orchestral", 101), ("concert_orchestral", 102),
        ("concert_orchestral", 103), ("mallet_choir", 201),
        ("mallet_choir", 202), ("mallet_choir", 203), ("full_field", 301),
        ("full_field", 302), ("full_field", 303), ("full_field", 304)]


def dir_size_mb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*")
               if f.is_file()) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PILOT_DIR)
    ap.add_argument("--max-total-mb", type=float, default=2000.0,
                    help="abort if cumulative output exceeds this")
    ap.add_argument("--min-free-gb", type=float, default=20.0,
                    help="abort if volume free space drops below this")
    ap.add_argument("--keep-stems", action="store_true",
                    help="keep per-instrument stems (budgeted like mixes)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mapping = json.loads((SFZ_DIR / "mapping.json").read_text())
    summary = []
    for preset, seed in PLAN:
        free_gb = shutil.disk_usage(args.out).free / 1e9
        if free_gb < args.min_free_gb:
            print(f"ABORT: only {free_gb:.1f} GB free "
                  f"(< {args.min_free_gb:.0f} GB floor)", file=sys.stderr)
            sys.exit(2)
        pid = f"{preset}_{seed}"
        outdir = args.out / pid
        t0 = time.time()
        p = generate(preset, seed, mapping)
        meta = write_outputs(p, outdir, mapping)
        res = render_piece(outdir, keep_stems=args.keep_stems)
        dt = time.time() - t0
        row = {"piece": pid, "preset": preset, "seed": seed,
               "tempo": meta["tempo"], "mode": meta["mode"],
               "density": meta["density"], "midi_dur": meta["duration_sec"],
               "audio_dur": round(res["duration"], 2),
               "n_onsets": meta["n_onsets"], "gen_render_sec": round(dt, 1)}
        summary.append(row)
        used_mb = dir_size_mb(args.out)
        print(f"{pid}: tempo={row['tempo']} dur={row['audio_dur']}s "
              f"onsets={row['n_onsets']} wall={dt:.1f}s "
              f"cum_out={used_mb:.0f}MB")
        if used_mb > args.max_total_mb:
            (args.out / "pilot_summary.json").write_text(
                json.dumps(summary, indent=1))
            print(f"ABORT: output dir {used_mb:.0f} MB exceeds "
                  f"--max-total-mb {args.max_total_mb:.0f}", file=sys.stderr)
            sys.exit(3)
    (args.out / "pilot_summary.json").write_text(json.dumps(summary, indent=1))
    tot_audio = sum(r["audio_dur"] for r in summary)
    tot_wall = sum(r["gen_render_sec"] for r in summary)
    print(f"TOTAL: {len(summary)} pieces, {tot_audio:.0f}s audio, "
          f"{tot_wall:.0f}s wall ({tot_wall / tot_audio:.2f}x realtime), "
          f"{dir_size_mb(args.out):.0f} MB on disk")


if __name__ == "__main__":
    main()
