"""Generate + render the WS5b full synthetic front-ensemble corpus.

Unlike run_pilot.py (10 short pieces, one fixed plan), this generates N
pieces spread across the 3 ensemble presets with piece lengths drawn from
--duration-min/--duration-max (default 60-90s) and unique seeds per piece,
so tempo/key/mode/density all vary (see arrange.Piece / arrange.generate).

Disk discipline (same in-code guards as run_pilot.py + render.py, not left
to operator care):
  --max-total-mb (default 6000): cumulative size of the output dir is
    checked after every completed piece; generation stops cleanly (not a
    crash) once exceeded -- whatever rendered so far is kept.
  --min-free-gb (default 20): free space on the volume is checked before
    dispatching each batch; aborts cleanly if below the floor.
  Stems are never kept (render_piece(keep_stems=False) default) -- final
  mixes + labels.json only. See corpus_report.md for the size rationale.
  Per-piece caps from render.py (MAX_PIECE_SEC=90s, 3x-expected-size sanity
  check, --use-eot, wall-clock timeout) apply unchanged to every piece.

Parallelism: pieces are independent (own seed, own output dir, own
mapping.json read), so a bounded ProcessPoolExecutor renders several at
once. --jobs is hard-capped at 4 in code (an MPS training workstream may be
running concurrently on this machine; this is a CPU-only renderer and must
not oversubscribe the 10-core machine).
"""
import argparse
import concurrent.futures as cf
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ASSETS, SFZ_DIR

CORPUS_DIR = ASSETS / "corpus"
MAX_JOBS_HARD_CAP = 4
PRESETS = ["concert_orchestral", "mallet_choir", "full_field"]


def dir_size_mb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*")
               if f.is_file()) / 1e6


def build_plan(n_pieces, seed_start):
    """Round-robin across the 3 presets; unique seed per piece, offset well
    away from the pilot's seeds (101-103/201-203/301-304) and from each
    other preset's block so there is no collision risk."""
    plan = []
    base = {"concert_orchestral": seed_start,
            "mallet_choir": seed_start + 100_000,
            "full_field": seed_start + 200_000}
    counters = dict.fromkeys(PRESETS, 0)
    for i in range(n_pieces):
        preset = PRESETS[i % len(PRESETS)]
        seed = base[preset] + counters[preset]
        counters[preset] += 1
        plan.append((preset, seed))
    return plan


def _render_one(args):
    """Runs in a worker process: generate + render a single piece."""
    preset, seed, out_root, dur_min, dur_max = args
    # re-import inside the worker (separate process / fresh interpreter)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from arrange import generate, write_outputs
    from render import render_piece, RenderBudgetError

    mapping = json.loads((SFZ_DIR / "mapping.json").read_text())
    pid = f"{preset}_{seed}"
    outdir = Path(out_root) / pid
    t0 = time.time()
    p = generate(preset, seed, mapping, duration_range=(dur_min, dur_max))
    meta = write_outputs(p, outdir, mapping)
    try:
        res = render_piece(outdir, keep_stems=False)
    except RenderBudgetError as e:
        return {"piece": pid, "preset": preset, "seed": seed, "error": str(e)}
    dt = time.time() - t0
    class_counts = {}
    for e in meta["events"]:
        class_counts[e["class"]] = class_counts.get(e["class"], 0) + 1
    return {"piece": pid, "preset": preset, "seed": seed,
            "tempo": meta["tempo"], "mode": meta["mode"], "root": meta["root"],
            "density": meta["density"], "midi_dur": meta["duration_sec"],
            "audio_dur": round(res["duration"], 3),
            "n_onsets": meta["n_onsets"], "class_counts": class_counts,
            "gen_render_sec": round(dt, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CORPUS_DIR)
    ap.add_argument("--n-pieces", type=int, default=280)
    ap.add_argument("--seed-start", type=int, default=1_000_000)
    ap.add_argument("--duration-min", type=float, default=60.0)
    ap.add_argument("--duration-max", type=float, default=90.0)
    ap.add_argument("--max-total-mb", type=float, default=6000.0)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--resume", action="store_true",
                    help="skip pieces whose mix.wav already exists")
    args = ap.parse_args()
    jobs = max(1, min(args.jobs, MAX_JOBS_HARD_CAP))
    args.out.mkdir(parents=True, exist_ok=True)

    plan = build_plan(args.n_pieces, args.seed_start)
    if args.resume:
        plan = [(preset, seed) for preset, seed in plan
                if not (args.out / f"{preset}_{seed}" / "mix.wav").exists()]
        print(f"resume: {len(plan)} pieces remaining")

    summary_path = args.out / "corpus_summary.json"
    summary = []
    if args.resume and summary_path.exists():
        summary = json.loads(summary_path.read_text())

    t_start = time.time()
    stop = False
    it = iter(plan)
    with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
        while not stop:
            free_gb = shutil.disk_usage(args.out).free / 1e9
            if free_gb < args.min_free_gb:
                print(f"ABORT: only {free_gb:.1f} GB free "
                      f"(< {args.min_free_gb:.0f} GB floor)", file=sys.stderr)
                break
            batch = []
            for _ in range(jobs):
                try:
                    preset, seed = next(it)
                except StopIteration:
                    stop = True
                    break
                batch.append((preset, seed, str(args.out),
                             args.duration_min, args.duration_max))
            if not batch:
                break
            for row in ex.map(_render_one, batch):
                if "error" in row:
                    print(f"ERROR {row['piece']}: {row['error']}",
                          file=sys.stderr)
                    continue
                summary.append(row)
                used_mb = dir_size_mb(args.out)
                elapsed = time.time() - t_start
                print(f"{row['piece']}: tempo={row['tempo']} "
                      f"dur={row['audio_dur']}s onsets={row['n_onsets']} "
                      f"wall={row['gen_render_sec']}s cum_out={used_mb:.0f}MB "
                      f"elapsed={elapsed:.0f}s")
                if used_mb > args.max_total_mb:
                    summary_path.write_text(json.dumps(summary, indent=1))
                    print(f"STOP: output dir {used_mb:.0f} MB exceeds "
                          f"--max-total-mb {args.max_total_mb:.0f} -- "
                          f"keeping {len(summary)} pieces rendered so far",
                          file=sys.stderr)
                    stop = True
                    break
            summary_path.write_text(json.dumps(summary, indent=1))

    tot_audio = sum(r["audio_dur"] for r in summary)
    tot_wall = time.time() - t_start
    print(f"TOTAL: {len(summary)} pieces, {tot_audio:.0f}s audio "
          f"({tot_audio / 3600:.2f}h), {tot_wall:.0f}s wall, "
          f"{dir_size_mb(args.out):.0f} MB on disk")


if __name__ == "__main__":
    main()
