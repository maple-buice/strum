"""Render a generated piece: per-instrument MIDI -> sfizz_render -> stems ->
gain-staged stereo mix.

Per-instrument mapping/gain comes from ws5_assets/sfz/mapping.json.
Each stem is peak-normalized to -6 dBFS (single scalar; within-stem dynamics
preserved), then the instrument's gain_db offset is applied, stems are summed
and the mix is scaled down if it would clip. Output: mix.wav (44.1 kHz,
16-bit PCM stereo) + optional stems/ (--keep-stems).

DISK-RUNAWAY DISCIPLINE (post-mortem 2026-07-02: a sfizz voice from MDL's
loop_continuous roll regions never went inactive, and without --use-eot
sfizz_render renders until all voices finish -> one stem wrote ~44 GB and
filled the disk). All of the following are enforced IN CODE, not by
operator care:
  1. sfizz_render is always invoked with --use-eot: rendering stops at the
     MIDI end-of-track marker unconditionally, stuck voices or not.
  2. A wall-clock timeout on every sfizz_render call (backstop; kills the
     child if it somehow ignores EOT).
  3. Per-piece duration cap (MAX_PIECE_SEC, default 90 s) checked against
     labels.json before any audio is rendered.
  4. Per-file size sanity check: every rendered file must stay under
     3x the size expected for its labeled duration at 44.1 kHz/16-bit/stereo
     (~10 MB per 60 s). Violation logs a warning and raises.
  5. Stems are deleted after the mix is written unless keep_stems is set.
     All soundfile writes pass subtype="PCM_16" explicitly.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SFIZZ_RENDER, SFZ_DIR

SR = 44100
BYTES_PER_SEC = SR * 2 * 2          # stereo, 16-bit PCM
MAX_PIECE_SEC = 90.0                # per-piece duration cap
FILE_SIZE_FACTOR = 3.0              # sanity: file may not exceed 3x expected
RENDER_TAIL_SEC = 5.0               # decay tail allowance past EOT
STEM_TIMEOUT_FLOOR_SEC = 120.0      # wall-clock backstop per sfizz call


class RenderBudgetError(RuntimeError):
    """A size/duration guard tripped; rendering was aborted."""


def _expected_bytes(duration_sec):
    return (duration_sec + RENDER_TAIL_SEC) * BYTES_PER_SEC + 4096


def _check_file_size(path, duration_sec, what):
    size = Path(path).stat().st_size
    limit = FILE_SIZE_FACTOR * _expected_bytes(duration_sec)
    if size > limit:
        print(f"WARNING: {what} is {size / 1e6:.1f} MB, exceeds "
              f"{FILE_SIZE_FACTOR:.0f}x expected "
              f"({limit / 1e6:.1f} MB for {duration_sec:.1f}s) -- aborting",
              file=sys.stderr)
        raise RenderBudgetError(
            f"{what}: {size} bytes > {limit:.0f} allowed")
    return size


def render_piece(piece_dir, keep_stems=False, verbose=False,
                 max_piece_sec=MAX_PIECE_SEC):
    piece_dir = Path(piece_dir)
    mapping = json.loads((SFZ_DIR / "mapping.json").read_text())["instruments"]
    meta = json.loads((piece_dir / "labels.json").read_text())

    dur = float(meta["duration_sec"])
    if dur > max_piece_sec:
        raise RenderBudgetError(
            f"{piece_dir.name}: labeled duration {dur:.1f}s exceeds the "
            f"per-piece cap of {max_piece_sec:.0f}s -- refusing to render")

    stems_dir = piece_dir / "stems"
    stems_dir.mkdir(exist_ok=True)
    timeout = max(STEM_TIMEOUT_FLOOR_SEC, 10.0 * dur)

    mix = None
    try:
        for mid in sorted((piece_dir / "midi").glob("*.mid")):
            inst = mid.stem
            base = "marimba" if inst == "marimba2" else inst
            info = mapping[base]
            sfz = (Path(info["sfz_abs"]) if "sfz_abs" in info
                   else SFZ_DIR / info["sfz"])
            wav = stems_dir / f"{inst}.wav"
            try:
                r = subprocess.run(
                    [str(SFIZZ_RENDER), "--sfz", str(sfz),
                     "--midi", str(mid), "--wav", str(wav),
                     "-s", str(SR), "--use-eot"],
                    capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                raise RenderBudgetError(
                    f"sfizz_render exceeded {timeout:.0f}s wall clock for "
                    f"{piece_dir.name}/{inst} -- killed (runaway backstop)")
            if r.returncode:
                raise RuntimeError(
                    f"sfizz_render failed for {inst}: {r.stderr[:400]}")
            _check_file_size(wav, dur, f"{piece_dir.name} stem {inst}")

            y, sr = sf.read(wav, always_2d=True)
            assert sr == SR
            if y.shape[1] == 1:
                y = np.repeat(y, 2, axis=1)
            peak = np.abs(y).max()
            if peak > 0:
                y = y * (10 ** (-6 / 20) / peak) * (10 ** (info["gain_db"] / 20))
            if verbose:
                print(f"  {inst}: {len(y)/SR:.1f}s raw_peak={peak:.4f} "
                      f"gain_db={info['gain_db']}")
            if mix is None:
                mix = y
            elif len(y) > len(mix):
                y[: len(mix)] += mix
                mix = y
            else:
                mix[: len(y)] += y
            if keep_stems:
                sf.write(wav, y, SR, subtype="PCM_16")
            else:
                wav.unlink()
    finally:
        # never leave intermediates behind on any exit path
        if not keep_stems and stems_dir.exists():
            shutil.rmtree(stems_dir)

    peak = np.abs(mix).max()
    if peak > 0.97:
        mix = mix * (0.97 / peak)
    out = piece_dir / "mix.wav"
    sf.write(out, mix, SR, subtype="PCM_16")
    _check_file_size(out, dur, f"{piece_dir.name} mix")
    return {"duration": len(mix) / SR, "peak": float(peak),
            "n_onsets": meta["n_onsets"]}


if __name__ == "__main__":
    res = render_piece(sys.argv[1], keep_stems="--keep-stems" in sys.argv,
                       verbose=True)
    print(f"{sys.argv[1]}: mix {res['duration']:.1f}s "
          f"(pre-limit peak {res['peak']:.3f}, {res['n_onsets']} onsets)")
