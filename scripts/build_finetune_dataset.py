#!/usr/bin/env python3
"""
Build an onset-classifier fine-tune data_dir from a corpus manifest.

Input:  a corpus manifest (JSON) listing Clone Hero song folders with a
        `split` ("train"/"val") and drum-stem availability
        (e.g. validation/ws4/corpus_manifest.json).
Output: a trainer-compatible data_dir for scripts/preprocess_onset_windows.py:

    <output-dir>/
      manifest.json                  {"songs": [{id, split, charts: {drums: true},
                                                 stems: {drums: <relpath>}}, ...]}
      <id>/drums_labels.json         (from notes.mid, via scripts/make_drums_labels.py)
      <id>/drums.ogg | drums.wav     (training audio, see below)
      separated/<id>/drums.wav       (expected pre-separated Demucs stems for
                                      full-mix-only songs — NOT created here)

Split naming: corpus "train" → "train"; corpus "val" → "test" (the trainer's
validation split is hard-named "test").

Training audio per drums_stem_kind:
  single  copy the song's drums.ogg unchanged
  split   sum drums_1/2/3... in Python (per-file resample to 44100 Hz if
          needed, downmix to mono, sum, peak-normalize only if the sum clips —
          same policy as resolve_drums_fallback in batch_infer_hybrid.py),
          written as <id>/drums.wav
  none    (full-mix-only) the song's training audio must already exist at
          <output-dir>/separated/<id>/drums.wav — produced by a Demucs
          separation pass over the song's full mix (see
          validation/ws4/RUNBOOK.md). If any are missing, this script FAILS
          LOUDLY and lists every missing song + expected path.

--list-missing-stems: print the full-mix-only songs whose separated stem is
absent (id, source folder, expected path), then exit without writing anything.

Usage:
    build_finetune_dataset.py --corpus-manifest validation/ws4/corpus_manifest.json \
        --output-dir datasets/ws4_edm [--list-missing-stems]
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from make_drums_labels import chart_to_labels, parse_drums_midi  # noqa: E402
from build_genre_benchmark import parse_song_ini  # noqa: E402

TARGET_SR = 44100
SANITIZE_CHARS = re.compile(r'[<>:"/\\|?*]')


def song_id(source_path: Path) -> str:
    """Deterministic id from the song folder's basename."""
    s = SANITIZE_CHARS.sub("", source_path.name)
    return re.sub(r"\s+", " ", s).strip()


def is_five_lane(song_dir: Path) -> bool:
    """Read song.ini five_lane_drums flag (tolerant parse; default False)."""
    ini = song_dir / "song.ini"
    if not ini.exists():
        return False
    val = parse_song_ini(ini).get("five_lane_drums", "")
    return val.strip().lower() in ("1", "true", "yes")


def load_mono_44k(path: Path) -> np.ndarray:
    """Read audio as float32 mono at 44100 Hz (torchaudio resample, like
    preprocess_onset_windows.py's phase 2)."""
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        import torch
        import torchaudio
        t = torch.from_numpy(y).float().unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, TARGET_SR)
        y = t.squeeze(0).numpy()
    return y


def mix_split_stems(stem_paths: list[Path], out_wav: Path) -> None:
    """Sum split drum stems to one mono track; peak-normalize only if clipping
    (same policy as resolve_drums_fallback in batch_infer_hybrid.py)."""
    parts = [load_mono_44k(p) for p in stem_paths]
    n = max(len(p) for p in parts)
    y = np.zeros(n, dtype=np.float32)
    for p in parts:
        y[: len(p)] += p
    peak = float(np.abs(y).max()) if n else 0.0
    if peak > 1.0:
        y = y / peak
    sf.write(str(out_wav), y, TARGET_SR)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--corpus-manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="data_dir to create (trainer's paths.data_dir)")
    ap.add_argument("--separated-dir", type=Path, default=None,
                    help="Where pre-separated Demucs drums stems live "
                         "(default: <output-dir>/separated)")
    ap.add_argument("--list-missing-stems", action="store_true",
                    help="Print full-mix-only songs needing Demucs and exit")
    args = ap.parse_args()

    with open(args.corpus_manifest) as f:
        corpus = json.load(f)
    songs = corpus["songs"]

    out_dir: Path = args.output_dir
    separated_dir: Path = args.separated_dir or (out_dir / "separated")

    # ── Pass 1: resolve ids, check inputs, find missing separated stems ──
    resolved = []
    seen_ids: dict[str, str] = {}
    missing_midi = []
    missing_stems = []
    for song in songs:
        src = Path(song["path"])
        sid = song_id(src)
        if sid in seen_ids:
            print(f"FATAL: song id collision: '{sid}' from both\n"
                  f"  {seen_ids[sid]}\n  {src}", file=sys.stderr)
            sys.exit(1)
        seen_ids[sid] = str(src)

        if not (src / "notes.mid").exists():
            missing_midi.append((sid, src))

        kind = song["drums_stem_kind"]
        sep_wav = separated_dir / sid / "drums.wav"
        if kind == "none" and not sep_wav.exists():
            missing_stems.append((sid, src, sep_wav))

        resolved.append((song, sid, kind, sep_wav))

    if args.list_missing_stems:
        print(f"# full-mix-only songs missing a pre-separated drums stem: "
              f"{len(missing_stems)}")
        print("# id\tsource_folder\texpected_stem_path")
        for sid, src, sep_wav in missing_stems:
            print(f"{sid}\t{src}\t{sep_wav}")
        return

    errors = []
    if missing_midi:
        errors.append("Songs missing notes.mid:")
        errors += [f"  {sid}: {src}" for sid, src in missing_midi]
    if missing_stems:
        errors.append(
            f"{len(missing_stems)} full-mix-only songs have no pre-separated "
            f"drums stem. Run the Demucs separation step first "
            f"(validation/ws4/RUNBOOK.md), expected at:")
        errors += [f"  {sid}: {sep_wav}" for sid, _, sep_wav in missing_stems]
    if errors:
        print("FATAL: dataset inputs incomplete.\n" + "\n".join(errors),
              file=sys.stderr)
        sys.exit(1)

    # ── Pass 2: build labels + audio + manifest ──
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_songs = []
    split_map = {"train": "train", "val": "test"}  # trainer val split is "test"
    n_hits_total = 0

    for song, sid, kind, sep_wav in resolved:
        src = Path(song["path"])
        song_out = out_dir / sid
        song_out.mkdir(parents=True, exist_ok=True)

        # Labels
        five_lane = is_five_lane(src)
        chart = parse_drums_midi(src / "notes.mid", five_lane_drums=five_lane)
        if not chart.hits:
            print(f"WARNING: {sid}: no Expert drum hits parsed — song will be "
                  f"skipped by preprocess_onset_windows", file=sys.stderr)
        with open(song_out / "drums_labels.json", "w") as f:
            json.dump(chart_to_labels(chart), f)
        n_hits_total += len(chart.hits)

        # Audio
        if kind == "single":
            stem_src = src / song["drums_stem_files"][0]
            stem_rel = f"{sid}/{stem_src.name}"
            dst = out_dir / stem_rel
            if not dst.exists():
                shutil.copy2(stem_src, dst)
        elif kind == "split":
            stem_rel = f"{sid}/drums.wav"
            dst = out_dir / stem_rel
            if not dst.exists():
                mix_split_stems(
                    [src / name for name in song["drums_stem_files"]], dst)
        elif kind == "none":
            stem_rel = str(sep_wav.relative_to(out_dir)) \
                if sep_wav.is_relative_to(out_dir) else str(sep_wav)
        else:
            print(f"FATAL: unknown drums_stem_kind '{kind}' for {sid}",
                  file=sys.stderr)
            sys.exit(1)

        manifest_songs.append({
            "id": sid,
            "split": split_map[song["split"]],
            "charts": {"drums": True},
            "stems": {"drums": stem_rel},
            # provenance (ignored by the trainer/preprocessor)
            "source_path": str(src),
            "title": song.get("title"),
            "artist": song.get("artist"),
            "drums_stem_kind": kind,
            "five_lane_drums": five_lane,
            "n_label_hits": len(chart.hits),
        })
        print(f"  [{len(manifest_songs)}/{len(resolved)}] {sid} "
              f"({split_map[song['split']]}, {kind}, {len(chart.hits)} hits)")

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_corpus_manifest": str(args.corpus_manifest),
        "songs": manifest_songs,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    n_train = sum(1 for s in manifest_songs if s["split"] == "train")
    n_test = sum(1 for s in manifest_songs if s["split"] == "test")
    print(f"\nWrote {out_dir / 'manifest.json'}: {len(manifest_songs)} songs "
          f"(train {n_train} / test {n_test}), {n_hits_total} label hits total")


if __name__ == "__main__":
    main()
