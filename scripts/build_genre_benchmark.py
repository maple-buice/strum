"""Rebuild the genre-benchmark full mixes from a local Clone Hero library.

The benchmark itself ships as metadata only (benchmarks/genre_manifest.json:
titles, artists, buckets, and MD5 hashes of each song's notes.mid).  This
script reconstructs the audio inputs from a song library the user already
owns -- no copyrighted content is ever distributed.

Usage:
    python scripts/build_genre_benchmark.py \\
        --library-dir "/path/to/Clone Hero/Songs" \\
        --manifest benchmarks/genre_manifest.json \\
        --output-dir bench_audio

For every song in the manifest the script:
  1. Locates the song folder in the library by matching the `artist` and
     `name` fields of song.ini (case-insensitive, whitespace-normalized);
     folder names are ignored because they are often mangled or suffixed.
  2. Verifies the folder's notes.mid MD5 against the manifest, so results
     are only reported against the exact ground-truth chart the benchmark
     was defined with.  If several folders match by artist+title, the one
     whose notes.mid MD5 matches wins.
  3. Builds the full mix: ffmpeg `amix` of ALL .ogg stems in the folder
     with `normalize=0`, followed by `alimiter=limit=0.97`, encoded as
     44.1 kHz stereo 16-bit PCM WAV.

Output filename: "<artist> - <name>.wav", sanitized deterministically:
the characters <>:"/\\|?* are removed (the same character class STRUM's
batch_infer_hybrid.py strips when it derives an output folder name, so
downstream folder names stay predictable), runs of whitespace collapse to
a single space, and leading/trailing whitespace is stripped.

Exit status is nonzero if any manifest song is missing from the library,
fails its notes.mid checksum, or fails to mix.  A per-song FOUND /
MISSING / MD5-MISMATCH / MIX-FAILED summary is always printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

SANITIZE_CHARS = re.compile(r'[<>:"/\\|?*]')


def sanitize_filename(s: str) -> str:
    """Deterministic filename sanitization (documented in module docstring)."""
    s = SANITIZE_CHARS.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def norm_key(s: str) -> str:
    """Normalization used to match manifest artist/title to song.ini values."""
    return re.sub(r"\s+", " ", s).strip().casefold()


def read_text_tolerant(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def parse_song_ini(p: Path) -> dict[str, str]:
    """Tolerant song.ini parser: [song]/[Song]/[SONG] or headerless files,
    `key = value` with any whitespace, case-insensitive keys, bad lines
    ignored. Returns lowercased-key -> raw string value."""
    data: dict[str, str] = {}
    in_song_section = False
    seen_any_section = False
    for line in read_text_tolerant(p).splitlines():
        line = line.strip()
        if not line or line.startswith((";", "#")):
            continue
        if line.startswith("[") and line.endswith("]"):
            seen_any_section = True
            in_song_section = line[1:-1].strip().lower() == "song"
            continue
        if not seen_any_section:
            in_song_section = True
        if not in_song_section or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().lower()
        if key and key not in data:
            data[key] = val.strip()
    return data


def index_library(library_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """Map (normalized artist, normalized title) -> song folders."""
    index: dict[tuple[str, str], list[Path]] = {}
    for ini_path in library_dir.rglob("*"):
        if not (ini_path.is_file() and ini_path.name.lower() == "song.ini"):
            continue
        try:
            ini = parse_song_ini(ini_path)
        except OSError:
            continue
        key = (norm_key(ini.get("artist", "")), norm_key(ini.get("name", "")))
        index.setdefault(key, []).append(ini_path.parent)
    return index


def md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def build_mix(song_dir: Path, out_wav: Path) -> tuple[bool, str]:
    """ffmpeg-amix ALL .ogg stems in song_dir into out_wav. Returns (ok, detail)."""
    stems = sorted(
        p for p in song_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".ogg"
    )
    if not stems:
        return False, "no .ogg stems in folder"
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for stem in stems:
        cmd += ["-i", str(stem)]
    cmd += [
        "-filter_complex",
        f"amix=inputs={len(stems)}:duration=longest:normalize=0,"
        "alimiter=limit=0.97",
        "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, f"ffmpeg exit {proc.returncode}: {proc.stderr.strip()[-500:]}"
    return True, f"{len(stems)} stems: {', '.join(s.name for s in stems)}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild genre-benchmark full mixes from a Clone Hero library."
    )
    parser.add_argument("--library-dir", type=Path, required=True,
                        help="Root of the Clone Hero song library to search")
    parser.add_argument("--manifest", type=Path,
                        default=Path("benchmarks/genre_manifest.json"),
                        help="Benchmark manifest (metadata + notes.mid MD5s)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write '<artist> - <name>.wav' mixes")
    parser.add_argument("--only", type=str, default=None,
                        help="Only process songs whose title contains this "
                             "substring (case-insensitive); missing songs "
                             "outside the filter are not checked")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    songs = manifest["songs"]
    if args.only:
        songs = [s for s in songs if args.only.casefold() in s["title"].casefold()]
        if not songs:
            print(f"No manifest songs match --only {args.only!r}")
            sys.exit(1)

    print(f"Indexing library {args.library_dir} ...")
    index = index_library(args.library_dir)
    print(f"  {sum(len(v) for v in index.values())} song folders indexed")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    found: list[str] = []
    missing: list[str] = []
    mismatched: list[str] = []
    failed: list[str] = []
    out_names: dict[str, str] = {}

    for song in songs:
        label = f"{song['artist']} - {song['title']}"
        out_name = sanitize_filename(f"{song['artist']} - {song['title']}") + ".wav"
        if out_name in out_names:
            print(f"[ERROR] filename collision: {label!r} and "
                  f"{out_names[out_name]!r} both sanitize to {out_name!r}")
            failed.append(label)
            continue
        out_names[out_name] = label

        candidates = index.get(
            (norm_key(song["artist"]), norm_key(song["title"])), []
        )
        if not candidates:
            print(f"[MISSING] {label}: not found in library")
            missing.append(label)
            continue

        song_dir = None
        for cand in candidates:
            mid = cand / "notes.mid"
            if mid.is_file() and md5_file(mid) == song["notes_md5"]:
                song_dir = cand
                break
        if song_dir is None:
            got = [
                md5_file(c / "notes.mid") if (c / "notes.mid").is_file() else "(no notes.mid)"
                for c in candidates
            ]
            print(f"[MD5-MISMATCH] {label}: found {len(candidates)} folder(s) "
                  f"but notes.mid MD5 {got} != manifest {song['notes_md5']}")
            mismatched.append(label)
            continue

        ok, detail = build_mix(song_dir, args.output_dir / out_name)
        if ok:
            print(f"[FOUND] {label} -> {out_name} ({detail})")
            found.append(label)
        else:
            print(f"[MIX-FAILED] {label}: {detail}")
            failed.append(label)

    print(
        f"\nSummary: {len(found)} built, {len(missing)} missing, "
        f"{len(mismatched)} checksum-mismatched, {len(failed)} failed "
        f"(of {len(songs)} manifest songs)"
    )
    for tag, items in (("missing", missing), ("checksum-mismatched", mismatched),
                       ("failed", failed)):
        for label in items:
            print(f"  {tag}: {label}")
    if missing or mismatched or failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
