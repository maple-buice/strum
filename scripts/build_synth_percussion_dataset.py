#!/usr/bin/env python3
"""
Build an onset-classifier fine-tune data_dir from the WS5b synthetic
percussion corpus (ws5_assets/corpus/<piece>/{mix.wav,labels.json}).

Style-matched to scripts/build_finetune_dataset.py (the WS4a real-chart
builder), but the input side is entirely different: there is no chart MIDI
here, just a flat list of instrument-tagged onset events per rendered piece
(see ws5_assets/corpus/corpus_summary.json for the authoritative 266-piece
list — the corpus directory itself contains 14 extra render dirs that were
excluded from corpus_summary.json upstream and are NOT used here).

Input:  ws5_assets/corpus/corpus_summary.json  (piece list: id, preset, seed)
        ws5_assets/corpus/<piece>/labels.json  ({"events": [{t, class,
            instrument, artic, midi, vel}, ...]}, t in seconds)
        ws5_assets/corpus/<piece>/mix.wav      (44.1kHz stereo SFZ render)

Output: <output-dir>/
          manifest.json           trainer-compatible ({"songs": [{id, split,
                                   charts: {drums: true}, stems: {drums:
                                   <ABSOLUTE path to mix.wav>}}, ...]})
          <piece>/drums_labels.json   {"hits": [{time_ms, lane, is_cymbal}]}

Audio is referenced, not copied: scripts/preprocess_onset_windows.py resolves
audio via `data_dir / song["stems"]["drums"]` (pathlib join); Python's
pathlib silently discards the left operand when the right operand is an
absolute path (`Path("/a/b") / "/c/d" == Path("/c/d")`), so an absolute path
string in the manifest's stems.drums resolves directly to the corpus's own
mix.wav with zero data movement. This avoids copying/symlinking 3.3 GB of
audio. (preprocess_onset_windows.py always reads
`data_dir/<id>/drums_labels.json` verbatim, so those small per-piece JSON
files are written under output-dir as usual.)

Instrument -> lane mapping: see INSTRUMENT_TO_LANE below. This is the exact
table from validation/ws4b/corpus_options.md section 2 ("Proposed
instrument -> lane mapping"), encoded with one entry per instrument and an
inline confidence/rationale comment so a future worker can revisit any single
row (e.g. if the Yo Shakespeare gate fails, marimba/timpani/cowbell are the
flagged first candidates to reconsider).

Split: 90/10 by piece, deterministic seed 20260702, stratified by preset
(concert_orchestral / full_field / mallet_choir), mirroring exactly the
method in validation/ws4/build_corpus.py (random.Random(SEED), sorted-then-
shuffled groups, round(0.1*n) per group val quota). Corpus "val" is renamed
"test" for the trainer, same as build_finetune_dataset.py.

Usage:
    build_synth_percussion_dataset.py \
        --corpus-dir ws5_assets/corpus \
        --output-dir datasets/ws4b_percussion
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SEED = 20260702

# ──────────────────────────────────────────────────────────────────────────
# Instrument -> (lane, is_cymbal) mapping.
#
# Lane numbering matches src/preprocessing/parsers/midi_parser.py / Clone
# Hero pro-drums convention: 0=Kick 1=Snare 2=Yellow 3=Blue 4=Green.
# is_cymbal selects the 8-class split for lanes 2-4 via
# src/models/onset_classifier_dataset.py's LANE_CYMBAL_TO_CLASS:
#   (2,True)->HiHat   (2,False)->HighTom
#   (3,True)->Ride    (3,False)->LowTom
#   (4,True)->Crash   (4,False)->FloorTom
# is_cymbal is ignored by the mapping for lanes 0/1 (Kick/Snare) — set False.
#
# Confidence labels and rationale are copied verbatim from
# validation/ws4b/corpus_options.md section 2's table (the settled Option C
# decision). Revise a single row here to change that instrument's mapping.
# ──────────────────────────────────────────────────────────────────────────
INSTRUMENT_TO_LANE: dict[str, tuple[int, bool]] = {
    # instrument:      (lane, is_cymbal)   # confidence — rationale
    "snareline":       (1, False),  # High — direct real-instrument analog (marching snare)
    "tenorline":       (2, False),  # High — tenor drums/"quads" ARE tuned toms (HighTom)
    "cymballine":      (4, True),   # High — name + artic=crash in the data itself (Crash)
    "glockenspiel":    (2, True),   # High — bright, highest-register mallet -> HiHat
    "xylophone":       (2, True),   # High — same register/timbre family as glockenspiel -> HiHat
    "crotales":        (2, True),   # High — highest-pitched class in corpus (midi 97-108) -> HiHat
    "triangle":        (4, True),   # High — bright ring/sustain, small count -> Crash
    "chimes":          (4, True),   # High — tubular bells, long sustain/ring -> Crash
    "vibraphone":      (3, True),   # Medium-High — bright sustained metal bar -> Ride
    "tambourine":      (4, True),   # Medium — bright metallic jingle/accent -> Crash
    "bassline":        (0, False),  # Medium — lowest-pitched family (midi 60-70) -> Kick
    "marimba":         (3, True),   # Medium (MOST CONSEQUENTIAL, most arguable) — 28% of corpus;
                                     #   warmer/lower wood tone sits between tom/cymbal; chosen
                                     #   cymbal (Ride) to match observed real-chart convention
    "woodblock":       (3, False),  # Medium — dry, short, non-ringing attack -> LowTom
    "cowbell":         (4, False),  # Low — genuinely ambiguous (metallic vs dry/short); arbitrary -> FloorTom
    "timpani":         (0, False),  # Low-Medium (2nd most debatable) — pitched/resonant/rolled,
                                     #   closer to a tuned drum than a kick punch, but task's
                                     #   suggested starting point (low pitch) -> Kick
    "cabasa":          (2, True),   # Low — continuous "rub" texture, weak onset boundary -> HiHat
    "shaker":          (2, True),   # Low — same rationale/caveats as cabasa -> HiHat
}

LANE_NAMES = {0: "Kick", 1: "Snare", 2: "Yellow", 3: "Blue", 4: "Green"}


def load_corpus_pieces(corpus_dir: Path) -> list[dict]:
    """Load the authoritative piece list from corpus_summary.json.

    The corpus directory itself has 14 extra render dirs (6
    concert_orchestral, 4 full_field, 4 mallet_choir) not present in
    corpus_summary.json — excluded upstream (WS5b), so not used here.
    """
    summary_path = corpus_dir / "corpus_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    pieces = []
    for row in summary:
        piece_dir = corpus_dir / row["piece"]
        labels_path = piece_dir / "labels.json"
        mix_path = piece_dir / "mix.wav"
        if not labels_path.exists() or not mix_path.exists():
            print(f"FATAL: {row['piece']}: missing labels.json or mix.wav "
                  f"under {piece_dir}", file=sys.stderr)
            sys.exit(1)
        pieces.append({
            "id": row["piece"],
            "preset": row["preset"],
            "seed": row["seed"],
            "piece_dir": piece_dir,
            "labels_path": labels_path,
            "mix_path": mix_path,
            "n_onsets_expected": row["n_onsets"],
        })
    return pieces


def events_to_hits(events: list[dict]) -> tuple[list[dict], Counter, list[str]]:
    """Map a piece's labels.json events to drums_labels.json hits.

    Returns (hits, per-8class-histogram Counter, unmapped-instrument list).
    Unmapped instruments (not in INSTRUMENT_TO_LANE) are dropped and reported
    — none are expected given corpus_summary.json's class inventory, but this
    guards against a corpus revision silently losing data.
    """
    hits = []
    hist = Counter()
    unmapped = []
    # (lane, is_cymbal) -> 8-class name, for histogram reporting only.
    class_name = {
        (0, False): "Kick", (0, True): "Kick",
        (1, False): "Snare", (1, True): "Snare",
        (2, True): "HiHat", (2, False): "HighTom",
        (3, True): "Ride", (3, False): "LowTom",
        (4, True): "Crash", (4, False): "FloorTom",
    }
    for ev in events:
        instrument = ev["class"]
        mapping = INSTRUMENT_TO_LANE.get(instrument)
        if mapping is None:
            unmapped.append(instrument)
            continue
        lane, is_cymbal = mapping
        time_ms = ev["t"] * 1000.0
        hits.append({"time_ms": time_ms, "lane": lane, "is_cymbal": is_cymbal})
        hist[class_name[(lane, is_cymbal)]] += 1
    return hits, hist, unmapped


def split_pieces(pieces: list[dict]) -> None:
    """Assign split in-place: 90/10 by piece, stratified by preset, seed
    SEED. Mirrors validation/ws4/build_corpus.py's genre-stratified split
    exactly (sorted-then-shuffled groups, per-group round(0.1*n) val quota,
    same rng instance threaded across groups in sorted key order)."""
    rng = random.Random(SEED)
    by_preset = defaultdict(list)
    for p in pieces:
        by_preset[p["preset"]].append(p)
    for preset in sorted(by_preset):
        grp = sorted(by_preset[preset], key=lambda p: p["id"])
        rng.shuffle(grp)
        n = len(grp)
        n_val = round(0.1 * n)
        if n_val == 0 and n >= 4:
            n_val = 1
        for p in grp[:n_val]:
            p["split"] = "test"     # trainer's val split is hard-named "test"
        for p in grp[n_val:]:
            p["split"] = "train"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--corpus-dir", type=Path, default=Path("ws5_assets/corpus"))
    ap.add_argument("--output-dir", type=Path, required=True,
                     help="data_dir to create (trainer's paths.data_dir)")
    args = ap.parse_args()

    pieces = load_corpus_pieces(args.corpus_dir)
    print(f"Loaded {len(pieces)} pieces from "
          f"{args.corpus_dir / 'corpus_summary.json'}")

    split_pieces(pieces)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_songs = []
    total_hist = Counter()
    split_hist = {"train": Counter(), "test": Counter()}
    n_hits_total = 0
    n_unmapped_total = 0
    unmapped_seen = Counter()

    for i, p in enumerate(pieces, 1):
        with open(p["labels_path"]) as f:
            label_data = json.load(f)
        events = label_data["events"]
        hits, hist, unmapped = events_to_hits(events)
        if unmapped:
            unmapped_seen.update(unmapped)
            n_unmapped_total += len(unmapped)

        song_out = out_dir / p["id"]
        song_out.mkdir(parents=True, exist_ok=True)
        with open(song_out / "drums_labels.json", "w") as f:
            json.dump({"hits": hits}, f)

        n_hits_total += len(hits)
        total_hist.update(hist)
        split_hist[p["split"]].update(hist)

        manifest_songs.append({
            "id": p["id"],
            "split": p["split"],
            "charts": {"drums": True},
            "stems": {"drums": str(p["mix_path"].resolve())},
            # provenance (ignored by the trainer/preprocessor)
            "preset": p["preset"],
            "seed": p["seed"],
            "n_label_hits": len(hits),
        })
        if i % 50 == 0 or i == len(pieces):
            print(f"  [{i}/{len(pieces)}] {p['id']} ({p['split']}, "
                  f"{p['preset']}, {len(hits)} hits)")

    if unmapped_seen:
        print(f"WARNING: {n_unmapped_total} events had unmapped instrument "
              f"classes (dropped): {dict(unmapped_seen)}", file=sys.stderr)

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_corpus_dir": str(args.corpus_dir),
        "source_corpus_summary": str(args.corpus_dir / "corpus_summary.json"),
        "seed": SEED,
        "instrument_to_lane": {
            k: {"lane": v[0], "is_cymbal": v[1]} for k, v in INSTRUMENT_TO_LANE.items()
        },
        "songs": manifest_songs,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    n_train = sum(1 for s in manifest_songs if s["split"] == "train")
    n_test = sum(1 for s in manifest_songs if s["split"] == "test")
    print(f"\nWrote {out_dir / 'manifest.json'}: {len(manifest_songs)} pieces "
          f"(train {n_train} / test {n_test}), {n_hits_total} label hits total")
    print(f"Per-preset split:")
    by_preset_split = defaultdict(lambda: Counter())
    for s in manifest_songs:
        by_preset_split[s["preset"]][s["split"]] += 1
    for preset in sorted(by_preset_split):
        c = by_preset_split[preset]
        print(f"  {preset}: train {c['train']} / test {c['test']}")
    print(f"\nOverall 8-class histogram (n={n_hits_total}):")
    for cls, n in total_hist.most_common():
        print(f"  {cls}: {n} ({100.0 * n / n_hits_total:.1f}%)")
    print(f"\nTrain hits: {sum(split_hist['train'].values())}, "
          f"Test hits: {sum(split_hist['test'].values())}")


if __name__ == "__main__":
    main()
