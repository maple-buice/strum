#!/usr/bin/env python3
"""
Chart MIDI (notes.mid) → drums_labels.json for onset-classifier training.

Produces the per-song label file consumed by scripts/preprocess_onset_windows.py
and src/models/onset_classifier_dataset.py:

    {"hits": [{"time_ms": <float>, "lane": <0-4>, "is_cymbal": <bool>}, ...]}

Semantics are exactly those of src/preprocessing/parsers/midi_parser.MidiParser
(the parser the rest of the pipeline uses for ground truth):
  - Expert pro-drums notes 96-100 → lanes 0-4 (101 = GH orange → lane 4)
  - Rock Band tom markers 110/111/112 at the same tick flip lanes 2/3/4
    from cymbal (default) to tom
  - tick → ms via the full multi-event tempo map (piecewise integration)

The only difference from MidiParser.parse() is that the MIDI file is opened
with mido clip=True, so community charts with out-of-range data bytes parse
instead of crashing. Everything downstream of file opening reuses MidiParser's
own methods, so semantics cannot drift.

--verify prints per-lane hit counts and cross-checks them against an
independent re-parse of the MIDI written directly against mido (separate
tempo-map integration and tom-marker logic, no MidiParser code).

Usage:
    make_drums_labels.py notes.mid -o drums_labels.json
    make_drums_labels.py notes.mid --verify
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import mido

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing.parsers.midi_parser import DrumChart, DrumHit, MidiParser  # noqa: E402

LANE_NAMES = {0: "Kick", 1: "Red/Snare", 2: "Yellow", 3: "Blue", 4: "Green"}


def parse_drums_midi(midi_path: Path, five_lane_drums: bool = False) -> DrumChart:
    """Parse a chart MIDI with MidiParser semantics, but mido clip=True.

    Reuses MidiParser's internal methods (tempo extraction, drum-track
    discovery, hit parsing, tick→ms integration) on a clip=True MidiFile.
    """
    midi = mido.MidiFile(str(midi_path), clip=True)
    parser = MidiParser()
    ticks_per_beat = midi.ticks_per_beat

    tempo_events = parser._extract_tempo_events(midi, ticks_per_beat)
    time_signatures = parser._extract_time_signatures(midi, ticks_per_beat, tempo_events)
    drum_track = parser._find_drum_track(midi)
    if drum_track is None:
        hits = []
    else:
        hits = parser._parse_drum_track(
            drum_track, ticks_per_beat, tempo_events, five_lane_drums
        )

    return DrumChart(
        hits=hits,
        tempo_events=tempo_events,
        time_signatures=time_signatures,
        ticks_per_beat=ticks_per_beat,
    )


def chart_to_labels(chart: DrumChart) -> dict:
    """DrumChart → drums_labels.json payload."""
    return {
        "hits": [
            {"time_ms": h.time_ms, "lane": h.lane, "is_cymbal": h.is_cymbal}
            for h in chart.hits
        ]
    }


# ──────────────────────────────────────────────────────────────
# Independent verification (no MidiParser code)
# ──────────────────────────────────────────────────────────────

def _independent_recount(midi_path: Path, five_lane_drums: bool) -> list[tuple[float, int, bool]]:
    """Re-parse the MIDI directly with mido: returns [(time_ms, lane, is_cymbal)].

    Deliberately re-implements the pro-drums semantics from scratch so a bug
    in the MidiParser-based path (or in JSON serialization) is caught rather
    than reproduced.
    """
    midi = mido.MidiFile(str(midi_path), clip=True)
    tpb = midi.ticks_per_beat

    # Tempo map: (abs_tick, tempo_us) from every track, sorted by tick.
    tempos: list[tuple[int, int]] = []
    for track in midi.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                tempos.append((tick, msg.tempo))
    tempos.sort(key=lambda t: t[0])
    if not tempos:
        tempos = [(0, 500000)]  # 120 BPM default

    def tick_to_ms(tick: int) -> float:
        # Match the pipeline convention: the first tempo event's tempo also
        # governs ticks before its own position.
        time_ms = 0.0
        prev_tick = 0
        prev_tempo = tempos[0][1]
        for ev_tick, ev_tempo in tempos:
            if ev_tick >= tick:
                break
            # tempo is µs per beat → /1000 = ms per beat; /tpb = ms per tick
            time_ms += (ev_tick - prev_tick) * (prev_tempo / 1000.0) / tpb
            prev_tick = ev_tick
            prev_tempo = ev_tempo
        time_ms += (tick - prev_tick) * (prev_tempo / 1000.0) / tpb
        return time_ms

    # Drum track: by name, else first track containing a drum note.
    note_to_lane = {96: 0, 97: 1, 98: 2, 99: 3, 100: 4, 101: 4}
    drum_names = {"PART DRUMS", "PART REAL_DRUMS_PS", "drums", "Drums", "DRUMS"}
    drum_track = None
    for track in midi.tracks:
        if track.name in drum_names:
            drum_track = track
            break
    if drum_track is None:
        for track in midi.tracks:
            if any(m.type == "note_on" and m.note in note_to_lane for m in track):
                drum_track = track
                break
    if drum_track is None:
        return []

    events_by_tick: dict[int, list[tuple[int, int]]] = defaultdict(list)
    tick = 0
    for msg in drum_track:
        tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            events_by_tick[tick].append((msg.note, msg.velocity))

    out: list[tuple[float, int, bool]] = []
    for t in sorted(events_by_tick):
        notes_here = {n for n, _ in events_by_tick[t]}
        t_ms = tick_to_ms(t)
        for note, _vel in events_by_tick[t]:
            if note not in note_to_lane:
                continue
            lane = note_to_lane[note]
            if five_lane_drums:
                is_cymbal = note in (98, 101)
            else:
                if lane == 2:
                    is_cymbal = 110 not in notes_here
                elif lane == 3:
                    is_cymbal = 111 not in notes_here
                elif lane == 4:
                    is_cymbal = 112 not in notes_here
                else:
                    is_cymbal = False
            out.append((t_ms, lane, is_cymbal))
    out.sort(key=lambda h: h[0])
    return out


def print_lane_counts(hits: list[tuple[float, int, bool]], header: str) -> None:
    counts = Counter((lane, cym) for _, lane, cym in hits)
    print(f"\n{header} ({len(hits)} hits):")
    for lane in range(5):
        cym = counts.get((lane, True), 0)
        tom = counts.get((lane, False), 0)
        if lane <= 1:
            print(f"  lane {lane} ({LANE_NAMES[lane]:<10}): {tom + cym:>6}")
        else:
            print(f"  lane {lane} ({LANE_NAMES[lane]:<10}): {tom + cym:>6}  "
                  f"(cymbal {cym}, tom {tom})")


def verify(midi_path: Path, chart: DrumChart, five_lane_drums: bool) -> bool:
    """Cross-check the MidiParser-based parse against the independent recount."""
    primary = [(h.time_ms, h.lane, h.is_cymbal) for h in chart.hits]
    recount = _independent_recount(midi_path, five_lane_drums)

    print(f"MIDI: {midi_path}")
    print(f"ticks_per_beat: {chart.ticks_per_beat}")
    n_tempo = len(chart.tempo_events)
    print(f"tempo events: {n_tempo}" + (" (multi-tempo)" if n_tempo > 1 else ""))
    if chart.hits:
        print(f"chart duration: {chart.get_duration_ms() / 1000:.1f} s")

    print_lane_counts(primary, "Parsed labels (MidiParser semantics, clip=True)")
    print_lane_counts(recount, "Independent mido recount")

    ok = True
    if len(primary) != len(recount):
        print(f"\nMISMATCH: hit count {len(primary)} vs {len(recount)}")
        ok = False
    else:
        max_dt = 0.0
        mismatches = 0
        for (t1, l1, c1), (t2, l2, c2) in zip(primary, recount):
            if l1 != l2 or c1 != c2:
                mismatches += 1
            max_dt = max(max_dt, abs(t1 - t2))
        if mismatches:
            print(f"\nMISMATCH: {mismatches} hits differ in lane/is_cymbal")
            ok = False
        if max_dt > 0.001:
            print(f"\nMISMATCH: max |Δtime_ms| = {max_dt:.6f} (tempo-map skew)")
            ok = False
        else:
            print(f"\nmax |Δtime_ms| between parses: {max_dt:.6f}")

    print("VERIFY: OK — parses identical" if ok else "VERIFY: FAILED")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("midi", type=Path, help="Path to notes.mid")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output drums_labels.json path")
    ap.add_argument("--five-lane", action="store_true",
                    help="GH 5-lane drums mapping (song.ini five_lane_drums)")
    ap.add_argument("--verify", action="store_true",
                    help="Print per-lane counts + independent mido recount cross-check")
    args = ap.parse_args()

    if not args.verify and args.out is None:
        ap.error("either -o/--out or --verify is required")

    chart = parse_drums_midi(args.midi, five_lane_drums=args.five_lane)
    if not chart.hits:
        print(f"WARNING: no drum hits parsed from {args.midi}", file=sys.stderr)

    if args.verify:
        ok = verify(args.midi, chart, args.five_lane)
        if not ok:
            sys.exit(1)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(chart_to_labels(chart), f)
        print(f"Wrote {len(chart.hits)} hits → {args.out}")


if __name__ == "__main__":
    main()
