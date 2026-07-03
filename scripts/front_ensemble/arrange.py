"""Programmatic MIDI arrangement generator for front-ensemble / orchestral
percussion music.

A piece is generated from (preset, tempo, key, density, seed) and emitted as
one single-track MIDI file per instrument (sfizz_render takes one SFZ + one
MIDI) plus a labels dict: every note-on becomes an onset label
{t (sec), class, instrument, artic, midi, vel}. Label times are computed from
the *tick-quantized* MIDI times, so labels and audio share the same clock.

Ensemble presets (>=3 distinct virtual ensembles):
  concert_orchestral  timpani + glock/chimes/xylo + triangle/tambourine/
                      woodblock/crotales (orchestral percussion section)
  mallet_choir        marimba x2 + vibraphone + xylo + glock + latin aux
                      (indoor front-ensemble pit)
  full_field          MDL battery (snare/tenor/bass/cymbal lines) + pit
                      (marimba/xylo/timpani/glock/crotales/tambourine)
"""
import json
import random
import sys
from pathlib import Path

import mido

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SFZ_DIR

TPQ = 480
MODES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10],
         "dorian": [0, 2, 3, 5, 7, 9, 10], "mixolydian": [0, 2, 4, 5, 7, 9, 10]}
PROGRESSIONS = {"major": [[0, 5, 3, 4], [0, 3, 4, 0], [0, 5, 1, 4], [0, 4, 5, 3]],
                "minor": [[0, 5, 2, 6], [0, 6, 5, 4], [0, 3, 6, 4], [0, 5, 3, 4]]}
PROGRESSIONS["dorian"] = PROGRESSIONS["minor"]
PROGRESSIONS["mixolydian"] = PROGRESSIONS["major"]


class Piece:
    def __init__(self, preset, tempo, root, mode, density, seed, bars):
        self.preset, self.tempo, self.root, self.mode = preset, tempo, root, mode
        self.density, self.seed, self.bars = density, seed, bars
        self.rng = random.Random(seed)
        self.scale = MODES[mode]
        prog = self.rng.choice(PROGRESSIONS[mode])
        self.chords = [prog[b // 1 % len(prog)] for b in range(bars)]
        # events: list of (tick, instrument, midi, vel, dur_ticks, artic)
        self.events = []

    # -- helpers ----------------------------------------------------------
    def sec(self, tick):
        return tick * 60.0 / (self.tempo * TPQ)

    def scale_note(self, degree, octave):
        return self.root + 12 * octave + self.scale[degree % 7] + 12 * (degree // 7)

    def chord_tones(self, bar):
        d = self.chords[bar]
        return [d, d + 2, d + 4]

    def fit(self, midi, lo, hi):
        while midi < lo:
            midi += 12
        while midi > hi:
            midi -= 12
        return midi if midi >= lo else lo

    def add(self, tick, inst, midi, vel, dur=TPQ // 8, artic="hit",
            jitter=True):
        if jitter:
            tick += round(self.rng.gauss(0, 2.5))
            tick = max(0, tick)
        vel = max(20, min(127, vel + self.rng.randint(-6, 6)))
        self.events.append((tick, inst, midi, vel, dur, artic))

    def bar_tick(self, bar, beat=0.0):
        return round((bar * 4 + beat) * TPQ)

    # -- roles -------------------------------------------------------------
    def role_mallet_ostinato(self, inst, lo, hi, subdiv=4):
        pat_len = self.rng.choice([1, 2])  # bars
        for bar in range(self.bars):
            if bar % pat_len == 0 or self.rng.random() < 0.15:
                tones = [self.fit(self.scale_note(d, 3), lo, hi)
                         for d in self.chord_tones(bar)]
                seq = [self.rng.choice(tones + [tones[0] + 12 if tones[0] + 12 <= hi else tones[0]])
                       for _ in range(4 * subdiv)]
            else:
                tones = [self.fit(self.scale_note(d, 3), lo, hi)
                         for d in self.chord_tones(bar)]
            for i in range(4 * subdiv):
                if self.rng.random() > self.density:
                    continue
                vel = 96 if i % subdiv == 0 else 72
                self.add(self.bar_tick(bar, i / subdiv), inst,
                         seq[i % len(seq)], vel)

    def role_block_chords(self, inst, lo, hi, beats=2):
        for bar in range(self.bars):
            for beat in (range(0, 4, beats)):
                if self.rng.random() > 0.8 * self.density + 0.15:
                    continue
                for d in self.chord_tones(bar):
                    m = self.fit(self.scale_note(d, 3), lo, hi)
                    self.add(self.bar_tick(bar, beat), inst, m, 70,
                             dur=beats * TPQ, artic="chord")

    def role_melody(self, inst, lo, hi, vel=100):
        deg = self.rng.randint(7, 14)
        self.melody_notes = []  # (tick, midi) for doubling
        for bar in range(self.bars):
            rhythm = self.rng.choice([[0, 1, 2, 3], [0, 1.5, 2, 3.5], [0, 0.5, 1, 2, 2.5, 3],
                                      [0, 2], [0, 1, 2, 3, 3.5]])
            for beat in rhythm:
                if self.rng.random() > self.density:
                    continue
                if beat == 0:  # chord tone on downbeat
                    deg = self.rng.choice(self.chord_tones(bar)) + 7
                else:
                    deg += self.rng.choice([-2, -1, -1, 1, 1, 2])
                m = self.fit(self.scale_note(deg, 3), lo, hi)
                t = self.bar_tick(bar, beat)
                self.add(t, inst, m, vel)
                self.melody_notes.append((t, m))

    def role_melody_double(self, inst, lo, hi, keep=0.5):
        for t, m in getattr(self, "melody_notes", []):
            if self.rng.random() < keep:
                self.add(t, inst, self.fit(m + 12, lo, hi), 84, jitter=False)

    def role_timpani(self, hit="timpani_hit", roll="timpani_roll",
                     lo=33, hi=56, active=1.0):
        for bar in range(self.bars):
            if self.rng.random() > active:
                continue
            root = self.fit(self.root + self.scale[self.chords[bar] % 7], lo, hi)
            fifth = self.fit(root + 7, lo, hi)
            self.add(self.bar_tick(bar, 0), hit, root, 105)
            if self.rng.random() < 0.6 * self.density:
                self.add(self.bar_tick(bar, 2), hit, fifth, 88)
            if bar % 4 == 3:  # phrase end: 8th pickups or roll
                if self.rng.random() < 0.5:
                    for k, beat in enumerate([3.0, 3.5]):
                        self.add(self.bar_tick(bar, beat), hit,
                                 fifth if k == 0 else root, 92 + 8 * k)
                else:
                    self.add(self.bar_tick(bar, 2), roll, root, 100,
                             dur=2 * TPQ, artic="roll")

    def role_aux(self, inst, artic_key, pattern, vel=90, dur=TPQ // 8,
                 artic="hit", prob=1.0):
        """pattern: beats within a bar; artic_key: articulation name."""
        for bar in range(self.bars):
            for beat in pattern(bar, self.rng):
                if self.rng.random() > prob * self.density:
                    continue
                self.add(self.bar_tick(bar, beat), inst, artic_key, vel,
                         dur=dur, artic=artic)

    # battery ---------------------------------------------------------------
    def role_snareline(self, inst="snareline"):
        accents = self.rng.choice([[0, 3, 6, 10, 12], [0, 4, 7, 10, 14],
                                   [0, 5, 8, 12], [0, 2, 4, 8, 11, 14]])
        for bar in range(self.bars):
            if bar % 4 == 3 and self.rng.random() < 0.5:  # phrase-end roll
                self.add(self.bar_tick(bar, 0), inst, "roll", 96,
                         dur=3 * TPQ, artic="roll")
                self.add(self.bar_tick(bar, 3), inst, "hit", 115, artic="accent")
                continue
            for i in range(16):
                if self.rng.random() > self.density * 0.95:
                    continue
                if i in accents:
                    art = "hit" if self.rng.random() < 0.8 else "rimshot"
                    self.add(self.bar_tick(bar, i / 4), inst, art, 112, artic=art)
                else:
                    self.add(self.bar_tick(bar, i / 4), inst, "hit", 62)
            if self.rng.random() < 0.25:
                self.add(self.bar_tick(bar, self.rng.choice([1.75, 3.75])),
                         inst, "crush_short", 84, artic="crush")

    def role_tenorline(self, inst="tenorline"):
        drums = ["hit_d1", "hit_d2", "hit_d3", "hit_d4"]
        for bar in range(self.bars):
            if self.rng.random() > 0.75:
                continue
            pat = self.rng.choice([[0, 0.5, 1, 1.5, 2, 3], [0, 1, 1.5, 2.5, 3, 3.5],
                                   [0, 0.5, 1.5, 2, 2.5, 3.5]])
            d = self.rng.randrange(4)
            for beat in pat:
                if self.rng.random() > self.density:
                    continue
                d = (d + self.rng.choice([-1, 0, 1])) % 4
                vel = 100 if beat == int(beat) else 74
                self.add(self.bar_tick(bar, beat), inst, drums[d], vel)

    def role_bassline(self, inst="bassline"):
        for bar in range(self.bars):
            self.add(self.bar_tick(bar, 0), inst, "unison", 110)
            if self.rng.random() < 0.7 * self.density:
                self.add(self.bar_tick(bar, 2), inst, "unison", 96)
            if self.rng.random() < 0.6 * self.density:  # split 8th run
                start = self.rng.choice([2.5, 3.0])
                for k, d in enumerate(self.rng.sample(["d1", "d2", "d3", "d4"], 3)):
                    self.add(self.bar_tick(bar, start + 0.25 * k), inst, d, 88)

    def role_cymballine(self, inst="cymballine"):
        for bar in range(self.bars):
            if bar % 4 == 0:
                self.add(self.bar_tick(bar, 0), inst, "crash_fff", 115,
                         dur=2 * TPQ, artic="crash")
            elif self.rng.random() < 0.3 * self.density:
                self.add(self.bar_tick(bar, 2), inst, "crash_mp", 80,
                         dur=TPQ, artic="crash")

    def role_crotales(self, inst="crotales", lo=84, hi=108):
        for bar in range(self.bars):
            if bar % 2 == 0 and self.rng.random() < 0.7 * self.density:
                d = self.rng.choice(self.chord_tones(bar))
                m = self.fit(self.scale_note(d, 6), lo, hi)
                beat = self.rng.choice([0, 0, 2, 3])
                self.add(self.bar_tick(bar, beat), inst, m, 100)

    def role_chimes(self, inst="chimes", lo=48, hi=64):
        for bar in range(self.bars):
            if bar % 2 == 0 or self.rng.random() < 0.3:
                root = self.fit(self.root + self.scale[self.chords[bar] % 7], lo, hi)
                self.add(self.bar_tick(bar, 0), inst, root, 96, dur=4 * TPQ)
                if self.rng.random() < 0.4 * self.density:
                    self.add(self.bar_tick(bar, 2), inst,
                             self.fit(root + 7, lo, hi), 84, dur=2 * TPQ)


# ---------------------------------------------------------------------------
# aux rhythm pattern factories
# ---------------------------------------------------------------------------
def every(beats):
    return lambda bar, rng: beats


def eighths():
    return lambda bar, rng: [i / 2 for i in range(8)]


def sixteenth_shaker():
    return lambda bar, rng: [i / 4 for i in range(16)]


def offbeats():
    return lambda bar, rng: [0.5, 1.5, 2.5, 3.5]


def backbeats():
    return lambda bar, rng: [1, 3]


def sparse_downbeat(period=2):
    return lambda bar, rng: ([0] if bar % period == 0 else [])


def clave():
    return lambda bar, rng: ([0, 0.75, 1.5] if bar % 2 == 0 else [2, 3])


def sparse_motif():
    return lambda bar, rng: (sorted(rng.sample([1.75, 2.25, 3.25, 3.5, 3.75], 2))
                             if rng.random() < 0.5 else [])


# ---------------------------------------------------------------------------
def generate(preset, seed, mapping, duration_range=(32, 56)):
    rng = random.Random(seed)

    def rg(inst, lo, hi):
        """Intersect a desired register with the instrument's sampled range."""
        mlo, mhi = mapping["instruments"][inst]["range"]
        lo2, hi2 = max(lo, mlo), min(hi, mhi)
        return (lo2, hi2) if lo2 < hi2 else (mlo, mhi)

    if preset == "concert_orchestral":
        tempo = rng.randint(84, 116)
        density = rng.uniform(0.55, 0.85)
    elif preset == "mallet_choir":
        tempo = rng.randint(100, 138)
        density = rng.uniform(0.6, 0.95)
    elif preset == "full_field":
        tempo = rng.randint(120, 160)
        density = rng.uniform(0.6, 0.95)
    else:
        raise ValueError(preset)
    root = rng.randrange(48, 60)
    mode = rng.choice(list(MODES))
    target = rng.uniform(*duration_range)  # seconds
    bars = max(8, 4 * round(target * tempo / 60 / 4 / 4))

    p = Piece(preset, tempo, root, mode, density, seed, bars)

    if preset == "concert_orchestral":
        p.role_timpani(lo=rg("timpani_hit",33,56)[0], hi=rg("timpani_hit",33,56)[1])
        p.role_melody("xylophone", *rg("xylophone",55,96))
        p.role_melody_double("glockenspiel", *rg("glockenspiel",67,96), keep=0.6)
        p.role_chimes(lo=rg("chimes",48,64)[0], hi=rg("chimes",48,64)[1])
        p.role_crotales(lo=rg("crotales",84,108)[0], hi=rg("crotales",84,108)[1])
        p.role_aux("triangle", "Triangle6_Hit", sparse_downbeat(2), vel=95)
        p.role_aux("triangle", "Triangle6_Roll",
                   lambda bar, rng: ([3] if bar % 8 == 7 else []),
                   vel=90, dur=TPQ, artic="roll")
        p.role_aux("tambourine", "Tamb1_Hit", backbeats(), vel=88, prob=0.8)
        p.role_aux("woodblock", "wood_click", sparse_motif(), vel=92)
    elif preset == "mallet_choir":
        p.role_mallet_ostinato("marimba", *rg("marimba",36,72), subdiv=rng.choice([2, 4]))
        p.role_block_chords("marimba2", *rg("marimba",48,84), beats=2)
        p.role_block_chords("vibraphone", *rg("vibraphone",50,80), beats=rng.choice([2, 4]))
        p.role_melody("xylophone", *rg("xylophone",60,96))
        p.role_melody_double("glockenspiel", *rg("glockenspiel",67,96), keep=0.5)
        p.role_aux("shaker_small", "Mid_ShakerLowFaster_Down", eighths(),
                   vel=80, prob=0.9, artic="shake")
        p.role_aux("cabasa", "Cabasa1_Rub", offbeats(), vel=84, artic="rub",
                   prob=0.7)
        p.role_aux("cowbell", "Cowbell1_Hit", clave(), vel=92, prob=0.8)
        p.role_aux("woodblock", "wood_click", sparse_motif(), vel=90)
    elif preset == "full_field":
        p.role_snareline()
        p.role_tenorline()
        p.role_bassline()
        p.role_cymballine()
        p.role_mallet_ostinato("marimba", *rg("marimba",36,72), subdiv=2)
        p.role_melody("xylophone", *rg("xylophone",60,96), vel=105)
        p.role_timpani(lo=rg("timpani_hit",33,56)[0], hi=rg("timpani_hit",33,56)[1], active=0.4)
        p.role_crotales(lo=rg("crotales",84,108)[0], hi=rg("crotales",84,108)[1])
        p.role_aux("tambourine2", "Tamb2_Shake", eighths(), vel=78,
                   prob=0.5, artic="shake")

    # marimba2 is the same SFZ as marimba, separate track
    return p


def write_outputs(p, outdir, mapping):
    outdir = Path(outdir)
    (outdir / "midi").mkdir(parents=True, exist_ok=True)
    insts = sorted({e[1] for e in p.events})
    labels = []
    end_tick = max(e[0] + e[4] for e in p.events)
    tempo_us = round(60_000_000 / p.tempo)
    for inst in insts:
        base = "marimba" if inst == "marimba2" else inst
        info = mapping["instruments"][base]
        evs = sorted([e for e in p.events if e[1] == inst])
        mid = mido.MidiFile(ticks_per_beat=TPQ)
        tr = mido.MidiTrack()
        mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
        msgs = []
        for tick, _, key, vel, dur, artic in evs:
            midi_key = key if isinstance(key, int) else info["artics"][key]
            msgs.append((tick, 1, midi_key, vel))          # 1 = note_on
            msgs.append((tick + dur, 0, midi_key, 0))      # 0 = note_off
            labels.append({"t": round(p.sec(tick), 6), "class": info["class"],
                           "instrument": inst, "artic": artic,
                           "midi": midi_key, "vel": vel})
        msgs.sort()
        prev = 0
        for tick, kind, key, vel in msgs:
            m = mido.Message("note_on" if kind else "note_off", note=key,
                             velocity=vel, time=tick - prev)
            tr.append(m)
            prev = tick
        tr.append(mido.MetaMessage("end_of_track", time=end_tick - prev + 2 * TPQ))
        mid.save(outdir / "midi" / f"{inst}.mid")

    labels.sort(key=lambda e: e["t"])
    meta = {"preset": p.preset, "tempo": p.tempo, "root": p.root,
            "mode": p.mode, "density": round(p.density, 3), "seed": p.seed,
            "bars": p.bars, "duration_sec": round(p.sec(end_tick), 3),
            "n_onsets": len(labels), "events": labels}
    (outdir / "labels.json").write_text(json.dumps(meta, indent=1))
    return meta


def main():
    mapping = json.loads((SFZ_DIR / "mapping.json").read_text())
    preset, seed, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    p = generate(preset, seed, mapping)
    meta = write_outputs(p, outdir, mapping)
    print(f"{outdir}: {preset} tempo={meta['tempo']} bars={meta['bars']} "
          f"dur={meta['duration_sec']}s onsets={meta['n_onsets']}")


if __name__ == "__main__":
    main()
