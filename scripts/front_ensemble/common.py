"""Shared paths & helpers for the front-ensemble synthetic-data pipeline.

Asset layout (samples, SFZ instruments, render output) lives outside the
repo tree; point STRUM_FE_ASSETS at it (default: <repo>/ws5_assets).
sfizz_render is resolved from $SFIZZ_RENDER, then PATH, then
<assets>/pipeline/bin/.
"""
import os
import re
import shutil
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent
_REPO = PIPELINE.parent.parent                # scripts/front_ensemble/ -> repo
ASSETS = Path(os.environ.get("STRUM_FE_ASSETS", _REPO / "ws5_assets"))
SAMPLES = ASSETS / "samples"
SFZ_DIR = ASSETS / "sfz"
PILOT_DIR = ASSETS / "pilot"
SFIZZ_RENDER = Path(
    os.environ.get("SFIZZ_RENDER")
    or shutil.which("sfizz_render")
    or (ASSETS / "pipeline" / "bin" / "sfizz_render")
)

VCSL = SAMPLES / "VCSL"
VCSL_SI = VCSL / "Idiophones" / "Struck Idiophones"
VCSL_SM = VCSL / "Membranophones" / "Struck Membranophones"
MDL_SOUND = SAMPLES / "MDL" / "resources" / "sound"
IOWA = SAMPLES / "Iowa_MIS"

_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_to_midi(tok: str) -> int:
    """'C4'->60, 'G#6'->92, 'Ab6'->92, 'A#3'->58. MIDI C4=60 convention."""
    m = re.fullmatch(r"([A-G])([#b]?)(-?\d)", tok)
    if not m:
        raise ValueError(f"bad note token {tok!r}")
    s = _SEMITONE[m.group(1)] + {"#": 1, "b": -1, "": 0}[m.group(2)]
    return 12 * (int(m.group(3)) + 1) + s


def midi_to_note(m: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[m % 12]}{m // 12 - 1}"
