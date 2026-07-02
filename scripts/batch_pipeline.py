"""
Batch Pipeline - Process multiple songs through the full STRUM pipeline.

Takes a folder of audio files and creates complete Clone Hero/YARG chart folders
with all instruments: Drums, Guitar, Bass, Vocals, Keys (including Pro Keys).

Usage:
    python scripts/batch_pipeline.py input/ output/
    python scripts/batch_pipeline.py input/ output/ --no-vocals --no-keys
    python scripts/batch_pipeline.py input/ output/ --continue  # Resume interrupted batch
"""

import argparse
import sys
from pathlib import Path

# Add project root and scripts dir to path for imports
_project_root = Path(__file__).parent.parent
_scripts_dir = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import shutil
import subprocess
import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import librosa
import mido

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class SongResult:
    """Result of processing a single song."""
    input_path: str
    output_path: str
    success: bool
    error: Optional[str] = None
    drums_hits: int = 0
    guitar_notes: int = 0
    bass_notes: int = 0
    vocals_phrases: int = 0
    keys_notes: int = 0


class BatchPipeline:
    """
    Complete batch pipeline for STRUM.
    
    Processes audio files through:
    1. Demucs stem separation
    2. Drums transcription (V11 model with tom/cymbal correction)
    3. Guitar transcription (Basic Pitch + rules)
    4. Bass transcription (Basic Pitch + rules)
    5. Vocals transcription (Whisper + pitch)
    6. Keys transcription (Basic Pitch)
    7. Chart enhancement (Star Power, difficulty, sections)
    8. Song.ini generation
    """
    
    def __init__(
        self,
        output_dir: Path,
        drums_checkpoint: str = "checkpoints/drums_v11/best.pt",
        demucs_model: str = "htdemucs_6s",  # 6-stem model: drums, bass, vocals, guitar, piano, other
        include_drums: bool = True,
        include_guitar: bool = True,
        include_bass: bool = True,
        include_vocals: bool = True,
        include_keys: bool = False,
        include_video: bool = False,
        device: str = None,
        use_v11: bool = True,  # Use V11 with tom/cymbal correction
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.drums_checkpoint = drums_checkpoint
        self.demucs_model = demucs_model
        self.include_drums = include_drums
        self.include_guitar = include_guitar
        self.include_bass = include_bass
        self.include_vocals = include_vocals
        self.include_keys = include_keys
        self.include_video = include_video
        self.device = device
        self.use_v11 = use_v11
        
        # Lazy-load engines
        self._drums_models = None
        self._vocals_charter = None
        self._keys_charter = None
        self.tomcym = None
    
    @property
    def drums_models(self):
        """Lazy load drums V14 onset detector + ensemble + phase3 + tom_refinement."""
        if self._drums_models is None and self.include_drums:
            try:
                from batch_infer_hybrid import (
                    load_v14_onset_detector, load_ensemble, load_tomcym_classifier,
                    load_phase3_model, load_tom_refinement, pick_device,
                )
                import torch
                device = torch.device(self.device) if self.device else pick_device()
                logger.info("Loading drums V14 onset detector + ensemble + phase3 + tom_refinement...")
                v14_model = load_v14_onset_detector(device)
                ensemble = load_ensemble(device)
                tomcym = load_tomcym_classifier(device)
                phase3_model = load_phase3_model(device)
                tom_refinement = load_tom_refinement(device)
                self.tomcym = tomcym
                self._drums_models = {
                    "v14": v14_model,
                    "ensemble": ensemble,
                    "phase3": phase3_model,
                    "tom_refinement": tom_refinement,
                    "device": device,
                }
            except Exception as e:
                logger.warning(f"  ⚠ Could not load drums models: {e}")
                self._drums_models = {}
        return self._drums_models
    
    @property
    def vocals_charter(self):
        """Lazy load vocals charter."""
        if self._vocals_charter is None and self.include_vocals:
            from vocals_charter import VocalsCharter
            logger.info("Loading vocals charter (Whisper)...")
            self._vocals_charter = VocalsCharter(whisper_model="medium")
        return self._vocals_charter
    
    @property
    def keys_charter(self):
        """Lazy load keys charter."""
        if self._keys_charter is None and self.include_keys:
            from keys_charter import KeysCharter
            logger.info("Loading keys charter...")
            # Tightened threshold: 0.3 -> 0.7. Default 0.3 fires on any tonal
            # content in the 'other' stem (guitar+vocals bleed), producing ~4x
            # GT keys note count across our held-out set.
            self._keys_charter = KeysCharter(detection_threshold=0.7)
        return self._keys_charter
    
    def _quick_musicbrainz_check(self, artist: str, title: str) -> bool:
        """
        Quick MusicBrainz lookup to verify artist/title combo exists.
        Returns True if a recording is found.
        """
        import urllib.request
        import urllib.parse
        import json
        
        try:
            query = f'recording:"{title}" AND artist:"{artist}"'
            url = f"https://musicbrainz.org/ws/2/recording?query={urllib.parse.quote(query)}&fmt=json&limit=1"
            
            req = urllib.request.Request(url, headers={
                'User-Agent': 'STRUM/1.0 (https://github.com/strum)'
            })
            
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            # Check if we got a result with matching artist
            if data.get('recordings'):
                recording = data['recordings'][0]
                # Verify the artist name matches somewhat
                if recording.get('artist-credit'):
                    found_artist = recording['artist-credit'][0].get('name', '').lower()
                    if artist.lower() in found_artist or found_artist in artist.lower():
                        return True
            return False
        except Exception:
            return False
    
    def parse_filename(self, path: Path) -> Tuple[str, str]:
        """Parse artist and title from filename, validating via MusicBrainz if possible."""
        import time
        
        name = path.stem
        
        # Try "X - Y" format
        if " - " in name:
            parts = name.split(" - ", 1)
            part1, part2 = parts[0].strip(), parts[1].strip()
            
            # Try both orders and validate with MusicBrainz
            # Order 1: "Title - Artist" (part1=title, part2=artist)
            order1_valid = self._quick_musicbrainz_check(part2, part1)
            
            if order1_valid:
                logger.debug(f"MusicBrainz confirmed: Artist='{part2}', Title='{part1}'")
                return part2, part1  # artist, title
            
            time.sleep(1)  # Rate limit
            
            # Order 2: "Artist - Title" (part1=artist, part2=title)
            order2_valid = self._quick_musicbrainz_check(part1, part2)
            
            if order2_valid:
                logger.debug(f"MusicBrainz confirmed: Artist='{part1}', Title='{part2}'")
                return part1, part2  # artist, title
            
            # Neither validated - default to "Title - Artist" format
            logger.debug(f"MusicBrainz couldn't validate, defaulting to Title-Artist format")
            return part2, part1  # artist, title
        
        # Try "(Artist)" at end
        match = re.search(r'^(.+?)\s*\(([^)]+)\)\s*$', name)
        if match:
            return match.group(2).strip(), match.group(1).strip()
        
        return "Unknown Artist", name
    
    def analyze_audio(self, audio_path: Path, artist: str = '', title: str = '') -> Dict:
        """Analyze audio for tempo, duration, and metadata."""
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        duration_sec = len(y) / sr
        
        # Use grid-alignment BPM refinement from drums pipeline if available.
        # IMPORTANT: also propagate phase_offset_ms and the multi-segment
        # tempo_events list so chart timing aligns with the audio. Without
        # phase alignment, bar lines render at a fixed offset from
        # downbeats and notes drift visibly off-beat against the backing track.
        phase_offset_ms = 0.0
        tempo_events_full = None
        try:
            from batch_infer_hybrid import analyze_audio as drums_analyze
            audio_info = drums_analyze(audio_path)
            tempo = audio_info["tempo_bpm"]
            phase_offset_ms = float(audio_info.get("phase_offset_ms", 0.0))
            tempo_events_full = audio_info.get("tempo_events", None)
            logger.info(f"  BPM (grid-aligned): {tempo:.1f}  phase_offset={phase_offset_ms:.1f}ms")
        except Exception:
            # Fallback: librosa.feature.rhythm.tempo (not beat_track which gives subharmonics)
            tempo = librosa.feature.rhythm.tempo(y=y, sr=sr)
            if hasattr(tempo, '__len__'):
                tempo = float(tempo[0])
            logger.info(f"  BPM (librosa): {tempo:.1f}")
        
        # Find good preview point (skip intro)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        if len(onset_frames) > 10:
            preview_frame = onset_frames[10]
            preview_sec = librosa.frames_to_time(preview_frame, sr=sr)
        else:
            preview_sec = min(30, duration_sec * 0.2)
        
        # Extract metadata from file
        metadata = self._extract_metadata(audio_path)
        
        # Fallback to MusicBrainz for missing metadata
        if artist and title and (not metadata['album'] or not metadata['year'] or not metadata['genre']):
            mb_metadata = self._fetch_musicbrainz_metadata(artist, title)
            if not metadata['album'] and mb_metadata['album']:
                metadata['album'] = mb_metadata['album']
            if not metadata['year'] and mb_metadata['year']:
                metadata['year'] = mb_metadata['year']
            if not metadata['genre'] and mb_metadata['genre']:
                metadata['genre'] = mb_metadata['genre']
        
        return {
            'tempo_bpm': round(tempo, 2),  # Keep 0.01 BPM precision (rounding to int caused 500ms+ drift)
            'duration_ms': int(duration_sec * 1000),
            'duration_sec': duration_sec,
            'preview_start_ms': int(preview_sec * 1000),
            'phase_offset_ms': phase_offset_ms,
            'tempo_events': tempo_events_full,
            'album': metadata.get('album', ''),
            'year': metadata.get('year', ''),
            'genre': metadata.get('genre', ''),
        }
    
    def _extract_metadata(self, audio_path: Path) -> Dict[str, str]:
        """Extract album, year, and genre from audio file metadata."""
        metadata = {'album': '', 'year': '', 'genre': ''}
        
        try:
            from mutagen import File
            from mutagen.id3 import ID3
            from mutagen.mp3 import MP3
            from mutagen.flac import FLAC
            from mutagen.mp4 import MP4
            
            audio = File(str(audio_path))
            if audio is None:
                return metadata
            
            # MP3 with ID3 tags
            if isinstance(audio, MP3) or hasattr(audio, 'tags'):
                tags = audio.tags if hasattr(audio, 'tags') else audio
                if tags:
                    # Album
                    for key in ['TALB', 'album', '\xa9alb']:
                        if key in tags:
                            val = tags[key]
                            metadata['album'] = str(val[0] if hasattr(val, '__getitem__') else val).strip()
                            break
                    
                    # Year/Date
                    for key in ['TDRC', 'TYER', 'date', '\xa9day']:
                        if key in tags:
                            val = tags[key]
                            year_str = str(val[0] if hasattr(val, '__getitem__') else val).strip()
                            # Extract just the year (first 4 digits)
                            import re
                            year_match = re.search(r'(\d{4})', year_str)
                            if year_match:
                                metadata['year'] = year_match.group(1)
                            break
                    
                    # Genre
                    for key in ['TCON', 'genre', '\xa9gen']:
                        if key in tags:
                            val = tags[key]
                            metadata['genre'] = str(val[0] if hasattr(val, '__getitem__') else val).strip()
                            break
            
            # FLAC
            elif isinstance(audio, FLAC):
                metadata['album'] = audio.get('album', [''])[0]
                metadata['year'] = audio.get('date', [''])[0][:4] if audio.get('date') else ''
                metadata['genre'] = audio.get('genre', [''])[0]
            
            # MP4/M4A
            elif isinstance(audio, MP4):
                metadata['album'] = audio.get('\xa9alb', [''])[0]
                metadata['year'] = audio.get('\xa9day', [''])[0][:4] if audio.get('\xa9day') else ''
                metadata['genre'] = audio.get('\xa9gen', [''])[0]
                
        except Exception as e:
            logger.debug(f"Metadata extraction failed: {e}")
        
        return metadata
    
    def _fetch_musicbrainz_metadata(self, artist: str, title: str) -> Dict[str, str]:
        """
        Fetch album, year, and genre from MusicBrainz API.
        Free API, no auth required, but rate limited (1 req/sec).
        """
        import urllib.request
        import urllib.parse
        import json
        import time
        
        metadata = {'album': '', 'year': '', 'genre': ''}
        
        try:
            # Search for recording by artist and title
            query = f'recording:"{title}" AND artist:"{artist}"'
            url = f"https://musicbrainz.org/ws/2/recording?query={urllib.parse.quote(query)}&fmt=json&limit=1"
            
            req = urllib.request.Request(url, headers={
                'User-Agent': 'STRUM/1.0 (https://github.com/strum)'
            })
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            if data.get('recordings'):
                recording = data['recordings'][0]
                
                # Get release (album) info
                if recording.get('releases'):
                    release = recording['releases'][0]
                    metadata['album'] = release.get('title', '')
                    
                    # Get release date
                    date = release.get('date', '')
                    if date and len(date) >= 4:
                        metadata['year'] = date[:4]
                
                # Get genre/tags from artist
                if recording.get('artist-credit'):
                    artist_id = recording['artist-credit'][0].get('artist', {}).get('id')
                    if artist_id:
                        time.sleep(1)  # Rate limit
                        tag_url = f"https://musicbrainz.org/ws/2/artist/{artist_id}?inc=tags&fmt=json"
                        tag_req = urllib.request.Request(tag_url, headers={
                            'User-Agent': 'STRUM/1.0 (https://github.com/strum)'
                        })
                        with urllib.request.urlopen(tag_req, timeout=10) as tag_response:
                            tag_data = json.loads(tag_response.read().decode('utf-8'))
                        
                        # Get most popular tag as genre
                        tags = tag_data.get('tags', [])
                        if tags:
                            # Sort by count, get top tag
                            tags.sort(key=lambda t: t.get('count', 0), reverse=True)
                            metadata['genre'] = tags[0].get('name', '').title()
                            
        except Exception as e:
            logger.debug(f"MusicBrainz lookup failed: {e}")
        
        return metadata

    def separate_stems(self, audio_path: Path, work_dir: Path) -> Dict[str, Path]:
        """Separate stems using Demucs Python API (avoids torchaudio save bug).

        Runs the configured `demucs_model` (default htdemucs_6s) for guitar/bass/
        vocals/keys/other. If `STRUM_DRUMS_DEMUCS` is set (default 'htdemucs_ft'),
        runs that model in addition and uses ITS drum stem — matches what the
        production drums pipeline (`batch_infer_hybrid`) and tom_refinement_demucs
        checkpoint were tuned on.
        """
        import os
        import soundfile as sf
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torch

        logger.info("  Separating stems with Demucs...")

        stem_dir = work_dir / "stems"
        stem_dir.mkdir(parents=True, exist_ok=True)

        drums_demucs = os.environ.get("STRUM_DRUMS_DEMUCS", "")
        use_separate_drums = bool(drums_demucs) and drums_demucs != self.demucs_model

        # Check if stems already exist
        expected_stems = ["drums", "bass", "other", "vocals"]
        if all((stem_dir / f"{s}.wav").exists() for s in expected_stems):
            logger.info("    Stems already exist, skipping separation")
            stems = {s: stem_dir / f"{s}.wav" for s in expected_stems}
            for extra in ["guitar", "piano"]:
                p = stem_dir / f"{extra}.wav"
                if p.exists():
                    stems[extra] = p
            return stems

        from batch_infer_hybrid import pick_device
        device = pick_device()

        def _run_model(model_name: str):
            model = get_model(model_name)
            model.eval()
            model.to(device)
            y, sr_orig = librosa.load(str(audio_path), sr=None, mono=False)
            if y.ndim == 1:
                y = np.stack([y, y])
            target_sr = model.samplerate
            if sr_orig != target_sr:
                y = librosa.resample(y, orig_sr=sr_orig, target_sr=target_sr)
            waveform = torch.from_numpy(y).float()
            ref = waveform.mean(0)
            waveform_norm = (waveform - ref.mean()) / ref.std()
            sources = apply_model(model, waveform_norm[None].to(device), device=device)[0]
            sources = sources * ref.std() + ref.mean()
            return model.sources, sources, target_sr

        # Primary model (6s for guitar/piano/etc)
        names, sources, target_sr = _run_model(self.demucs_model)
        stems = {}
        for i, name in enumerate(names):
            stem = sources[i].cpu().numpy()
            path = stem_dir / f"{name}.wav"
            sf.write(str(path), stem.T, target_sr)
            stems[name] = path
        logger.info(f"    Separated stems ({self.demucs_model}): {list(stems.keys())}")

        # Drums-specialist pass (htdemucs_ft) — overwrites drums.wav with the
        # higher-quality drum stem that production drums pipeline expects.
        if use_separate_drums:
            try:
                logger.info(f"    Re-separating drums with {drums_demucs}...")
                names_d, sources_d, target_sr_d = _run_model(drums_demucs)
                if "drums" in names_d:
                    idx = list(names_d).index("drums")
                    stem = sources_d[idx].cpu().numpy()
                    path = stem_dir / "drums.wav"
                    sf.write(str(path), stem.T, target_sr_d)
                    stems["drums"] = path
                    logger.info(f"    Drums stem replaced from {drums_demucs}")
            except Exception as e:
                logger.warning(f"    ⚠ {drums_demucs} drums pass failed, keeping {self.demucs_model} drums: {e}")

        return stems
    
    def transcribe_drums(self, drums_stem: Path, tempo_bpm: float,
                         song_folder: Path | None = None,
                         phase_offset_ms: float = 0.0):
        """Transcribe drums using the production batch_infer_hybrid pipeline.

        Mirrors `batch_infer_hybrid.process_song` post-separation flow so that
        batch_pipeline drums quality matches the standalone production pipeline.

        Args:
            phase_offset_ms: Audio phase offset.  Hits are shifted by this
                amount BEFORE quantization so that postprocess snaps them to
                a grid anchored at MIDI tick 0 (= bar 1 in Clone Hero).
                Without this, postprocess auto-derives its own grid origin
                from hit times, the audio is trimmed by phase_offset_ms
                later, and the two grids drift out of phase by up to half
                a 96th-note (≈15 ms at 84 BPM) — visible as notes sitting
                slightly in front of or behind the bar lines in CH editors.
        """
        if not self.include_drums:
            return None

        logger.info("  Transcribing drums (V14 hybrid + phase3 + tom_refinement)...")

        try:
            models = self.drums_models
            if not models or "v14" not in models:
                logger.warning("  ⚠ Drums models not loaded, skipping")
                return None

            import os
            from batch_infer_hybrid import (
                detect_onsets_v14, extract_onset_windows, classify_onsets_ensemble,
                build_context_vectors, build_chart, postprocess_chart,
                _compute_spectral_centroid_features, _compute_onset_rms,
                run_phase3_inference, phase3_reclassify, phase3_onset_rescue,
                phase3_cymbal_cooccurrence_rescue, spectral_reclassify,
                apply_tom_refinement_filter, apply_cymbal_to_tom_rescue,
                separate_drums_6stem, apply_drumsep_lane_arbiter,
                apply_drumsep_hits_arbiter,
                apply_snare_in_hh_roll_veto, apply_crash_ride_costack_veto,
                compute_spectral_features, apply_spectral_reclassification,
                DEFAULT_CLASS_THRESHOLDS,
                PHASE3_RECLASS_ENABLED, PHASE3_RESCUE_ENABLED,
                PHASE3_COOCCURRENCE_ENABLED, SPECTRAL_RECLASS_ENABLED,
            )
            from src.preprocessing.parsers.midi_parser import (
                DrumChart, TimeSignature, TempoEvent,
            )
            import numpy as np

            v14_model = models["v14"]
            ensemble = models["ensemble"]
            phase3_model = models.get("phase3")
            tom_refinement = models.get("tom_refinement")
            device = models["device"]

            # Stage 1: V14 onset detection (+ class probs + MC onset probs)
            onset_times_ms, v14_class_probs, _mc_onset_probs = detect_onsets_v14(
                v14_model, drums_stem, device, onset_threshold=0.4
            )
            logger.info(f"    Stage 1: {len(onset_times_ms)} onsets")

            # Use pipeline-level BPM (from full mix) for tempo events
            tempo_events = [TempoEvent(tick=0, tempo_bpm=tempo_bpm, time_ms=0.0)]

            if not onset_times_ms:
                chart = DrumChart(
                    hits=[],
                    tempo_events=tempo_events,
                    time_signatures=[TimeSignature(tick=0, numerator=4, denominator=4, time_ms=0.0)],
                    ticks_per_beat=480,
                )
            else:
                # Stage 2: extract windows + ensemble classify (two-pass context)
                any_needs_cqt = any(e["needs_cqt"] for e in ensemble)
                windows = extract_onset_windows(
                    drums_stem, onset_times_ms, needs_cqt=any_needs_cqt
                )

                context = build_context_vectors(onset_times_ms)
                logits_pass1 = classify_onsets_ensemble(ensemble, windows, context, device)
                probs_pass1 = 1.0 / (1.0 + np.exp(-logits_pass1))
                context = build_context_vectors(onset_times_ms, probs_pass1)
                logits = classify_onsets_ensemble(ensemble, windows, context, device)
                probs = 1.0 / (1.0 + np.exp(-logits))

                # Spectral + RMS features
                spectral_centroids, spectral_high_pcts = _compute_spectral_centroid_features(
                    drums_stem, onset_times_ms
                )
                onset_rms = _compute_onset_rms(drums_stem, onset_times_ms)

                # 6-stem drumsep arbiter on PROBS (v25 critical step):
                # ≥98% Tom→Cym fix-rate. Reads/writes demucs_temp/stems_6/.
                stems_dir = None
                if song_folder is not None and os.environ.get("STRUM_USE_DRUMSEP", "1") == "1":
                    try:
                        stems_dir = separate_drums_6stem(drums_stem, song_folder)
                        if stems_dir is not None:
                            probs = apply_drumsep_lane_arbiter(
                                probs, onset_times_ms, stems_dir,
                                win_ms=float(os.environ.get("STRUM_DRUMSEP_WIN_MS", "50")),
                                margin_db=float(os.environ.get("STRUM_DRUMSEP_MARGIN_DB", "20")),
                                toms_floor_db=float(os.environ.get("STRUM_DRUMSEP_TOMS_FLOOR_DB", "-40")),
                            )
                    except Exception as _e:
                        logger.warning(f"    Drumsep prob arbiter failed: {_e}")

                # Fill-aware cymbal→tom rescue on PROBS (v25 critical step)
                if os.environ.get("STRUM_FILL_RESCUE", "1") == "1":
                    try:
                        spectral_lr = compute_spectral_features(drums_stem, onset_times_ms)
                        probs = apply_spectral_reclassification(
                            probs, spectral_lr, onset_times_ms,
                            tom_ratio_threshold=float(os.environ.get("STRUM_FILL_TOM_RATIO", "5.0")),
                            transfer_pct=float(os.environ.get("STRUM_FILL_TRANSFER", "0.4")),
                        )
                    except Exception as _e:
                        logger.warning(f"    Fill rescue failed: {_e}")

                # Build chart with production thresholds
                chart = build_chart(
                    onset_times_ms, probs, windows["valid_mask"],
                    tempo_events=tempo_events,
                    thresholds=DEFAULT_CLASS_THRESHOLDS,
                    spectral_centroids=spectral_centroids,
                    spectral_high_pcts=spectral_high_pcts,
                    onset_rms=onset_rms,
                    probs_pass1=probs_pass1,
                    v14_class_probs=v14_class_probs,
                )

            def _trace(stage: str):
                if os.environ.get("STRUM_TRACE_TOM_CHORDS", "0") != "1":
                    return
                from collections import Counter
                buckets = Counter()
                examples = []
                hits_sorted = sorted(chart.hits, key=lambda h: h.time_ms)
                i = 0
                while i < len(hits_sorted):
                    j = i
                    grp = [hits_sorted[i]]
                    while j + 1 < len(hits_sorted) and hits_sorted[j+1].time_ms - hits_sorted[i].time_ms < 5.0:
                        j += 1
                        grp.append(hits_sorted[j])
                    toms = [h for h in grp if not h.is_cymbal and h.lane in (2,3,4)]
                    if len(toms) >= 2:
                        buckets[tuple(sorted(h.lane for h in toms))] += 1
                        if len(examples) < 5:
                            examples.append((grp[0].time_ms, [(h.lane, 'C' if h.is_cymbal else 'T') for h in grp]))
                    i = j + 1
                logger.info(f"  [TRACE {stage}] tom-chord groups: {dict(buckets)} (total={sum(buckets.values())}); ex: {examples[:3]}")

            _trace("after_build_chart")

            # NOTE: phase pre-shift was moved to AFTER all rescue/refinement
            # passes.  Doing it before postprocess broke every audio-coupled
            # pass (phase3_onset_rescue, phase3_cymbal_cooccurrence_rescue,
            # apply_cymbal_to_tom_rescue, apply_drumsep_hits_arbiter,
            # spectral_reclassify, apply_tom_refinement_filter) because
            # they index `mc_frame_probs[hit.time_ms * sr / hop]` — once
            # hit.time_ms is shifted by -phase_offset_ms, those passes
            # sample audio at the wrong frame, giving false rescues
            # (overcharting) and adding new hits at raw audio frame times
            # (un-shifted, so visibly off the rest of the chart).
            # Post-processing runs on ORIGINAL audio times.
            if chart.hits:
                chart = postprocess_chart(chart)
            _trace("after_postprocess")

            # Phase 3 reclassification + onset rescue + co-occurrence rescue
            phase3_probs = None
            if phase3_model is not None and (PHASE3_RECLASS_ENABLED or PHASE3_RESCUE_ENABLED):
                phase3_probs = run_phase3_inference(phase3_model, drums_stem, device)
                if PHASE3_RECLASS_ENABLED:
                    chart = phase3_reclassify(chart, phase3_probs)
            if phase3_probs is not None and PHASE3_RESCUE_ENABLED:
                chart = phase3_onset_rescue(chart, phase3_probs)
            if phase3_probs is not None and PHASE3_COOCCURRENCE_ENABLED:
                chart = phase3_cymbal_cooccurrence_rescue(chart, phase3_probs)
            _trace("after_phase3")

            # Spectral reclassification (HiHat↔Snare, Crash↔Ride)
            if SPECTRAL_RECLASS_ENABLED and chart.hits:
                chart = spectral_reclassify(chart, drums_stem)
            _trace("after_spectral_reclassify")

            # Tom refinement filter (verifies all tom hits, flips false-positives)
            if tom_refinement is not None:
                chart = apply_tom_refinement_filter(chart, drums_stem, tom_refinement, device)
                # Symmetric rescue: cymbals on shared lanes that are toms in a fill
                chart = apply_cymbal_to_tom_rescue(chart, drums_stem, tom_refinement, device)
            _trace("after_tom_refinement")

            # FINAL drumsep arbiter on EMITTED HITS (v25 critical step):
            # +40dB margins, ≥98% Tom→Cym correction rate. Reasserts the
            # physical signal after Phase 3 may have reverted prob-level fixes.
            if (
                song_folder is not None
                and os.environ.get("STRUM_USE_DRUMSEP", "1") == "1"
                and os.environ.get("STRUM_DRUMSEP_FINAL_PASS", "1") == "1"
                and chart.hits
            ):
                _stems_dir = song_folder / "demucs_temp" / "stems_6"
                if _stems_dir.exists():
                    try:
                        chart = apply_drumsep_hits_arbiter(
                            chart, _stems_dir,
                            win_ms=float(os.environ.get("STRUM_DRUMSEP_WIN_MS", "50")),
                            margin_db=float(os.environ.get("STRUM_DRUMSEP_MARGIN_DB", "20")),
                            toms_floor_db=float(os.environ.get("STRUM_DRUMSEP_TOMS_FLOOR_DB", "-40")),
                        )
                    except Exception as _e:
                        logger.warning(f"    Drumsep final arbiter failed: {_e}")

                    # Snare-in-HH-roll veto (v25.2)
                    if os.environ.get("STRUM_SNARE_HH_ROLL_VETO", "1") == "1":
                        try:
                            chart = apply_snare_in_hh_roll_veto(
                                chart, _stems_dir,
                                hh_window_ms=float(os.environ.get("STRUM_SHHR_HH_WIN_MS", "300")),
                                min_hh_neighbours=int(os.environ.get("STRUM_SHHR_MIN_HH", "4")),
                                max_other_snares=int(os.environ.get("STRUM_SHHR_MAX_OTHER", "1")),
                                attack_db=float(os.environ.get("STRUM_SHHR_ATTACK_DB", "-8.0")),
                            )
                        except Exception as _e:
                            logger.warning(f"    Snare-in-HH-roll veto failed: {_e}")

                    # Crash+Ride co-stack veto (v25.4)
                    if os.environ.get("STRUM_CR_COSTACK_VETO", "1") == "1":
                        try:
                            chart = apply_crash_ride_costack_veto(
                                chart,
                                stack_tol_ms=float(os.environ.get("STRUM_CRCV_TOL_MS", "30.0")),
                            )
                        except Exception as _e:
                            logger.warning(f"    Crash+Ride co-stack veto failed: {_e}")

            # FINAL hand-cap re-pass: phase3_onset_rescue,
            # phase3_cymbal_cooccurrence_rescue, complete_cymbal_patterns and
            # spectral_reclassify can all add or move hits AFTER the
            # postprocess_chart hand-cap step ran, leaving 3+ hand notes at
            # the same tick. Rerun resolve_playability to cap to 2 hands.
            try:
                from scripts.chart_postprocess import resolve_playability, protect_tom_fills, cap_close_hands, enforce_single_tom_per_onset
                if chart.hits:
                    chart = protect_tom_fills(chart)
                    chart = resolve_playability(chart)
                    chart = cap_close_hands(chart)
                    chart = enforce_single_tom_per_onset(chart)
            except Exception as _e:
                logger.warning(f"  Final hand-cap pass failed: {_e}")
            _trace("after_final_handcap")

            # FINAL drumsep arbiter on EMITTED HITS (v25 critical step):
            # Phase 3 reclass can revert prob-level fixes; this re-asserts
            # the physical signal at hit level. Cymbal hits whose toms.wav
            # RMS dominates by margin_db get flipped to tom counterpart.
            if (
                song_folder is not None
                and os.environ.get("STRUM_USE_DRUMSEP", "1") == "1"
                and os.environ.get("STRUM_DRUMSEP_FINAL_PASS", "1") == "1"
                and chart.hits
            ):
                _stems_dir = song_folder / "demucs_temp" / "stems_6"
                if _stems_dir.exists():
                    try:
                        chart = apply_drumsep_hits_arbiter(
                            chart, _stems_dir,
                            win_ms=float(os.environ.get("STRUM_DRUMSEP_WIN_MS", "50")),
                            margin_db=float(os.environ.get("STRUM_DRUMSEP_MARGIN_DB", "20")),
                            toms_floor_db=float(os.environ.get("STRUM_DRUMSEP_TOMS_FLOOR_DB", "-40")),
                        )
                    except Exception as _e:
                        logger.warning(f"    Drumsep final hits arbiter failed: {_e}")

            # Apply single-tom constraint one more time after the final
            # drumsep arbiter (which can flip cymbal→tom and create a
            # tom-tom pair with an adjacent existing tom).
            try:
                from scripts.chart_postprocess import enforce_single_tom_per_onset
                if chart.hits:
                    chart = enforce_single_tom_per_onset(chart)
            except Exception as _e:
                logger.warning(f"  Final single-tom pass failed: {_e}")

            _trace("after_final_drumsep")

            # FINAL phase shift + tick-0 grid snap (was previously the
            # pre-shift before postprocess).  All audio-coupled passes
            # have completed and the chart's time domain still matches the
            # original audio, so it's safe to translate everything onto
            # MIDI tick 0 = bar 1 now.  After the shift we snap drum hits
            # to the same 32nd-note grid the other instruments use so
            # bar lines render aligned in CH/Moonscraper editors.
            if phase_offset_ms >= 1.0 and chart.hits:
                kept = []
                for h in chart.hits:
                    nt = h.time_ms - phase_offset_ms
                    if nt >= 0.0:
                        h.time_ms = nt
                        kept.append(h)
                chart.hits = kept
                for te in (chart.tempo_events or []):
                    te.time_ms = max(0.0, te.time_ms - phase_offset_ms)
                logger.info(f"    Phase post-shift: -{phase_offset_ms:.1f}ms "
                            f"({len(chart.hits)} hits remain)")

            # Snap drum hits to the tick-0 32nd-note grid (matches the
            # snap pass applied to guitar/bass/keys/vocals later).  Hits
            # inside a fast same-lane roll (neighbor within 1.1 grid
            # cells) are left alone so the roll keeps its micro-timing —
            # otherwise two hits would collapse onto the same tick.
            if chart.hits and chart.tempo_events:
                _bpm = chart.tempo_events[0].tempo_bpm
                _beat_ms = 60_000.0 / max(_bpm, 1.0)
                _grid_ms = _beat_ms / 8.0
                _roll_window_ms = _grid_ms * 1.1
                # Build per-(lane,is_cymbal) sorted timelines for roll detection
                from collections import defaultdict as _dd
                import bisect as _bisect
                lane_times: dict[tuple[int, bool], list[float]] = _dd(list)
                for h in chart.hits:
                    lane_times[(h.lane, h.is_cymbal)].append(h.time_ms)
                for v in lane_times.values():
                    v.sort()
                snapped = 0
                for h in chart.hits:
                    if h.time_ms <= 0:
                        continue
                    times = lane_times[(h.lane, h.is_cymbal)]
                    pos = _bisect.bisect_left(times, h.time_ms)
                    in_roll = False
                    if pos > 0 and 0.5 < (h.time_ms - times[pos - 1]) <= _roll_window_ms:
                        in_roll = True
                    if not in_roll and pos + 1 < len(times) \
                            and 0.5 < (times[pos + 1] - h.time_ms) <= _roll_window_ms:
                        in_roll = True
                    if in_roll:
                        continue
                    g = round(h.time_ms / _grid_ms) * _grid_ms
                    if g != h.time_ms:
                        h.time_ms = g
                        snapped += 1
                if snapped:
                    logger.info(f"    Drums tick-0 32nd snap: {snapped} hits")

            logger.info(f"    Drums: {len(chart.hits)} Expert hits")
            return chart

        except Exception as e:
            logger.warning(f"  ⚠ Drums transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def transcribe_guitar(self, other_stem: Path, tempo_bpm: float, full_mix: Path | None = None):
        """Transcribe guitar.

        Backend selection (env STRUM_GUITAR_BACKEND, default 'hybrid'):
          * 'hybrid' (default): V2 onset CRNN (F1 0.81) + basic-pitch
            polyphonic pitch transcription + rule-based pitch→fret. Sidesteps
            the weak V2 fret head (Event F1 0.17). Runs on the htdemucs_6s
            guitar stem so basic-pitch transcribes only guitar pitches and
            the onset CRNN doesn't fire on vocals/drums.
          * 'neural': V2 onset CRNN + V2 fret classifier (full mix).
          * 'rule': legacy librosa+pYIN+rule on the Demucs other stem.
        """
        if not self.include_guitar:
            return None

        import os as _os_g
        backend = _os_g.environ.get("STRUM_GUITAR_BACKEND", "hybrid").lower()
        # Legacy compat
        if _os_g.environ.get("STRUM_GUITAR_RULE", "0") == "1":
            backend = "rule"

        # Source selection per backend:
        #   hybrid → guitar stem (clean signal for basic-pitch + onset model)
        #   neural → full mix (model trained on full mix)
        #   rule   → other stem (legacy)
        if backend == "neural" and full_mix is not None:
            src_path = full_mix
        else:
            src_path = other_stem  # hybrid uses the passed-in guitar stem
        logger.info(f"  Transcribing guitar ({backend}, src={src_path.name})...")

        try:
            if backend == "rule":
                from src.inference.guitar_bass import transcribe_guitar
                chart = transcribe_guitar(src_path, tempo_bpm=tempo_bpm, confidence_threshold=0.3)
            elif backend == "neural":
                from src.inference.guitar_neural import transcribe_guitar_neural
                chart = transcribe_guitar_neural(src_path, tempo_bpm=tempo_bpm)
            elif backend == "basicpitch":
                import sys as _sys_g
                _scripts_dir = str(Path(__file__).resolve().parent)
                if _scripts_dir not in _sys_g.path:
                    _sys_g.path.insert(0, _scripts_dir)
                from guitar_basicpitch import transcribe_guitar_basicpitch
                chart = transcribe_guitar_basicpitch(src_path, tempo_bpm=tempo_bpm)
            else:  # hybrid
                from src.inference.guitar_hybrid_v2 import transcribe_guitar_hybrid
                chart = transcribe_guitar_hybrid(src_path, tempo_bpm=tempo_bpm)
            logger.info(f"    Guitar: {len(chart.notes)} notes + {len(chart.chords)} chords")
            return chart
        except Exception as e:
            logger.warning(f"  ⚠ Guitar transcription failed: {e}")
            return None
    
    def transcribe_bass(self, bass_stem: Path, tempo_bpm: float, full_mix: Path | None = None):
        """Transcribe bass.

        Backend selection (env STRUM_BASS_BACKEND, default 'hybrid'):
          * 'hybrid': V2 onset CRNN + basic-pitch + rule pitch->fret on
            bass.wav stem (same pipeline as guitar, with min_pitch=24).
          * 'basicpitch': pure Basic Pitch onsets + amplitude filter +
            pitch-class quintile binning. Monophonic, no chords. Validated
            on benchmark (Pearson up to +0.89 on Yellowcard).
          * 'rule': legacy librosa+pYIN on bass.wav.
          * 'neural': V2 fret head on full mix (broken — = guitar output).
        """
        if not self.include_bass:
            return None

        import os as _os_b
        backend = _os_b.environ.get("STRUM_BASS_BACKEND", "hybrid").lower()
        # legacy compat
        if _os_b.environ.get("STRUM_BASS_NEURAL", "0") == "1":
            backend = "neural"

        if backend == "neural" and full_mix is not None:
            src_path = full_mix
        else:
            src_path = bass_stem
        logger.info(f"  Transcribing bass ({backend}, src={src_path.name}, no chords per C3 rules)...")

        try:
            if backend == "neural" and full_mix is not None:
                from src.inference.guitar_neural import transcribe_guitar_neural
                chart = transcribe_guitar_neural(src_path, tempo_bpm=tempo_bpm, is_bass=True)
            elif backend == "rule":
                from src.inference.guitar_bass import transcribe_bass
                chart = transcribe_bass(src_path, tempo_bpm=tempo_bpm, confidence_threshold=0.4)
            elif backend == "basicpitch":
                import sys as _sys
                _scripts_dir = str(Path(__file__).resolve().parent)
                if _scripts_dir not in _sys.path:
                    _sys.path.insert(0, _scripts_dir)
                from bass_basicpitch import transcribe_bass_basicpitch
                chart = transcribe_bass_basicpitch(src_path, tempo_bpm=tempo_bpm)
            else:  # hybrid
                # Bass pitch range: E1 (28) to G4 (67). Override basic-pitch
                # range via env so the same hybrid pipeline doesn't drop low
                # notes. Restored after the call.
                #
                # Bass also needs its OWN onset peak threshold — guitar runs
                # at STRUM_GUITAR_PEAK_THR=0.55 (raised to fix density), but
                # bass naturally has ~half the onset density and that same
                # threshold causes severe undercharting. Default 0.35.
                prev_min = _os_b.environ.get("STRUM_BP_MIN_PITCH")
                prev_max = _os_b.environ.get("STRUM_BP_MAX_PITCH")
                prev_thr = _os_b.environ.get("STRUM_GUITAR_PEAK_THR")
                _os_b.environ["STRUM_BP_MIN_PITCH"] = "24"
                _os_b.environ["STRUM_BP_MAX_PITCH"] = "67"
                # Bass naturally sits ~6dB below guitar in the mix and the
                # onset CRNN was trained on guitar attacks, so it produces
                # weaker peaks for bass even at matched RMS. Hybrid pipeline
                # also normalizes stem RMS, but bass attack envelopes remain
                # softer. 0.15 default avoids dropping the song entirely
                # when the Demucs bass stem is unusually quiet.
                bass_thr = _os_b.environ.get("STRUM_BASS_PEAK_THR", "0.15")
                _os_b.environ["STRUM_GUITAR_PEAK_THR"] = bass_thr
                try:
                    from src.inference.guitar_hybrid_v2 import transcribe_guitar_hybrid
                    chart = transcribe_guitar_hybrid(src_path, tempo_bpm=tempo_bpm, is_bass=True)
                finally:
                    if prev_min is None:
                        _os_b.environ.pop("STRUM_BP_MIN_PITCH", None)
                    else:
                        _os_b.environ["STRUM_BP_MIN_PITCH"] = prev_min
                    if prev_max is None:
                        _os_b.environ.pop("STRUM_BP_MAX_PITCH", None)
                    else:
                        _os_b.environ["STRUM_BP_MAX_PITCH"] = prev_max
                    if prev_thr is None:
                        _os_b.environ.pop("STRUM_GUITAR_PEAK_THR", None)
                    else:
                        _os_b.environ["STRUM_GUITAR_PEAK_THR"] = prev_thr
            # transcribe_guitar_hybrid(is_bass=True) defaults max_chord_size=1
            # and harmonic_collapse=True, so chords are produced ONLY when
            # 2+ truly distinct (non-octave) bass pitches sound at an onset
            # — i.e. genuine double-stops. No further filtering needed.
            logger.info(f"    Bass: {len(chart.notes)} notes + {len(chart.chords)} chords")
            return chart
        except Exception as e:
            logger.warning(f"  ⚠ Bass transcription failed: {e}")
            return None
    
    def transcribe_vocals(self, vocals_stem: Path, artist: str, title: str):
        """Transcribe vocals using Whisper + pitch detection."""
        if not self.include_vocals:
            return None, None
        
        logger.info("  Transcribing vocals...")
        
        try:
            lead_phrases, harmony_phrases = self.vocals_charter.transcribe(
                str(vocals_stem),
                artist=artist,
                title=title
            )
            return lead_phrases, harmony_phrases
        except Exception as e:
            logger.warning(f"  ⚠ Vocals transcription failed: {e}")
            return None, None
    
    def transcribe_keys(self, other_stem: Path, guitar_stem: Path | None = None):
        """Transcribe keys/keyboards from other stem.

        Uses the keyboard-presence detector (force=False) so we don't hallucinate
        keys onto songs that don't have a keyboard part. Most songs in our test
        set lack PART KEYS — gating here avoids massive precision loss.

        guitar_stem (when provided) is used by the detector to reject piano
        stems that are mostly guitar leakage (htdemucs_6s often bleeds heavy
        guitar into the piano stem).
        """
        if not self.include_keys:
            return None

        logger.info("  Transcribing keys...")

        try:
            notes, details = self.keys_charter.transcribe(
                str(other_stem), force=False,
                guitar_stem=str(guitar_stem) if guitar_stem else None,
            )
            if notes is None:
                logger.info("    Keys: no keyboard detected, skipping")
            return notes
        except Exception as e:
            logger.warning(f"  ⚠ Keys transcription failed: {e}")
            return None
    
    def create_combined_midi(
        self,
        output_path: Path,
        tempo_bpm: float,
        drums_chart=None,
        guitar_chart=None,
        bass_chart=None,
        lead_phrases=None,
        harmony_phrases=None,
        keys_notes=None,
        ticks_per_beat: int = 480,
    ):
        """Create combined MIDI with all instruments."""
        from src.inference.guitar_bass import reduce_to_difficulty as reduce_guitar_difficulty, FRET_NOTE_OFFSETS
        from src.export.midi import (
            DIFFICULTY_NOTE_OFFSETS,
            _compute_hit_ticks,
            _get_midi_note,
            _get_tom_marker,
            _reduce_to_hard,
            _reduce_to_medium,
            _reduce_to_easy,
        )
        
        mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
        ticks_per_sec = ticks_per_beat * tempo_bpm / 60
        
        # Tempo track
        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.name = "TEMPO TRACK"
        tempo_us = int(60_000_000 / tempo_bpm)
        tempo_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
        tempo_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        tempo_track.append(mido.MetaMessage("end_of_track", time=0))
        
        # Beat track (for practice mode)
        beat_track = mido.MidiTrack()
        mid.tracks.append(beat_track)
        beat_track.name = "BEAT"
        beat_track.append(mido.MetaMessage("end_of_track", time=0))
        
        # Events track (for sections - will be populated by enhancer)
        events_track = mido.MidiTrack()
        mid.tracks.append(events_track)
        events_track.name = "EVENTS"
        events_track.append(mido.MetaMessage("end_of_track", time=0))
        
        # PART DRUMS
        if drums_chart and drums_chart.hits:
            drums_track = self._create_drums_track(drums_chart, ticks_per_beat)
            mid.tracks.append(drums_track)
            logger.info(f"  Added PART DRUMS ({len(drums_chart.hits)} expert hits)")
        
        # PART GUITAR
        if guitar_chart and guitar_chart.notes:
            guitar_track = self._create_guitar_track(guitar_chart, "PART GUITAR", ticks_per_beat)
            mid.tracks.append(guitar_track)
            logger.info(f"  Added PART GUITAR ({len(guitar_chart.notes)} expert notes)")
        
        # PART BASS
        if bass_chart and bass_chart.notes:
            bass_track = self._create_guitar_track(bass_chart, "PART BASS", ticks_per_beat)
            mid.tracks.append(bass_track)
            logger.info(f"  Added PART BASS ({len(bass_chart.notes)} expert notes)")
        
        # PART VOCALS
        if lead_phrases:
            vocals_track = self._create_vocals_track(lead_phrases, "PART VOCALS", tempo_bpm, ticks_per_beat)
            mid.tracks.append(vocals_track)
            logger.info(f"  Added PART VOCALS ({len(lead_phrases)} phrases)")
            
            # HARM1 (harmonies)
            if harmony_phrases:
                harm_track = self._create_vocals_track(harmony_phrases, "HARM1", tempo_bpm, ticks_per_beat)
                mid.tracks.append(harm_track)
                logger.info(f"  Added HARM1 ({len(harmony_phrases)} phrases)")
        
        # PART KEYS (5-lane)
        if keys_notes:
            keys_track = self._create_keys_track(keys_notes, "PART KEYS", tempo_bpm, ticks_per_beat)
            mid.tracks.append(keys_track)
            logger.info(f"  Added PART KEYS ({len(keys_notes)} notes)")
            
            # Pro Keys tracks
            for diff, suffix in [("E", "Easy"), ("M", "Medium"), ("H", "Hard"), ("X", "Expert")]:
                prokeys_track = self._create_prokeys_track(
                    keys_notes, f"PART REAL_KEYS_{diff}", tempo_bpm, ticks_per_beat, suffix.lower()
                )
                mid.tracks.append(prokeys_track)
        
        mid.save(output_path)
    
    def _create_drums_track(self, chart, ticks_per_beat: int) -> mido.MidiTrack:
        """Create drums track with all difficulties."""
        from src.export.midi import (
            DIFFICULTY_NOTE_OFFSETS,
            _compute_hit_ticks,
            _get_midi_note,
            _get_tom_marker,
            _reduce_to_hard,
            _reduce_to_medium,
            _reduce_to_easy,
        )
        
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name='PART DRUMS', time=0))
        
        hard_chart = _reduce_to_hard(chart)
        medium_chart = _reduce_to_medium(hard_chart)
        easy_chart = _reduce_to_easy(medium_chart)
        
        events = []
        # Tom markers (110/111/112) are GLOBAL flags that apply across all
        # difficulties. They must NOT be offset per-difficulty (a -12 offset
        # would turn marker 112 into note 100, colliding with Expert Green).
        # Emit each unique (tick, marker) only once.
        tom_marker_set: set[tuple[int, int]] = set()

        for difficulty, diff_chart in [("expert", chart), ("hard", hard_chart),
                                        ("medium", medium_chart), ("easy", easy_chart)]:
            note_offset = DIFFICULTY_NOTE_OFFSETS[difficulty]
            hits_with_ticks = _compute_hit_ticks(diff_chart, ticks_per_beat)

            for hit in hits_with_ticks:
                base_note = _get_midi_note(hit) + note_offset
                note_duration = ticks_per_beat // 8

                events.append(("on", hit.tick, base_note, hit.velocity))
                events.append(("off", hit.tick + note_duration, base_note, 0))

                tom_marker = _get_tom_marker(hit)
                if tom_marker is not None:
                    tom_marker_set.add((hit.tick, tom_marker))

        # Emit deduplicated tom markers (no per-difficulty offset).
        note_duration = ticks_per_beat // 8
        for tick, marker in tom_marker_set:
            events.append(("on", tick, marker, 100))
            events.append(("off", tick + note_duration, marker, 0))

        # Final (tick, note, type) dedup — guards against multi-pass
        # classification artifacts where two DrumHits with same lane
        # collapse to the same export tick.
        seen: set[tuple[str, int, int]] = set()
        unique_events = []
        for etype, tick, note, vel in events:
            key = (etype, tick, note)
            if key in seen:
                continue
            seen.add(key)
            unique_events.append((etype, tick, note, vel))
        events = unique_events

        events.sort(key=lambda e: (e[1], e[0] == "off"))
        
        prev_tick = 0
        for etype, tick, note, vel in events:
            delta = tick - prev_tick
            track.append(mido.Message('note_on' if etype == 'on' else 'note_off', 
                                      note=note, velocity=vel, time=delta))
            prev_tick = tick
        
        track.append(mido.MetaMessage('end_of_track', time=0))
        return track
    
    def _create_guitar_track(self, chart, name: str, ticks_per_beat: int) -> mido.MidiTrack:
        """Create guitar/bass track with all difficulties + C3 rule enforcement."""
        from src.inference.guitar_bass import reduce_to_difficulty, FRET_NOTE_OFFSETS, GuitarNote
        from src.inference.c3_rules import GuitarEvent as C3Event, apply_c3_rules
        
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name=name, time=0))
        
        tempo_bpm = chart.tempo_bpm
        ms_per_tick = (60_000 / tempo_bpm) / ticks_per_beat
        
        events = []
        for difficulty in ["expert", "hard", "medium", "easy"]:
            diff_chart = reduce_to_difficulty(chart, difficulty)
            note_offset = FRET_NOTE_OFFSETS[difficulty]
            
            # Group notes + chords into per-onset C3Events for rule pass
            buckets: dict[float, C3Event] = {}
            def _add(t_ms: float, fret: int, dur_ms: float, vel: int):
                k = round(t_ms, 1)
                if k not in buckets:
                    buckets[k] = C3Event(time_ms=t_ms, frets=[fret],
                                         duration_ms=dur_ms, velocity=vel)
                else:
                    if fret not in buckets[k].frets:
                        buckets[k].frets.append(fret)
                    buckets[k].duration_ms = max(buckets[k].duration_ms, dur_ms)

            for n in diff_chart.notes:
                _add(n.time_ms, int(n.fret), float(n.duration_ms), int(n.velocity))
            for c in diff_chart.chords:
                for f in c.frets:
                    _add(c.time_ms, int(f), float(c.duration_ms), int(c.velocity))

            c3_events = apply_c3_rules(
                list(buckets.values()), tempo_bpm, difficulty,
            )

            for ev in c3_events:
                start_tick = int(ev.time_ms / ms_per_tick)
                duration_ticks = max(int(ev.duration_ms / ms_per_tick), ticks_per_beat // 16)
                for f in ev.frets:
                    midi_note = note_offset + f
                    events.append(("on", start_tick, midi_note, ev.velocity))
                    events.append(("off", start_tick + duration_ticks, midi_note, 0))
        
        # off-before-on at the same tick to avoid same-pitch off cancelling a fresh on
        events.sort(key=lambda e: (e[1], 0 if e[0] == "off" else 1))
        
        prev_tick = 0
        for etype, tick, note, vel in events:
            delta = tick - prev_tick
            track.append(mido.Message('note_on' if etype == 'on' else 'note_off',
                                      note=note, velocity=vel, time=delta))
            prev_tick = tick
        
        track.append(mido.MetaMessage('end_of_track', time=0))
        return track
    
    def _create_vocals_track(self, phrases, name: str, tempo_bpm: float, ticks_per_beat: int) -> mido.MidiTrack:
        """Create vocals track with phrase markers, pitch notes, and slide connections.

        connects_to_next on a VocalNote means YARG/CH should draw a smooth
        slide between this note and the next \u2014 we encode it by ensuring the
        note_off lines up with (or 1 tick before) the next note_on, with no
        gap between them. Without this the chart looks "stair-step".
        """
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name=name, time=0))

        ticks_per_sec = ticks_per_beat * tempo_bpm / 60
        events = []

        for phrase in phrases:
            # Phrase marker (note 105)
            start_tick = int(phrase.start_time * ticks_per_sec)
            end_tick = int(phrase.end_time * ticks_per_sec)

            events.append(("on", start_tick, 105, 100))
            events.append(("off", end_tick, 105, 0))

            phrase_notes = phrase.notes
            for i, note in enumerate(phrase_notes):
                note_start = int(note.start_time * ticks_per_sec)
                note_end = int(note.end_time * ticks_per_sec)

                # Slide handling: when this note connects to the next, force
                # note_off to land EXACTLY at the next note's note_on tick.
                # YARG/CH renders these as a single legato slide instead of
                # two stair-step notes.
                if note.connects_to_next and i + 1 < len(phrase_notes):
                    next_start = int(phrase_notes[i + 1].start_time * ticks_per_sec)
                    note_end = next_start  # touch, no gap

                # Guarantee at least 1-tick duration
                if note_end <= note_start:
                    note_end = note_start + 1

                midi_pitch = note.midi_pitch  # VocalNote uses midi_pitch attribute

                events.append(("on", note_start, midi_pitch, 100))
                events.append(("off", note_end, midi_pitch, 0))

                # Add lyric text
                if note.lyric:
                    events.append(("lyric", note_start, note.lyric, 0))

        # Sort: at same tick, lyric first, then off (so prev pitch ends before
        # new pitch starts for slides), then on.
        events.sort(key=lambda e: (e[1], 0 if e[0] == "lyric" else (1 if e[0] == "off" else 2)))
        
        prev_tick = 0
        for event in events:
            etype, tick, data, vel = event
            delta = tick - prev_tick
            
            if etype == "lyric":
                track.append(mido.MetaMessage('lyrics', text=data, time=delta))
            elif etype == "on":
                track.append(mido.Message('note_on', note=data, velocity=vel, time=delta))
            else:
                track.append(mido.Message('note_off', note=data, velocity=0, time=delta))
            
            prev_tick = tick
        
        track.append(mido.MetaMessage('end_of_track', time=0))
        return track
    
    def _create_keys_track(self, notes, name: str, tempo_bpm: float, ticks_per_beat: int) -> mido.MidiTrack:
        """Create 5-lane keys track."""
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name=name, time=0))
        
        ticks_per_sec = ticks_per_beat * tempo_bpm / 60

        # Map pitches to 5 lanes
        pitches = [n.midi_pitch for n in notes]
        if not pitches:
            track.append(mido.MetaMessage('end_of_track', time=0))
            return track
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        pitch_range = max(max_pitch - min_pitch, 1)

        # Deterministic difficulty reduction (every Nth note dropped, in order):
        #   expert: keep all
        #   hard:   drop every 4th  (75% kept)
        #   medium: keep every other (50%)
        #   easy:   keep every 3rd  (~33%)
        diff_keep = {
            "expert": lambda i: True,
            "hard":   lambda i: i % 4 != 0,
            "medium": lambda i: i % 2 == 0,
            "easy":   lambda i: i % 3 == 0,
        }

        events = []
        for difficulty, offset in [("expert", 96), ("hard", 84), ("medium", 72), ("easy", 60)]:
            keep = diff_keep[difficulty]
            for i, note in enumerate(notes):
                if not keep(i):
                    continue
                # Map pitch to 0-4 lanes
                lane = int((note.midi_pitch - min_pitch) / pitch_range * 4.99)
                lane = max(0, min(4, lane))

                start_tick = int(note.start_time * ticks_per_sec)
                end_tick = max(start_tick + ticks_per_beat // 8,
                               int(note.end_time * ticks_per_sec))
                midi_note = offset + lane

                events.append(("on", start_tick, midi_note, 100))
                events.append(("off", end_tick, midi_note, 0))
        
        events.sort(key=lambda e: (e[1], e[0] == "off"))
        
        prev_tick = 0
        for etype, tick, note, vel in events:
            delta = tick - prev_tick
            track.append(mido.Message('note_on' if etype == 'on' else 'note_off',
                                      note=note, velocity=vel, time=delta))
            prev_tick = tick
        
        track.append(mido.MetaMessage('end_of_track', time=0))
        return track
    
    def _create_prokeys_track(self, notes, name: str, tempo_bpm: float, ticks_per_beat: int, difficulty: str) -> mido.MidiTrack:
        """Create Pro Keys track with actual pitches and range-shift markers.

        Pro Keys MIDI spec (Rock Band / Clone Hero):
          * 25-key playable range is fixed at MIDI notes 48..72 (C3..C5).
          * Notes outside that 2-octave window are OCTAVE-FOLDED to fit so
            high-piano content isn't silently dropped.
          * Range-shift markers are MIDI notes 0/2/4/5/7/9 (C/D/E/F/G/A),
            ONE active at a time. We pick the range covering the most notes
            in the song and emit a single shift at t=0.
          * Difficulty reduction uses deterministic stride (every Nth note),
            NOT hash() which is randomized per Python run.
        """
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name=name, time=0))

        if not notes:
            track.append(mido.MetaMessage('end_of_track', time=0))
            return track

        ticks_per_sec = ticks_per_beat * tempo_bpm / 60

        # Deterministic difficulty reduction (stride): keep every 1, 4/3, 2, 3rd note
        keep_stride = {"expert": 1, "hard": 1, "medium": 2, "easy": 3}[difficulty]
        # Hard keeps 75% — emulate by dropping every 4th note
        if difficulty == "hard":
            filtered = [n for i, n in enumerate(notes) if i % 4 != 0]
        else:
            filtered = notes[::keep_stride]

        if not filtered:
            track.append(mido.MetaMessage('end_of_track', time=0))
            return track

        # Pick best range shift: try each (C/D/E/F/G/A) and pick the one
        # that puts the most notes (after octave-fold) closest to center.
        # For simplicity, pick the median pitch and choose the range whose
        # window center is closest.
        pitches_all = [n.midi_pitch for n in filtered]
        median_pitch = int(np.median(pitches_all))
        # Range shift -> chart's 48 maps to that pitch class an octave below median area
        # Range options: C(0), D(2), E(4), F(5), G(7), A(9). The visible window
        # center is offset+60 (midpoint of 48..72=60). Pick offset minimizing
        # |median_pitch - (60 + offset)|.
        range_options = [(0, 'C'), (2, 'D'), (4, 'E'), (5, 'F'), (7, 'G'), (9, 'A')]
        best_offset, _ = min(range_options, key=lambda r: abs(median_pitch - (60 + r[0])))

        events = []

        # Single range shift at t=0
        events.append(("range", 0, best_offset))

        # Octave-fold every note into [48, 72]
        for note in filtered:
            p = int(note.midi_pitch)
            while p < 48:
                p += 12
            while p > 72:
                p -= 12

            start_tick = int(note.start_time * ticks_per_sec)
            end_tick = max(start_tick + ticks_per_beat // 8,
                           int(note.end_time * ticks_per_sec))
            events.append(("on", start_tick, p, int(note.velocity)))
            events.append(("off", end_tick, p, 0))

        # Sort: at same tick, range first, then off, then on (avoid same-pitch
        # off cancelling a fresh on)
        def _sort_key(e):
            order = {"range": 0, "off": 1, "on": 2}
            return (e[1], order[e[0]])
        events.sort(key=_sort_key)

        prev_tick = 0
        for event in events:
            etype = event[0]
            tick = event[1]
            delta = max(0, tick - prev_tick)

            if etype == "range":
                # Range shift = brief note pulse on offset (0/2/4/5/7/9)
                shift_note = event[2]
                track.append(mido.Message('note_on', note=shift_note, velocity=100, time=delta))
                track.append(mido.Message('note_off', note=shift_note, velocity=0, time=ticks_per_beat // 16))
                prev_tick = tick + ticks_per_beat // 16
            elif etype == "on":
                track.append(mido.Message('note_on', note=event[2], velocity=event[3], time=delta))
                prev_tick = tick
            else:
                track.append(mido.Message('note_off', note=event[2], velocity=0, time=delta))
                prev_tick = tick

        track.append(mido.MetaMessage('end_of_track', time=0))
        return track
    
    def run_chart_enhancer(self, midi_path: Path, audio_path: Path):
        """Run the chart enhancer for SP, difficulty reduction, and sections."""
        logger.info("  Enhancing chart (SP, difficulty, sections)...")
        
        from chart_enhancer import ChartEnhancer
        enhancer = ChartEnhancer()
        enhancer.enhance_chart(str(midi_path), str(audio_path), str(midi_path))
    
    def convert_to_ogg(self, input_path: Path, output_path: Path,
                       trim_start_ms: float = 0.0):
        """Convert audio to OGG format, optionally trimming N ms from the start."""
        cmd = ["ffmpeg", "-y"]
        if trim_start_ms and trim_start_ms >= 1.0:
            cmd += ["-ss", f"{trim_start_ms / 1000.0:.3f}"]
        cmd += [
            "-i", str(input_path),
            "-vn",  # Strip any video/image streams (album art)
            "-c:a", "libvorbis", "-q:a", "6",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Try with powershell/system ffmpeg
            logger.warning("ffmpeg failed, audio will be copied instead")
            shutil.copy(input_path, output_path.with_suffix(input_path.suffix))
    
    def create_song_ini(
        self,
        song_folder: Path,
        artist: str,
        title: str,
        tempo_bpm: float,
        duration_ms: int,
        preview_start_ms: int,
        album: str = '',
        year: str = '',
        genre: str = '',
    ):
        """Create song.ini file."""
        ini_content = f"""[song]
name = {title}
artist = {artist}
charter = STRUM
album = {album}
year = {year}
genre = {genre}
diff_drums = -1
diff_guitar = -1
diff_bass = -1
diff_vocals = -1
diff_keys = -1
diff_keys_real = -1
preview_start_time = {preview_start_ms}
song_length = {duration_ms}
"""
        
        ini_path = song_folder / "song.ini"
        ini_path.write_text(ini_content, encoding='utf-8')
    
    def fetch_album_art(self, audio_path: Path, song_folder: Path, artist: str, title: str) -> bool:
        """
        Fetch album art from multiple sources and save to song folder.
        
        Priority:
        1. Embedded ID3 tag in audio file
        2. iTunes Search API (free, no auth needed)
        3. Music file in same folder named album.png/jpg
        
        Returns True if album art was found and saved.
        """
        album_path = song_folder / "album.png"
        
        # 1. Try extracting from audio file embedded art
        if self._extract_embedded_art(audio_path, album_path):
            logger.info("  Album art: extracted from audio file")
            return True
        
        # 2. Try iTunes Search API
        if self._fetch_itunes_art(artist, title, album_path):
            logger.info("  Album art: fetched from iTunes")
            return True
        
        # 3. Check for existing album art in source folder
        source_folder = audio_path.parent
        for art_name in ['album.png', 'album.jpg', 'cover.png', 'cover.jpg', 'folder.jpg', 'folder.png']:
            art_source = source_folder / art_name
            if art_source.exists():
                shutil.copy(art_source, album_path.with_suffix(art_source.suffix))
                logger.info(f"  Album art: copied from source folder ({art_name})")
                return True
        
        logger.warning("  ⚠ No album art found")
        return False
    
    def _extract_embedded_art(self, audio_path: Path, output_path: Path) -> bool:
        """Extract album art embedded in audio file."""
        try:
            from mutagen import File
            from mutagen.id3 import ID3
            from mutagen.mp3 import MP3
            from mutagen.flac import FLAC
            from mutagen.mp4 import MP4
            
            audio = File(str(audio_path))
            if audio is None:
                return False
            
            art_data = None
            
            # MP3 with ID3 tags
            if str(audio_path).lower().endswith('.mp3'):
                try:
                    tags = ID3(str(audio_path))
                    for key in tags.keys():
                        if key.startswith('APIC'):
                            art_data = tags[key].data
                            break
                except Exception:
                    pass
            
            # FLAC
            elif str(audio_path).lower().endswith('.flac'):
                try:
                    flac = FLAC(str(audio_path))
                    if flac.pictures:
                        art_data = flac.pictures[0].data
                except Exception:
                    pass
            
            # M4A/MP4
            elif str(audio_path).lower().endswith(('.m4a', '.mp4')):
                try:
                    mp4 = MP4(str(audio_path))
                    if 'covr' in mp4.tags:
                        art_data = bytes(mp4.tags['covr'][0])
                except Exception:
                    pass
            
            if art_data:
                # Determine format from magic bytes
                if art_data[:8] == b'\x89PNG\r\n\x1a\n':
                    output_path = output_path.with_suffix('.png')
                else:
                    output_path = output_path.with_suffix('.jpg')
                
                output_path.write_bytes(art_data)
                return True
            
            return False
            
        except ImportError:
            logger.debug("mutagen not installed, skipping embedded art extraction")
            return False
        except Exception as e:
            logger.debug(f"Failed to extract embedded art: {e}")
            return False
    
    def _fetch_itunes_art(self, artist: str, title: str, output_path: Path) -> bool:
        """Fetch album art from iTunes Search API."""
        try:
            import urllib.request
            import urllib.parse
            import json as json_lib
            
            # Build search query
            query = f"{artist} {title}"
            encoded_query = urllib.parse.quote(query)
            url = f"https://itunes.apple.com/search?term={encoded_query}&media=music&entity=song&limit=5"
            
            # Make request
            req = urllib.request.Request(url, headers={'User-Agent': 'STRUM/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json_lib.loads(response.read().decode('utf-8'))
            
            if data['resultCount'] == 0:
                return False
            
            # Find best match (look for artist/title match)
            art_url = None
            for result in data['results']:
                result_artist = result.get('artistName', '').lower()
                result_track = result.get('trackName', '').lower()
                
                if artist.lower() in result_artist or result_artist in artist.lower():
                    if title.lower() in result_track or result_track in title.lower():
                        art_url = result.get('artworkUrl100', '')
                        break
            
            # Fallback to first result
            if not art_url and data['results']:
                art_url = data['results'][0].get('artworkUrl100', '')
            
            if not art_url:
                return False
            
            # Get higher resolution (600x600)
            art_url = art_url.replace('100x100', '600x600')
            
            # Download image
            req = urllib.request.Request(art_url, headers={'User-Agent': 'STRUM/1.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                image_data = response.read()
            
            # Save as jpg (iTunes returns jpg)
            output_path = output_path.with_suffix('.jpg')
            output_path.write_bytes(image_data)
            return True
            
        except Exception as e:
            logger.debug(f"Failed to fetch from iTunes: {e}")
            return False
    
    def fetch_music_video(self, song_folder: Path, artist: str, title: str) -> bool:
        """
        Optionally fetch music video from YouTube (if yt-dlp is available).
        
        Returns True if video was found and downloaded.
        """
        video_path = song_folder / "video.mp4"
        
        try:
            # Check if yt-dlp is available
            result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                return False
            
            # Search and download video
            query = f"ytsearch1:{artist} - {title} official music video"
            cmd = [
                "yt-dlp",
                "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
                "-o", str(video_path),
                "--no-playlist",
                "--max-filesize", "200M",
                query
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if video_path.exists():
                logger.info(f"  Video: downloaded music video")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Video download failed: {e}")
            return False

    def process_song(self, audio_path: Path) -> SongResult:
        """Process a single song through the full pipeline."""
        artist, title = self.parse_filename(audio_path)
        safe_name = re.sub(r'[<>:"/\\|?*]', '', f"{artist} - {title}")
        song_folder = self.output_dir / safe_name
        
        result = SongResult(
            input_path=str(audio_path),
            output_path=str(song_folder),
            success=False
        )
        
        stems = {}
        tempo_bpm = 120  # Default fallback
        audio_info = None
        
        try:
            song_folder.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing: {artist} - {title}")
            
            # Analyze audio
            audio_info = self.analyze_audio(audio_path, artist, title)
            tempo_bpm = audio_info['tempo_bpm']
            logger.info(f"  Tempo: {tempo_bpm} BPM, Duration: {audio_info['duration_sec']:.1f}s")
            
            # Separate stems
            stems = self.separate_stems(audio_path, song_folder)
            
        except Exception as e:
            result.error = f"Preprocessing failed: {e}"
            logger.error(f"  ✗ Preprocessing failed: {e}")
            return result
        
        # Transcribe each instrument - isolated errors per instrument
        drums_chart = None
        guitar_chart = None
        bass_chart = None
        lead_phrases = None
        harmony_phrases = None
        keys_notes = None
        errors = []
        
        # Phase offset for MIDI-time alignment (must be applied to ALL
        # instruments + audio trim so bar lines render under audio downbeats).
        # Drums get it pre-postprocess; other instruments are shifted below
        # in the combined-MIDI build block.
        import os as _os_phase
        phase_offset_ms_for_align = float(audio_info.get('phase_offset_ms', 0.0)) \
            if _os_phase.environ.get("STRUM_PHASE_ALIGN", "1") == "1" else 0.0

        # Drums
        if "drums" in stems:
            try:
                # Low-RMS drums-stem fallback (shared with batch_infer_hybrid):
                # if the separated drums stem is near-silent, detect on a
                # drums+other mix or the original full mix instead.
                from batch_infer_hybrid import (
                    STEM_FALLBACK_ENABLED,
                    resolve_drums_fallback,
                )
                drums_input = stems["drums"]
                if STEM_FALLBACK_ENABLED:
                    drums_input = resolve_drums_fallback(
                        drums_input,
                        full_mix_path=audio_path,
                        other_path=stems.get("other"),
                    )
                drums_chart = self.transcribe_drums(
                    drums_input, tempo_bpm,
                    song_folder=song_folder,
                    phase_offset_ms=phase_offset_ms_for_align,
                )
                if drums_chart:
                    result.drums_hits = len(drums_chart.hits)
                    logger.info(f"    Drums: {result.drums_hits} hits")
            except Exception as e:
                errors.append(f"drums: {e}")
                logger.warning(f"  ⚠ Drums failed: {e}")
        
        # Guitar - prefer dedicated guitar stem (htdemucs_6s), fallback to other
        guitar_stem = stems.get("guitar") or stems.get("other")
        if guitar_stem:
            try:
                stem_type = "guitar" if "guitar" in stems else "other"
                logger.info(f"    Using {stem_type} stem for guitar transcription")
                guitar_chart = self.transcribe_guitar(guitar_stem, tempo_bpm, full_mix=audio_path)
                if guitar_chart:
                    result.guitar_notes = len(guitar_chart.notes)
                    logger.info(f"    Guitar: {result.guitar_notes} notes")
            except Exception as e:
                errors.append(f"guitar: {e}")
                logger.warning(f"  ⚠ Guitar failed: {e}")
        
        # Keys - prefer dedicated piano stem (htdemucs_6s), fallback to other
        keys_stem = stems.get("piano") or stems.get("other")
        if keys_stem:
            try:
                stem_type = "piano" if "piano" in stems else "other"
                logger.info(f"    Using {stem_type} stem for keys transcription")
                keys_notes = self.transcribe_keys(keys_stem, guitar_stem=stems.get("guitar"))
                if keys_notes:
                    result.keys_notes = len(keys_notes)
                    logger.info(f"    Keys: {result.keys_notes} notes")
            except Exception as e:
                errors.append(f"keys: {e}")
                logger.warning(f"  ⚠ Keys failed: {e}")
        
        # Bass
        if "bass" in stems:
            try:
                bass_chart = self.transcribe_bass(stems["bass"], tempo_bpm, full_mix=audio_path)
                if bass_chart:
                    result.bass_notes = len(bass_chart.notes)
                    logger.info(f"    Bass: {result.bass_notes} notes")
            except Exception as e:
                errors.append(f"bass: {e}")
                logger.warning(f"  ⚠ Bass failed: {e}")
        
        # Vocals
        if "vocals" in stems:
            try:
                lead_phrases, harmony_phrases = self.transcribe_vocals(
                    stems["vocals"], artist, title
                )
                if lead_phrases:
                    result.vocals_phrases = len(lead_phrases)
                    logger.info(f"    Vocals: {result.vocals_phrases} phrases")
            except Exception as e:
                errors.append(f"vocals: {e}")
                logger.warning(f"  ⚠ Vocals failed: {e}")
        
        # Check if we have ANY transcribed content
        has_content = any([
            drums_chart and drums_chart.hits,
            guitar_chart and guitar_chart.notes,
            bass_chart and bass_chart.notes,
            lead_phrases,
            keys_notes,
        ])
        
        if not has_content:
            result.error = "No instruments transcribed successfully"
            logger.error(f"  ✗ No instruments transcribed")
            # Still cleanup stems
            self._cleanup_stems(song_folder)
            return result
        
        # Create combined MIDI (even if some instruments failed)
        try:
            import os
            logger.info("  Creating combined chart...")

            # Phase alignment: shift all hits/notes back by phase_offset_ms and
            # trim the audio start by the same amount, so MIDI tick 0 (= bar 1
            # in Clone Hero) coincides with the audio's first detected beat.
            # Without this, bar lines render at chart_time 0, bar_period, …
            # while audio downbeats sit at +phase_offset_ms — every note
            # drifts visibly off the backing track.
            phase_offset_ms = float(audio_info.get('phase_offset_ms', 0.0)) \
                if os.environ.get("STRUM_PHASE_ALIGN", "1") == "1" else 0.0

            if phase_offset_ms >= 1.0:
                logger.info(f"  Phase align: shifting charts by -{phase_offset_ms:.1f}ms")
                # Drums are already shifted inside transcribe_drums (BEFORE
                # postprocess) so that grid quantization snaps to a 0-anchored
                # MIDI grid. Skipping here avoids a double-shift.
                # Guitar / Bass (have .notes[].time_ms or .start_ms)
                for chart_obj in (guitar_chart, bass_chart):
                    if chart_obj is None:
                        continue
                    notes = getattr(chart_obj, 'notes', None) or []
                    kept_notes = []
                    for n in notes:
                        for attr in ('time_ms', 'start_ms', 'start_time_ms'):
                            if hasattr(n, attr):
                                v = getattr(n, attr)
                                if v is None:
                                    continue
                                nv = v - phase_offset_ms
                                if nv < 0.0:
                                    break
                                setattr(n, attr, nv)
                        else:
                            kept_notes.append(n)
                            continue
                        # broke -> drop
                    # Only assign if we actually filtered (some chart types
                    # may have notes inside their own structure)
                    if kept_notes and len(kept_notes) != len(notes):
                        try:
                            chart_obj.notes = kept_notes
                        except Exception:
                            pass
                # Vocals phrases (start/end ms or start/end_time in SECONDS)
                phase_offset_s = phase_offset_ms / 1000.0
                for phrases in (lead_phrases, harmony_phrases):
                    if not phrases:
                        continue
                    kept_p = []
                    for p in phrases:
                        for s_attr in ('start_ms', 'start_time_ms'):
                            if hasattr(p, s_attr) and getattr(p, s_attr) is not None:
                                ns = getattr(p, s_attr) - phase_offset_ms
                                if ns < 0.0:
                                    ns = 0.0
                                setattr(p, s_attr, ns)
                        for e_attr in ('end_ms', 'end_time_ms'):
                            if hasattr(p, e_attr) and getattr(p, e_attr) is not None:
                                setattr(p, e_attr,
                                        max(0.0, getattr(p, e_attr) - phase_offset_ms))
                        # VocalPhrase uses start_time / end_time in SECONDS
                        if hasattr(p, 'start_time') and getattr(p, 'start_time') is not None:
                            p.start_time = max(0.0, p.start_time - phase_offset_s)
                        if hasattr(p, 'end_time') and getattr(p, 'end_time') is not None:
                            p.end_time = max(0.0, p.end_time - phase_offset_s)
                        # Per-syllable times if present
                        for syl_attr in ('syllables', 'words', 'notes'):
                            syls = getattr(p, syl_attr, None)
                            if not syls:
                                continue
                            for s in syls:
                                for a in ('time_ms', 'start_ms', 'start_time_ms'):
                                    if hasattr(s, a) and getattr(s, a) is not None:
                                        setattr(s, a, max(0.0, getattr(s, a) - phase_offset_ms))
                                for a in ('end_ms', 'end_time_ms'):
                                    if hasattr(s, a) and getattr(s, a) is not None:
                                        setattr(s, a, max(0.0, getattr(s, a) - phase_offset_ms))
                                # VocalNote start_time / end_time in SECONDS
                                if hasattr(s, 'start_time') and getattr(s, 'start_time') is not None:
                                    s.start_time = max(0.0, s.start_time - phase_offset_s)
                                if hasattr(s, 'end_time') and getattr(s, 'end_time') is not None:
                                    s.end_time = max(0.0, s.end_time - phase_offset_s)
                        kept_p.append(p)
                    phrases[:] = kept_p
                # Keys (list of dicts or simple objects with time_ms)
                if keys_notes:
                    kept_k = []
                    for n in keys_notes:
                        if isinstance(n, dict):
                            for a in ('time_ms', 'start_ms'):
                                if a in n and n[a] is not None:
                                    nv = n[a] - phase_offset_ms
                                    if nv < 0:
                                        break
                                    n[a] = nv
                            else:
                                for a in ('end_ms', 'end_time_ms'):
                                    if a in n and n[a] is not None:
                                        n[a] = max(0.0, n[a] - phase_offset_ms)
                                kept_k.append(n)
                                continue
                            continue
                        else:
                            for a in ('time_ms', 'start_ms', 'start_time_ms'):
                                if hasattr(n, a) and getattr(n, a) is not None:
                                    nv = getattr(n, a) - phase_offset_ms
                                    if nv < 0:
                                        break
                                    setattr(n, a, nv)
                            else:
                                for a in ('end_ms', 'end_time_ms'):
                                    if hasattr(n, a) and getattr(n, a) is not None:
                                        setattr(n, a, max(0.0, getattr(n, a) - phase_offset_ms))
                                kept_k.append(n)
                                continue
                    keys_notes = kept_k
                # Update reported duration so song.ini matches trimmed audio.
                audio_info['duration_ms'] = max(0, audio_info['duration_ms'] - int(phase_offset_ms))

            # Snap non-drum instruments to the 32nd-note MIDI grid anchored at
            # tick 0.  Drums already went through chart_postprocess with the
            # pre-shift, so they're already on-grid.  Guitar/bass/keys/vocals
            # come straight from neural transcribers with no quantization, so
            # they sit on raw audio onset times \u2014 visible in editors as
            # notes a few ms off the bar lines.
            beat_ms = 60_000.0 / max(tempo_bpm, 1.0)
            grid_ms = beat_ms / 8.0  # 32nd-note grid
            grid_s = grid_ms / 1000.0
            def _snap(t):
                if t is None or t <= 0:
                    return t
                return round(t / grid_ms) * grid_ms
            def _snap_s(t):
                if t is None or t <= 0:
                    return t
                return round(t / grid_s) * grid_s

            for chart_obj in (guitar_chart, bass_chart):
                if chart_obj is None:
                    continue
                for n in (getattr(chart_obj, 'notes', None) or []):
                    for a in ('time_ms', 'start_ms', 'start_time_ms'):
                        if hasattr(n, a) and getattr(n, a) is not None:
                            setattr(n, a, _snap(getattr(n, a)))
                    for a in ('end_ms', 'end_time_ms'):
                        if hasattr(n, a) and getattr(n, a) is not None:
                            setattr(n, a, _snap(getattr(n, a)))
                # GuitarChord also has time_ms — was missed before, leaving
                # all chord events off-grid in the editor.
                for c in (getattr(chart_obj, 'chords', None) or []):
                    for a in ('time_ms', 'start_ms', 'start_time_ms'):
                        if hasattr(c, a) and getattr(c, a) is not None:
                            setattr(c, a, _snap(getattr(c, a)))
                    for a in ('end_ms', 'end_time_ms'):
                        if hasattr(c, a) and getattr(c, a) is not None:
                            setattr(c, a, _snap(getattr(c, a)))
            if keys_notes:
                for n in keys_notes:
                    if isinstance(n, dict):
                        for a in ('time_ms', 'start_ms', 'end_ms', 'end_time_ms'):
                            if a in n and n[a] is not None:
                                n[a] = _snap(n[a])
                    else:
                        for a in ('time_ms', 'start_ms', 'start_time_ms',
                                  'end_ms', 'end_time_ms'):
                            if hasattr(n, a) and getattr(n, a) is not None:
                                setattr(n, a, _snap(getattr(n, a)))
            # Vocal phrases use start_time / end_time in SECONDS (not ms)
            # and contain VocalNote objects also in seconds.  These were
            # being missed entirely by the *_ms attribute scan, leaving
            # vocal pitches at raw Whisper timestamps (frame-aligned ~20ms
            # off the grid).
            for phrases in (lead_phrases, harmony_phrases):
                if not phrases:
                    continue
                for p in phrases:
                    for a in ('start_time', 'end_time'):
                        if hasattr(p, a) and getattr(p, a) is not None:
                            setattr(p, a, _snap_s(getattr(p, a)))
                    for a in ('start_ms', 'end_ms', 'start_time_ms', 'end_time_ms'):
                        if hasattr(p, a) and getattr(p, a) is not None:
                            setattr(p, a, _snap(getattr(p, a)))
                    for syl_attr in ('syllables', 'words', 'notes'):
                        for s in (getattr(p, syl_attr, None) or []):
                            for a in ('start_time', 'end_time'):
                                if hasattr(s, a) and getattr(s, a) is not None:
                                    setattr(s, a, _snap_s(getattr(s, a)))
                            for a in ('time_ms', 'start_ms', 'start_time_ms',
                                      'end_ms', 'end_time_ms'):
                                if hasattr(s, a) and getattr(s, a) is not None:
                                    setattr(s, a, _snap(getattr(s, a)))

            notes_path = song_folder / "notes.mid"
            self.create_combined_midi(
                notes_path,
                tempo_bpm,
                drums_chart=drums_chart,
                guitar_chart=guitar_chart,
                bass_chart=bass_chart,
                lead_phrases=lead_phrases,
                harmony_phrases=harmony_phrases,
                keys_notes=keys_notes,
            )
            
            # Convert audio (trim leading phase_offset_ms so audio downbeats
            # align with bar lines in the chart).
            song_ogg = song_folder / "song.ogg"
            self.convert_to_ogg(audio_path, song_ogg, trim_start_ms=phase_offset_ms)
            
            # Create song.ini
            self.create_song_ini(
                song_folder,
                artist=artist,
                title=title,
                tempo_bpm=tempo_bpm,
                duration_ms=audio_info['duration_ms'],
                preview_start_ms=audio_info['preview_start_ms'],
                album=audio_info.get('album', ''),
                year=audio_info.get('year', ''),
                genre=audio_info.get('genre', ''),
            )
            
            # Run chart enhancer
            try:
                audio_for_enhancer = song_ogg if song_ogg.exists() else audio_path
                self.run_chart_enhancer(notes_path, audio_for_enhancer)
            except Exception as e:
                logger.warning(f"  ⚠ Chart enhancement failed: {e}")
            
            # Fetch album art (required)
            try:
                self.fetch_album_art(audio_path, song_folder, artist, title)
            except Exception as e:
                logger.warning(f"  ⚠ Album art fetch failed: {e}")
            
            # Optionally fetch music video
            if self.include_video:
                try:
                    self.fetch_music_video(song_folder, artist, title)
                except Exception as e:
                    logger.debug(f"  Video fetch skipped: {e}")
            
            result.success = True
            if errors:
                result.error = f"Partial: {'; '.join(errors)}"
                logger.info(f"  ✓ Complete (with warnings): {song_folder.name}")
            else:
                logger.info(f"  ✓ Complete: {song_folder.name}")
            
        except Exception as e:
            result.error = f"Chart creation failed: {e}"
            logger.error(f"  ✗ Chart creation failed: {e}")
        
        # Always cleanup stems at the end
        self._cleanup_stems(song_folder)
        
        return result
    
    def _cleanup_stems(self, song_folder: Path):
        """Remove temporary stem files."""
        demucs_temp = song_folder / "demucs_temp"
        if demucs_temp.exists():
            try:
                shutil.rmtree(demucs_temp)
                logger.info("  Cleaned up stem files")
            except Exception as e:
                logger.warning(f"  ⚠ Failed to cleanup stems: {e}")
    
    def process_batch(self, input_dir: Path, continue_from: bool = False) -> List[SongResult]:
        """Process all audio files in a directory."""
        input_dir = Path(input_dir)
        
        # Find audio files
        extensions = ["*.mp3", "*.wav", "*.ogg", "*.flac", "*.m4a"]
        audio_files = []
        for ext in extensions:
            audio_files.extend(input_dir.glob(ext))
        
        audio_files = sorted(set(audio_files))
        
        if not audio_files:
            logger.error(f"No audio files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(audio_files)} audio files")
        logger.info("=" * 60)
        
        # Check for already processed if continuing
        if continue_from:
            existing = set(p.name for p in self.output_dir.iterdir() if p.is_dir())
            original_count = len(audio_files)
            audio_files = [
                f for f in audio_files
                if re.sub(r'[<>:"/\\|?*]', '', f"{self.parse_filename(f)[0]} - {self.parse_filename(f)[1]}") not in existing
            ]
            if len(audio_files) < original_count:
                logger.info(f"Skipping {original_count - len(audio_files)} already processed songs")
        
        # Process each file
        results = []
        for i, audio_path in enumerate(audio_files, 1):
            logger.info(f"\n[{i}/{len(audio_files)}] {audio_path.name}")
            result = self.process_song(audio_path)
            results.append(result)
        
        # Summary
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        logger.info("\n" + "=" * 60)
        logger.info("BATCH COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Processed: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        
        if failed:
            logger.info("\nFailed songs:")
            for r in failed:
                logger.info(f"  - {Path(r.input_path).name}: {r.error}")
        
        # Save results
        results_path = self.output_dir / "batch_results.json"
        with open(results_path, 'w') as f:
            json.dump([vars(r) for r in results], f, indent=2)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="STRUM Batch Pipeline - Process multiple songs"
    )
    parser.add_argument("input_dir", help="Input directory with audio files")
    parser.add_argument("output_dir", help="Output directory for chart folders")
    parser.add_argument("--no-drums", action="store_true", help="Skip drums")
    parser.add_argument("--no-guitar", action="store_true", help="Skip guitar")
    parser.add_argument("--no-bass", action="store_true", help="Skip bass")
    parser.add_argument("--no-vocals", action="store_true", help="Skip vocals")
    parser.add_argument("--no-keys", action="store_true", help="(deprecated; keys are off by default)")
    parser.add_argument("--keys", action="store_true",
                        help="Enable keys transcription (off by default; detector is unreliable)")
    parser.add_argument("--include-video", action="store_true",
                        help="Download music videos from YouTube (requires yt-dlp)")
    parser.add_argument("--continue", dest="continue_from", action="store_true",
                        help="Skip already processed songs")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    pipeline = BatchPipeline(
        output_dir=Path(args.output_dir),
        include_drums=not args.no_drums,
        include_guitar=not args.no_guitar,
        include_bass=not args.no_bass,
        include_vocals=not args.no_vocals,
        include_keys=args.keys and not args.no_keys,
        include_video=args.include_video,
        device=args.device,
    )
    
    pipeline.process_batch(Path(args.input_dir), continue_from=args.continue_from)


if __name__ == "__main__":
    main()
