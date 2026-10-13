"""Transcribe a Notetaker recording (German / English) with speaker labels.

Two-track mode (from the recorder):
    python transcribe.py --mic mic.wav --system system.wav --me Elias --speakers 3
    -> mic track is always you; the system track is diarized into Speaker 1..n

Single-file mode (any prerecorded audio / video):
    python transcribe.py --file call.m4a --speakers 4
    -> the whole file is diarized into Speaker 1..n

Speakers are separated first; each speaker turn is then transcribed on its own with its own
language (Whisper otherwise locks one language per 30 s window and drops or translates the other).

Writes transcript.json (data + speaker names) and transcript.md next to the audio.
Lines starting with "@progress " are machine-readable progress for the GUI.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

SR = 16000
ME = "ME"
OTHERS = "SPEAKER_00"     # label for the other side when it is not separated
LANGS = ("de", "en")       # "auto" chooses between these per turn
MAX_TURN = 30.0            # seconds of audio per Whisper call
TURN_GAP = 1.0             # same-speaker turns closer than this are transcribed together
MIN_TURN = 0.8             # shorter diarization turns are handed to the neighbouring speaker
LANG_MIN = 2.0             # shorter turns keep the speaker's previous language unless detection is clear
MERGE_GAP = 1.5            # transcript lines of the same speaker closer than this are joined
ECHO_SIMILARITY = 0.6      # share of a mic line's words also heard on the system track at that moment = bleed, dropped
HALLUCINATION = re.compile(r"untertitel|amara\.org|thanks for watching|vielen dank fürs zuschauen", re.I)


def log(msg: str) -> None:
    print(msg, flush=True)


def progress(stage: str, frac: float) -> None:
    print(f"@progress {stage} {max(0.0, min(1.0, frac)):.3f}", flush=True)


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


# ------------------------------------------------------------------------------------------
# Runtime setup
# ------------------------------------------------------------------------------------------
def pick_device(requested: str) -> str:
    """faster-whisper (CTranslate2) is built for CUDA 12: point it at the nvidia-cublas/cudnn-cu12 wheels.

    torch (CUDA 13) brings its own DLLs and does not need these folders.
    """
    import logging
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)  # "triton not found", irrelevant here
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        import nvidia
        bins = [p / "bin" for root in nvidia.__path__ for p in Path(root).iterdir() if (p / "bin").is_dir()]
        for b in bins:
            os.add_dll_directory(str(b))
        os.environ["PATH"] = os.pathsep.join(map(str, bins)) + os.pathsep + os.environ.get("PATH", "")
        return "cuda"
    except Exception as e:
        if requested == "cuda":
            raise
        log(f"GPU not usable ({e}), running on CPU")
        return "cpu"


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_audio(path: Path):
    """Any audio/video file -> 16 kHz mono float32 (own PyAV loop: faster-whisper's decode_audio breaks on PyAV 19)."""
    import av
    import numpy as np
    chunks = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
        for frame in container.decode(audio=0):
            chunks += [f.to_ndarray() for f in resampler.resample(frame)]
        chunks += [f.to_ndarray() for f in resampler.resample(None)]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=1).reshape(-1).astype(np.float32) / 32768.0


# ------------------------------------------------------------------------------------------
# Who speaks when
# ------------------------------------------------------------------------------------------
def speech_turns(audio, speaker: str) -> list[tuple[float, float, str]]:
    """Voice activity (Silero VAD) as turns of one speaker."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    if len(audio) < SR // 2:
        return []
    ts = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500, speech_pad_ms=200))
    return [(t["start"] / SR, t["end"] / SR, speaker) for t in ts]


def diarize(audio, device: str, num_speakers: int | None) -> list[tuple[float, float, str]]:
    """Speaker turns (start, end, label) via pyannote. Needs a Hugging Face login (see README)."""
    import torch
    import pyannote.audio
    from pyannote.audio import Pipeline

    v4 = int(pyannote.audio.__version__.split(".")[0]) >= 4
    model_id = "pyannote/speaker-diarization-community-1" if v4 else "pyannote/speaker-diarization-3.1"
    token = os.environ.get("HF_TOKEN")  # None = use the cached `hf auth login`
    log(f"Loading diarization model {model_id} ...")
    pipeline = Pipeline.from_pretrained(model_id, **({"token": token} if v4 else {"use_auth_token": token or True}))
    if pipeline is None:
        raise RuntimeError("model download refused: accept its terms on huggingface.co and run `hf auth login`")
    pipeline.to(torch.device(device))

    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    log("Separating speakers" + (f" ({num_speakers})" if num_speakers else " (count auto)") + " ...")
    progress("diarize", 0.0)
    result = pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SR}, **kwargs)
    ann = getattr(result, "exclusive_speaker_diarization", None) or getattr(result, "speaker_diarization", result)
    progress("diarize", 1.0)
    turns = [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]
    del pipeline
    free_gpu()
    return turns


def group_turns(turns: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Hand slivers to the nearest neighbouring speaker, then join same-speaker fragments into chunks of at most MAX_TURN seconds.

    Diarization often flips speaker for a few hundred ms at the start or end of a sentence;
    transcribed alone those slivers come out as wrong-language noise.
    """
    turns = sorted(t for t in turns if t[1] > t[0])
    long_turns = [t for t in turns if t[1] - t[0] >= MIN_TURN]
    out: list[tuple[float, float, str]] = []
    for s, e, spk in turns:
        if e - s < MIN_TURN and long_turns:
            near = min(long_turns, key=lambda t: max(t[0] - e, s - t[1], 0.0))
            if max(near[0] - e, s - near[1], 0.0) < TURN_GAP:
                spk = near[2]
        if out and out[-1][2] == spk and s - out[-1][1] < TURN_GAP and e - out[-1][0] <= MAX_TURN:
            out[-1] = (out[-1][0], max(e, out[-1][1]), spk)
        else:
            out.append((s, e, spk))
    return [t for t in out if t[1] - t[0] >= 0.3]


# ------------------------------------------------------------------------------------------
# Speech to text
# ------------------------------------------------------------------------------------------
def load_whisper(model_name: str, device: str):
    from faster_whisper import WhisperModel
    compute = "int8_float16" if device == "cuda" else "int8"
    log(f"Loading Whisper '{model_name}' on {device} ({compute}) ...")
    return WhisperModel(model_name, device=device, compute_type=compute)


def pick_language(model, chunk, fallback: str) -> str:
    _, _, probs = model.detect_language(chunk)
    p = dict(probs)
    best = max(LANGS, key=lambda lang: p.get(lang, 0.0))
    if len(chunk) < LANG_MIN * SR and p.get(best, 0.0) < 0.8:
        return fallback
    return best


def transcribe_turns(model, audio, turns, language: str, stage: str) -> list[dict]:
    """One Whisper call per speaker turn, each with its own language."""
    lines: list[dict] = []
    last_lang: dict[str, str] = {}
    used: dict[str, int] = {}
    for i, (s, e, spk) in enumerate(turns):
        chunk = audio[int(max(0.0, s - 0.2) * SR):int((e + 0.2) * SR)]
        if language == "auto":
            lang = pick_language(model, chunk, last_lang.get(spk) or next(iter(last_lang.values()), LANGS[0]))
            last_lang[spk] = lang
        else:
            lang = language
        segments, _ = model.transcribe(chunk, language=lang, beam_size=5, condition_on_previous_text=False,
                                       vad_filter=False)
        # Skip silence guesses and repetition loops ("the link to the link to the link")
        text = " ".join(seg.text.strip() for seg in segments
                        if not (seg.no_speech_prob > 0.6 and seg.avg_logprob < -0.8) and seg.compression_ratio <= 2.4).strip()
        if text and not (HALLUCINATION.search(text) and len(text) < 80):
            lines.append({"start": s, "end": e, "speaker": spk, "text": text, "lang": lang})
            used[lang] = used.get(lang, 0) + 1
        progress(stage, (i + 1) / len(turns))
    if used:
        log("  turns per language: " + ", ".join(f"{k} {v}" for k, v in sorted(used.items())))
    return lines


# ------------------------------------------------------------------------------------------
# Merging
# ------------------------------------------------------------------------------------------
def drop_echo(mine: list[dict], others: list[dict]) -> tuple[list[dict], int]:
    """Without headphones the mic also hears the call; drop mic lines that repeat what the system track says."""
    kept, dropped = [], 0
    for m in mine:
        words = _words(m["text"])
        overlap = set(_words(" ".join(o["text"] for o in others if o["start"] < m["end"] + 1 and o["end"] > m["start"] - 1)))
        # Short replies ("ja", "okay") are too ambiguous to call echo; keep them
        if len(words) >= 3 and sum(w in overlap for w in words) / len(words) >= ECHO_SIMILARITY:
            dropped += 1
        else:
            kept.append(m)
    return kept, dropped


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def merge_lines(lines: list[dict]) -> list[dict]:
    lines = sorted(lines, key=lambda x: x["start"])
    out: list[dict] = []
    for ln in lines:
        if out and out[-1]["speaker"] == ln["speaker"] and ln["start"] - out[-1]["end"] < MERGE_GAP:
            out[-1]["text"] += " " + ln["text"]
            out[-1]["end"] = max(out[-1]["end"], ln["end"])
        else:
            out.append(dict(ln))
    return out


def name_speakers(lines: list[dict], me: str | None) -> dict[str, str]:
    """ME -> your name; diarization labels -> Speaker 1..n in order of first appearance."""
    names: dict[str, str] = {}
    n = 0
    for ln in lines:
        key = ln["speaker"]
        if key in names:
            continue
        if key == ME:
            names[key] = me or "Me"
        else:
            n += 1
            names[key] = f"Speaker {n}"
    return names


# ------------------------------------------------------------------------------------------
# Output (also used by the GUI's rename dialog)
# ------------------------------------------------------------------------------------------
def render_markdown(data: dict) -> str:
    names = data["speakers"]
    lines = [
        f"# {data.get('title') or 'Transcript'}",
        "",
        f"- Recorded: {data.get('recorded', '')}",
        f"- Source: {', '.join(data.get('sources', []))}",
        f"- Duration: {fmt_ts(data.get('duration', 0))}",
        f"- Speakers: {', '.join(names[k] for k in names)}",
        "",
    ]
    for u in data["utterances"]:
        lines.append(f"**[{fmt_ts(u['start'])}] {names.get(u['speaker'], u['speaker'])}:** {u['text']}")
        lines.append("")
    return "\n".join(lines)


def save(data: dict, json_path: Path) -> Path:
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    md_path.write_text(render_markdown(data), encoding="utf-8")
    return md_path


# ------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> Path:
    t_start = time.time()
    device = pick_device(args.device)
    n_spk = args.speakers or None
    two_track = bool(args.mic)

    if two_track:
        mic_audio, sys_audio = load_audio(Path(args.mic)), load_audio(Path(args.system))
        duration = max(len(mic_audio), len(sys_audio)) / SR
        out_dir = Path(args.out or Path(args.mic).parent)
        json_path = out_dir / "transcript.json"
        sources = [Path(args.mic).name, Path(args.system).name]
        recorded = datetime.fromtimestamp(Path(args.mic).stat().st_mtime - duration)
    else:
        sys_audio = load_audio(Path(args.file))
        duration = len(sys_audio) / SR
        out_dir = Path(args.out or Path(args.file).parent)
        json_path = out_dir / f"{Path(args.file).stem}_transcript.json"
        sources = [Path(args.file).name]
        recorded = datetime.fromtimestamp(Path(args.file).stat().st_mtime)
    log(f"Audio: {fmt_ts(duration)}")

    # 1. Who speaks when (diarization first, so every Whisper call sees one speaker and one language)
    my_turns = speech_turns(mic_audio, ME) if two_track else []
    other_turns = speech_turns(sys_audio, OTHERS)
    if other_turns and n_spk != 1:
        try:
            other_turns = diarize(sys_audio, device, n_spk)
        except Exception as e:
            log(f"WARNING: speaker separation failed, the other side stays one speaker: {e}")
    log(f"Speech turns: {len(my_turns)} mic, {len(other_turns)} " + ("PC sound" if two_track else "file"))

    # 2. What was said
    model = load_whisper(args.model, device)
    mine: list[dict] = []
    if my_turns:
        log("Transcribing your microphone ...")
        mine = transcribe_turns(model, mic_audio, group_turns(my_turns), args.language, "mic")
    others: list[dict] = []
    if other_turns:
        log("Transcribing " + ("the other side ..." if two_track else "file ..."))
        others = transcribe_turns(model, sys_audio, group_turns(other_turns), args.language, "system")
    del model
    free_gpu()

    if mine and others:
        mine, dropped = drop_echo(mine, others)
        if dropped:
            log(f"Dropped {dropped} mic line(s) that were the call audio picked up by your microphone")

    lines = merge_lines(mine + others)
    data = {
        "title": args.title or recorded.strftime("Transcript %d.%m.%Y %H:%M"),
        "recorded": recorded.strftime("%d.%m.%Y %H:%M"),
        "sources": sources,
        "duration": duration,
        "model": args.model,
        "speakers": name_speakers(lines, args.me),
        "utterances": [{"start": round(u["start"], 2), "end": round(u["end"], 2),
                        "speaker": u["speaker"], "text": u["text"]} for u in lines],
    }
    md_path = save(data, json_path)
    log(f"Done in {fmt_ts(time.time() - t_start)}: {md_path}")
    return md_path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Transcribe a two-track recording or a single audio file with speaker labels")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mic", help="microphone track (you)")
    src.add_argument("--file", help="single prerecorded file, every voice diarized")
    ap.add_argument("--system", help="PC sound track (the other side); required with --mic")
    ap.add_argument("--me", default=None, help="your name for the mic track")
    ap.add_argument("--speakers", type=int, default=0, help="number of speakers to separate (other side in two-track mode); 0 = auto")
    ap.add_argument("--language", default="auto", choices=["auto", "de", "en"])
    ap.add_argument("--model", default="large-v3-turbo", help="Whisper model: large-v3-turbo, large-v3, medium, small")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None, help="output folder (default: next to the audio)")
    args = ap.parse_args()
    if args.mic and not args.system:
        ap.error("--mic needs --system")
    run(args)


if __name__ == "__main__":
    main()
