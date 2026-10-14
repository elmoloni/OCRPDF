"""Transcribe a Notetaker recording (German / English) with speaker labels.

Two-track mode (from the recorder):
    python transcribe.py --mic mic.wav --system system.wav --me Elias --speakers 3
    -> mic track is always you; the system track is diarized into Speaker 1..n

Single-file mode (any prerecorded audio / video):
    python transcribe.py --file call.m4a --speakers 4
    -> the whole file is diarized into Speaker 1..n

Speakers are separated first; each speaker turn is then transcribed on its own with its own
language (Whisper otherwise locks one language per 30 s window and drops or translates the other).

Writes <title>.json (data + speaker names) and <title>.md next to the audio, where "<title> - mic.wav"
gives the title (plain mic.wav -> transcript.*).
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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

SR = 16000
ME = "ME"
OTHERS = "SPEAKER_00"     # label for the other side when it is not separated
LANGS = ("de", "en")       # "auto" chooses between these per turn
MAX_TURN = 29.0            # seconds per Whisper clip (30 s window minus padding)
PAD = 0.2                  # seconds of context around each turn
BATCH = 8                  # Whisper windows per GPU call (fits the 4 GB RTX A500 next to the model)
LID_BATCH = 32             # turns per language-detection call (small model)
# Packing several turns per 30 s window: 2 s gaps + Whisper timestamp tokens gave 0.6% word error rate vs 2.9%
# for one turn per window, at half the time (10-min DE/EN test call); without timestamps Whisper skips turns.
GAP = 2.0                  # seconds of silence between turns packed into one window
PACK_SECONDS = 29.4        # fill Whisper windows up to this length
WITHOUT_TIMESTAMPS = False
LID_MODEL = "base"         # language detection only
TURN_GAP = 1.0             # same-speaker turns closer than this are transcribed together
MIN_TURN = 0.8             # shorter diarization turns are handed to the neighbouring speaker
LANG_MIN = 2.0             # shorter turns keep the speaker's previous language unless detection is clear
MERGE_GAP = 1.5            # transcript lines of the same speaker closer than this are joined
ECHO_SIMILARITY = 0.6      # share of a mic line's words also heard on the system track at that moment = bleed, dropped
HALLUCINATION = re.compile(r"untertitel|amara\.org|thanks for watching|vielen dank fürs zuschauen", re.I)


def log(msg: str) -> None:
    print(msg, flush=True)


class Lap:
    """Log how long each step took (shown in the app log)."""
    def __init__(self):
        self.t = time.time()

    def __call__(self, what: str) -> None:
        now = time.time()
        log(f"  [{what}: {now - self.t:.1f} s]")
        self.t = now


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


def wait_for_gpu():
    """One transcription at a time across all Notetaker windows: two would not fit into 4 GB of GPU memory.
    The lock is a byte-range lock on a file; Windows releases it when this process ends, even on a crash."""
    import msvcrt
    path = Path.home() / ".notetaker" / "gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")
    waited = False
    while True:
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return f  # keep the handle open for the lifetime of the process
        except OSError:
            if not waited:
                log("Waiting for another transcription to finish ...")
                progress("wait", 0.0)
                waited = True
            time.sleep(2)


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
    if device == "cuda":
        half_precision_embeddings(pipeline)

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


def split_long_turns(audio, turns: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Cut turns longer than MAX_TURN (monologues) at short pauses, so no Whisper window loses its tail.
    Silero VAD with max_speech_duration_s splits at the last pause before the limit; without any pause, hard cut."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    out: list[tuple[float, float, str]] = []
    for s, e, spk in turns:
        if e - s <= MAX_TURN:
            out.append((s, e, spk))
            continue
        a = audio[int(s * SR):int(e * SR)]
        pieces = get_speech_timestamps(a, VadOptions(min_silence_duration_ms=100, speech_pad_ms=100,
                                                     max_speech_duration_s=MAX_TURN - 1)) or [{"start": 0, "end": len(a)}]
        chunk: list[float] = []  # pauses split finely; re-join pieces up to MAX_TURN so cuts happen only where needed
        for piece in pieces:
            ps, pe = s + piece["start"] / SR, min(e, s + piece["end"] / SR)
            if chunk and pe - chunk[0] <= MAX_TURN:
                chunk[1] = pe
                continue
            if chunk:
                out.append((chunk[0], chunk[1], spk))
            while pe - ps > MAX_TURN:
                out.append((ps, ps + MAX_TURN, spk))
                ps += MAX_TURN
            chunk = [ps, pe]
        if chunk:
            out.append((chunk[0], chunk[1], spk))
    return out


def half_precision_embeddings(pipeline) -> None:
    """Run the speaker-embedding ResNet in float16: ~2x faster, same speakers (99% of frames identical in tests).
    The filterbank in front of it must stay float32, or all voices collapse into one."""
    import torch
    resnet = getattr(getattr(getattr(pipeline, "_embedding", None), "model_", None), "resnet", None)
    if resnet is None:
        return
    forward = resnet.forward

    def half_forward(*args, **kwargs):
        with torch.autocast("cuda", dtype=torch.float16):
            out = forward(*args, **kwargs)
        return tuple(o.float() if torch.is_tensor(o) else o for o in out) if isinstance(out, tuple) else out.float()

    resnet.forward = half_forward


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
    # cpu_threads: faster-whisper defaults to 4; use every core (feature extraction, and all of it on CPU)
    return WhisperModel(model_name, device=device, compute_type=compute, cpu_threads=os.cpu_count() or 4)


def turn_clips(audio, turns) -> list[tuple[float, float]]:
    """Audio span per turn, padded a little so first and last syllables are not clipped."""
    total = len(audio) / SR
    return [(max(0.0, s - PAD), min(total, e + PAD)) for s, e, _ in turns]


def detect_languages(lid_model, audio, turns, clips) -> list[str]:
    """German or English per turn, from the small Whisper 'base' model (batched; ~100x cheaper than turbo)."""
    import numpy as np
    from faster_whisper.audio import pad_or_trim
    probs: list[dict[str, float]] = []
    for b in range(0, len(clips), LID_BATCH):
        feats = np.stack([pad_or_trim(lid_model.feature_extractor(audio[int(s * SR):int(e * SR)])[..., :-1])
                          for s, e in clips[b:b + LID_BATCH]])
        for res in lid_model.model.detect_language(lid_model.encode(feats)):
            probs.append({tok[2:-2]: pr for tok, pr in res})
        progress("language", min(1.0, (b + LID_BATCH) / len(clips)))
    langs: list[str] = []
    last: dict[str, str] = {}
    for (s, e, spk), p in zip(turns, probs):
        best = max(LANGS, key=lambda lang: p.get(lang, 0.0))
        if e - s < LANG_MIN and p.get(best, 0.0) < 0.8:  # too short to tell: keep the speaker's language
            best = last.get(spk) or next(iter(last.values()), LANGS[0])
        last[spk] = best
        langs.append(best)
    return langs


def pack_windows(idx: list[int], clips) -> list[list[int]]:
    """Fill 30 s Whisper windows with consecutive same-language turns (the encoder costs the same for 3 s or 30 s)."""
    windows: list[list[int]] = []
    used = MAX_TURN + 1
    for i in idx:
        dur = clips[i][1] - clips[i][0]
        if windows and used + GAP + dur <= PACK_SECONDS:
            windows[-1].append(i)
            used += GAP + dur
        else:
            windows.append([i])
            used = dur
    return windows


def transcribe_turns(model, lid_model, audio, turns, language: str) -> list[dict]:
    """Every turn transcribed in its own language. Same-language turns are packed into full windows with
    GAP seconds of silence between them, transcribed in batches, and split back by word timestamps."""
    import numpy as np
    from faster_whisper import BatchedInferencePipeline
    if not turns:
        return []
    turns = sorted(turns)
    clips = turn_clips(audio, turns)
    langs = detect_languages(lid_model, audio, turns, clips) if language == "auto" else [language] * len(turns)
    pipe = BatchedInferencePipeline(model)
    words: dict[int, list[str]] = {i: [] for i in range(len(turns))}
    plan = {lang: pack_windows([i for i, l in enumerate(langs) if l == lang], clips) for lang in sorted(set(langs))}
    n_windows = sum(len(w) for w in plan.values())
    done = 0
    for lang, windows in plan.items():
        # Synthetic audio: windows one after another; remember where each turn sits in it
        parts, places, wins, pos = [], [], [], 0.0
        for w in windows:
            w_start = pos
            for k, i in enumerate(w):
                if k:
                    parts.append(np.zeros(int(GAP * SR), np.float32))
                    pos += GAP
                chunk = audio[int(clips[i][0] * SR):int(clips[i][1] * SR)]
                parts.append(chunk)
                places.append((pos, pos + len(chunk) / SR, i))
                pos += len(chunk) / SR
            wins.append({"start": w_start, "end": pos})
            parts.append(np.zeros(SR, np.float32))
            pos += 1.0
        syn = np.concatenate(parts)
        segments, _ = pipe.transcribe(syn, language=lang, vad_filter=False, batch_size=BATCH, beam_size=5,
                                      condition_on_previous_text=False, without_timestamps=WITHOUT_TIMESTAMPS,
                                      word_timestamps=True, clip_timestamps=wins)
        for seg in segments:
            done += 1
            progress("text", done / n_windows)
            # Skip silence guesses and repetition loops ("the link to the link to the link")
            if (seg.no_speech_prob > 0.6 and seg.avg_logprob < -0.8) or seg.compression_ratio > 2.4:
                continue
            for w in (seg.words or [type("W", (), {"start": seg.start, "end": seg.end, "word": seg.text})]):
                mid = (w.start + w.end) / 2
                i = min(places, key=lambda pl: 0.0 if pl[0] <= mid <= pl[1] else min(abs(pl[0] - mid), abs(pl[1] - mid)))[2]
                words[i].append(w.word)
    progress("text", 1.0)
    lines: list[dict] = []
    for i, (s, e, spk) in enumerate(turns):
        text = "".join(words[i]).strip()
        if text and not (HALLUCINATION.search(text) and len(text) < 80):
            lines.append({"start": s, "end": e, "speaker": spk, "text": text, "lang": langs[i]})
    log("  turns per language: " + ", ".join(f"{lang} {langs.count(lang)}" for lang in sorted(set(langs)))
        + f"; {n_windows} Whisper windows for {len(turns)} turns")
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


def render_text(data: dict) -> str:
    """Plain text for pasting into Word / OneNote / mails: one line per utterance, no Markdown."""
    names = data["speakers"]
    lines = [
        data.get("title") or "Transcript",
        f"Recorded {data.get('recorded', '')}, duration {fmt_ts(data.get('duration', 0))}, "
        f"speakers: {', '.join(names[k] for k in names)}",
        "",
    ]
    lines += [f"[{fmt_ts(u['start'])}] {names.get(u['speaker'], u['speaker'])}: {u['text']}" for u in data["utterances"]]
    return "\n".join(lines)


def session_base(mic: Path) -> str:
    """'<title> - mic.wav' -> '<title>' (transcript name); a plain 'mic.wav' from older recordings -> 'transcript'."""
    return re.sub(r"(?:\s*-\s*)?mic$", "", mic.stem, flags=re.I).strip() or "transcript"


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
    gpu_lock = wait_for_gpu()  # noqa: F841 (held until exit)
    lap = Lap()
    device = pick_device(args.device)
    n_spk = args.speakers or None
    two_track = bool(args.mic)

    if two_track:
        with ThreadPoolExecutor(2) as ex:  # decoding and VAD release the GIL: both tracks at once
            mic_audio, sys_audio = ex.map(load_audio, (Path(args.mic), Path(args.system)))
        duration = max(len(mic_audio), len(sys_audio)) / SR
        out_dir = Path(args.out or Path(args.mic).parent)
        json_path = out_dir / f"{session_base(Path(args.mic))}.json"
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
    lap("load audio")

    # 1. Who speaks when (diarization first, so every Whisper clip holds one speaker and one language)
    with ThreadPoolExecutor(2) as ex:
        mine_f = ex.submit(speech_turns, mic_audio, ME) if two_track else None
        other_turns = speech_turns(sys_audio, OTHERS)
        my_turns = mine_f.result() if mine_f else []
    lap("voice activity")
    if other_turns and n_spk != 1:
        try:
            other_turns = diarize(sys_audio, device, n_spk)
        except Exception as e:
            log(f"WARNING: speaker separation failed, the other side stays one speaker: {e}")
        lap("speaker separation")
    log(f"Speech turns: {len(my_turns)} mic, {len(other_turns)} " + ("PC sound" if two_track else "file"))

    # 2. What was said: both tracks in one batched run (PC sound placed after the mic audio plus 1 s of silence)
    import numpy as np
    off = len(mic_audio) / SR + 1.0 if two_track else 0.0
    audio = np.concatenate([mic_audio, np.zeros(SR, np.float32), sys_audio]) if two_track else sys_audio
    my_chunks = split_long_turns(mic_audio, group_turns(my_turns)) if two_track else []
    other_chunks = split_long_turns(sys_audio, group_turns(other_turns))
    turns = my_chunks + [(s + off, e + off, k) for s, e, k in other_chunks]
    model = load_whisper(args.model, device)
    lid_model = load_whisper(LID_MODEL, device) if args.language == "auto" else None
    lap("load Whisper")
    log(f"Transcribing {len(turns)} turns ...")
    lines = transcribe_turns(model, lid_model, audio, turns, args.language)
    lap("transcribe")
    del model, lid_model
    free_gpu()
    mine = [ln for ln in lines if ln["speaker"] == ME]
    others = [dict(ln, start=ln["start"] - off, end=ln["end"] - off) for ln in lines if ln["speaker"] != ME]

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
