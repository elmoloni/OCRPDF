"""Two-track recorder: microphone on one WAV, PC sound output (WASAPI loopback) on the other.

Both tracks share one clock (t0), so a word at 00:12:03 in mic.wav and a word at
00:12:03 in system.wav were spoken at the same moment. Gaps in either stream
(WASAPI loopback delivers nothing while the PC is silent) are padded with zeros.
"""
from __future__ import annotations

import queue
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudiowpatch as pyaudio

CHUNK_SECONDS = 0.1
GAP_TOLERANCE = 0.5  # seconds a stream may lag the clock before zeros are inserted


def _wasapi(p: pyaudio.PyAudio) -> dict:
    return p.get_host_api_info_by_type(pyaudio.paWASAPI)


def list_devices() -> tuple[list[str], list[str]]:
    """Return (microphones, outputs) as WASAPI device names; first entry is the Windows default."""
    p = pyaudio.PyAudio()
    try:
        api = _wasapi(p)
        mics, outs = [], []
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d["hostApi"] != api["index"] or d.get("isLoopbackDevice"):
                continue
            if d["maxInputChannels"] > 0:
                mics.append(d["name"])
            if d["maxOutputChannels"] > 0:
                outs.append(d["name"])
        def_mic = p.get_device_info_by_index(api["defaultInputDevice"])["name"] if api["defaultInputDevice"] >= 0 else None
        def_out = p.get_device_info_by_index(api["defaultOutputDevice"])["name"] if api["defaultOutputDevice"] >= 0 else None
        mics.sort(key=lambda n: n != def_mic)
        outs.sort(key=lambda n: n != def_out)
        return mics, outs
    finally:
        p.terminate()


class _Track:
    def __init__(self, path: Path, channels: int, rate: int):
        self.path = path
        self.channels = channels
        self.rate = rate
        self.frames = 0
        self.level = 0.0
        self.wf = wave.open(str(path), "wb")
        self.wf.setnchannels(channels)
        self.wf.setsampwidth(2)
        self.wf.setframerate(rate)

    def write(self, data: bytes, arrived: float, t0: float) -> None:
        n = len(data) // (2 * self.channels)
        # Where this chunk should start on the shared clock vs. where the file currently ends
        expected_start = int((arrived - t0) * self.rate) - n
        if expected_start - self.frames > GAP_TOLERANCE * self.rate:
            self._silence(expected_start - self.frames)
        self.wf.writeframes(data)
        self.frames += n
        a = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        self.level = float(np.sqrt(np.mean(a * a)) / 32768.0) if a.size else 0.0

    def pad_to(self, seconds: float) -> None:
        missing = int(seconds * self.rate) - self.frames
        if missing > 0:
            self._silence(missing)

    def _silence(self, frames: int) -> None:
        self.wf.writeframes(b"\x00" * (frames * 2 * self.channels))
        self.frames += frames

    def close(self) -> None:
        self.wf.close()


class DualRecorder:
    """Record mic + PC sound to <out_dir>/mic.wav and <out_dir>/system.wav."""

    def __init__(self, out_dir: Path, mic_name: Optional[str] = None, output_name: Optional[str] = None):
        self.out_dir = Path(out_dir)
        self.mic_name = mic_name
        self.output_name = output_name
        self.p: Optional[pyaudio.PyAudio] = None
        self.streams: list = []
        self.tracks: dict[str, _Track] = {}
        self.q: queue.Queue = queue.Queue()
        self.writer: Optional[threading.Thread] = None
        self.t0 = 0.0
        self.running = False

    # ---------------------------------------------------------------- devices
    def _find(self, name: Optional[str], want_input: bool) -> dict:
        api = _wasapi(self.p)
        if name:
            for i in range(self.p.get_device_count()):
                d = self.p.get_device_info_by_index(i)
                if (d["hostApi"] == api["index"] and d["name"] == name and not d.get("isLoopbackDevice")
                        and (d["maxInputChannels"] if want_input else d["maxOutputChannels"]) > 0):
                    return d
        idx = api["defaultInputDevice"] if want_input else api["defaultOutputDevice"]
        if idx < 0:
            raise RuntimeError("No default %s device found" % ("microphone" if want_input else "output"))
        return self.p.get_device_info_by_index(idx)

    def _loopback_for(self, speakers: dict) -> dict:
        for lb in self.p.get_loopback_device_info_generator():
            if speakers["name"] in lb["name"]:
                return lb
        raise RuntimeError(f"No loopback device for '{speakers['name']}'")

    # ---------------------------------------------------------------- control
    def start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.p = pyaudio.PyAudio()
        mic = self._find(self.mic_name, want_input=True)
        speakers = self._find(self.output_name, want_input=False)
        loop = self._loopback_for(speakers)

        sources = {
            "mic": (mic, min(int(mic["maxInputChannels"]), 2)),
            "system": (loop, min(int(loop["maxInputChannels"]), 2)),
        }
        self.devices = {"mic": mic["name"], "system": speakers["name"]}
        for key, (dev, ch) in sources.items():
            self.tracks[key] = _Track(self.out_dir / f"{key}.wav", ch, int(dev["defaultSampleRate"]))

        self.running = True
        self.writer = threading.Thread(target=self._write_loop, daemon=True)
        self.writer.start()
        self.t0 = time.perf_counter()

        # Silent playback keeps the loopback stream ticking while nobody is talking
        out_rate = int(speakers["defaultSampleRate"])
        out_ch = max(1, min(int(speakers["maxOutputChannels"]), 2))

        def silent(in_data, frame_count, time_info, status):
            return (b"\x00" * (frame_count * out_ch * 2), pyaudio.paContinue)

        self.streams.append(self.p.open(format=pyaudio.paInt16, channels=out_ch, rate=out_rate, output=True,
                                        output_device_index=speakers["index"], stream_callback=silent))

        for key, (dev, ch) in sources.items():
            rate = int(dev["defaultSampleRate"])

            def cb(in_data, frame_count, time_info, status, key=key):
                self.q.put((key, in_data, time.perf_counter()))
                return (None, pyaudio.paContinue)

            self.streams.append(self.p.open(format=pyaudio.paInt16, channels=ch, rate=rate, input=True,
                                            input_device_index=dev["index"],
                                            frames_per_buffer=int(rate * CHUNK_SECONDS), stream_callback=cb))

    def _write_loop(self) -> None:
        while self.running or not self.q.empty():
            try:
                key, data, arrived = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            self.tracks[key].write(data, arrived, self.t0)

    def stop(self) -> tuple[Path, Path]:
        elapsed = self.elapsed
        for s in self.streams:
            try:
                s.stop_stream()
                s.close()
            except Exception:
                pass
        self.streams.clear()
        self.running = False
        if self.writer:
            self.writer.join()
        for t in self.tracks.values():
            t.pad_to(elapsed)  # equal length, so trailing silence on one side doesn't shift anything
            t.close()
        if self.p:
            self.p.terminate()
            self.p = None
        return self.tracks["mic"].path, self.tracks["system"].path

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0 if self.running else 0.0

    @property
    def levels(self) -> tuple[float, float]:
        return (self.tracks["mic"].level if "mic" in self.tracks else 0.0,
                self.tracks["system"].level if "system" in self.tracks else 0.0)
