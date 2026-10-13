"""Notetaker: record a call on two tracks (your mic / PC sound) and transcribe it with speaker labels.

Each recording goes to <output folder>/<YYYY MM DD - title>/ with mic.wav, system.wav,
transcript.md and transcript.json. Transcription runs transcribe.py in a child process.
"""
from __future__ import annotations

import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recorder import DualRecorder, list_devices  # noqa: E402
import transcribe  # noqa: E402  (heavy libraries load only inside its functions)

CONFIG = Path.home() / ".notetaker" / "config.json"
LANGUAGES = {"Auto (DE/EN mixed)": "auto", "Deutsch": "de", "English": "en"}
MODELS = ["large-v3-turbo", "large-v3", "medium", "small"]
STAGES = {"mic": "Transcribing your mic", "system": "Transcribing the other side", "diarize": "Separating speakers"}


def documents_dir() -> Path:
    """Real Documents folder, also when Windows redirects it into OneDrive."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as k:
            return Path(os.path.expandvars(winreg.QueryValueEx(k, "Personal")[0]))
    except OSError:
        return Path.home() / "Documents"


def load_config() -> dict:
    cfg = {"me": "", "out_dir": str(documents_dir() / "Notetaker"), "language": "auto", "speakers": "auto",
           "model": MODELS[0], "mic": "", "output": "", "auto_transcribe": True}
    try:
        cfg.update(json.loads(CONFIG.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")


def date_prefix() -> str:
    return datetime.now().strftime("%Y %m %d - ")


def unique_folder(parent: Path, name: str) -> Path:
    folder, n = parent / name, 2
    while folder.exists():
        folder, n = parent / f"{name} ({n})", n + 1
    return folder


def level_to_meter(rms: float) -> float:
    return max(0.0, min(100.0, (20 * math.log10(rms + 1e-9) + 60) / 60 * 100))  # -60 dBFS .. 0 dBFS


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.first_run = not CONFIG.exists()
        self.cfg = load_config()
        self.rec: DualRecorder | None = None
        self.proc: subprocess.Popen | None = None
        self.jobs: list[tuple[list[str], Path]] = []  # queued (args, expected json path)
        self.msgs: queue.Queue = queue.Queue()
        self.last_json: Path | None = None

        root.title("Notetaker")
        root.minsize(620, 520)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build()
        self.refresh_devices()
        self.ent_title.focus_set()
        self.ent_title.icursor("end")
        self.root.after(100, self.tick)
        if self.first_run:
            self.root.after(300, self.first_start)
        elif not self.cfg["me"]:
            self.root.after(300, self.ask_name)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        pad = {"padx": 6, "pady": 3}
        f = ttk.Frame(self.root, padding=10)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)

        self.v_me = tk.StringVar(value=self.cfg["me"])
        self.v_title = tk.StringVar(value=date_prefix())
        self.v_mic = tk.StringVar()
        self.v_out = tk.StringVar()
        self.v_lang = tk.StringVar(value=next(k for k, v in LANGUAGES.items() if v == self.cfg["language"]))
        self.v_spk = tk.StringVar(value=str(self.cfg["speakers"]))
        self.v_model = tk.StringVar(value=self.cfg["model"])
        self.v_auto = tk.BooleanVar(value=self.cfg["auto_transcribe"])
        self.v_dir = tk.StringVar(value=self.cfg["out_dir"])

        r = 0
        ttk.Label(f, text="You").grid(row=r, column=0, sticky="w", **pad)
        ttk.Label(f, textvariable=self.v_me).grid(row=r, column=1, sticky="w", **pad)
        ttk.Button(f, text="Change...", width=10, command=self.ask_name).grid(row=r, column=2, **pad)
        r += 1
        ttk.Label(f, text="Title").grid(row=r, column=0, sticky="w", **pad)
        self.ent_title = ttk.Entry(f, textvariable=self.v_title)
        self.ent_title.grid(row=r, column=1, columnspan=2, sticky="ew", **pad)
        r += 1
        ttk.Label(f, text="Microphone (you)").grid(row=r, column=0, sticky="w", **pad)
        self.cb_mic = ttk.Combobox(f, textvariable=self.v_mic, state="readonly")
        self.cb_mic.grid(row=r, column=1, sticky="ew", **pad)
        ttk.Button(f, text="Refresh", width=8, command=self.refresh_devices).grid(row=r, column=2, **pad)
        r += 1
        ttk.Label(f, text="PC sound (the others)").grid(row=r, column=0, sticky="w", **pad)
        self.cb_out = ttk.Combobox(f, textvariable=self.v_out, state="readonly")
        self.cb_out.grid(row=r, column=1, sticky="ew", **pad)
        r += 1
        ttk.Label(f, text="Pick the device your call app plays to (e.g. your headset).",
                  foreground="gray").grid(row=r, column=1, columnspan=2, sticky="w", padx=6)
        r += 1

        opts = ttk.Frame(f)
        opts.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(6, 3))
        ttk.Label(opts, text="Language").pack(side="left", padx=(6, 3))
        ttk.Combobox(opts, textvariable=self.v_lang, values=list(LANGUAGES), state="readonly", width=18).pack(side="left")
        ttk.Label(opts, text="Other speakers").pack(side="left", padx=(12, 3))
        ttk.Combobox(opts, textvariable=self.v_spk, values=["auto"] + [str(i) for i in range(1, 11)],
                     state="readonly", width=5).pack(side="left")
        ttk.Label(opts, text="Model").pack(side="left", padx=(12, 3))
        ttk.Combobox(opts, textvariable=self.v_model, values=MODELS, width=14).pack(side="left")
        r += 1

        dirf = ttk.Frame(f)
        dirf.grid(row=r, column=0, columnspan=3, sticky="ew")
        ttk.Label(dirf, text="Save to").pack(side="left", padx=6)
        ttk.Entry(dirf, textvariable=self.v_dir).pack(side="left", fill="x", expand=True)
        ttk.Button(dirf, text="...", width=3, command=self.pick_dir).pack(side="left", padx=6)
        r += 1

        ttk.Separator(f).grid(row=r, column=0, columnspan=3, sticky="ew", pady=8)
        r += 1

        recf = ttk.Frame(f)
        recf.grid(row=r, column=0, columnspan=3, sticky="ew")
        self.btn_rec = tk.Button(recf, text="●  Record", fg="white", bg="#c62828", activebackground="#b71c1c",
                                 font=("Segoe UI", 11, "bold"), width=14, command=self.record_clicked)
        self.btn_rec.pack(side="left", padx=(6, 3))
        self.btn_stop = tk.Button(recf, text="■  Stop", font=("Segoe UI", 11, "bold"), width=8,
                                  command=self.stop_record, state="disabled")
        self.btn_stop.pack(side="left", padx=3)
        self.lbl_time = ttk.Label(recf, text="00:00:00", font=("Segoe UI", 14))
        self.lbl_time.pack(side="left", padx=10)
        meters = ttk.Frame(recf)
        meters.pack(side="left", fill="x", expand=True, padx=6)
        meters.columnconfigure(1, weight=1)
        ttk.Label(meters, text="Mic").grid(row=0, column=0, sticky="w")
        self.m_mic = ttk.Progressbar(meters, maximum=100)
        self.m_mic.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(meters, text="PC sound").grid(row=1, column=0, sticky="w")
        self.m_sys = ttk.Progressbar(meters, maximum=100)
        self.m_sys.grid(row=1, column=1, sticky="ew", padx=4)
        r += 1
        self.lbl_rec = ttk.Label(f, text="", foreground="#c62828")
        self.lbl_rec.grid(row=r, column=0, columnspan=3, sticky="w", padx=6)
        r += 1
        ttk.Checkbutton(f, text="Transcribe automatically when a recording ends", variable=self.v_auto).grid(
            row=r, column=0, columnspan=3, sticky="w", padx=6, pady=(4, 0))
        r += 1

        btns = ttk.Frame(f)
        btns.grid(row=r, column=0, columnspan=3, sticky="ew", pady=6)
        ttk.Button(btns, text="Transcribe file...", command=self.transcribe_file).pack(side="left", padx=6)
        self.btn_open = ttk.Button(btns, text="Open transcript", command=self.open_transcript, state="disabled")
        self.btn_open.pack(side="left", padx=6)
        ttk.Button(btns, text="Copy transcript", command=self.copy_transcript).pack(side="left", padx=6)
        ttk.Button(btns, text="Rename speakers...", command=self.rename_speakers).pack(side="left", padx=6)
        ttk.Button(btns, text="Open folder", command=lambda: self._startfile(self.v_dir.get())).pack(side="left", padx=6)
        r += 1

        self.lbl_status = ttk.Label(f, text="Ready")
        self.lbl_status.grid(row=r, column=0, columnspan=3, sticky="w", padx=6)
        r += 1
        self.pbar = ttk.Progressbar(f, maximum=100)
        self.pbar.grid(row=r, column=0, columnspan=3, sticky="ew", padx=6, pady=3)
        r += 1
        self.txt = tk.Text(f, height=10, wrap="word", font=("Consolas", 9), state="disabled")
        self.txt.grid(row=r, column=0, columnspan=3, sticky="nsew", padx=6, pady=3)
        f.rowconfigure(r, weight=1)

    # ------------------------------------------------------------------ helpers
    def log(self, msg: str) -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def refresh_devices(self) -> None:
        try:
            mics, outs = list_devices()
        except Exception as e:
            messagebox.showerror("Notetaker", f"Could not list audio devices:\n{e}")
            return
        self.cb_mic["values"], self.cb_out["values"] = mics, outs
        self.v_mic.set(self.cfg["mic"] if self.cfg["mic"] in mics else (mics[0] if mics else ""))
        self.v_out.set(self.cfg["output"] if self.cfg["output"] in outs else (outs[0] if outs else ""))

    def pick_dir(self) -> None:
        d = filedialog.askdirectory(initialdir=self.v_dir.get())
        if d:
            self.v_dir.set(d)

    def remember(self) -> None:
        self.cfg.update(me=self.v_me.get().strip(), out_dir=self.v_dir.get(), language=LANGUAGES[self.v_lang.get()],
                        speakers=self.v_spk.get(), model=self.v_model.get().strip() or MODELS[0],
                        mic=self.v_mic.get(), output=self.v_out.get(), auto_transcribe=self.v_auto.get())
        save_config(self.cfg)

    @staticmethod
    def _startfile(path) -> None:
        if path and Path(path).exists():
            os.startfile(str(path))

    # ------------------------------------------------------------------ settings
    def first_start(self) -> None:
        self.ask_name()
        self.ask_save_dir()

    def ask_name(self) -> None:
        """Your name labels the microphone track; asked once, changeable via 'Change...'."""
        name = simpledialog.askstring("Notetaker", "Your name (labels everything you say on your microphone):",
                                      initialvalue=self.v_me.get(), parent=self.root)
        if name and name.strip():
            self.v_me.set(name.strip())
            self.remember()

    def ask_save_dir(self) -> None:
        """First start: choose where recordings and transcripts go."""
        messagebox.showinfo("Notetaker", "Where should Notetaker save your recordings and transcripts?\n\n"
                                         "Each recording gets its own folder there. You can change this later under 'Save to'.")
        d = filedialog.askdirectory(title="Folder for recordings and transcripts",
                                    initialdir=str(documents_dir()), mustexist=False)
        if d:
            self.v_dir.set(str(Path(d)))
        self.remember()
        self.log(f"Recordings are saved to {self.v_dir.get()}")

    def record_clicked(self) -> None:
        """Record; during a recording: save it (and transcribe) and start the next one right away."""
        if self.rec:
            self.stop_record()
        self.start_record()

    def start_record(self) -> None:
        self.remember()
        title = self.v_title.get().strip()
        if not title or title == date_prefix().strip():
            title = datetime.now().strftime("%Y %m %d - %H%M")
        title = re.sub(r'[<>:"/\\|?*]+', "", title).strip().rstrip(".")
        folder = unique_folder(Path(self.v_dir.get()), title)
        try:
            self.rec = DualRecorder(folder, self.v_mic.get() or None, self.v_out.get() or None)
            self.rec.start()
        except Exception as e:
            self.rec = None
            messagebox.showerror("Notetaker", f"Could not start recording:\n{e}")
            return
        self.rec_title = folder.name
        self.v_title.set(date_prefix())  # ready for the next recording's title
        self.ent_title.icursor("end")
        self.btn_rec.configure(text="●  New recording")
        self.btn_stop.configure(state="normal")
        self.lbl_rec.configure(text=f"Recording: {folder.name}")
        self.log(f"Recording to {folder}\n  mic: {self.rec.devices['mic']}\n  PC sound: {self.rec.devices['system']}")

    def stop_record(self) -> None:
        if not self.rec:
            return
        mic, system = self.rec.stop()
        self.rec = None
        self.btn_rec.configure(text="●  Record")
        self.btn_stop.configure(state="disabled")
        self.lbl_rec.configure(text="")
        self.lbl_time.configure(text="00:00:00")
        self.m_mic["value"] = self.m_sys["value"] = 0
        self.log(f"Saved {mic.parent.name}")
        if self.v_auto.get():
            self.queue_two_track(mic, system, self.rec_title)
        else:
            self.lbl_status.configure(text=f"Recording saved in {mic.parent}")

    # ------------------------------------------------------------------ transcription
    def common_args(self, speakers: str) -> list[str]:
        self.remember()
        return ["--language", self.cfg["language"], "--model", self.cfg["model"],
                "--speakers", "0" if speakers in ("", "auto") else speakers]

    def queue_two_track(self, mic: Path, system: Path, title: str = "") -> None:
        args = ["--mic", str(mic), "--system", str(system), "--me", self.v_me.get().strip() or "Me"]
        args += self.common_args(self.v_spk.get())
        if title:
            args += ["--title", title]
        self.enqueue(args, mic.parent / "transcript.json")

    def transcribe_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Audio or video file",
            filetypes=[("Audio / video", "*.wav *.mp3 *.m4a *.mp4 *.ogg *.flac *.webm *.wma *.aac *.mkv"), ("All", "*.*")])
        if not path:
            return
        p = Path(path)
        # A Notetaker session folder: redo it in two-track mode
        if p.name.lower() in ("mic.wav", "system.wav") and (p.parent / "mic.wav").exists() and (p.parent / "system.wav").exists():
            self.queue_two_track(p.parent / "mic.wav", p.parent / "system.wav")
            return
        n = simpledialog.askstring("Speakers", "How many different voices are in this file?\n(leave empty = detect automatically)",
                                   parent=self.root)
        if n is None:
            return
        n = n.strip()
        if n and not n.isdigit():
            messagebox.showerror("Notetaker", "Please enter a number or leave it empty.")
            return
        self.enqueue(["--file", str(p)] + self.common_args(n), p.with_name(f"{p.stem}_transcript.json"))

    def enqueue(self, args: list[str], json_path: Path) -> None:
        self.jobs.append((args, json_path))
        if self.proc is None:
            self.next_job()
        else:
            self.log(f"Queued: {json_path.parent.name}")

    def next_job(self) -> None:
        if not self.jobs:
            return
        args, json_path = self.jobs.pop(0)
        exe = Path(sys.executable)
        if exe.name.lower() == "pythonw.exe":
            exe = exe.with_name("python.exe")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore",
                   HF_HUB_DISABLE_PROGRESS_BARS="1", TQDM_DISABLE="1")
        self.log("")
        self.lbl_status.configure(text="Starting transcription ...")
        self.pbar["value"] = 0
        self.proc = subprocess.Popen([str(exe), str(HERE / "transcribe.py")] + args, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                     env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        proc = self.proc

        def pump():
            for line in proc.stdout:
                self.msgs.put(("line", line.rstrip()))
            self.msgs.put(("done", (proc.wait(), json_path)))

        threading.Thread(target=pump, daemon=True).start()

    def handle(self, kind: str, payload) -> None:
        if kind == "line":
            if payload.startswith("@progress "):
                _, stage, frac = payload.split()
                self.pbar["value"] = float(frac) * 100
                self.lbl_status.configure(text=f"{STAGES.get(stage, stage)} ... {float(frac):.0%}")
            elif payload.strip():
                self.log(payload)
            return
        code, json_path = payload
        self.proc = None
        if code == 0 and json_path.exists():
            self.last_json = json_path
            self.btn_open.configure(state="normal")
            self.lbl_status.configure(text=f"Done: {json_path.with_suffix('.md')}")
            self.pbar["value"] = 100
        else:
            self.lbl_status.configure(text=f"Transcription failed (exit code {code}), see log")
        self.next_job()

    def open_transcript(self) -> None:
        if self.last_json:
            self._startfile(self.last_json.with_suffix(".md"))

    def pick_transcript(self) -> Path | None:
        """The latest transcript, or one chosen from disk."""
        if self.last_json and self.last_json.exists():
            return self.last_json
        f = filedialog.askopenfilename(title="Transcript", initialdir=self.v_dir.get(), filetypes=[("Transcript", "*.json")])
        return Path(f) if f else None

    def copy_transcript(self) -> None:
        path = self.pick_transcript()
        if not path:
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.root.clipboard_clear()
        self.root.clipboard_append(transcribe.render_text(data))
        self.root.update()  # hand the text to Windows now, so it survives closing the app
        self.lbl_status.configure(text=f"Copied: {path.parent.name} ({len(data['utterances'])} lines), paste with Ctrl+V")

    def rename_speakers(self) -> None:
        path = self.pick_transcript()
        if not path:
            return
        data = json.loads(path.read_text(encoding="utf-8"))

        win = tk.Toplevel(self.root)
        win.title("Rename speakers")
        win.transient(self.root)
        fr = ttk.Frame(win, padding=10)
        fr.pack(fill="both", expand=True)
        ttk.Label(fr, text=path.parent.name, foreground="gray").grid(row=0, column=0, columnspan=3, sticky="w")
        entries = {}
        for i, (key, name) in enumerate(data["speakers"].items(), start=1):
            first = next((u["text"] for u in data["utterances"] if u["speaker"] == key), "")
            ttk.Label(fr, text=name).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            v = tk.StringVar(value=name)
            ttk.Entry(fr, textvariable=v, width=22).grid(row=i, column=1, padx=4, pady=2)
            ttk.Label(fr, text=f"„{first[:60]}…“", foreground="gray").grid(row=i, column=2, sticky="w", padx=4)
            entries[key] = v

        def apply():
            data["speakers"] = {k: (v.get().strip() or data["speakers"][k]) for k, v in entries.items()}
            md = transcribe.save(data, path)
            self.last_json = path
            self.btn_open.configure(state="normal")
            self.log(f"Renamed speakers in {md.name}")
            win.destroy()

        ttk.Button(fr, text="Save", command=apply).grid(row=len(entries) + 1, column=1, pady=8)

    # ------------------------------------------------------------------ loop / close
    def tick(self) -> None:
        if self.rec:
            s = int(self.rec.elapsed)
            self.lbl_time.configure(text=f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}")
            lm, ls = self.rec.levels
            self.m_mic["value"], self.m_sys["value"] = level_to_meter(lm), level_to_meter(ls)
        while not self.msgs.empty():
            self.handle(*self.msgs.get())
        self.root.after(100, self.tick)

    def on_close(self) -> None:
        if self.rec:
            if not messagebox.askyesno("Notetaker", "A recording is running. Stop and save it, then quit?"):
                return
            self.rec.stop()
            self.rec = None
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("Notetaker", "A transcription is running. Cancel it and quit?\n"
                                                    "(The audio is saved; you can transcribe it later.)"):
                return
            self.proc.kill()
        self.remember()
        self.root.destroy()


def main() -> None:
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)  # sharp text on scaled displays; must precede Tk()
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
