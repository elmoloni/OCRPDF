# Notetaker

Records a call on two tracks and turns it into a German/English transcript with speaker labels.
Everything runs locally on this laptop; no audio leaves the machine.

- **Track 1, your microphone:** always labelled with your name.
- **Track 2, PC sound** (WASAPI loopback, i.e. what Teams / Zoom / a video plays): split into
  Speaker 1, Speaker 2, ... by voice (pyannote diarization). Rename them afterwards.
- **Prerecorded files** (mp3, m4a, mp4, wav, ...): every voice in the file is separated.

Speech-to-text is [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`large-v3-turbo`, MIT);
speaker separation is [pyannote.audio](https://github.com/pyannote/pyannote-audio) (`speaker-diarization-community-1`, MIT code,
free model behind a Hugging Face login).

## Install (once)

```
powershell -ExecutionPolicy Bypass -File setup.ps1
```

This installs Python packages (~4 GB, PyTorch with CUDA) into `%USERPROFILE%\.notetaker\venv`, outside OneDrive.
The Whisper model (~1.6 GB) downloads on the first transcription.

### Hugging Face login, for speaker separation

1. Create a free account at huggingface.co.
2. Open https://huggingface.co/pyannote/speaker-diarization-community-1 and accept the conditions.
3. Create a read token under Settings → Access Tokens.
4. Run once and paste the token:
   ```
   %USERPROFILE%\.notetaker\venv\Scripts\hf.exe auth login
   ```

Without the login everything still works, but the other side is one speaker ("Speaker 1").
After the first download the model is cached and runs offline.

## Use

Start `Notetaker.bat`. On the first start it asks for your name (labels your microphone track; change it
with *Change…* next to *You*) and where to save recordings (change later under *Save to*).

1. **Title:** starts with today's date (`2026 10 13 - `); type the rest, e.g. `2026 10 13 - Call Francis`.
   Left as just the date, the time is used (`2026 10 13 - 1405`).
2. **Microphone:** the mic you speak into. **PC sound:** the device your call app plays to.
   If Teams plays to your headset but Windows' default is the laptop speakers, pick the headset here.
3. **Other speakers:** leave on *auto*. Set a number only if auto merges or splits people; in tests a fixed
   count was not better than auto. *1* skips separation.
4. **● Record** starts. During a recording the button
   becomes **● New recording**: it saves the current recording, starts its transcript and starts the next
   recording at once. **■ Stop** ends recording. Watch the two level bars: both should move when the
   respective side talks.
5. **Rename speakers…** replaces "Speaker 1" etc. with names; the first sentence of each speaker is shown
   as a hint.
6. **Copy transcript** puts the latest transcript on the clipboard as plain text, one line per
   utterance (`[00:00:09] Speaker 1: …`), ready to paste into Word, OneNote or a mail.

While recording, files go to a temporary `2026 10 13 - 1405 (recording)` folder. When the recording ends
(**Stop**, **New recording** or closing the app), the folder and files are named after what is in *Title*
at that moment, so you can type or fix the title during the call. A second recording with the same title
gets ` (2)`:

```
2026 10 13 - Call Francis    2026 10 13 - Call Francis - mic.wav       your side
    2026 10 13 - Call Francis - system.wav    the other side
    2026 10 13 - Call Francis.md              readable transcript
    2026 10 13 - Call Francis.json            same data + speaker names (used for renaming)
```

**Transcribe file…** takes any audio/video file and asks how many voices are in it. Choosing the
`… - mic.wav` of a Notetaker recording re-runs it in two-track mode.

The transcript can go straight into the `onenote` skill in Claude for call notes.

### Command line

```
%USERPROFILE%\.notetaker\venv\Scripts\python.exe transcribe.py --mic mic.wav --system system.wav --me Elias --speakers 3
%USERPROFILE%\.notetaker\venv\Scripts\python.exe transcribe.py --file call.m4a --speakers 4 --language de
```

## Good to know

- **Headphones help.** Without them your mic also picks up the call. Mic lines that repeat what the
  system track says at the same moment are dropped automatically, but headphones give cleaner results.
- **Language:** *Auto* re-detects the language per segment, so mixed German/English calls work.
  If a call is entirely in one language, choose it: slightly more accurate.
- **Speed:** on the RTX A500 a call transcribes in about a fifth of its length (10-min test call: 2 min;
  a 1-hour call roughly 12 min). Most of that is speaker separation; the log shows the time per step.
  Without a GPU it falls back to the CPU, using all cores, several times slower. *medium* / *small* models
  are faster but less accurate. The first transcription downloads the models (~1,8 GB, once).
- **Back-to-back calls:** use **● New recording** in one window: it saves the call that just ended, queues
  its transcript and keeps recording the next one. Several windows also work: each recording gets its own
  folder, and transcriptions take turns on the GPU (a second one shows *Waiting for the transcription in
  another Notetaker window*), since two at once would not fit into the GPU's 4 GB.
- **Speaker separation is not perfect:** overlapping speech, similar voices and compressed call audio can
  put a sentence under the wrong speaker. Telling it the number of speakers helps most.
- **Consent:** in Germany, recording a non-public conversation without the participants' consent is a
  criminal offence (§ 201 StGB). Announce the recording at the start of the call.
