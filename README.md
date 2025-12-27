# Voicemail Drop Detector

A small utility that scans WAV files and detects likely voicemail "drops" — i.e., when speech stops and long continuous silence follows. The detection logic is implemented in `main.py`.

This README explains how to set up and run the project locally, including optional Whisper instructions and commands to push the project to GitHub.

## Quick start (5 minutes)

1. Ensure you have Python 3.8 or newer installed.
2. Create a virtual environment (recommended) and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `Voicemails` folder next to `main.py` and put your `.wav` files there (one or more):

```bash
mkdir -p Voicemails
# copy or move your .wav files into Voicemails/
```

5. Run the detector:

```bash
python main.py
```

You should see per-file drop timestamps printed and a final summary.

## Files and important symbols

- `main.py` — the detector script. Key variables you may want to edit:
   - `VOICEMAIL_DIR` — directory scanned for `.wav` files (by default it's the `Voicemails` folder next to `main.py`).
   - `SAMPLE_RATE`, `CHUNK_DURATION`, `SILENCE_DB_THRESHOLD`, `MIN_CONT_SILENCE` — detection tuning values.

## Requirements

Minimal dependencies are listed in `requirements.txt`. Important ones:

- `numpy` — numeric operations
- `librosa` — audio loading/processing
- `soundfile` — backend used by `librosa` to read/write files
- `openai-whisper` and `torch` — optional; the script can run without them. If Whisper is installed, the project attempts to load the `base` Whisper model but proceeds without it if unavailable.

Install everything with:

```bash
pip install -r requirements.txt
```

If you do not need Whisper (transcription), you may omit `openai-whisper` and `torch` from the environment to save space.

### Installing Whisper (optional)

Whisper requires `torch`. On Linux CPU-only, an example install is:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper
```

For GPU support, install the appropriate `torch` wheel for your CUDA version; see https://pytorch.org/ for the correct command.

## Usage notes

- Place `.wav` files in the `Voicemails` folder (next to `main.py`) or change `VOICEMAIL_DIR` in `main.py` to point at another directory.
- The script prints `None` for files where a drop timestamp cannot be determined.
- If you get errors reading audio, make sure `soundfile` is installed (`pip install soundfile`).

## Example output

```
Processing voicemails...

=== FINAL RESULTS ===
message1.wav: 12.5
message2.wav: None
```

## Troubleshooting

- If `librosa` or `soundfile` fails to read a WAV, verify the file format (PCM WAV) and that `soundfile` is installed.
- If `openai-whisper` import fails but you don't need transcription, ignore the warning — the timing/detection code will still run.
- If `torch` install fails, follow the platform-specific install guide at https://pytorch.org/.

## Running tests / quick check

This repository doesn't include an automated test suite. To verify runtime, run `python -m py_compile main.py` or just run `python main.py` against a small sample .wav file.

## Git: push this repo to GitHub

If you haven't already created a git repo and remote, here's a minimal set of commands (replace `<YOUR_REMOTE_URL>` with the remote repository URL):

```bash
# initialize (if not a git repo already)
git init
git add .
git commit -m "Initial commit - voicemail drop detector"
git branch -M main
git remote add origin <YOUR_REMOTE_URL>
git push -u origin main
```

If the repo already exists, you can skip `git init` and `git remote add` steps.

## Next steps / improvements

- Add CLI flags (argparse) to pass the voicemail directory and tune thresholds at runtime.
- Add unit tests for the silence detector logic and an integration test using a short sample WAV file.
- Add logging instead of print statements for better observability in production.

## License

Choose and add a license if you intend to publish this repository publicly.

---

If you'd like, I can also:
- Add an `argparse` interface to `main.py` to pass `--dir` and tuning params.
- Add a small sample WAV (or a test that generates a synthetic WAV) so you can verify behavior immediately.

Tell me which of these you'd like next.