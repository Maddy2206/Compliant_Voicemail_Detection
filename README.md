# Voicemail Drop Detector

This project detects likely voicemail drops (long continuous silence after speech) in WAV files using the detector implemented in [main.py](main.py).

Key symbols:
- [`VOICEMAIL_DIR`](main.py) — path scanned for .wav files
- [`AudioStreamer`](main.py) — streams audio in chunks
- [`SilenceDetector`](main.py) — detects silence per chunk
- [`VoicemailDropDetector`](main.py) — main detector logic

Requirements
1. Python 3.8+
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

Notes
- Whisper is optional; the script will work for timing detection without it (see the top of [main.py](main.py)).
- By default the script uses the hardcoded [`VOICEMAIL_DIR`](main.py). Edit that variable in [main.py](main.py) or set the path to your voicemail directory before running.

Run
```sh
python main.py
```

Outputs
- Prints per-file drop timestamps to stdout and a final summary.

Troubleshooting
- If librosa fails to read WAV files, ensure `soundfile` is installed (included above).
- Installing `torch` may require choosing the correct build for your platform (CPU vs GPU). If you don't need Whisper, you can omit `openai-whisper`/`torch`.