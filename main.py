import os
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

VOICEMAIL_DIR = "/kaggle/input/voicemails/Voicemails - SWE Intern"
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5        
SILENCE_DB_THRESHOLD = -40  
MIN_CONT_SILENCE = 1.5      
try:
    import whisper
    whisper_model = whisper.load_model("base")
    WHISPER_AVAILABLE = True
except:
    WHISPER_AVAILABLE = False
    whisper_model = None
    print("⚠️ Whisper not available (this does NOT affect timing detection)")

class AudioStreamer:
    def __init__(self, file_path):
        self.audio, self.sr = librosa.load(file_path, sr=SAMPLE_RATE)
        self.chunk_size = int(CHUNK_DURATION * self.sr)

    def stream(self):
        for i in range(0, len(self.audio), self.chunk_size):
            start = i / self.sr
            end = min(len(self.audio), i + self.chunk_size) / self.sr
            yield self.audio[i:i+self.chunk_size], start, end

class SilenceDetector:
    def is_silent(self, audio_chunk):
        rms = np.sqrt(np.mean(audio_chunk**2))
        db = 20 * np.log10(rms + 1e-10)
        return db < SILENCE_DB_THRESHOLD

class VoicemailDropDetector:
    def __init__(self):
        self.silence_detector = SilenceDetector()

    def process(self, audio_file):
        streamer = AudioStreamer(audio_file)

        speech_seen = False
        last_speech_end = None
        silence_start = None

        for audio_chunk, start, end in streamer.stream():
            silent = self.silence_detector.is_silent(audio_chunk)

            if not silent:
                speech_seen = True
                last_speech_end = end
                silence_start = None

            else:
                if speech_seen:
                    if silence_start is None:
                        silence_start = start
                    if end - silence_start >= MIN_CONT_SILENCE:
                        return round(silence_start, 2)

        if last_speech_end is not None:
            return round(last_speech_end, 2)

        return None

detector = VoicemailDropDetector()
results = {}

print("Processing voicemails...\n")

for file in sorted(os.listdir(VOICEMAIL_DIR)):
    if file.lower().endswith(".wav"):
        path = os.path.join(VOICEMAIL_DIR, file)
        drop_ts = detector.process(path)
        results[file] = drop_ts

print("\n=== FINAL RESULTS ===")
for k, v in results.items():
    print(f"{k}: {v}")
