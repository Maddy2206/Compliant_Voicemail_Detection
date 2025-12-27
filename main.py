import os
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICEMAIL_DIR = os.path.join(BASE_DIR, "Voicemails")

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5        
SILENCE_DB_THRESHOLD = -40  
MIN_CONT_SILENCE = 2.0      
 
VOICEMAIL_END_PHRASES = [
    "please leave",
    "leave your name",
    "leave your number",
    "leave a message",
    "leave your message",
    "leave a brief message",
    "leave your contact information",
    "record your message",
    "after the beep",
    "after the tone",
    "thank you for calling",
    "have a great day",
    "have a nice day",
    "i will get back to you",
    "i will return your call",
    "i will call you back",
    "as soon as possible",
    "at the earliest opportunity"
]
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

        recent_audio = np.array([], dtype=np.float32)

        semantic_end_time = None
        last_speech_end = None
        silence_start = None

        last_whisper_call = 0
        WHISPER_INTERVAL = 3.0          
        WHISPER_WINDOW = 4.0            

        for audio_chunk, start, end in streamer.stream():

            recent_audio = np.concatenate([recent_audio, audio_chunk])
            max_samples = int(WHISPER_WINDOW * SAMPLE_RATE)
            if len(recent_audio) > max_samples:
                recent_audio = recent_audio[-max_samples:]

            silent = self.silence_detector.is_silent(audio_chunk)

            if not silent:
                last_speech_end = end
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = start

                anchor = semantic_end_time if semantic_end_time else last_speech_end
                if anchor is not None and silence_start >= anchor:
                    if end - silence_start >= MIN_CONT_SILENCE:
                        return round(silence_start, 2)


            if (
                WHISPER_AVAILABLE
                and semantic_end_time is None
                and end - last_whisper_call >= WHISPER_INTERVAL
                and len(recent_audio) >= SAMPLE_RATE
            ):
                try:

                    result = whisper_model.transcribe(
                        recent_audio,
                        language="en",
                        fp16=False,
                        verbose=False
                    )

                    for seg in result.get("segments", []):
                        text = seg["text"].lower()
                        if any(p in text for p in VOICEMAIL_END_PHRASES):
                            semantic_end_time = end - (len(recent_audio) / SAMPLE_RATE) + seg["end"]
                            break
                except Exception:
                    
                    pass

                last_whisper_call = end

        # Final fallback
        if semantic_end_time is not None:
            return round(semantic_end_time, 2)

        if last_speech_end is not None:
            return round(last_speech_end, 2)

        return None

def main():
    detector = VoicemailDropDetector()
    results = {}

    print("Processing voicemails...\n")

    if not os.path.isdir(VOICEMAIL_DIR):
        print(f"Voicemail directory not found: {VOICEMAIL_DIR}")
        print("Place .wav files in the Voicemails directory (next to main.py) or update VOICEMAIL_DIR.")
        return

    wav_files = [f for f in sorted(os.listdir(VOICEMAIL_DIR)) if f.lower().endswith(".wav")]
    if not wav_files:
        print(f"No .wav files found in {VOICEMAIL_DIR}")
        return

    for file in wav_files:
        path = os.path.join(VOICEMAIL_DIR, file)
        try:
            drop_ts = detector.process(path)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            drop_ts = None
        results[file] = drop_ts

    print("\n=== FINAL RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
