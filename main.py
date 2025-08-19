
import json
import threading
import queue
import sys
import os
import time
from vosk import Model, KaldiRecognizer
import sounddevice as sd

from kivy.app import App
from kivy.uix.label import Label
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.clock import Clock

# Optional webrtcvad for Voice Activity Detection (will be used if available and enabled in config)
try:
    import webrtcvad
    HAVE_VAD = True
except Exception:
    HAVE_VAD = False

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
if not os.path.exists(CONFIG_PATH):
    raise SystemExit("config.json not found. Please create it next to this file (see README).")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

# Register custom font if provided
try:
    if "fonte" in config and config["fonte"]:
        font_path = os.path.join("fonts", f'{config["fonte"]}.ttf')
        if os.path.exists(font_path):
            LabelBase.register(name=config["fonte"], fn_regular=font_path)
except Exception as e:
    print("Warning registering font:", e)

# Window colors and text default
if "cor_fundo" in config:
    try:
        Window.clearcolor = tuple(config["cor_fundo"])
    except Exception as e:
        print("Warning setting background color:", e)

SAMPLE_RATE = int(config.get("sample_rate", 16000))
MODEL_PATH = config.get("model_path", "models/vosk-model-small-pt-0.3")
DEVICE = config.get("device", None)  # can be index or None

# Low-latency tuning params (can be changed in config.json)
BLOCKSIZE = int(config.get("blocksize", 1600))   # samples per callback (1600 @ 16k = 100 ms)
FRAME_MS = int(config.get("frame_ms", 30))      # webrtcvad frame size (10,20,30 allowed)
USE_VAD = bool(config.get("use_vad", True))     # enable VAD if possible
VAD_MODE = int(config.get("vad_mode", 2))       # 0-3 for aggressiveness (webrtcvad)

# Validate FRAME_MS
if FRAME_MS not in (10, 20, 30):
    print("frame_ms must be one of 10,20,30. Falling back to 30ms.")
    FRAME_MS = 30

# Create queue and control event
audio_q = queue.Queue()
running = threading.Event()

# Load model
if not os.path.exists(MODEL_PATH):
    print(f"Model path '{MODEL_PATH}' not found. Please download a Vosk model and set model_path in config.json.")
    sys.exit(1)

print("Loading Vosk model (this may take a while)...")
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

# Setup VAD if enabled and available
vad = None
if USE_VAD:
    if HAVE_VAD:
        vad = webrtcvad.Vad(VAD_MODE)
        print("webrtcvad available and initialized (mode=%d)." % VAD_MODE)
    else:
        print("webrtcvad requested in config but not installed — continuing without VAD.")
        vad = None

def audio_callback(indata, frames, time_info, status):
    """
    sounddevice callback: push raw bytes to a queue.
    dtype expected 'int16' and channels=1.
    """
    if status:
        print("SoundDevice status:", status, file=sys.stderr)
    # put immutable bytes snapshot into queue
    audio_q.put(indata.copy().tobytes())

def frame_generator(frame_ms, sample_rate, source_bytes):
    """
    Generator that yields frames of `frame_ms` milliseconds from source bytes.
    Each frame is raw bytes (16-bit little-endian PCM mono).
    """
    bytes_per_sample = 2  # 16-bit
    frame_bytes = int(sample_rate * (frame_ms / 1000.0) * bytes_per_sample)
    offset = 0
    length = len(source_bytes)
    while offset + frame_bytes <= length:
        yield source_bytes[offset:offset + frame_bytes]
        offset += frame_bytes

def recognizer_thread_fn(app_ref):
    """
    Consumes audio buffers from audio_q, splits into frames, applies optional VAD,
    feeds Vosk frame-by-frame for low-latency partials and finals.
    """
    print("Recognizer thread started.")
    in_speech = False
    silence_frames = 0
    speech_seen = False
    # when VAD indicates end of speech, we will trigger a final result after some hangover frames
    vad_hangover_frames = int(300 / FRAME_MS)  # ~300ms hangover

    # simple timing for diagnostics
    last_feed_time = time.time()

    while running.is_set():
        try:
            data = audio_q.get(timeout=0.2)  # bytes
        except queue.Empty:
            continue

        # split callback bytes into frame_ms frames for VAD
        for frame in frame_generator(FRAME_MS, SAMPLE_RATE, data):
            # Check VAD (if available)
            is_speech = True
            if vad is not None:
                try:
                    is_speech = vad.is_speech(frame, SAMPLE_RATE)
                except Exception as e:
                    # If VAD fails for any reason, assume speech to avoid losing data
                    is_speech = True
            # If using VAD and we are currently in silence, skip feeding many silent frames to Vosk
            if vad is not None and not is_speech and not in_speech:
                # accumulate a small amount of silence to keep vad state stable, but skip heavy processing
                silence_frames += 1
                # only every few silent frames, feed a tiny chunk to recognizer to keep it in sync (optional)
                if silence_frames < 2:
                    try:
                        if recognizer.AcceptWaveform(frame):
                            res = recognizer.Result()
                            text = json.loads(res).get("text", "")
                            Clock.schedule_once(lambda dt, t=text: app_ref.update_label(t, finalized=True))
                        else:
                            pres = recognizer.PartialResult()
                            partial = json.loads(pres).get("partial", "")
                            Clock.schedule_once(lambda dt, p=partial: app_ref.update_label(p, finalized=False))
                    except Exception as e:
                        pass
                continue
            # if we reach here, either VAD says speech, or VAD not available
            silence_frames = 0
            # Feed frame to Vosk recognizer (small chunks -> lower latency)
            try:
                if recognizer.AcceptWaveform(frame):
                    res = recognizer.Result()
                    try:
                        text = json.loads(res).get("text", "")
                    except Exception:
                        text = ""
                    Clock.schedule_once(lambda dt, t=text: app_ref.update_label(t, finalized=True))
                    in_speech = False
                    speech_seen = True
                else:
                    pres = recognizer.PartialResult()
                    try:
                        partial = json.loads(pres).get("partial", "")
                    except Exception:
                        partial = ""
                    Clock.schedule_once(lambda dt, p=partial: app_ref.update_label(p, finalized=False))
                    in_speech = True
                    speech_seen = True
            except Exception as e:
                # If recognizer errors, print and continue
                print("Recognizer error:", e)
                continue

            # If VAD is enabled and we were in speech but now have some consecutive silence frames,
            # consider this the end of the utterance and request a final result (if recognizer didn't already finalize).
            if vad is not None and speech_seen and not is_speech:
                silence_frames += 1
                if silence_frames >= vad_hangover_frames:
                    # Try to get a final result (some models return best guess with Result())
                    try:
                        res = recognizer.Result()
                        try:
                            text = json.loads(res).get("text", "")
                        except Exception:
                            text = ""
                        Clock.schedule_once(lambda dt, t=text: app_ref.update_label(t, finalized=True))
                    except Exception as e:
                        pass
                    speech_seen = False
                    in_speech = False
                    silence_frames = 0

    print("Recognizer thread finished.")

import json as _json  # used in scheduled lambdas (safe reference)

class SonorisApp(App):
    def build(self):
        self.label = Label(
            text="Aguardando transcrição...",
            font_name=config.get("fonte", None),
            font_size=config.get("tamanho", 36),
            color=tuple(config.get("cor_texto", [1,1,1,1])),
            halign='center',
            valign='middle'
        )
        self.label.bind(size=self.label.setter('text_size'))
        return self.label

    def on_start(self):
        # start thread and audio stream
        running.set()
        self._rec_thread = threading.Thread(target=recognizer_thread_fn, args=(self,), daemon=True)
        self._rec_thread.start()

        # detect device sample rate for info
        try:
            if DEVICE is None:
                dev_info = sd.query_devices(None, 'input')
            else:
                dev_info = sd.query_devices(int(DEVICE), 'input')
            print("Device info:", dev_info.get('name', 'unknown'))
            print("Device default samplerate:", int(dev_info['default_samplerate']))
        except Exception as e:
            print("Could not query device info:", e)

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                device=DEVICE,
                dtype='int16',
                channels=1,
                callback=audio_callback,
                blocksize=BLOCKSIZE,
                latency='low'
            )
            self._stream.start()
            print("Audio stream started (blocksize=%d, samplerate=%d, frame_ms=%d)." % (BLOCKSIZE, SAMPLE_RATE, FRAME_MS))
        except Exception as e:
            print("Failed to start audio stream:", e)
            running.clear()

    def update_label(self, text, finalized=False):
        """
        Update label on Kivy main thread.
        """
        if not text and not finalized:
            return
        if finalized:
            display = text.strip() if text.strip() else "Aguardando transcrição..."
        else:
            display = text + "…" if text else ""
        self.label.text = display

    def on_stop(self):
        print("Stopping app...")
        try:
            if hasattr(self, "_stream") and self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception as e:
            print("Error closing stream:", e)
        running.clear()
        if hasattr(self, "_rec_thread"):
            self._rec_thread.join(timeout=1.0)
        print("Stopped.")

if __name__ == "__main__":
    SonorisApp().run()
