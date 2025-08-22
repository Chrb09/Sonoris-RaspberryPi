#!/usr/bin/env python3
"""
main_console_pi.py
Console-only, low-latency Vosk transcription optimized for Raspberry Pi 3B.
Features:
 - Minimal overhead (no GUI)
 - Optional webrtcvad integration (uses it only if installed)
 - Small queue with drop-old policy to avoid lag accumulation
 - Small blocksize and short frames for low latency
 - Reads configuration from config.json if present, otherwise uses sensible defaults
"""

import os
import sys
import json
import queue
import threading
import time
import signal

try:
    from vosk import Model, KaldiRecognizer
except Exception as e:
    print("Error: vosk not installed or model not found.", e, file=sys.stderr)
    raise

import sounddevice as sd

# Optional VAD (used only if available)
try:
    import webrtcvad
    HAVE_VAD = True
except Exception:
    HAVE_VAD = False

# Load optional config.json (same folder)
BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
else:
    cfg = {}

# Defaults (tuned for low-latency on Pi 3)
MODEL_PATH = cfg.get("model_path", "models/vosk-model-small-pt-0.3")
SAMPLE_RATE = int(cfg.get("sample_rate", 16000))
# blocksize: how many samples per sounddevice callback.
# 1600 samples @16k = 100 ms. Use 800 for ~50ms if CPU allows.
BLOCKSIZE = int(cfg.get("blocksize", 800))
# frame length for VAD processing (10,20,30 allowed). 20 is a good compromise.
FRAME_MS = int(cfg.get("frame_ms", 20))
USE_VAD = bool(cfg.get("use_vad", True)) and HAVE_VAD
VAD_MODE = int(cfg.get("vad_mode", 2)) if HAVE_VAD else None
DEVICE = cfg.get("device", None)  # optional device index

# Safety checks
if FRAME_MS not in (10, 20, 30):
    FRAME_MS = 20

# Check model
if not os.path.exists(MODEL_PATH):
    print("Model not found at", MODEL_PATH, file=sys.stderr)
    sys.exit(1)

# Instantiate model and recognizer
print("Loading Vosk model...", file=sys.stderr)
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

# Setup VAD if requested & available
vad = None
if USE_VAD:
    try:
        vad = webrtcvad.Vad(VAD_MODE)
        print("webrtcvad enabled (mode=%d)." % VAD_MODE, file=sys.stderr)
    except Exception as e:
        print("Failed to initialize webrtcvad, continuing without VAD:", e, file=sys.stderr)
        vad = None
        USE_VAD = False

# Queue to transfer audio bytes from callback to processing thread
# small maxsize to avoid unbounded memory and high latency. Drop old chunks if queue full.
AUDIO_QUEUE_MAX = 6
audio_q = queue.Queue(maxsize=AUDIO_QUEUE_MAX)

running = threading.Event()
running.set()

# Helper: split bytes into frames of frame_ms milliseconds
def frame_generator(frame_ms, sample_rate, data_bytes):
    bytes_per_sample = 2  # 16-bit
    frame_bytes = int(sample_rate * (frame_ms / 1000.0) * bytes_per_sample)
    offset = 0
    length = len(data_bytes)
    while offset + frame_bytes <= length:
        yield data_bytes[offset:offset + frame_bytes]
        offset += frame_bytes

# audio callback: non-blocking, drop if queue full (keeps latency bounded)
def audio_callback(indata, frames, time_info, status):
    if status:
        # only print status rarely to avoid IO overhead
        print("Audio status:", status, file=sys.stderr)
    try:
        b = indata.tobytes()
        try:
            audio_q.put_nowait(b)
        except queue.Full:
            # drop oldest and push new - keeps queue fresh for low latency
            try:
                _ = audio_q.get_nowait()
                audio_q.put_nowait(b)
            except queue.Empty:
                pass
    except Exception as e:
        # rare, but don't crash callback
        print("Callback error:", e, file=sys.stderr)

# Processing thread: reads small frames, optional VAD, feeds recognizer, prints partials/finals
def recognizer_thread():
    partial = ""
    last_printed = ""
    in_speech = False
    silence_frames = 0
    vad_hangover = int(200 / FRAME_MS)  # 200ms hangover after VAD says non-speech

    try:
        while running.is_set():
            try:
                data = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            # split into frames for VAD and feeding recognizer
            for frame in frame_generator(FRAME_MS, SAMPLE_RATE, data):
                is_speech = True
                if vad is not None:
                    try:
                        is_speech = vad.is_speech(frame, SAMPLE_RATE)
                    except Exception:
                        is_speech = True

                if vad is not None and not is_speech and not in_speech:
                    # if in silence before speech starts, skip most frames to save CPU
                    silence_frames += 1
                    if silence_frames < 2:
                        # feed a tiny frame to keep Vosk internal state moving
                        if recognizer.AcceptWaveform(frame):
                            res = recognizer.Result()
                            try:
                                final_text = json.loads(res).get("text", "")
                            except Exception:
                                final_text = ""
                            if final_text:
                                print(final_text)
                                last_printed = ""
                        else:
                            pr = recognizer.PartialResult()
                            try:
                                p = json.loads(pr).get("partial", "")
                            except Exception:
                                p = ""
                            # print partial only if non-empty
                            if p:
                                # print partial in-place
                                sys.stdout.write("\r" + p + "…")
                                sys.stdout.flush()
                                last_printed = p
                    continue

                # we have speech or no vad; feed to recognizer
                silence_frames = 0
                if recognizer.AcceptWaveform(frame):
                    res = recognizer.Result()
                    try:
                        final = json.loads(res).get("text", "")
                    except Exception:
                        final = ""
                    if final:
                        # finalize line
                        print("\r" + final)
                        last_printed = ""
                        in_speech = False
                    else:
                        # no text in result, just clear partial
                        if last_printed:
                            sys.stdout.write("\r")
                            sys.stdout.flush()
                            last_printed = ""
                        in_speech = False
                else:
                    pr = recognizer.PartialResult()
                    try:
                        ptext = json.loads(pr).get("partial", "")
                    except Exception:
                        ptext = ""
                    if ptext:
                        # overwrite previous partial in-place
                        sys.stdout.write("\r" + ptext + "…")
                        sys.stdout.flush()
                        last_printed = ptext
                        in_speech = True
    except Exception as e:
        print("Recognizer thread exception:", e, file=sys.stderr)

# Graceful shutdown on signals
def stop(signum=None, frame=None):
    running.clear()
    # give threads a moment to stop
    time.sleep(0.2)
    try:
        sd.stop()
    except Exception:
        pass
    print("\nStopping...", file=sys.stderr)
    sys.exit(0)

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

def main():
    print("Starting console transcription (press Ctrl+C to stop).", file=sys.stderr)
    print("Model:", MODEL_PATH, "Sample rate:", SAMPLE_RATE, "Blocksize:", BLOCKSIZE, file=sys.stderr)
    if vad is not None:
        print("VAD enabled.", file=sys.stderr)
    else:
        print("VAD disabled or not available.", file=sys.stderr)

    # Start recognizer thread
    th = threading.Thread(target=recognizer_thread, daemon=True)
    th.start()

    # Start audio stream
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            device=DEVICE,
                            dtype='int16',
                            channels=1,
                            callback=audio_callback,
                            blocksize=BLOCKSIZE,
                            latency='low'):
            while running.is_set():
                time.sleep(0.1)
    except Exception as e:
        print("Failed to start audio stream:", e, file=sys.stderr)
        running.clear()
        time.sleep(0.2)

    th.join(timeout=1.0)

if __name__ == "__main__":
    main()
