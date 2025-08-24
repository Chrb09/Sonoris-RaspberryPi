#!/usr/bin/env python3
"""
main_console_pi_accuracy.py
Variation of the console transcription script that sacrifices latency for accuracy.
Behavior changes:
 - When speech is detected (webrtcvad if available, otherwise energy-based VAD), the script
   accumulates the audio for the whole utterance.
 - After end-of-speech (hangover), the script decodes the whole utterance with a fresh
   KaldiRecognizer to obtain a single, more accurate final result.
 - This reduces streaming/partial errors and uses more CPU per utterance (but still light overall).
 - Configurable via config.json:
    "accumulate_utterance": true,
    "end_speech_hangover_ms": 300,
    "utterance_max_seconds": 8,
    "show_partials": false
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
import numpy as np

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

# Defaults
MODEL_PATH = cfg.get("model_path", "models/vosk-model-small-pt-0.3")
SAMPLE_RATE = int(cfg.get("sample_rate", 16000))
BLOCKSIZE = int(cfg.get("blocksize", 800))
FRAME_MS = int(cfg.get("frame_ms", 20))
DEVICE = cfg.get("device", None)

# Accuracy-vs-latency options
ACCUMULATE_UTTERANCE = bool(cfg.get("accumulate_utterance", True))
END_SPEECH_HANGOVER_MS = int(cfg.get("end_speech_hangover_ms", 300))
UTTERANCE_MAX_SECONDS = int(cfg.get("utterance_max_seconds", 8))
SHOW_PARTIALS = bool(cfg.get("show_partials", False))

USE_VAD = bool(cfg.get("use_vad", True)) and HAVE_VAD
VAD_MODE = int(cfg.get("vad_mode", 2)) if HAVE_VAD else None

# Safety checks
if FRAME_MS not in (10, 20, 30):
    FRAME_MS = 20
if BLOCKSIZE <= 0:
    BLOCKSIZE = 800
if END_SPEECH_HANGOVER_MS < 50:
    END_SPEECH_HANGOVER_MS = 50

# Check model
if not os.path.exists(MODEL_PATH):
    print("Model not found at", MODEL_PATH, file=sys.stderr)
    sys.exit(1)

print("Loading Vosk model...", file=sys.stderr)
model = Model(MODEL_PATH)

# If not accumulating utterances we keep a streaming recognizer (original behavior)
streaming_recognizer = None
if not ACCUMULATE_UTTERANCE and SHOW_PARTIALS:
    streaming_recognizer = KaldiRecognizer(model, SAMPLE_RATE)

# Setup VAD if enabled
vad = None
if USE_VAD:
    try:
        vad = webrtcvad.Vad(VAD_MODE)
        print("webrtcvad available and initialized (mode=%d)." % VAD_MODE, file=sys.stderr)
    except Exception as e:
        print("Failed to init webrtcvad, disabling VAD:", e, file=sys.stderr)
        vad = None

# Queue
AUDIO_QUEUE_MAX = 8
audio_q = queue.Queue(maxsize=AUDIO_QUEUE_MAX)
running = threading.Event()
running.set()

# Helpers
def frame_generator(frame_ms, sample_rate, data_bytes):
    bytes_per_sample = 2
    frame_bytes = int(sample_rate * (frame_ms / 1000.0) * bytes_per_sample)
    offset = 0
    length = len(data_bytes)
    while offset + frame_bytes <= length:
        yield data_bytes[offset:offset + frame_bytes]
        offset += frame_bytes

def is_speech_frame_vad(frame_bytes):
    try:
        return vad.is_speech(frame_bytes, SAMPLE_RATE)
    except Exception:
        return True

def is_speech_frame_energy(frame_bytes, db_threshold=-40.0):
    # quick RMS energy check (cheap): convert to int16 then float
    arr = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(arr*arr)) if arr.size else 0.0
    if rms <= 0:
        return False
    db = 20 * np.log10(rms)
    return db >= db_threshold

# audio callback - minimal: push raw bytes into queue
def audio_callback(indata, frames, time_info, status):
    if status:
        # occasional log
        print("Audio status:", status, file=sys.stderr)
    try:
        b = indata.tobytes()
        try:
            audio_q.put_nowait(b)
        except queue.Full:
            # drop oldest then enqueue
            try:
                _ = audio_q.get_nowait()
                audio_q.put_nowait(b)
            except queue.Empty:
                pass
    except Exception as e:
        print("Callback error:", e, file=sys.stderr)

# recognizer thread: accumulate utterance when speech is present, then decode whole utterance
def recognizer_thread():
    print("Recognizer thread started (accumulate_utterance=%s, show_partials=%s)." % (ACCUMULATE_UTTERANCE, SHOW_PARTIALS), file=sys.stderr)
    try:
        utterance_buf = bytearray()
        in_speech = False
        silence_ms = 0
        hangover_frames = max(1, int(END_SPEECH_HANGOVER_MS / FRAME_MS))
        max_utterance_frames = max(1, int((UTTERANCE_MAX_SECONDS*1000) / FRAME_MS))

        # if showing partials but not accumulating, we will use streaming_recognizer
        while running.is_set():
            try:
                data = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            for frame in frame_generator(FRAME_MS, SAMPLE_RATE, data):
                # Determine speech using VAD or energy
                if vad is not None:
                    speech = is_speech_frame_vad(frame)
                else:
                    speech = is_speech_frame_energy(frame, db_threshold=-42.0)

                if ACCUMULATE_UTTERANCE:
                    if speech:
                        utterance_buf.extend(frame)
                        in_speech = True
                        silence_ms = 0
                        # if utterance too long, force decode to avoid memory growth
                        max_bytes = max_utterance_frames * int(SAMPLE_RATE*(FRAME_MS/1000.0)) * 2
                        if len(utterance_buf) >= max_bytes:
                            decode_buf = bytes(utterance_buf)
                            utterance_buf.clear()
                            # decode with fresh recognizer for best result on the whole utterance
                            rec = KaldiRecognizer(model, SAMPLE_RATE)
                            rec.AcceptWaveform(decode_buf)
                            res = rec.Result()
                            try:
                                text = json.loads(res).get("text", "")
                            except Exception:
                                text = ""
                            if text:
                                print("\r" + text)
                    else:
                        if in_speech:
                            silence_ms += FRAME_MS
                            if silence_ms >= END_SPEECH_HANGOVER_MS:
                                # end of utterance -> decode accumulated bytes
                                if utterance_buf:
                                    decode_buf = bytes(utterance_buf)
                                    utterance_buf.clear()
                                    in_speech = False
                                    silence_ms = 0
                                    # decode with fresh recognizer (better accuracy)
                                    rec = KaldiRecognizer(model, SAMPLE_RATE)
                                    rec.AcceptWaveform(decode_buf)
                                    res = rec.Result()
                                    try:
                                        text = json.loads(res).get("text", "")
                                    except Exception:
                                        text = ""
                                    if text:
                                        print("\r" + text)
                                else:
                                    in_speech = False
                                    silence_ms = 0
                        else:
                            # not in speech, optionally keep showing ambient partials if requested
                            if SHOW_PARTIALS and streaming_recognizer is not None:
                                if streaming_recognizer.AcceptWaveform(frame):
                                    res = streaming_recognizer.Result()
                                    try:
                                        txt = json.loads(res).get("text", "")
                                    except Exception:
                                        txt = ""
                                    if txt:
                                        print("\r" + txt)
                                else:
                                    pr = streaming_recognizer.PartialResult()
                                    try:
                                        ptxt = json.loads(pr).get("partial", "")
                                    except Exception:
                                        ptxt = ""
                                    if ptxt:
                                        sys.stdout.write("\r" + ptxt + "…")
                                        sys.stdout.flush()
                else:
                    # original streaming behavior (lower latency) with optional partials
                    if streaming_recognizer is None:
                        streaming_recognizer = KaldiRecognizer(model, SAMPLE_RATE)
                    if streaming_recognizer.AcceptWaveform(frame):
                        res = streaming_recognizer.Result()
                        try:
                            text = json.loads(res).get("text", "")
                        except Exception:
                            text = ""
                        if text:
                            print("\r" + text)
                    else:
                        pr = streaming_recognizer.PartialResult()
                        try:
                            ptxt = json.loads(pr).get("partial", "")
                        except Exception:
                            ptxt = ""
                        if SHOW_PARTIALS and ptxt:
                            sys.stdout.write("\r" + ptxt + "…")
                            sys.stdout.flush()
    except Exception as e:
        print("Recognizer thread exception:", e, file=sys.stderr)

# Graceful shutdown signals
def stop(signum=None, frame=None):
    running.clear()
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
    print("Starting accuracy-focused transcription (press Ctrl+C to stop).", file=sys.stderr)
    print("Model:", MODEL_PATH, "Sample rate:", SAMPLE_RATE, "Blocksize:", BLOCKSIZE, file=sys.stderr)
    print("Accumulate utterance:", ACCUMULATE_UTTERANCE, "Hangover(ms):", END_SPEECH_HANGOVER_MS, "Max utterance(s):", UTTERANCE_MAX_SECONDS, file=sys.stderr)
    if vad is not None:
        print("VAD enabled.", file=sys.stderr)
    else:
        print("VAD disabled (using energy-based detection).", file=sys.stderr)

    th = threading.Thread(target=recognizer_thread, daemon=True)
    th.start()

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
