#!/usr/bin/env python3
"""
main_console_pi_monitor.py
Console transcription (Vosk) with simple live audio monitoring and ring-buffer recording.
Features:
 - Minimal transcription pipeline (based on previous console scripts)
 - Optional live playback (monitor) through an output device (use headphones to avoid feedback)
 - Ring buffer stores last N seconds of audio; press 's' to save it to WAV for inspection
 - Simple command interface in stdin: s=save, p=toggle playback, q=quit
 - Designed to be lightweight on Raspberry Pi 3B
"""

import os
import sys
import json
import queue
import threading
import time
import signal
import math
import wave
from collections import deque

try:
    from vosk import Model, KaldiRecognizer
except Exception as e:
    print("Error: vosk not installed or model not found.", e, file=sys.stderr)
    raise

import sounddevice as sd
import numpy as np

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
DEVICE = cfg.get("device", None)
# Monitor / ring buffer settings
MONITOR_PLAYBACK = bool(cfg.get("monitor_playback", False))  # whether to play incoming audio
MONITOR_DEVICE = cfg.get("monitor_device", None)  # output device index or None for default
MONITOR_SECONDS = int(cfg.get("monitor_seconds", 30))  # how many seconds to keep in ring buffer

# Safety checks
if BLOCKSIZE <= 0:
    BLOCKSIZE = 800

# Check model
if not os.path.exists(MODEL_PATH):
    print("Model not found at", MODEL_PATH, file=sys.stderr)
    sys.exit(1)

# Instantiate model and recognizer
print("Loading Vosk model...", file=sys.stderr)
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

# Queues and control
AUDIO_QUEUE_MAX = 6
audio_q = queue.Queue(maxsize=AUDIO_QUEUE_MAX)   # for ASR processing
playback_q = queue.Queue(maxsize=16)             # for live playback (optional)
running = threading.Event()
running.set()

# Ring buffer: store last MONITOR_SECONDS of raw audio (int16 bytes)
bytes_per_block = BLOCKSIZE * 2  # 2 bytes per sample (int16)
max_blocks = max(1, math.ceil((SAMPLE_RATE * MONITOR_SECONDS) / BLOCKSIZE))
ring = deque(maxlen=max_blocks)

# Helper: split bytes into smaller frames (optional)
def frame_generator(frame_bytes, data_bytes):
    offset = 0
    length = len(data_bytes)
    while offset + frame_bytes <= length:
        yield data_bytes[offset:offset + frame_bytes]
        offset += frame_bytes

# Audio callback: push to ASR queue, ring buffer, and playback queue (if enabled)
def audio_callback(indata, frames, time_info, status):
    if status:
        # don't spam, but print occasional status
        print("Audio status:", status, file=sys.stderr)
    try:
        b = indata.copy().tobytes()  # int16 bytes
        # push to ASR queue (drop-old policy)
        try:
            audio_q.put_nowait(b)
        except queue.Full:
            try:
                _ = audio_q.get_nowait()
                audio_q.put_nowait(b)
            except queue.Empty:
                pass
        # store to ring buffer
        ring.append(b)
        # send to playback queue if enabled (drop if full)
        if MONITOR_PLAYBACK:
            try:
                playback_q.put_nowait(b)
            except queue.Full:
                # drop oldest playback item
                try:
                    _ = playback_q.get_nowait()
                    playback_q.put_nowait(b)
                except queue.Empty:
                    pass
    except Exception as e:
        print("Audio callback error:", e, file=sys.stderr)

# ASR processing thread: similar to console script but simplified (no VAD here)
def recognizer_thread():
    last_printed = ""
    try:
        while running.is_set():
            try:
                data = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            # feed entire block to recognizer; use PartialResult for low latency
            if recognizer.AcceptWaveform(data):
                res = recognizer.Result()
                try:
                    text = json.loads(res).get("text", "")
                except Exception:
                    text = ""
                if text:
                    print("\r" + text)
                    last_printed = ""
            else:
                pr = recognizer.PartialResult()
                try:
                    ptext = json.loads(pr).get("partial", "")
                except Exception:
                    ptext = ""
                if ptext:
                    sys.stdout.write("\r" + ptext + "…")
                    sys.stdout.flush()
                    last_printed = ptext
    except Exception as e:
        print("Recognizer thread exception:", e, file=sys.stderr)

# Playback callback: fill output buffer with next block from playback_q or silence
def playback_callback(outdata, frames, time_info, status):
    if status:
        # minimal logging
        pass
    try:
        # expected bytes = frames * 2
        need_bytes = frames * 2
        try:
            b = playback_q.get_nowait()
            # if size differs, trim or pad with zeros
            if len(b) == need_bytes:
                out = np.frombuffer(b, dtype=np.int16)
            else:
                # convert to desired frame size
                arr = np.frombuffer(b, dtype=np.int16)
                if len(arr) >= frames:
                    out = arr[:frames]
                else:
                    out = np.zeros(frames, dtype=np.int16)
                    out[:len(arr)] = arr
        except queue.Empty:
            out = np.zeros(frames, dtype=np.int16)
        outdata[:] = out.reshape(outdata.shape)
    except Exception as e:
        # do not crash audio callback
        outdata[:] = np.zeros(outdata.shape, dtype=np.int16)

# Input thread for simple commands: s=save ring buffer to WAV, p=toggle play, q=quit
def input_thread():
    global MONITOR_PLAYBACK
    print("\nCommands: s = save last %d seconds to WAV, p = toggle playback (current=%s), q = quit\n" % (MONITOR_SECONDS, MONITOR_PLAYBACK))
    while running.is_set():
        try:
            cmd = input().strip().lower()
        except EOFError:
            # stdin closed
            break
        if not cmd:
            continue
        if cmd == 's':
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            fname = os.path.join(BASE_DIR, f"ring_{timestamp}.wav")
            try:
                # join ring buffer bytes in chronological order
                with wave.open(fname, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(b"".join(ring))
                print("Saved ring buffer to", fname)
            except Exception as e:
                print("Failed to save WAV:", e)
        elif cmd == 'p':
            MONITOR_PLAYBACK = not MONITOR_PLAYBACK
            print("Monitor playback toggled ->", MONITOR_PLAYBACK)
        elif cmd == 'q':
            print("Quitting...")
            running.clear()
            break
        else:
            print("Unknown command:", cmd)

# Graceful stop
def stop(signum=None, frame=None):
    running.clear()
    time.sleep(0.2)
    try:
        sd.stop()
    except Exception:
        pass
    print("\nStopped.", file=sys.stderr)
    sys.exit(0)

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

def main():
    print("Starting transcription with monitor support (Ctrl+C to stop).", file=sys.stderr)
    print("Model:", MODEL_PATH, "Sample rate:", SAMPLE_RATE, "Blocksize:", BLOCKSIZE, file=sys.stderr)
    print("Monitor playback:", MONITOR_PLAYBACK, "Monitor seconds:", MONITOR_SECONDS, file=sys.stderr)

    # Start recognizer thread
    th_asr = threading.Thread(target=recognizer_thread, daemon=True)
    th_asr.start()

    # Start input thread
    th_input = threading.Thread(target=input_thread, daemon=True)
    th_input.start()

    # Start playback stream if monitor enabled (keep stream open regardless to avoid popps)
    out_stream = None
    try:
        out_stream = sd.OutputStream(samplerate=SAMPLE_RATE,
                                     device=MONITOR_DEVICE,
                                     dtype='int16',
                                     channels=1,
                                     blocksize=BLOCKSIZE,
                                     callback=playback_callback,
                                     latency='low')
        out_stream.start()
    except Exception as e:
        print("Warning: could not start output stream for monitoring:", e, file=sys.stderr)
        out_stream = None

    # Start input stream (microphone)
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
        print("Failed to start audio input stream:", e, file=sys.stderr)
        running.clear()
        time.sleep(0.2)

    # cleanup
    if out_stream is not None:
        out_stream.stop()
        out_stream.close()

    th_asr.join(timeout=1.0)
    th_input.join(timeout=0.1)

if __name__ == "__main__":
    main()
