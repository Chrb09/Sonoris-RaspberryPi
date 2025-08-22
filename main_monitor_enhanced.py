#!/usr/bin/env python3
"""
main_console_pi_enhanced_monitor_save.py
Enhanced console transcription with processed-audio monitoring and ring-buffer saving.
- Plays back processed audio (pre-emphasis, HPF, RMS normalization, noise gate)
- Keeps a ring buffer of the processed int16 frames for the last MONITOR_SECONDS seconds
- Interactive commands on stdin:
    s -> save the last MONITOR_SECONDS to WAV
    p -> toggle playback on/off (monitor_playback)
    q -> quit
Designed to be lightweight for Raspberry Pi 3B.
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

# Optional VAD (used only if available)
try:
    import webrtcvad
    HAVE_VAD = True
except Exception:
    HAVE_VAD = False

# Optional SymSpell for post-correction (not used in ring buffer logic)
try:
    from symspellpy import SymSpell, Verbosity
    HAVE_SYMSPELL = True
except Exception:
    HAVE_SYMSPELL = False

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
BLOCKSIZE = int(cfg.get("blocksize", 800))
FRAME_MS = int(cfg.get("frame_ms", 20))
USE_VAD = bool(cfg.get("use_vad", True)) and HAVE_VAD
VAD_MODE = int(cfg.get("vad_mode", 2)) if HAVE_VAD else None
DEVICE = cfg.get("device", None)  # optional input device index

# Monitor / playback settings and ring buffer size
MONITOR_PLAYBACK = bool(cfg.get("monitor_playback", False))
MONITOR_DEVICE = cfg.get("monitor_device", None)  # output device index or None for default
MONITOR_SECONDS = int(cfg.get("monitor_seconds", 30))  # how many seconds to keep in ring buffer
PLAYBACK_QUEUE_MAX = int(cfg.get("playback_queue_max", 8))

# Preprocessing params
PREEMPHASIS = float(cfg.get("preemphasis", 0.97))
RMS_TARGET = float(cfg.get("rms_target", 0.08))
NOISE_GATE_DB = float(cfg.get("noise_gate_db", -40.0))

# Safety checks
if FRAME_MS not in (10, 20, 30):
    FRAME_MS = 20
if BLOCKSIZE <= 0:
    BLOCKSIZE = 800
if MONITOR_SECONDS <= 0:
    MONITOR_SECONDS = 30

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

# Queues and control
AUDIO_QUEUE_MAX = 6
audio_q = queue.Queue(maxsize=AUDIO_QUEUE_MAX)
playback_q = queue.Queue(maxsize=PLAYBACK_QUEUE_MAX)  # holds int16 bytes ready for playback
running = threading.Event()
running.set()

# Ring buffer for processed int16 bytes (chronological order)
# Compute number of blocks needed to hold MONITOR_SECONDS seconds
blocks_for_seconds = max(1, math.ceil((SAMPLE_RATE * MONITOR_SECONDS) / BLOCKSIZE))
ring = deque(maxlen=blocks_for_seconds)

# Helper functions: preprocessing
def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def dbfs(x):
    eps = 1e-9
    rms = np.sqrt(np.mean(x*x)) + eps
    return 20 * np.log10(rms)

def float32_to_bytes(x):
    arr = np.clip(x * 32768.0, -32768, 32767).astype(np.int16)
    return arr.tobytes()

def frame_generator(frame_ms, sample_rate, data_bytes):
    bytes_per_sample = 2
    frame_bytes = int(sample_rate * (frame_ms / 1000.0) * bytes_per_sample)
    offset = 0
    length = len(data_bytes)
    while offset + frame_bytes <= length:
        yield data_bytes[offset:offset + frame_bytes]
        offset += frame_bytes

# ---------- Início do patch (substituir no seu script) ----------
# Variáveis globais persistentes para o HPF e crossfade
_hp_prev_y = 0.0
_hp_prev_x = 0.0
_prev_tail = None  # para crossfade entre blocos
CROSSFADE_MS = 10  # comprimento do crossfade em ms (ajuste: 8-15ms é bom)

def highpass_iir(x, sr, cutoff=80.0):
    """Single-pole HPF com estado persistente entre blocos (evita descontinuidades)."""
    global _hp_prev_y, _hp_prev_x
    dt = 1.0 / sr
    rc = 1.0 / (2 * np.pi * cutoff)
    a = rc / (rc + dt)
    y = np.empty_like(x)
    prev_y = _hp_prev_y
    prev_x = _hp_prev_x
    for i, xi in enumerate(x):
        yi = a * (prev_y + xi - prev_x)
        y[i] = yi
        prev_y = yi
        prev_x = xi
    # salvar estado para o próximo bloco
    _hp_prev_y = prev_y
    _hp_prev_x = prev_x
    return y

def rms_normalize(x, target=0.08, max_gain=6.0, noise_gate_db=-42.0):
    """
    Normalize RMS but avoid boosting noise:
    - Se o RMS está muito baixo (abaixo do gate), retorna zeros (silêncio).
    - Limita ganho máximo para evitar amplificação exagerada.
    """
    eps = 1e-9
    rms = np.sqrt(np.mean(x * x)) + eps
    db = 20 * np.log10(rms) if rms > 0 else -200.0
    if db < noise_gate_db:
        # Considera silêncio/ruído -> não amplificar
        return np.zeros_like(x)
    gain = target / rms
    if gain > max_gain:
        gain = max_gain
    return x * gain

def _apply_crossfade(arr):
    """Suaviza a fronteira entre blocos para evitar clicks.
    Faz crossfade com o _prev_tail global (comprimento CROSSFADE_MS)."""
    global _prev_tail
    fade_len = int((CROSSFADE_MS / 1000.0) * SAMPLE_RATE)
    if fade_len <= 0:
        _prev_tail = arr[-min(len(arr), 1):].copy()
        return arr
    if _prev_tail is None:
        # ainda não há bloco anterior: apenas set tail e retorna
        _prev_tail = arr[-fade_len:].copy()
        return arr
    # garante tamanho
    head_len = min(fade_len, len(arr))
    prev_len = min(fade_len, len(_prev_tail))
    N = min(head_len, prev_len)
    if N > 0:
        win = np.linspace(0.0, 1.0, N)
        arr[:N] = _prev_tail[-N:] * (1.0 - win) + arr[:N] * win
    # atualiza prev_tail com final deste bloco
    _prev_tail = arr[-fade_len:].copy()
    return arr

# Substitua sua audio_callback pelo abaixo (ou ajuste a existente):
def audio_callback(indata, frames, time_info, status):
    if status:
        # mostrar pouco para não congestionar I/O
        print("Audio status:", status, file=sys.stderr)
    try:
        # converter int16 -> float32 (-1..1)
        arr = indata.copy().astype(np.int16).astype(np.float32) / 32768.0

        # HIGH-PASS com estado persistente (remove rumble)
        arr = highpass_iir(arr, SAMPLE_RATE, cutoff=80.0)

        # PRE-EMPHASIS (opcional): se quiser desabilitar, comente a próxima linha
        if PREEMPHASIS and PREEMPHASIS > 0:
            arr = np.append(arr[0], arr[1:] - PREEMPHASIS * arr[:-1])

        # CROSSFADE com bloco anterior para evitar pops
        arr = _apply_crossfade(arr)

        # NORMALIZAÇÃO condicionada (não amplifica silêncio)
        arr = rms_normalize(arr, target=RMS_TARGET, max_gain=6.0, noise_gate_db=NOISE_GATE_DB)

        # Se todo frame virou zero (gate), converte para zeros int16 sem processamento
        if np.all(arr == 0):
            b = (np.zeros(frames, dtype=np.int16)).tobytes()
        else:
            # converte de volta para int16
            arr_int16 = np.clip(arr * 32768.0, -32768, 32767).astype(np.int16)
            b = arr_int16.tobytes()

        # push para filas (como antes), com política drop-old
        try:
            audio_q.put_nowait(b)
        except queue.Full:
            try:
                _ = audio_q.get_nowait()
                audio_q.put_nowait(b)
            except queue.Empty:
                pass

        # se tiver monitor playback, enfileira também
        if MONITOR_PLAYBACK:
            try:
                playback_q.put_nowait(b)
            except queue.Full:
                try:
                    _ = playback_q.get_nowait()
                    playback_q.put_nowait(b)
                except queue.Empty:
                    pass

        # guarde no ring buffer (processed)
        try:
            ring.append(b)
        except Exception:
            pass

    except Exception as e:
        # não deixar callback travar
        print("Audio callback error:", e, file=sys.stderr)

# ASR processing thread: feeds Vosk and prints partials/finals
def recognizer_thread():
    last_printed = ""
    in_speech = False
    silence_frames = 0
    try:
        while running.is_set():
            try:
                data = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            for frame in frame_generator(FRAME_MS, SAMPLE_RATE, data):
                is_speech = True
                if vad is not None:
                    try:
                        is_speech = vad.is_speech(frame, SAMPLE_RATE)
                    except Exception:
                        is_speech = True

                if vad is not None and not is_speech and not in_speech:
                    silence_frames += 1
                    if silence_frames < 2:
                        if recognizer.AcceptWaveform(frame):
                            res = recognizer.Result()
                            try:
                                final_text = json.loads(res).get("text", "")
                            except Exception:
                                final_text = ""
                            if final_text:
                                print("\r" + final_text)
                                last_printed = ""
                    continue

                silence_frames = 0
                if recognizer.AcceptWaveform(frame):
                    res = recognizer.Result()
                    try:
                        final = json.loads(res).get("text", "")
                    except Exception:
                        final = ""
                    if final:
                        print("\r" + final)
                        last_printed = ""
                        in_speech = False
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
                        in_speech = True
    except Exception as e:
        print("Recognizer thread exception:", e, file=sys.stderr)

# Playback callback: feed output buffer with next available processed block or silence
def playback_callback(outdata, frames, time_info, status):
    if status:
        # minimal logging
        pass
    try:
        need_bytes = frames * 2
        try:
            b = playback_q.get_nowait()
            arr = np.frombuffer(b, dtype=np.int16)
            # if different length, pad/trim
            if len(arr) < frames:
                out = np.zeros(frames, dtype=np.int16)
                out[:len(arr)] = arr
            else:
                out = arr[:frames]
        except queue.Empty:
            out = np.zeros(frames, dtype=np.int16)
        outdata[:] = out.reshape(outdata.shape)
    except Exception:
        outdata[:] = np.zeros(outdata.shape, dtype=np.int16)

# Input thread for simple commands: s=save ring buffer to WAV, p=toggle play, q=quit
def input_thread():
    global MONITOR_PLAYBACK
    print("\nCommands: s = save last %d seconds to WAV, p = toggle playback (current=%s), q = quit\n" % (MONITOR_SECONDS, MONITOR_PLAYBACK))
    while running.is_set():
        try:
            cmd = input().strip().lower()
        except EOFError:
            break
        if not cmd:
            continue
        if cmd == 's':
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            fname = os.path.join(BASE_DIR, f"processed_ring_{timestamp}.wav")
            try:
                with wave.open(fname, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    # join ring buffer bytes in chronological order
                    wf.writeframes(b"".join(ring))
                print("Saved processed last %d seconds to %s" % (MONITOR_SECONDS, fname))
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

# Graceful shutdown
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
    global MONITOR_PLAYBACK, MONITOR_DEVICE
    print("Starting enhanced transcription with processed-audio monitor + save (Ctrl+C to stop).", file=sys.stderr)
    print("Model:", MODEL_PATH, "Sample rate:", SAMPLE_RATE, "Blocksize:", BLOCKSIZE, file=sys.stderr)
    print("Monitor playback:", MONITOR_PLAYBACK, "Monitor device:", MONITOR_DEVICE, file=sys.stderr)
    print("Ring buffer seconds:", MONITOR_SECONDS, "Blocks stored:", blocks_for_seconds, file=sys.stderr)

    th = threading.Thread(target=recognizer_thread, daemon=True)
    th.start()

    th_in = threading.Thread(target=input_thread, daemon=True)
    th_in.start()

    # Start output stream for playback if monitor enabled
    out_stream = None
    if MONITOR_PLAYBACK:
        try:
            out_stream = sd.OutputStream(samplerate=SAMPLE_RATE,
                                         device=MONITOR_DEVICE,
                                         dtype='int16',
                                         channels=1,
                                         blocksize=BLOCKSIZE,
                                         callback=playback_callback,
                                         latency='low')
            out_stream.start()
            print("Playback output stream started.", file=sys.stderr)
        except Exception as e:
            print("Warning: could not start playback stream:", e, file=sys.stderr)
            out_stream = None

    # Start input stream
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

    th.join(timeout=1.0)
    th_in.join(timeout=0.1)

if __name__ == "__main__":
    main()
