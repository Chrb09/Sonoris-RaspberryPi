#!/usr/bin/env python3

import os
import sys
import json
import queue
import threading
import time
import signal

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.core.text import LabelBase
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.button import Button

try:
    from vosk import Model, KaldiRecognizer
except Exception as e:
    print("Error: vosk not installed or model not found.", e, file=sys.stderr)
    raise

import sounddevice as sd

# Optional VAD
try:
    import webrtcvad
    HAVE_VAD = True
except Exception:
    HAVE_VAD = False

# Load config.json or use defaults
BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
else:
    cfg = {}

MODEL_PATH = cfg.get("model_path", "models/vosk-model-small-pt-0.3")
SAMPLE_RATE = int(cfg.get("sample_rate", 16000))
BLOCKSIZE = int(cfg.get("blocksize", 800))
FRAME_MS = int(cfg.get("frame_ms", 30))
DEVICE = cfg.get("device", None)
USE_VAD = bool(cfg.get("use_vad", True)) and HAVE_VAD
VAD_MODE = int(cfg.get("vad_mode", 2)) if HAVE_VAD else None

# UI / partial limits
MAX_PARTIAL_CHARS = int(cfg.get("max_partial_chars", 240))  # limit partial length to avoid overflow
PARTIAL_UPDATE_MIN_MS = int(cfg.get("partial_update_min_ms", 80))  # min interval between UI partial updates
HISTORY_MAX_LINES = int(cfg.get("history_max_lines", 200))
PARTIAL_RESET_MS = int(cfg.get("partial_reset_ms", 3000))  # ms to return to "Aguardando..." after last partial

# Kivy style config
FONT_NAME = cfg.get("fonte", None)  # optional custom font name (ttf file in fonts/)
FONT_SIZE_PARTIAL = cfg.get("tamanho_parcial", 28)
FONT_SIZE_HISTORY = cfg.get("tamanho_historico", 20)
BACKGROUND_COLOR = tuple(cfg.get("cor_fundo", [0.0, 0.0, 0.0, 1.0]))
TEXT_COLOR = tuple(cfg.get("cor_texto", [1.0, 1.0, 1.0, 1.0]))

if FONT_NAME and os.path.exists(os.path.join(BASE_DIR, "fonts", f"{FONT_NAME}.ttf")):
    LabelBase.register(name=FONT_NAME, fn_regular=os.path.join(BASE_DIR, "fonts", f"{FONT_NAME}.ttf"))

Window.clearcolor = BACKGROUND_COLOR

# Validate FRAME_MS
if FRAME_MS not in (10, 20, 30):
    FRAME_MS = 30

# Check model path
if not os.path.exists(MODEL_PATH):
    print(f"Vosk model not found at '{MODEL_PATH}'. Please download and set model_path in config.json.", file=sys.stderr)
    sys.exit(1)

print("Loading Vosk model (this may take a while)...", file=sys.stderr)
model = Model(MODEL_PATH)

# Recognizer used for streaming partials (single instance)
streaming_recognizer = KaldiRecognizer(model, SAMPLE_RATE)

# Setup VAD if requested
vad = None
if USE_VAD:
    try:
        vad = webrtcvad.Vad(VAD_MODE)
        print("webrtcvad enabled (mode=%d)." % VAD_MODE, file=sys.stderr)
    except Exception as e:
        print("Failed to initialize webrtcvad, continuing without VAD:", e, file=sys.stderr)
        vad = None

# Audio queue and control
AUDIO_QUEUE_MAX = 8
audio_q = queue.Queue(maxsize=AUDIO_QUEUE_MAX)
running = threading.Event()
running.set()

# Timing for partial UI updates
_last_partial_update = 0.0

# Helper to split bytes into frames for VAD/feeding recognizer
def frame_generator(frame_ms, sample_rate, data_bytes):
    bytes_per_sample = 2
    frame_bytes = int(sample_rate * (frame_ms / 1000.0) * bytes_per_sample)
    offset = 0
    length = len(data_bytes)
    while offset + frame_bytes <= length:
        yield data_bytes[offset:offset + frame_bytes]
        offset += frame_bytes

# Audio callback: push raw bytes to queue (drop-old policy)
def audio_callback(indata, frames, time_info, status):
    if status:
        # don't spam too much
        print("Audio status:", status, file=sys.stderr)
    try:
        b = indata.tobytes()
        try:
            audio_q.put_nowait(b)
        except queue.Full:
            try:
                _ = audio_q.get_nowait()
                audio_q.put_nowait(b)
            except queue.Empty:
                pass
    except Exception as e:
        print("Callback error:", e, file=sys.stderr)

# Kivy UI classes
class TranscriptHistory(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 1
        self.size_hint_y = None
        self.bind(minimum_height=self.setter('height'))
        self.lines = []

    def add_line(self, text):
        # append and trim history
        lbl = Label(text=text, size_hint_y=None, height=FONT_SIZE_HISTORY*1.6, halign='left', valign='middle',
                    text_size=(self.width, None), font_size=FONT_SIZE_HISTORY, color=TEXT_COLOR, markup=False)
        lbl.bind(width=lambda inst, w: inst.setter('text_size')(inst, (w, None)))
        self.add_widget(lbl)
        self.lines.append(lbl)
        if len(self.lines) > HISTORY_MAX_LINES:
            old = self.lines.pop(0)
            self.remove_widget(old)

    def clear_all(self):
        for w in list(self.lines):
            try:
                self.remove_widget(w)
            except Exception:
                pass
        self.lines = []

class MainLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=8, spacing=6, **kwargs)

        # Top toolbar with Clear History button
        toolbar = BoxLayout(size_hint=(1, None), height=40, spacing=6)
        btn_clear = Button(text="Limpar histórico", size_hint=(None, 1), width=160)
        btn_clear.bind(on_release=self._on_clear_history)
        toolbar.add_widget(btn_clear)
        # spacer
        toolbar.add_widget(Label(text="", size_hint=(1,1)))
        self.add_widget(toolbar)

        # history scrollview (middle)
        self.scroll = ScrollView(size_hint=(1, 0.65))
        self.history = TranscriptHistory()
        self.scroll.add_widget(self.history)
        self.add_widget(self.scroll)

        # partial label (bottom)
        self.partial_label = Label(text="Aguardando...", size_hint=(1, 0.25), halign='center', valign='middle',
                                   text_size=(None, None), font_size=FONT_SIZE_PARTIAL, color=TEXT_COLOR)
        self.partial_label.bind(size=self._update_partial_text_size)
        self.add_widget(self.partial_label)

        # internal for partial reset scheduling
        self._partial_reset_ev = None

    def _update_partial_text_size(self, inst, val):
        inst.text_size = (inst.width - 20, inst.height)

    def add_final(self, text):
        # add final recognized line to history and scroll to bottom
        sanitized = text.strip().capitalize() if text else ""
        if sanitized:
            self.history.add_line(sanitized)
            Clock.schedule_once(lambda dt: self.scroll.scroll_to(self.history.lines[-1]))
        # reset partial to awaiting after a final is produced
        Clock.schedule_once(lambda dt: self.set_partial("Aguardando..."), 0.01)

    def set_partial(self, text):
        # set the partial label (thread-safe via Clock)
        # if this is called from background thread it will be scheduled through Clock,
        # but we still protect scheduling of the reset timer here.
        self.partial_label.text = text
        # cancel previous reset if any
        if self._partial_reset_ev:
            try:
                self._partial_reset_ev.cancel()
            except Exception:
                pass
            self._partial_reset_ev = None
        # schedule reset to "Aguardando..." after PARTIAL_RESET_MS if text is non-empty and not already "Aguardando..."
        txt = (text or "").strip()
        if txt and txt.lower() != "aguardando..." and PARTIAL_RESET_MS > 0:
            self._partial_reset_ev = Clock.schedule_once(lambda dt: self._reset_partial(), PARTIAL_RESET_MS / 1000.0)

    def _reset_partial(self):
        self._partial_reset_ev = None
        self.partial_label.text = "Aguardando..."

    def _on_clear_history(self, instance):
        # clear UI history and reset partial
        self.history.clear_all()
        self._reset_partial()

# Worker thread: consumes audio frames, feeds Vosk recognizer, updates UI via Clock
def recognizer_worker(app_ref):
    global streaming_recognizer, _last_partial_update
    print("Recognizer worker started.", file=sys.stderr)
    last_printed_partial = ""
    in_speech = False
    silence_frames = 0
    vad_hangover = int(250 / FRAME_MS)  # 250ms hangover

    while running.is_set():
        try:
            data = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        for frame in frame_generator(FRAME_MS, SAMPLE_RATE, data):
            # VAD check if available
            is_speech = True
            if vad is not None:
                try:
                    is_speech = vad.is_speech(frame, SAMPLE_RATE)
                except Exception:
                    is_speech = True

            if vad is not None and not is_speech and not in_speech:
                # in silence before speech starts: feed occasional frame to keep recognizer state
                silence_frames += 1
                if silence_frames < 2:
                    try:
                        if streaming_recognizer.AcceptWaveform(frame):
                            res = streaming_recognizer.Result()
                            text = json.loads(res).get("text", "")
                            if text:
                                # schedule add_final on UI thread
                                Clock.schedule_once(lambda dt, t=text: app_ref.add_final(t))
                                last_printed_partial = ""
                        else:
                            pres = streaming_recognizer.PartialResult()
                            partial = json.loads(pres).get("partial", "")
                            # update partial UI at limited rate
                            now = time.time() * 1000
                            if partial and partial != last_printed_partial and (now - _last_partial_update) >= PARTIAL_UPDATE_MIN_MS:
                                # truncate safely
                                p = _truncate_partial(partial)
                                Clock.schedule_once(lambda dt, p=p: app_ref.set_partial(p))
                                last_printed_partial = partial
                                _last_partial_update = now
                    except Exception:
                        pass
                continue

            # speech or no VAD: feed to recognizer continuously
            silence_frames = 0
            try:
                if streaming_recognizer.AcceptWaveform(frame):
                    res = streaming_recognizer.Result()
                    final = json.loads(res).get("text", "")
                    if final:
                        # final result: push to history in UI thread
                        Clock.schedule_once(lambda dt, t=final: app_ref.add_final(t))
                        last_printed_partial = ""
                        in_speech = False
                else:
                    pres = streaming_recognizer.PartialResult()
                    partial = json.loads(pres).get("partial", "")
                    if partial is not None:
                        # limit partials to avoid runaway length causing issues
                        now = time.time() * 1000
                        if partial and partial != last_printed_partial and (now - _last_partial_update) >= PARTIAL_UPDATE_MIN_MS:
                            p = _truncate_partial(partial)
                            Clock.schedule_once(lambda dt, p=p: app_ref.set_partial(p))
                            last_printed_partial = partial
                            _last_partial_update = now
                        in_speech = True
            except Exception as e:
                print("Recognizer error:", e, file=sys.stderr)
                # try to reset recognizer to recover from errors
                try:
                    streaming_recognizer = KaldiRecognizer(model, SAMPLE_RATE)
                except Exception:
                    pass

    print("Recognizer worker finished.", file=sys.stderr)

def _truncate_partial(text):
    """Truncate partial safely to MAX_PARTIAL_CHARS without breaking words badly."""
    if not text:
        return ""
    t = text.strip()
    if len(t) <= MAX_PARTIAL_CHARS:
        return t.capitalize()
    # try to cut at last space before limit
    cut = t[:MAX_PARTIAL_CHARS]
    last_space = cut.rfind(' ')
    if last_space > int(MAX_PARTIAL_CHARS * 0.6):
        cut = cut[:last_space]
    return (cut + '…').capitalize()

class TranscriberApp(App):
    def build(self):
        self.title = "Sonoris - Transcrição"
        self.layout = MainLayout()
        return self.layout

    def on_start(self):
        # start worker thread
        running.set()
        self._worker = threading.Thread(target=recognizer_worker, args=(self.layout,), daemon=True)
        self._worker.start()

        # start audio stream
        try:
            self._stream = sd.InputStream(samplerate=SAMPLE_RATE,
                                         device=DEVICE,
                                         dtype='int16',
                                         channels=1,
                                         callback=audio_callback,
                                         blocksize=BLOCKSIZE,
                                         latency='low')
            self._stream.start()
            print("Audio stream started.", file=sys.stderr)
        except Exception as e:
            print("Failed to start input stream:", e, file=sys.stderr)
            running.clear()

    def on_stop(self):
        running.clear()
        try:
            if hasattr(self, "_stream") and self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        # give worker a moment
        if hasattr(self, "_worker"):
            self._worker.join(timeout=1.0)

if __name__ == "__main__":
    TranscriberApp().run()
