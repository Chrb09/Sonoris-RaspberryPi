import json # Json das configurações da legenda
from vosk import Model, KaldiRecognizer # Vosk para transcrição do áudio
import sounddevice as sd # microfone
import queue
from kivy.app import App
from kivy.uix.label import Label
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.clock import Clock

# Load config
with open('config.json') as f:
    config = json.load(f)

# Register custom font
LabelBase.register(name=config["fonte"], fn_regular=f'fonts/{config["fonte"]}.ttf')

Window.clearcolor = tuple(config["cor_fundo"])  # Set background

class SonorisApp(App):
    def build(self):
        self.label = Label(
            text="Aguardando transcrição...",
            font_name=config["fonte"],
            font_size=config["tamanho"],
            color=tuple(config["cor_texto"]),
            halign='center',
            valign='middle'
        )
        self.label.bind(size=self.label.setter('text_size'))
        return self.label

SonorisApp().run()