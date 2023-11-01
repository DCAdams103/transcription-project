# https://realpython.com/playing-and-recording-sound-python/
import threading
import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog
from ctypes import windll
import pyaudio
import wave
import shutil
import whisper
import time
import torch
from tempfile import NamedTemporaryFile
import speech_recognition as sr
from queue import Queue
from datetime import datetime, timedelta
import io
from time import sleep
import os

# Fixes blurry text
windll.shcore.SetProcessDpiAwareness(1)

# Set chunk size to 1024 samples per data frame
chunk = 1024
sample_format = pyaudio.paInt16 # 16 bits per sample
channels = 1
fs = 44100
seconds = 3
filename = "output.wav"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Voice Recognition")
        self.geometry("800x600")

        self.recordThread = threading.Event()
        self.playThread = threading.Event()
        self.stream = None
        self.frames = []
        self.recording = False
        self.playing = False
        self.paused = False

        # Set up SpeechRecognition
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = 1000
        self.recorder.dynamic_energy_threshold = False
        self.source = sr.Microphone(sample_rate=16000)

        self.record_timeout = 2
        self.phase_timeout = 3

        self.phase_time = None
        self.last_sample = bytes()
        self.data_queue = Queue()

        self.temp_file = NamedTemporaryFile().name

        self.model = whisper.load_model("base")

        self.transcription_text = ['']

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        mainframe = ttk.Frame(self)
        mainframe.pack(side=TOP)
        label = ttk.Label(mainframe, text="Voice Recognition Transcriber", font=("Arial", 25))
        label.pack(side=TOP)
        label2 = ttk.Label(mainframe, text="By Team 2")
        label2.pack(side=TOP)

        # Define buttons
        record = ttk.Button(mainframe, text="Record")
        pause = ttk.Button(mainframe, text="Pause")
        play = ttk.Button(mainframe, text="Play")
        upload = ttk.Button(mainframe, text="Upload File")
        transcribe = ttk.Button(mainframe, text="Transcribe")
        self.textarea = Text(mainframe, height = 5, width = 52)
        live = ttk.Button(mainframe, text='Start Live')
        
        record.config(command=lambda: [self.recordcallback(), self.updateRecordText(record)])
        record.pack()

        upload.config(command=lambda: [self.upload_file()])
        upload.pack()
        
        play.config(command=lambda: [self.playaudioback(play, pause)])
        play.pack()
        
        pause.config(command=lambda: [self.setPaused(), self.disableBtn(pause), self.enable_btn(play)])
        self.disableBtn(pause)
        pause.pack()

        transcribe.config(command=lambda: [self.transcriptAudio()])
        transcribe.pack()

        live.config(command=lambda:[self.sr_test()])
        live.pack()
        
        self.textarea.pack(pady=15)

    def sr_callback(self, _, audio: sr.AudioData) -> None:
        self.data = audio.get_raw_data()
        self.data_queue.put(self.data)

    def sr_test(self):
        s = threading.Thread(target=self.sr_loop)
        s.start()

    def sr_loop(self):
        self.recorder.listen_in_background(self.source, self.sr_callback, phrase_time_limit=self.record_timeout)

        while True:
            try:
                now = datetime.utcnow()

                if not self.data_queue.empty():
                    phrase_complete = False

                    if self.phase_time and now - self.phase_time > timedelta(seconds=self.phase_timeout):
                        self.last_sample = bytes()
                        phrase_complete = True

                    self.phase_time = now

                    while not self.data_queue.empty():
                        data = self.data_queue.get()
                        self.last_sample += data

                    audio_data = sr.AudioData(self.last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                    wav_data = io.BytesIO(audio_data.get_wav_data())

                    with open(self.temp_file, 'w+b') as f:
                        f.write(wav_data.read())

                    result = self.model.transcribe(self.temp_file, fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                    if phrase_complete:
                        self.transcription_text.append(text)
                    else:
                        self.transcription_text[-1] = text

                    os.system('cls' if os.name=='nt' else 'clear')

                    for line in self.transcription_text:
                        print(line)

                    print('', end='', flush=True)
                    sleep(0.25)
            except KeyboardInterrupt:
                break

    def updateRecordText(self, record):
        if record['text'] == 'Record':
            record.config(text='Stop Recording')
        else:
            record.config(text='Record')

    def disableBtn(self, btn):
        btn['state'] = 'disabled'

    def enable_btn(self, btn):
        btn['state'] = 'enabled'

    def setPaused(self):
        self.paused = not self.paused

    def upload_file(self):
        
        filePath = filedialog.askopenfilename(initialdir='/', title='Select a file', filetypes=(('wav', '*.wav'), ('mp3', '*.mp3')))
        
        shutil.copy(r''+filePath, 'output.wav')

    def recordcallback(self):

        if self.recording:
            self.recording = not self.recording
        else:
            self.recording = True
            p = pyaudio.PyAudio()

            print("Recording")
            self.stream = p.open(format=sample_format,
                            channels=channels,
                            rate=fs,
                            frames_per_buffer=chunk,
                            input=True)

            self.frames = []

            # Begin recording
            self.recordThread.set()
            t = threading.Thread(target=self.record_loop, args=(p,))
            t.start()
            

    def record_loop(self, p):
        while self.recording:
            data = self.stream.read(chunk)
            self.frames.append(data)

        # Stop and close the stream
        self.stream.stop_stream()
        self.stream.close()

        # Clear the thread
        self.recordThread.clear()

        # Terminate PyAudio
        p.terminate()

        # Save recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        print("finished recording")

    def play_loop(self, play, pause):
        try:
            # Open sound file
            wf = wave.open(filename, 'rb')
            p = pyaudio.PyAudio()

            self.playing = True

            self.enable_btn(pause)
            self.disableBtn(play)

            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

            # Read data in chunks
            data = wf.readframes(chunk)

            # Play the sound by writing audio data to stream
            while data != b'' and self.playing:
                if not self.paused:
                    stream.write(data)
                    data = wf.readframes(chunk)

            # Close and terminate the stream
            stream.close()
            p.terminate()
            self.playThread.clear()
            self.playing = False
            self.paused = False
            self.disableBtn(pause)
            self.enable_btn(play)

            print("Play finished")
        except (FileNotFoundError):
            print('File not found')
            self.playThread.clear()

    def playaudioback(self, play, pause):

        if self.playing:
            self.paused = False
            self.enable_btn(pause)
        else:
            self.playThread.set()
            t = threading.Thread(target=self.play_loop, args=(play, pause,))
            t.start()

    def transcript_thread(self):
        result = self.model.transcribe("output.wav", fp16=torch.cuda.is_available())
        self.textarea.insert(INSERT, result['text'])
        
    def transcriptAudio(self):
        t = threading.Thread(target=self.transcript_thread())
        t.start()

    def reset_recording(self):
        if self.recording:
            self.recording = False
            #self.recordcallback()
        else:
            self.recordcallback()


    def live_loop(self):
        tic_one = time.perf_counter()
        count = 0
        self.reset_recording()
        while True:
            if(time.perf_counter() - tic_one >= 2):
                #count += 1
                self.transcriptAudio()
                tic_one = time.perf_counter()
                self.reset_recording()

            if count == 3:
                break

    def startLive(self):
        l = threading.Thread(target=self.live_loop)
        l.start()

app = App()
app.mainloop()
