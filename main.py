# https://realpython.com/playing-and-recording-sound-python/
import threading
import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog
from ctypes import windll
import pyaudio                              # Record / Play audio
import wave                                 # Audio file
import shutil                               # File copy
import whisper                              # Transcription engine
import time                         
import torch                                # GPU Acceleration
from tempfile import NamedTemporaryFile     
import speech_recognition as sr             
from queue import Queue
from datetime import datetime, timedelta
import io
import os
from gradio_client import Client # Whisper JAX
from pickle import Pickler, Unpickler

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

        
        # # Set up SpeechRecognition
        # self.recorder = sr.Recognizer()
        # self.recorder.energy_threshold = 1000
        # self.recorder.dynamic_energy_threshold = False
        # self.source = sr.Microphone(sample_rate=16000)
        # self.record_timeout = 2
        # self.phase_timeout = 3
        # self.phase_time = None
        # self.last_sample = bytes()
        # self.data_queue = Queue()
        # self.model = whisper.load_model("medium")
        # self.transcription_text = ['']

        # with self.source as source:
        #     self.recorder.adjust_for_ambient_noise(source)

        self.mainframe = ttk.Frame(self)
        self.mainframe.pack(side=TOP)
        

        # Define buttons
        
        # live = ttk.Button(mainframe, text='Start Live')

        # live.config(command=lambda:[self.sr_test()])
        # live.pack()

        # switch.config(command=self.switchframe)

        self.mainframe.pack(side="top", fill="both", expand=True)
        self.mainframe.grid_rowconfigure(0, weight=1)
        self.mainframe.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, TranscribePage, LivePage):
            page = F.__name__
            frame = F(parent=self.mainframe, controller=self)
            self.frames[page] = frame

            frame.grid(row=0, column=0, sticky="nsew")
        
        self.show_frame("StartPage")


    def show_frame(self, page):
        frame = self.frames[page]
        frame.tkraise()

    def sr_callback(self, _, audio: sr.AudioData) -> None:
        self.data = audio.get_raw_data()
        self.data_queue.put(self.data)

    # Live Transcription
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
                    time.sleep(0.25)

            except KeyboardInterrupt:
                break

    def live_loop(self):
        tic_one = time.perf_counter()
        count = 0
        self.reset_recording()
        while True:
            if(time.perf_counter() - tic_one >= 2):
                #count += 1
                self.transcript_audio()
                tic_one = time.perf_counter()
                self.reset_recording()

            if count == 3:
                break

    def start_live_recording(self):
        l = threading.Thread(target=self.live_loop)
        l.start()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label = ttk.Label(self, text="Voice Recognition Transcriber", font=("Arial", 25))
        label.pack(side=TOP)

        label2 = ttk.Label(self, text="By Team 2")
        label2.pack(side=TOP)

        button = tk.Button(self, text="Transcribe", command=lambda:controller.show_frame("TranscribePage"))
        button.pack()

        button1 = tk.Button(self, text="Live", command=lambda:controller.show_frame("LivePage"))
        button1.pack()

class TranscribePage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        home = ttk.Button(self, text="Back")
        record = ttk.Button(self, text="Record")
        pause = ttk.Button(self, text="Pause")
        play = ttk.Button(self, text="Play")
        upload = ttk.Button(self, text="Upload File")
        transcribe = ttk.Button(self, text="Transcribe")
        self.textarea = Text(self, height = 5, width = 52)
        
        home.config(command=lambda:controller.show_frame("StartPage"))
        home.pack()

        record.config(command=lambda: [self.record_callback(), self.update_record_text(record)])
        record.pack()

        upload.config(command=lambda: [self.upload_file()])
        upload.pack()
        
        play.config(command=lambda: [self.play_recording(play, pause)])
        play.pack()
        
        pause.config(command=lambda: [self.setPaused(), self.disable_button(pause), self.enable_btn(play)])
        self.disable_button(pause)
        pause.pack()

        transcribe.config(command=lambda: [self.transcript_audio()])
        transcribe.pack()

        self.textarea.pack(pady=15)

        self.recordThread = threading.Event()
        self.playThread = threading.Event()
        self.stream = None
        self.frames = []
        self.recording = False
        self.playing = False
        self.paused = False

        self.temp_file = NamedTemporaryFile().name


    def record_callback(self):

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

    # Record Audio
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

    # Upload File
    def upload_file(self):
        filePath = filedialog.askopenfilename(initialdir='/', title='Select a file', filetypes=(('wav', '*.wav'), ('mp3', '*.mp3')))
        shutil.copy(r''+filePath, 'output.wav')

    def play_recording(self, play, pause):

        if self.playing:
            self.paused = False
            self.enable_btn(pause)
        else:
            self.playThread.set()
            t = threading.Thread(target=self.play_thread, args=(play, pause,))
            t.start()

    # Play Recording 
    def play_thread(self, play, pause):
        try:
            # Open sound file
            wf = wave.open(filename, 'rb')
            p = pyaudio.PyAudio()

            self.playing = True

            self.enable_btn(pause)
            self.disable_button(play)

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
            self.disable_button(pause)
            self.enable_btn(play)

            print("Play finished")
        except (FileNotFoundError):
            print('File not found')
            self.playThread.clear()

    def transcript_thread(self):
        # result = self.model.transcribe("output.wav", fp16=torch.cuda.is_available())
        client = Client("https://sanchit-gandhi-whisper-jax.hf.space/")
        result = client.predict(
				"output.wav",	# str (filepath or URL to file) in 'inputs' Audio component
				"transcribe",	# str in 'Task' Radio component
				True,	# bool in 'Return timestamps' Checkbox component
				api_name="/predict"
        )
        print(result)

        #self.textarea.insert(INSERT, result['text'])
        
    def transcript_audio(self):
        t = threading.Thread(target=self.transcript_thread)
        t.start()

    # Update GUI Functions
    def update_record_text(self, record):
        if record['text'] == 'Record':
            record.config(text='Stop Recording')
        else:
            record.config(text='Record')

    def disable_button(self, btn):
        btn['state'] = 'disabled'

    def enable_btn(self, btn):
        btn['state'] = 'enabled'

    # Reset Value Functions
    def setPaused(self):
        self.paused = not self.paused

    def reset_recording(self):
        if self.recording:
            self.recording = False
            #self.record_callback()
        else:
            self.record_callback()

class LivePage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Live")
        label.pack()

if __name__ == "__main__":
    app = App()
    app.mainloop()
