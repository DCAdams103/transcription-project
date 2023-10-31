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

        live.config(command=lambda:[self.startLive()])
        live.pack()
        
        self.textarea.pack(pady=15)

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
        
    def transcriptAudio(self):
        model = whisper.load_model("base")
        result = model.transcribe("output.wav")
        self.textarea.insert(INSERT, result['text'])

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
