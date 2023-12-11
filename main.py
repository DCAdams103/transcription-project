# https://realpython.com/playing-and-recording-sound-python/
import threading
import tkinter as tk
from tkinter import *
from tkinter import filedialog
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
import pickle
import customtkinter
from CTkToolTip import *
import re
from functools import partial
from datetime import datetime
import librosa
import soundfile as sf

# Fixes blurry text
windll.shcore.SetProcessDpiAwareness(1)

# Set chunk size to 1024 samples per data frame
chunk = 1024
sample_format = pyaudio.paInt16 # 16 bits per sample
channels = 1
fs = 44100
seconds = 3


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Voice Recognition")
        self.geometry("800x600")
        #self.resizable(False, False)
        customtkinter.set_appearance_mode("Dark")
        customtkinter.set_default_color_theme("dark-blue")
        
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

        self.mainframe = customtkinter.CTkFrame(self, bg_color="black")
        self.mainframe.pack(side=TOP)

        # Define buttons
        
        # live = ttk.Button(mainframe, text='Start Live')

        # live.config(command=lambda:[self.sr_test()])
        # live.pack()

        # switch.config(command=self.switchframe)

        self.mainframe.pack(side="top", fill="both", expand=True)
        self.mainframe.grid_rowconfigure(0, weight=1)
        self.mainframe.grid_columnconfigure(0, weight=1)

        try:
            self.transcriptions = pickle.load(open("save.p", "rb"))
        except FileNotFoundError:
            print("file not found")

        self.key = ""

        self.frames = {}
        for F in (StartPage, TranscribePage, LivePage):
            page = F.__name__
            frame = F(parent=self.mainframe, controller=self)
            self.frames[page] = frame

            frame.grid(row=0, column=0, sticky="nsew")
        
        self.show_frame("StartPage")


    def show_frame(self, page, text=""):
        frame = self.frames[page]
        frame.update()
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


class StartPage(customtkinter.CTkFrame):

    def __init__(self, parent, controller):
        customtkinter.CTkFrame.__init__(self, parent)
        self.controller = controller
        self.text=""

        label = customtkinter.CTkLabel(self, text="Voice Recognition Transcriber", font=("Arial", 25))
        label.pack(side=TOP)

        label2 = customtkinter.CTkLabel(self, text="By Team 2")
        label2.pack(side=TOP)

        #button = customtkinter.CTkButton(self, text="Transcribe", command=lambda:controller.show_frame("TranscribePage"))
        #button.pack()

        #button1 = customtkinter.CTkButton(self, text="Live", command=lambda:controller.show_frame("LivePage"))
        #button1.pack()
        
        # db = {}
        # pickle.dump(db, open("save.p", "wb"))
        
        self.buttons = []

        try:
            for index, key in enumerate(controller.transcriptions):
                btn = customtkinter.CTkButton(self, text=key, width=150, height=150)
                btn.configure(command=partial(self.set_key, btn.cget('text')))
                btn.pack(side=LEFT, padx=20)
                self.buttons.append(btn)
        except Exception as e:
            print(e)


        self.create_new = customtkinter.CTkButton(self, text="+", width=150, height=150, command=lambda:[self.new_transcription(), controller.show_frame("TranscribePage")])
        self.create_new.pack(side=LEFT, padx=20)

    def update(self):
        try:

            # Unpack existing buttons
            for button in self.buttons:
                button.pack_forget()

            self.create_new.pack_forget()
            
            existing = pickle.load(open("save.p", "rb"))
            
            # Pack buttons again, showing any new entries or changes
            for index, key in enumerate(existing):
                btn = customtkinter.CTkButton(self, text=key, width=150, height=150)
                btn.configure(command=partial(self.set_key, btn.cget('text')))
                btn.pack(side=LEFT, padx=20)
                self.buttons.append(btn)
            
            self.create_new.pack(side=LEFT, padx=20)
        except Exception as e:
            print(e)

    def set_key(self, key):
        self.controller.key = key
        self.controller.show_frame("TranscribePage")

    def new_transcription(self):

        self.controller.key = "New Transcription 1"
        temp = self.controller.key

        # Last character of title
        last = temp[len(temp)-1]

        try:
            existing = pickle.load(open("save.p", "rb"))
            
            for key in existing:
                # Check to see if the transcription already exists
                if temp in existing:
                    if last.isdigit():
                        last = str(int(last) + 1)
                        # If new title does not exists, use it
                        if 'New Transcription ' + str(last) not in existing:
                            self.controller.key = self.controller.key[:-1] + str(last)
                            break

        except FileNotFoundError:
            print("file not found")
        

class TranscribePage(customtkinter.CTkFrame):

    def __init__(self, parent, controller):
        customtkinter.CTkFrame.__init__(self, parent)
        self.controller = controller

        self.title = customtkinter.CTkEntry(self, textvariable=tk.StringVar(self, self.controller.key), justify="center")

        home = customtkinter.CTkButton(self, text="Back")
        
        record = customtkinter.CTkButton(self, text="Record")
        record_tooltip = CTkToolTip(record, delay=0.5, message="Begin recording audio")
        
        self.pause = customtkinter.CTkButton(self, text="Pause")
        pause_tooltip = CTkToolTip(self.pause, delay=0.5, message="Pause audio playback")

        self.play = customtkinter.CTkButton(self, text="Play")
        play_tooltip = CTkToolTip(self.play, delay=0.5, message="Begin playing saved audio")

        upload = customtkinter.CTkButton(self, text="Upload File")
        upload_tooltip = CTkToolTip(upload, delay=0.5, message="Upload an audio file")

        transcribe = customtkinter.CTkButton(self, text="Transcribe")
        transcribe_tooltip = CTkToolTip(transcribe, delay=0.5, message="Create transcription from given audio")

        self.textarea = customtkinter.CTkTextbox(self, width = 600, corner_radius=10)
        self.textarea.bind("<Motion>", self.hover)
        self.textarea.bind("<Button-1>", self.click)
        
        self.title.configure(font=('Arial', 25), width=300, height=10)
        self.title.pack(side=TOP)

        home.configure(command=lambda:controller.show_frame("StartPage"))
        home.pack(side=TOP, anchor=NW, padx=10, pady=10)

        record.configure(command=lambda: [self.record_callback(), self.update_record_text(record)])
        record.pack(pady=5)

        upload.configure(command=lambda: [self.upload_file()])
        upload.pack()
        
        self.play.configure(command=lambda: [self.play_recording(self.pause)])
        self.play.pack(pady=5)
        
        self.pause.configure(command=lambda: [self.setPaused(), self.disable_button(self.pause), self.enable_btn(self.play)])
        self.disable_button(self.pause)
        self.pause.pack()

        transcribe.configure(command=lambda: [self.transcript_audio()])
        transcribe.pack(pady=5)

        self.textarea.pack(pady=10)

        save = customtkinter.CTkButton(self, text="Save")
        save_tooltop = CTkToolTip(save, delay=0.5, message="Save your transcription")
        save.configure(command=lambda: [self.save()])
        save.pack()

        self.recordThread = threading.Event()
        self.playThread = threading.Event()
        self.stream = None
        self.frames = []
        self.recording = False
        self.playing = False
        self.paused = False

        self.save_transcript_copy = ""
        self.title_orig = ""

        self.last_hover_start = None
        self.last_hover_end = None

        self.clicked_timestamp = ""

        self.temp_file = NamedTemporaryFile().name

    def click(self, event):
        txt = event.widget
        keyword_begin = txt.index(f"@{event.x},{event.y} linestart")
        keyword_end = txt.index(f"@{event.x},{event.y} lineend")
        word = txt.get(keyword_begin, keyword_end)
        self.clicked_timestamp = self.textarea.tag_names(txt.index('current'))[0]
        dt = datetime.strptime(self.clicked_timestamp, "%M:%S.%f")

        if not self.playing and not self.paused:
            secs = dt.second + (60*dt.minute)
            self.play_recording(secs)
        

    def hover(self, event):
        txt = event.widget
        keyword_begin = txt.index(f"@{event.x},{event.y} linestart")
        keyword_end = txt.index(f"@{event.x},{event.y} lineend")
        
        if self.last_hover_start != keyword_begin:
            txt.tag_delete("highlight")
            self.last_hover_start = keyword_begin
            self.last_hover_end = keyword_end

            word = txt.get(keyword_begin, keyword_end)
            txt.tag_add("highlight", keyword_begin, keyword_end)
            txt.tag_config("highlight", background="firebrick")
            print(self.textarea.tag_names(txt.index('current')))
        #print(word)

    def update(self):
        try:     
            self.textarea.delete("0.0", "end")
            self.title.configure(textvariable=tk.StringVar(self, self.controller.key))
            self.title_orig = self.controller.key
            file = pickle.load(open("save.p", "rb"))
            self.add_tags([file[self.controller.key][1]])

        except Exception as e:
            print(e)

    def save(self):
        file = {}

        try:
            file = pickle.load(open("save.p", "rb"))

            # If this is a new, unsaved transcription
            if self.title_orig not in file:
                file.update({self.title_orig.rstrip(): ["".join(self.title_orig.split()) + '.wav', self.save_transcript_copy]})
                pickle.dump(file, open("save.p", "wb"))

            # If the new title is not already used, change the sound file's name and update the name in the dictionary, then save.
            if (self.title.cget('textvariable').get() not in file) and (self.title_orig != self.title.cget('textvariable').get()):
                    if os.path.exists("".join(self.title_orig.split()) + '.wav'):
                        os.rename("".join(self.title_orig.split()) + '.wav', "".join(self.title.cget('textvariable').get().split()) + '.wav')
                    file[self.title.cget('textvariable').get()] = file.pop(self.title_orig)
                    pickle.dump(file, open("save.p", "wb"))
        
        except Exception as e:
            print(e)

            
        # Error msg


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
        wf = wave.open("".join(self.title_orig.split()) + '.wav', 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        print("finished recording")

    # Upload File
    def upload_file(self):
        filePath = filedialog.askopenfilename(initialdir='/', title='Select a file', filetypes=(('wav', '*.wav'), ('mp3', '*.mp3')))
        print("FILE ", self.title_orig)
        shutil.copy(r''+filePath, "".join(self.title_orig.split()) + '.wav')

    def play_recording(self, skip_seconds=0):

        if self.playing:
            self.paused = False
            self.enable_btn(self.pause)
        else:
            self.playThread.set()
            t = threading.Thread(target=self.play_thread, args=(skip_seconds,))
            t.start()

    # Play Recording 
    def play_thread(self, skip_seconds=0):
        try:
            
            filename = ""
            # Determine if sound file is .wav or .mp3
            if os.path.exists("".join(self.title_orig.split()) + ".wav"):
                filename = "".join(self.title_orig.split()) + ".wav"
            elif os.path.exists("".join(self.title_orig.split()) + ".mp3"):
                filename = "".join(self.title_orig.split()) + ".mp3"

            print("FILENAME ", filename)
            
            try:
                wf = wave.open(filename, 'rb')
            except wave.Error:
                x,_ = librosa.load(filename, sr=16000)
                sf.write('tmp.wav', x, 16000)
                wf = wave.open('tmp.wav', 'rb')
                print('converted')

            p = pyaudio.PyAudio()
            self.playing = True

            self.enable_btn(self.pause)
            self.disable_button(self.play)

            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

            # Skip ahead in the audio
            if skip_seconds > 0:
                sample_rate = wf.getframerate()
                count = 0
                samples = int(skip_seconds * sample_rate)
                while count < samples:
                    wf.readframes(chunk)
                    count += chunk

            # Read data in chunks
            data = wf.readframes(chunk)

            # Play the sound by writing audio data to stream
            while data != b'' and self.playing:
                if not self.paused:
                    stream.write(data)
                    data = wf.readframes(chunk)
                else:
                    time.sleep(0.5)

            # Close and terminate the stream
            stream.close()
            p.terminate()
            self.playThread.clear()
            self.playing = False
            self.paused = False
            self.disable_button(self.pause)
            self.enable_btn(self.play)

            print("Play finished")
        except (FileNotFoundError):
            print('File not found')
            self.playThread.clear()

    def transcript_thread(self):
        
        # Whisper JAX API
        client = Client("https://sanchit-gandhi-whisper-jax.hf.space/")

        # Determine if the file is .wav or .mp3
        filename = ""
        if os.path.exists("".join(self.title_orig.split()) + ".wav"):
            filename = "".join(self.title_orig.split()) + ".wav"
        elif os.path.exists("".join(self.title_orig.split()) + ".mp3"):
            filename = "".join(self.title_orig.split()) + ".mp3"

        # Use Whisper JAX to transcribe the audio
        try:
            result = client.predict(
                    filename,	# str (filepath or URL to file) in 'inputs' Audio component
                    "transcribe",	# str in 'Task' Radio component
                    True,	# bool in 'Return timestamps' Checkbox component
                    api_name="/predict"
            )
        except FileNotFoundError:
            print("File does not exist")
        
        # Save unaltered result to save into the save file
        self.save_transcript_copy = result[0]

        # Add timestamp tags and insert text into textbox
        self.add_tags(result)
        
    
    def add_tags(self, result):

        result_copy = result
        time = []
        indexes = []

        # RegEx to find all timestamp values
        test = re.finditer("\[(.*?)\]", result[0])
        for match in test:
            indexes.append(match.span())
            time.append(result[0][match.start():match.end()])
        
        # Clear Textbox
        self.textarea.delete("0.0", "end")

        # Assign tags and update textbox
        last = indexes[0][1]
        indexes.pop(0)

        for ind, x in enumerate(indexes):
            stringtxt = result_copy[0][last:x[0]]
            t = time.pop(0)
            last=x[1]
            self.textarea.insert("end", stringtxt, t[1:10])

    def transcript_audio(self):
        t = threading.Thread(target=self.transcript_thread)
        t.start()

    # Update GUI Functions
    def update_record_text(self, record):
        if record.cget('text') == 'Record':
            record.configure(text='Stop Recording')
        else:
            record.configure(text='Record')

    def disable_button(self, btn):
        btn.configure(state='disabled')

    def enable_btn(self, btn):
        btn.configure(state='enabled')

    # Reset Value Functions
    def setPaused(self):
        self.paused = not self.paused

    def reset_recording(self):
        if self.recording:
            self.recording = False
            #self.record_callback()
        else:
            self.record_callback()

class LivePage(customtkinter.CTkFrame):

    def __init__(self, parent, controller):
        customtkinter.CTkFrame.__init__(self, parent)
        self.controller = controller

        home = customtkinter.CTkButton(self, text="Back")

        home.configure(command=lambda:controller.show_frame("StartPage"))
        home.pack(side=TOP, anchor=NW, padx=10, pady=10)

        label = customtkinter.CTkLabel(self, text="Live")
        label.pack()



if __name__ == "__main__":
    app = App()
    app.mainloop()
