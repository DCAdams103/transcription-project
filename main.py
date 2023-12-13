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
import sys
from deep_translator import GoogleTranslator

# Fixes blurry text
windll.shcore.SetProcessDpiAwareness(1)

# Set chunk size to 1024 samples per data frame
chunk = 1024
sample_format = pyaudio.paInt16 # 16 bits per sample
channels = 1
fs = 44100
seconds = 3
thread_end = False


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Voice Recognition")
        self.geometry("800x600")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.resizable(False, False)
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

    def on_closing(self):
        global thread_end
        thread_end = True
        self.destroy()

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
        
        if not os.path.exists("save.p"):
            db = {}
            pickle.dump(db, open("save.p", "wb"))
        
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
        
class Loading(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Loading...")
        self.geometry("400x150")
        self.protocol("WM_DELETE_WINDOW", None)
        self.resizable(False, False)
        
        self.label = customtkinter.CTkLabel(self, text="", font=('Arial', 25))
        self.label.pack(padx=20, pady=20)

    def set_message(self, message):
        self.label.configure(text=message)

class TranscribePage(customtkinter.CTkFrame):

    def __init__(self, parent, controller):
        customtkinter.CTkFrame.__init__(self, parent)
        self.controller = controller
        #self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

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

        self.textarea = customtkinter.CTkTextbox(self, width=600, corner_radius=10)
        
        self.title.configure(font=('Arial', 25), width=300, height=10)
        self.title.grid(column=1, row=1)

        home.configure(command=lambda:controller.show_frame("StartPage"))
        home.grid(column=1, row=1, padx=10, pady=10, sticky="w")

        record.configure(command=lambda: [self.record_callback(), self.update_record_text(record)])
        record.grid(column=1, row=2, pady=5)

        upload.configure(command=lambda: [self.upload_file()])
        upload.grid(column=1, row=3)
        
        self.play.configure(command=lambda: [self.play_recording(0)])
        self.play.grid(column=1, row=4)
        
        self.pause.configure(command=lambda: [self.setPaused(), self.disable_button(self.pause), self.enable_btn(self.play)])
        self.disable_button(self.pause)
        self.pause.grid(column=1, row=5)

        transcribe.configure(command=lambda: [self.transcript_audio()])
        transcribe.grid(column=1, row=6, pady=5)
        
        self.translate_menu = customtkinter.CTkOptionMenu(self, values=["English", "Spanish", "French", "German"])
        self.translate_menu.grid(column=1, row=3, padx=(150, 0), sticky="w")

        translate = customtkinter.CTkButton(self, text="Translate Text")
        translate.configure(command=lambda: [self.translate()])
        translate.grid(column=1, row=4, pady=5, padx=(150, 0), sticky="w")

        self.textarea.grid(column=1,row=7)

        export = customtkinter.CTkButton(self, text="Export")
        export_tooltip = CTkToolTip(export, delay=0.5, message="Export your transcription to a *.txt file")
        export.configure(command=lambda: [self.export()])
        export.grid(column=1, row=8, pady=5)

        save = customtkinter.CTkButton(self, text="Save")
        save_tooltop = CTkToolTip(save, delay=0.5, message="Save your transcription")
        save.configure(command=lambda: [self.save()])
        save.grid(column=1, row=9)

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

        self.toplevel_window = None

        self.temp_file = NamedTemporaryFile().name

    def export(self):
        save_file = tk.filedialog.asksaveasfilename(initialdir='/', title="Select Directory and Filename", defaultextension='.txt', filetypes=[(".txt",".txt")])
        f = open(save_file, "w")
        f.write(self.textarea.get("0.0", "end"))
        f.close()

    def click(self, event):
        
        try:
            txt = event.widget
            keyword_begin = txt.index(f"@{event.x},{event.y} linestart")
            keyword_end = txt.index(f"@{event.x},{event.y} lineend")
            word = txt.get(keyword_begin, keyword_end)

            self.clicked_timestamp = self.textarea.tag_names(txt.index('current'))[0]
            dt = datetime.strptime(self.clicked_timestamp, "%M:%S.%f")

            if not self.playing and not self.paused:
                secs = dt.second + (60*dt.minute)
                self.play_recording(secs)
        except Exception as e:
            print(e)
      
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

            if self.save_transcript_copy == "":
                self.save_transcript_copy = self.textarea.get("0.0", "end")
                return
            
            # If this is a new, unsaved transcription
            if self.title_orig not in file:
                file.update({self.title_orig.rstrip(): ["".join(self.title_orig.split()) + '.wav', self.save_transcript_copy]})
                pickle.dump(file, open("save.p", "wb"))
                return

            # If the new title is not already used, change the sound file's name and update the name in the dictionary, then save.
            if (self.title.cget('textvariable').get() not in file) and (self.title_orig != self.title.cget('textvariable').get()):
                    if os.path.exists("".join(self.title_orig.strip()) + '.wav'):
                        os.rename("".join(self.title_orig.strip()) + '.wav', "".join(self.title.cget('textvariable').get().split()) + '.wav')
                    file[self.title.cget('textvariable').get().rstrip()] = file.pop(self.title_orig)
                    pickle.dump(file, open("save.p", "wb"))
                    return

            if (self.title.cget('textvariable').get() in file):
                file.update({self.title.cget('textvariable').get().rstrip(): ["".join(self.title_orig.split()) + '.wav', self.save_transcript_copy]})
                pickle.dump(file, open("save.p", "wb"))
                return
        
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

    def play_recording(self, skip_seconds: int):

        if self.playing:
            self.paused = False
            self.enable_btn(self.pause)
        else:
            self.playThread.set()
            t = threading.Thread(target=self.play_thread, args=(skip_seconds,))
            t.start()

    # Play Recording 
    def play_thread(self, skip_seconds: int):
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
                # If the wave file is in the incorrect format, create a copy and use that instead
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
            while data != b'' and self.playing and not thread_end:
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
        
        self.create_loading_window("Please wait while\nwe transcribe your audio.....\nThis may take a while")

        # Whisper JAX API
        client = Client("https://sanchit-gandhi-whisper-jax.hf.space/")

        # Determine if the file is .wav or .mp3
        filename = ""
        if os.path.exists("".join(self.title_orig.split()) + ".wav"):
            filename = "".join(self.title_orig.split()) + ".wav"
        elif os.path.exists("".join(self.title_orig.split()) + ".mp3"):
            filename = "".join(self.title_orig.split()) + ".mp3"

        print(filename)

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

        self.destroy_loading_window()
        
        # Save unaltered result to save into the save file
        self.save_transcript_copy = result[0]

        print(result)

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
        if(time and indexes):
            self.textarea.bind("<Motion>", self.hover)
            self.textarea.bind("<Button-1>", self.click)
            self.textarea.delete("0.0", "end")

            # Assign tags and update textbox
            last = indexes[0][1]
            indexes.pop(0)

            # If there is only one timestamp
            if(len(indexes) == 0):
                stringtxt = result_copy[0][last:len(result_copy[0])]
                t = time.pop(0)
                self.textarea.insert("end", stringtxt, t[1:10])
                return

            for ind, x in enumerate(indexes):
                stringtxt = result_copy[0][last:x[0]]
                t = time.pop(0)
                last=x[1]
                self.textarea.insert("end", stringtxt, t[1:10])
        else: # If there are no timestamps (such as translations)
            self.textarea.insert("0.0", result[0])

    def transcript_audio(self):
        t = threading.Thread(target=self.transcript_thread)
        t.start()

    def translate(self):
        t = threading.Thread(target=self.translate_thread)
        t.start()

    def translate_thread(self):

        self.create_loading_window("Please wait while\nwe translate your text...")
        
        translator = GoogleTranslator(source='auto', target=self.translate_menu.get().lower()).translate(self.textarea.get("0.0", "end"))
        
        self.save_transcript_copy = translator
        
        self.textarea.delete("0.0", "end")

        self.textarea.insert("0.0", translator)
        self.textarea.unbind("<Motion>")
        self.textarea.unbind("<Button-1>")
        for tag in self.textarea.tag_names():
            self.textarea.tag_delete(tag)

        self.destroy_loading_window()

    def create_loading_window(self, message):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = Loading(self)
            self.toplevel_window.grab_set()
            self.toplevel_window.set_message(message)
        else:
            self.toplevel_window.focus()

    def destroy_loading_window(self):
        if self.toplevel_window is not None:
            self.toplevel_window.destroy()

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
