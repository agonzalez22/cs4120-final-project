""" Utils functions for final project
Alina G and Jesse E
Text
 """ 
import numpy as np 
import librosa
import speech_recognition as sr

def load_wavs(pathname: str, files:list, get_labels=True) -> list: 
    """ takes a list of paths leading to .wav files and reads them. Also gets 
        the labels if needed from the titles. Pathname is the path the files
        are in, assuming there is a folder. 
    """
    values = []
    for file in files: 
        if file.endswith('.wav'):
            sig_rate = librosa.load(pathname + '/' + file, sr=None)
            code_lst = file.split('-') # splits the file_name up into fragments 
            emotion_label = int(code_lst[2]) 
            values.append((sig_rate, emotion_label)) 
    return values 

def get_audio_text(pathname:str, files:list) -> list: 
    """ using pathname and list of files, attempts to convert audio to text
    *** ASSIGNS 0 FOR AN UNRECOGNIZED AUDIO FILE***
    """
    texts = []
    # intialize the recognizer...
    r = sr.Recognizer()
    for file in files: 
        with sr.AudioFile(pathname + '/' + file) as source:
            audio = r.record(source)
            try: 
                s = r.recognize_google(audio)
            except: 
                s = 0 # in case our recognizer fails, we will just assign 0...
            texts.append(s)
    return texts

def seperate_tups(lst:list) -> tuple: 
    """ takes in a list of tuple parings and returns a tuple of 2 lists """ 
    x, y = [], []
    for pair in lst: 
        x.append(pair[0])
        y.append(pair[1])
    return x, y
