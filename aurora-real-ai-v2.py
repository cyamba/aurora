import io
import os
import sys
import wave
import zipfile
import torch
import pygame
import sounddevice as sd
import numpy as np
import time
from gtts import gTTS
from vosk import Model, KaldiRecognizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Singleton LLM Class
class SingletonLLM:
    _instance = None

    def __new__(cls, model_name="facebook/blenderbot-400M-distill", cache_dir="./model_cache"):
        if cls._instance is None:
            cls._instance = super(SingletonLLM, cls).__new__(cls)
            cls._instance._initialize(model_name, cache_dir)
        return cls._instance

    def _initialize(self, model_name, cache_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def generate_response(self, prompt, max_length=100, temperature=1.0, repetition_penalty=1.2, top_k=50, top_p=0.95):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p
            )
        generated_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return generated_response


# TTS Function
def speak_english(text, gender='female'):
    tts = gTTS(text=text, lang='en', tld='com.au' if gender == 'male' else 'com')
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    pygame.mixer.music.load(audio_buffer, 'mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue


# STT Initialization
data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')
model_zip_path = os.path.join(data_dir, 'vosk-model-small-en-us-0.15.zip')
model_path = os.path.join(data_dir, 'vosk-model-small-en-us-0.15')

if not os.path.exists(model_path):
    if os.path.exists(model_zip_path):
        print("Extracting VOSK model...")
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Model extraction complete.")
    else:
        print(f"Error: Vosk model zip file not found in {model_zip_path}. Please download and place it there.")
        sys.exit(1)

model = Model(model_path)  # Initialize the Vosk model once
recognizer = KaldiRecognizer(model, 16000)  # Use this recognizer throughout


# Recording Function
def record_audio(filename, sample_rate=16000, silence_threshold=2):
    print("Recording...")
    audio = []
    last_speech_time = time.time()
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
        while True:
            data, _ = stream.read(1024)
            audio.append(data)
            if recognizer.AcceptWaveform(data.tobytes()):  # Convert numpy array to bytes
                result = recognizer.Result().lower()
                last_speech_time = time.time()
                if 'good bye' in result:
                    return 'exit'
            # Check for silence
            if time.time() - last_speech_time > silence_threshold:
                break
    audio = np.concatenate(audio, axis=0)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    print(f"Audio saved as {filename}")


# Transcribe Function
def transcribe_audio(filename):
    wf = wave.open(filename, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000, 44100]:
        print("Audio file must be WAV format mono PCM.")
        sys.exit(1)
    recognizer = KaldiRecognizer(model, wf.getframerate())
    final_transcription = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            final_transcription += eval(result)['text'] + " "
    final_transcription += eval(recognizer.FinalResult())['text']
    return final_transcription


# Main Loop
if __name__ == "__main__":
    llm = SingletonLLM()
    pygame.mixer.init()  # Initialize pygame once

    while True:
        record_filename = "microphone_recording.wav"
        recording_status = record_audio(record_filename)
        if recording_status == 'exit':
            speak_english("Good bye")
            break

        transcribed_text = transcribe_audio(record_filename).lower()
        # Generate a response if not exiting
        response = llm.generate_response(transcribed_text)
        # Ensure the response does not contain the prompt
        response_without_prompt = response.replace(transcribed_text, '').strip()
        speak_english(response_without_prompt)

        print("Say something. You will get a response after 2 seconds of silence, or say 'good bye' to exit.")
