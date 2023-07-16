import os
import tkinter as tk
from tkinter import messagebox, filedialog
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import speech_recognition as sr
from tensorflow.keras.models import load_model

def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def feature_extractors(data):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data).T, axis=0)
    result = np.hstack((result, mel))

    return result

def get_features(path):
    y, sr = librosa.load(path, duration=2.5, offset=0.6)
    features = feature_extractors(y)
    reshaped_features = np.array(features).reshape(1, -1)
    return reshaped_features

def speech_to_text(audio_file, recognizer):
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language='tr-TR')
        return text
    except sr.UnknownValueError:
        return "Ses anlaşılamadı. Lütfen tekrar deneyin."
    except sr.RequestError as e:
        return f"Hata oluştu: {e}. Lütfen hatayı kontrol edin."

def convert_to_waveform(recording, fs):
    normalized_recording = recording / np.max(np.abs(recording))
    duration = len(recording) / fs
    time = np.linspace(0, duration, len(recording))

    plt.figure()
    plt.plot(time, normalized_recording)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.show()

class SoundRecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sound Recorder")
        self.root.geometry("800x600")
        self.root.minsize(300, 300)
        self.root.configure(background="#AAB6FB")

        self.welcome_label = tk.Label(root, text="SOUND RECOGNITION", font=("Georgia", 36, "bold"), pady=10, background="#AAB6FB")
        self.welcome_label.pack()

        self.record_button = tk.Button(root, text="Ses Kaydet", command=self.record_sound, font=("Arial", 15), bg="dark grey", fg="white")
        self.record_button.pack(ipady=10, ipadx=10, padx=30, pady=20)

        self.predict_button = tk.Button(root, text="Tahmin Et", command=self.make_prediction, font=("Arial", 15), state=tk.DISABLED, bg="dark grey", fg="white")
        self.predict_button.pack(ipadx=10, ipady=10, padx=30, pady=20)

    def record_sound(self):
        global record_count  # Erişim için global değişken tanımlanması

        fs = 44100
        duration = 5
        recording = sd.rec(int(fs * duration), samplerate=fs, channels=1)
        sd.wait()

        save_path = "Kayıtlar"
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"Kayıt{record_count}.wav")
        sf.write(save_file, recording, fs)

        messagebox.showinfo("Recording Complete", f"Recording saved as {save_file}")
        record_count += 1  # Kaydedilen dosya sayısını artırma
        self.predict_button.config(state=tk.NORMAL)

        convert_to_waveform(recording, fs)

    def make_prediction(self):
        model = load_model("kisimodel.h5")
        recognizer = sr.Recognizer()

        file_path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        if not file_path:
            return

        feature = get_features(file_path)
        prediction = model.predict(feature)
        predicted_class = np.argmax(prediction)

        recognized_text = speech_to_text(file_path, recognizer)

        accuracy = prediction[0][predicted_class]
        fm = 2 * (prediction.max() / prediction.sum())  # Calculate FM value

        messagebox.showinfo("Tahmin", f"Konuşan Kişi Tahmini: {predicted_class}\nSöylenen Kelimeler: {recognized_text}\nAccuracy: {accuracy}\nFM: {fm}")


if __name__ == "__main__":
    record_count = 1
    root = tk.Tk()
    app = SoundRecorderGUI(root)
    root.mainloop()
