import tensorflow.keras as keras
import librosa
import numpy as np


MODEL_PATH = "model.h5"
NUM_SAMPLES_TO_CONSIDER = 33075 # jumlah ini merupakan total sample rate yang digunakan per segment
# (sample rate*durasi)/segment -> (22050 * 15)/10

class _Keyword_Spotting_Service:
    
    model = None
    _mappings = [
        "Anak",
        "Ayah"
    ]

    _instance = None # instance ini merupakan trik untuk membuat class sebagai single trun
    # maksud dari single turn adalah setelah kelas nanti dijalankan->kelas tidak perlu dijalankan kembali
    # contoh pada program ini: sebelum prediksi.py dijalankan maka sistem akan menjalankan semua code termasuk
    # menggunakan tensorflow, setelah sekali berjalan, sistem tidak perlu lagi menjalankan code semuanya
    # seperti sebelumnya, jika bisa langsung mengeluarkan prediksi ->kelebihan menghemat waktu running

    def predict(self, file_path):

        # extract MFCCs
        MFCCs = self.preprocess(file_path) # (# segment, # coeficient)
        # print(f"MFCCs hasil preprocssing: {MFCCs}")

        # convert 2d MFCCs array into 4d array -> (# sample that we predict, # segment, # coeficient, # channel)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        # print(f"MFCCs yang dirubah ke 4d array: {MFCCs}")
        
        # make prediction
        prediction = self.model.predict(MFCCs)
        predicted_index = np.argmax(prediction)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        
        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T


def Keyword_Spotting_Service():
    
    # ensure that we only have 1 instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":
    kss = Keyword_Spotting_Service()

    k1 = kss.predict("suaraAyah.wav")

    print(f"Predicted keyword: {k1}")