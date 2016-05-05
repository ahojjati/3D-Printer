from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import wave
import numpy as np
import scipy.signal as signal
import pdb
import wave
import struct
import pickle
import stfft
import train

def main():
    with open("pipe.model") as f:
        model = pickle.load(f)

    preds = []
    """ samplerate, samples = read("/Users/sunny/workspace/3d-printer/recordings/smartphone/1430177546499/1430177546499.wav")

    for i in range(0, len(samples), 220500):
        data, _ = train.prepare_data(samples[i:i+220500], samplerate, "00", 220500)
        preds.append(model.predict(data))
"""

    f = wave.open("/Users/sunny/workspace/3d-printer/recordings/smartphone/1430177546499/1430177546499.wav")
    #f = wave.open('/Users/sunny/workspace/3d-printer/recordings/smartphone/1461799968313/1461801907950.wav')
    f = wave.open("/Users/sunny/Desktop/1461950958243.wav")
    data_tot = []
    pdb.set_trace()
    for i in range(0, f.getnframes(), 4410000):
        print i
        waveData = f.readframes(4410000)
        data_raw = np.reshape(np.fromstring(waveData, dtype='int16'), (-1, 2))

        data = train.prepare_data((data_raw[:, 1] + data_raw[:, 0]) / 2, f.getframerate(), "00", 4410)
        data_tot.append(data[0])
        preds.extend([int(elem) for elem in model.predict(data[0])])
        #data = struct.unpack('hh', np.array_split(np.fromstring(waveData), 220500))

    pdb.set_trace()
    
    preds_smoothed = signal.medfilt(preds, kernel_size=21)
    pdb.set_trace()


if __name__ == '__main__':
    main()
