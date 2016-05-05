import csv
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks_cwt

from scipy.io.wavfile import read


from collections import deque
from math import pi, log
import pylab
import pdb
from scipy import fft, ifft
from scipy import signal
from scipy.optimize import curve_fit

import wave  
import time  
import sys
from scipy.fftpack import rfft
from sklearn.preprocessing import normalize

import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

from scipy.interpolate import interp1d

slice_length = 4410
max_slices = 10

def stfft(samples, samplerate, segment_size):
    #s = stfft.stft(samples, 2**10)
    ims = np.array([rfft(elem, samplerate) for elem in 
                np.array_split(samples, int(len(samples) / segment_size) + 1)],
                dtype="float64")
    #sshow, freq = stfft.logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(ims)/10e-6)

    ims = ims[np.isfinite(ims).all(1)]
    sample = normalize(ims, axis=1)
    #sample_chunked = [np.max(elem, axis=0) for elem in np.array_split(sample, int(len(sample) / segment_size) + 1)]
    return sample

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def maxfilter(samples, filter_size):
    vals = deque()
    max_samples = np.empty(len(samples))
    for i in xrange(len(samples)):
        vals.append(samples[i])
        if len(vals) > filter_size:
            vals.popleft()
        max_samples[i] = max(vals)
    return max_samples

def peakfind(peak_samples, data_samples, time, samplerate):
    print "Finding Peaks..."
    samples = maxfilter(peak_samples, 100)
    indexes = signal.find_peaks_cwt(samples, np.arange(60, 200, 20), min_snr=1.5)
    
    print "%d Peaks detected." % len(indexes)
    #adjusting for times
    basetime = int(time[0])

    for i in xrange(len(time)):
        time[i] = int(time[i]) - basetime

    translated_indexes = [int(time[idx] / 1000.0 * samplerate) for idx in indexes]
    filteredidx = peakfilter(translated_indexes, data_samples, samplerate)
    return filteredidx

def peakfilter(peak_times, data_samples, samplerate):
    print "Filtering peaks..."
    if len(peak_times) < 1:
        return []
    prev_fft = maxfilter(rfft(data_samples[peak_times[0] 
                            : peak_times[1]], samplerate), 100)
    filtered_peaks = []
    ffts = []
    for i in xrange(len(peak_times) - 1):
        cur_fft = maxfilter(rfft(data_samples[peak_times[i] 
                            : peak_times[i+1]], samplerate), 100)
        ffts.append(np.sum(np.abs(cur_fft - prev_fft) ** 2))
        if np.sum(np.abs(cur_fft - prev_fft) ** 2) > 5e15:
            filtered_peaks.append(peak_times[i])
        prev_fft = cur_fft
    cur_fft = rfft(data_samples[peak_times[i] :], samplerate)
    if np.sum(np.abs(cur_fft - prev_fft) ** 2) > 5e15:
        filtered_peaks.append(peak_times[i])

    print "Removed %d peaks." % (len(peak_times) - len(filtered_peaks))
    return filtered_peaks

def splice(data, splittimes):
    return_array = []
    prev_time = 0
    for i in xrange(len(splittimes)):
        if splittimes[i] - prev_time > max_slices * slice_length:
            return_array.append(data[prev_time:prev_time+slice_length*max_slices])
        elif splittimes[i] - prev_time > slice_length:
            return_array.append(data[prev_time : splittimes[i]])
        prev_time = splittimes[i]

    if splittimes[i] - prev_time > max_slices * slice_length:
        return_array.append(data[prev_time:prev_time+slice_length*max_slices])
    elif splittimes[i] - prev_time > slice_length:
        return_array.append(data[prev_time:])
    return return_array

def preprocess(wavfile, magfile, accelfile, truthfile):
    with open(accelfile, 'r') as f:
        csv_f = csv.reader(f)
        dataraw = np.array(list(csv_f), dtype='float64')
        data = normalize(dataraw, axis=0)
        accel_data = data[:,0] ** 2 + data[:,1] ** 2 + data[:,2] ** 2
        accel_time = dataraw[:,3]
    
    samplerate, wav_samples = read(wavfile)
    wav_data = (wav_samples[:,0] + wav_samples[:,1])/2
    peaktimes = peakfind(accel_data, wav_data, accel_time, samplerate)
    with open(magfile, 'rb') as f:
        reader = csv.reader(f)
        mag_data = np.array(list(reader), dtype='float64')
    y = normalize(mag_data[:, 0] ** 2 + mag_data[:, 1] ** 2 + mag_data[:, 2] ** 2)
    x = mag_data[:, 3] - min(mag_data[:,3])
    fun = interp1d(x, y)
    xnew = np.arange(min(x), max(x), 1000/44100.0)
    

    wav_array = splice(wav_data, peaktimes); # splice wav file into array based on filtered_times
    mag_array = splice(fun(xnew)[0], peaktimes); # splice mag file into array based on filtered_times

    with open(truthfile, 'r') as f:
        csv_f = csv.reader(f)
        truth_array = np.array(list(csv_f), dtype='float64')

    prep_array = np.empty((0,88200))
    truth_label = []
    print "Preparing data..."
    for i in xrange(len(wav_array)):
        wav_prep = stfft(wav_array[i], 44100, 4410)
        mag_prep = stfft(mag_array[i], 44100, 4410)
        prep_array = np.concatenate((prep_array, (np.concatenate((mag_prep,wav_prep), axis=1))))
        truth_label.extend([truth_array[i][0]] * wav_prep.shape[0])

    return prep_array, np.array(truth_label)

def main():
    #preprocess(wavfile, magfile, accelfile, trutharray)

    par_dir = 'recordings/smartphone/'
    rec_num = '1430177546499'
    wavfile = par_dir + rec_num + '/' + rec_num + '.wav'
    magfile = par_dir + rec_num + '/' + rec_num + 'Mag.csv'
    accelfile = par_dir + rec_num + '/' + rec_num + 'Accel.csv'
    truthfile = par_dir + rec_num + '/' + rec_num + 'Truth.csv'

    data, truth = preprocess(wavfile, magfile, accelfile, truthfile)

if __name__ == '__main__':
    main()
