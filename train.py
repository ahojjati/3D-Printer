from scipy.fftpack import rfft
from scipy.io.wavfile import read
import os
import fnmatch
from sklearn import svm, decomposition, linear_model
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
import numpy as np
import pdb
import pickle
import preprocessing
from numpy import amax
import matplotlib.pyplot as plt
new_model = True
def mlformat(dir, segment_size):
    # read in audio files, take the fft, and return the normalized fft plus
    # the class

    mlsamples = []
    mlclasses = []

    flst = sorted(os.listdir(dir))      # list the available audio files

    for angle in [
            '00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
            '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
            '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
            '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47',
            '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59',
            '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71',
            '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83',
            '84', '85', '86', '87', '88', '89', '90']:
        for filename in flst:
            if fnmatch.fnmatch(filename, '%sdeg*' % angle):
                samplerate, samples = read(dir + '%s' % filename)
                sample, label = preprocessing.prepare_data(samples, samplerate, angle, segment_size)    # take the fft
                mlsamples.extend(sample[0:1])
                mlclasses.extend(label[0:1])         # the class of each fft

    return mlsamples, mlclasses
def run_prelabelled():
    traindir = '/Users/sunny/Desktop/train/'
    testdir = '/Users/sunny/Desktop/test/'

    trnsam, trnclass = mlformat(traindir, 4410)        # format the training data

    tstsam, tstclass = mlformat(testdir, 4410)         # format the test data


    print "Begin Predicting"
    if new_model:
        pipe = linear_model.LogisticRegression(solver='lbfgs', verbose=1)
        #logreg = svm.SVC(kernel='linear', C=1)
        #pca = decomposition.PCA()
        #pipe = Pipeline(steps=[('pca', pca), ('logistic', logreg)])
    else:
        with open("pipe.model", "rb") as f:
            pipe = pickle.load(f)

    pdb.set_trace()
    pipe.fit(trnsam, trnclass)

    f = open("pipe.model", "wb")
    pickle.dump(pipe.sparsify(), f)
    prd1 = pipe.predict(tstsam)
    print "Log Predictions:"
    print prd1


    print "Training Score:"
    print pipe.score(trnsam, trnclass)
    print "Test Score:"
    print pipe.score(tstsam, tstclass)

def main():

    par_dir = 'recordings/smartphone/'
    rec_num = '1430177546499'
    wavfile = par_dir + rec_num + '/' + rec_num + '.wav'
    magfile = par_dir + rec_num + '/' + rec_num + 'Mag.csv'
    accelfile = par_dir + rec_num + '/' + rec_num + 'Accel.csv'
    truthfile = par_dir + rec_num + '/' + rec_num + 'Truth.csv'

    data, truth = preprocessing.preprocess(wavfile, magfile, accelfile, truthfile)
    print "Begin Training..."
    if new_model:
        pipe = linear_model.LogisticRegression(solver='lbfgs', verbose=1)
    else:
        with open("pipe.model", "rb") as f:
            pipe = pickle.load(f)

    pipe.fit(data, truth)

    f = open("pipe.model", "wb")
    pickle.dump(pipe.sparsify(), f)

    print "Training Score:"
    print pipe.score(data, truth)

if __name__ == '__main__':
    main()
