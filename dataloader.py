from utils import getFiles
from sklearn.utils import _joblib
from python_speech_features import mfcc, logfbank
from scipy.io import wavfile
import numpy as np
from hmmlearn import hmm
from matplotlib import pyplot as plt
import os
import time

class Dataloader(object):
    def __init__(self, class_list: list, root: str = 'fakes', format_filiter: str = '.wav'):
        self.datas = {}
        self.class_list = class_list
        for cl in class_list:
            cl_data = []
            clp = os.path.join(root, cl)
            data_fl = getFiles(clp)
            X = np.array([])
            for f in data_fl:
                if format_filiter in f:
                    print('Loading {} ---> for {}'.format(f, cl))
                    sampling_freq, audio = wavfile.read(f)# Extract MFCC features
                    # Append to the variable X
                    mfcc_features = mfcc(audio, sampling_freq)
                    if len(X) == 0:
                        X = mfcc_features
                    else:
                        X = np.append(X, mfcc_features, axis=0)
            self.datas[cl] = X
        print("All data load finished--->")


def single_loader(wav_path:str, is_print_info = True, is_vision = True):
    sampling_freq, audio = wavfile.read(wav_path)
    mfcc_features = mfcc(audio, sampling_freq)
    filterbank_features = logfbank(audio, sampling_freq)
    if is_print_info:
        print('Open {}'.format(wav_path))
        print('='*30)
        print('MFCC:\nNumber of windows =', mfcc_features.shape[0])
        print('Length of each feature =', mfcc_features.shape[1])
        print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
        print('Length of each feature =', filterbank_features.shape[1])
        print('='*30)
    if is_vision:
        plt.matshow((mfcc_features.T)[:, :300])
        plt.text(150, -10, wav_path, horizontalalignment='center', fontsize=20)
        plt.show()
    return mfcc_features

