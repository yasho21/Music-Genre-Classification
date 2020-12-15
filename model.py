from python_speech_features import mfcc, logfbank
from scipy.io import wavfile
import numpy as np
from hmmlearn import hmm
from matplotlib import pyplot as plt
import itertools
import os
import time
from sklearn.utils import _joblib
from utils import getFiles


class HMM_Model(object):
    def __init__(self):
        self.model = None
        self.label = None #str

    def load(self,label:str)->bool: #load model
        try:
            self.model = _joblib.load('weights/'+label+".m")
            with open('weights/'+label+'.lb','r') as f:
                self.label = f.read()
        except Exception as e:
            print('Load model failed:{}'.format(e))

    def train(self, X:np.array, label:str, n_components:int=10, cov_type:str='diag', n_iter:int=1000) -> None:
        print('Now is trining model for {} ----->'.format(label))
        self.label = label
        self.model = hmm.GaussianHMM(
                    n_components=n_components, covariance_type=cov_type, n_iter=n_iter)
        np.seterr(all='ignore')
        self.model.fit(X) # Train here
        _joblib.dump(self.model,'weights/'+label+".m")
        with open('weights/'+label+'.lb','w') as f:
            f.write(label)

    def predict(self,X:np.array) -> float:
        score = self.model.score(X)
        return score



def get_result(X, models:list):
    scores = {}
    for m in models:
        scores[m.label] = m.predict(X)
    return max(scores, key=lambda x: scores[x])


def train(dataloader)->list:
    print('Start training, you may need more than 1.5 hour to wait for it---->')
    start_time = time.clock()
    models = []
    for cl in dataloader.class_list:
        hmm_model = None
        hmm_model = HMM_Model()
        hmm_model.train(dataloader.datas[cl] ,cl, n_components=10)
        #hmm_model.load(cl)
        models.append(hmm_model)
    print('All models are trained---> cost {} s'.format(time.clock()-start_time))
    return models


def evaluate(eval_root:list,eval_class_list:list, models:list):
    from sklearn.metrics import confusion_matrix
    from dataloader import single_loader
    real = []
    pred = []
    for idx,cl in enumerate(eval_class_list):
        wav_fl = getFiles(os.path.join(eval_root,cl))
        for wav_f in wav_fl:
            real.append(idx)
            print('Evaluating for {}'.format(wav_f))
            X = single_loader(wav_f, is_print_info=False, is_vision=False)
            pred.append(eval_class_list.index(get_result(X, models)))
    cm = confusion_matrix(real,pred)
    return cm, real, pred
        



def load_models(class_list:list):
    models = []
    for cl in class_list:
        hmm_model = None
        hmm_model = HMM_Model()
        #hmm_model.train(dataloader.datas[cl] ,cl, n_components=10)
        hmm_model.load(cl)
        models.append(hmm_model)
    return models
