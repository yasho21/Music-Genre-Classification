from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import os
from utils import getFiles, plot_confusion_matrix
from model import HMM_Model, get_result, train, load_models, evaluate
from dataloader import Dataloader,single_loader
import numpy as np


genre_list = ['blues', 'classical', 'jazz', 'country',
              'pop', 'rock', 'metal', 'disco', 'hiphop', 'reggae']


# dl = Dataloader(genre_list, root='genres')
# train(dl)


models = load_models(genre_list)

# Evaluate method 1:
cm, real, pred = evaluate('genres_small', genre_list, models)
plot_confusion_matrix(cm, genre_list, True) # get plot
print(classification_report(real, pred, target_names=genre_list)) # get report


# Evaluate method 2:
fl = getFiles('genres_small')
for f in fl:
    X = single_loader(f, is_print_info=False, is_vision= False)
    print('Truth:{}, predict:{}'.format(f, get_result(X,models)))


# Single test:
X = single_loader('blues.00000.wav')
result = get_result(X, models)
print(result)