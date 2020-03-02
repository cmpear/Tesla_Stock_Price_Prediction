from io import StringIO
import requests
import json
import pandas as pd
import types
from botocore.client import Config
import datetime
import numpy as np
import os
from keras.preprocessing import sequence
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.layers import Input, LSTM
from keras.models import Model, model_from_json
import h5py
import pickle
from sklearn.externals import joblib # for saving minmaxscalar


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
            os.makedirs(dir)

def load_data(stock_name, model_name, load_dataset = False):
    this_dir = os.path.dirname(os.path.realpath('__file__') )
    this_dir = os.path.join(this_dir, stock_name)

    this_dir = os.path.join(this_dir, model_name)
    prefix = stock_name + '_' + model_name + '_'

    # LOAD MODEL    
    f_name = stock_name + '_' + model_name + '_' + 'model.json'
    target_dir = os.path.join(this_dir, f_name)
    with open (target_dir, 'r') as json_file:
        model = json_file.read()
    model = model_from_json(model)
    f_name = stock_name + '_' + model_name + '_weights.h5'
    target_dir = os.path.join(this_dir, f_name)
    model.load_weights(target_dir)
    # LOAD X_all and y_all
    f_name = prefix + 'X_all.npy'
    target_dir = os.path.join(this_dir, f_name)
    X_all = np.load(target_dir)

    f_name = prefix + 'y_all.npy'
    target_dir = os.path.join(this_dir, f_name)
    y_all = np.load(target_dir)
    # LOAD bPar and sc
    f_name = prefix + 'bPar.pickle'
    target_dir = os.path.join(this_dir, f_name)
    with open (target_dir, 'rb') as handle:
        bPar = pickle.load(handle)

    f_name = prefix + 'sc.save'
    target_dir = os.path.join(this_dir, f_name)
    sc = joblib.load(target_dir)

    # GET DATASET
    if (load_dataset):
        f_name = stock_name + '.npy'
        target_dir = os.path.join(this_dir, f_name)
        dataset = pd.read_csv(target_dir, delimiter=',')
        return (model, X_all, y_all, bPar, sc, dataset)
    return(model, X_all, y_all, bPar, sc)

def save_data(stock_name, model_name, model, X_all, y_all, bPar, sc = None, scaled = None, dataset = None):
    this_dir = os.path.dirname(os.path.realpath('__file__') )

    this_dir = os.path.join(this_dir, stock_name)
    ensure_dir_exists(this_dir)
    if (dataset != None):
        f_name = stock_name + '.csv' 
        target_dir = os.path.join(this_dir, f_name)
        dataset.to_csv(target_dir, index = False)

    this_dir = os.path.join(this_dir, model_name)
    ensure_dir_exists(this_dir)

    f_name = stock_name + '_' + model_name + '_model.json'
    target_dir = os.path.join(this_dir, f_name)
    model_json = model.to_json()
    with open (target_dir, 'w') as json_file:
        json_file.write(model_json)
    f_name = stock_name + '_' + model_name + '_weights.h5'
    target_dir = os.path.join(this_dir, f_name)
    model.save_weights(target_dir)

    prefix = stock_name + '_' + model_name + '_'
    f_name = prefix + 'X_all.npy'
    target_dir = os.path.join(this_dir, f_name)
    np.save(target_dir, X_all)

    f_name = prefix + 'y_all.npy'
    target_dir = os.path.join(this_dir, f_name)
    np.save(f_name, y_all)

    f_name = prefix + 'bPar.json'
    target_dir = os.path.join(this_dir, f_name)
    with open (target_dir, 'w') as json_file:
        json.dump(bPar, target_dir)
    

    if (sc != None):
        f_name = prefix + 'sc.save'
        target_dir = os.path.join(this_dir, f_name)
        joblib.dump(sc, target_dir)
    
    if (scaled != None):
        f_name = prefix + 'scaled.npy'
        target_dir = os.path.join(this_dir, f_name)
        np.save(target_dir, scaled)

model, X_all, y_all, bPar, sc = load_data('TSLA','closing')
print(X_all)