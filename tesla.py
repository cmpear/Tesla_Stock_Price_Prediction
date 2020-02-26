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

####################################################################################################################################################
####################################################################################################################################################
# FUNCTIONS #
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
# batch_params: creators parameters for grouping data into scraps, training,
#               testing, and future batches
####################################################################################################################################################
def batch_params(dataset, batch_size, timesteps, test_percent):
    # for dividing by: scraps, train, test, future
    d =  ( { 'batch_size': batch_size, 
    'timesteps' : timesteps, 
    'test_percent' : test_percent } )

    d['future_end'] = len(dataset)
    d['future_start'] = d['future_end'] - batch_size
    d['test_end'] = d['future_start']

    d['batches'] = d['test_end'] // batch_size

    d['train_start'] = d['test_end'] - batch_size * d['batches']
    d['train_end'] = int(((d['batches'] * (1 - test_percent) )//1) * batch_size + d['train_start'])
    d['test_start'] = d['train_end']

    d['scrap_end'] = d['train_start']

    return(d)
####################################################################################################################################################
# ensure_dir_exists: checks a target director to make a file, creates it
#                    if it does not exist
####################################################################################################################################################

def ensure_dir_exists(dir):
    if not os.path.exists(dir):
            os.makedirs(dir)

####################################################################################################################################################
# streamlined model: reshapes data for use with a stateful LSTM model,
#                    builds said model, and saves said model
#                    originaly just a long chunk of code that was generalized into a function
####################################################################################################################################################

# NOTE this 'function' was not originally intended to be such.  Turned into a function to make testing other features easier
def streamlined_model(data, bPar, epochs, df_name):
    # Feature Scaling
    #scale between 0 and 1. the weights are esier to find.
    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler(feature_range = (0, 1))
    df_scaled = sc.fit_transform(np.float64(data) )


#    print('head of normalized training data')
#    print(df_scaled[0:15])
    # numpy arrays don't have heads, only pandas

    X_all = []
    y_all = []

    # Creating a data structure with n timesteps
    for i in range(bPar['train_start'], bPar['future_end'] ):
        X_all.append(df_scaled[i - bPar['timesteps'] : i, 0 ] )
        if (i < bPar['future_start']):
            y_all.append(df_scaled[i : i + bPar['timesteps'], 0 ] )
    # y_all looks in the future, while x looks back.  We are butting against the end of our data, thus y_all will have to stop first

    # Reshaping: need numpy, not lists
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    #X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1) )
    X_all = np.reshape(X_all, (X_all.shape[0], X_all.shape[1], 1) )
    y_all = np.reshape(y_all, (y_all.shape[0], y_all.shape[1], 1) )

    # we removed the scraps, so bPar is off by train_start / scrap_end
    X_train = X_all[ 0 : bPar['train_end'] - bPar['scrap_end'] ]
    y_train = y_all[ 0 : bPar['train_end'] - bPar['scrap_end'] ]

    # Building the LSTM
    # Importing the Keras libraries and packages
    from keras.layers import Dense
    from keras.layers import Input, LSTM
    from keras.models import Model
    import h5py

    # Initialising the LSTM Model with MAE Loss-Function
    # Using Functional API  ##### two APIs in Keras, one is sequential, other is function
    # can do much easier and complicated things with functional api`

    inputs_1_mae = Input(batch_shape=(bPar['batch_size'], bPar['timesteps'],1) )
    #each layer is the input of the next layer
    lstm_1_mae = LSTM(10, stateful=True, return_sequences=True)(inputs_1_mae)
    lstm_2_mae = LSTM(10, stateful=True, return_sequences=True)(lstm_1_mae)

    # units, essentially dimensions
    output_1_mae = Dense(units = 1)(lstm_2_mae)

    regressor_mae = Model(inputs=inputs_1_mae, outputs = output_1_mae)

    #adam is fast starting off and then gets slower and more precise
    #mae -> mean absolute error loss function
    regressor_mae.compile(optimizer='adam', loss = 'mae')
    regressor_mae.summary()

    #Statefull
    for i in range(epochs):
        print("Epoch: " + str(i))
        #run through all data but the cell, hidden state are used for the next batch.
        regressor_mae.fit(X_train, y_train, shuffle=False, epochs = 1, batch_size = bPar['batch_size'])
        #resets only the states but the weights, cell and hidden are kept.
        regressor_mae.reset_states()

    #save model and data

    import pickle
    from sklearn.externals import joblib # for saving minmaxscalar

    this_dir = os.path.dirname(os.path.realpath('__file__'))
    rel_dir = df_name + '/'
    this_dir = os.path.join(this_dir, rel_dir)

    ensure_dir_exists(this_dir)

    model_name = df_name + '_with_mae_32_ts.h5'
    target_dir = os.path.join(this_dir, model_name)
    regressor_mae.save(filepath=target_dir)

    target_dir = os.path.join(this_dir, 'X_all.csv')
    pickle.dump(X_all, open (target_dir, 'wb') )

    target_dir = os.path.join(this_dir, 'y_all.csv')    
    pickle.dump(y_all, open (target_dir, 'wb') )

    target_dir = os.path.join(this_dir, 'bPar.json')
    pickle.dump(bPar, open (target_dir, 'wb') )

    target_dir = os.path.join(this_dir, 'sc.save')
    joblib.dump(sc, target_dir)
####################################################################################################################################################
####################################################################################################################################################
# MAIN #
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
# RETRIEVE DATA #
####################################################################################################################################################

this_dir = os.path.dirname(os.path.realpath('__file__') )
target_dir = os.path.join(this_dir, 'data/TSLA.csv')
TSLA = pd.read_csv(target_dir, delimiter=',') 
####################################################################################################################################################
# FEATURE CREATION #
####################################################################################################################################################
TSLA.columns = ['date','open','high','low','close','adj_close','volume']

TSLA['date'] = pd.to_datetime(TSLA.date)
TSLA['daily_change'] = TSLA['close'] - TSLA['open']
TSLA['days_after_ipo'] = TSLA.date - datetime.datetime(2010, 1, 29)
TSLA.days_after_ipo = TSLA.days_after_ipo // np.timedelta64(1, 'D')
TSLA.days_after_ipo = TSLA.days_after_ipo.astype('int')
####################################################################################################################################################
# SAVE CLEANED DATA #
####################################################################################################################################################
this_dir = os.path.dirname(os.path.realpath('__file__') )
target_dir = os.path.join(this_dir, 'TSLA/')
ensure_dir_exists(target_dir)
target_dir = os.path.join(target_dir, 'TSLA.csv')
TSLA.to_csv(target_dir)

# decided to have batch_size be a multiple of timesteps
epochs = 120

bPar = batch_params (TSLA, batch_size = 64, timesteps = 32, test_percent = 0.1)
####################################################################################################################################################
# CREATE AND SAVE MODELS #
####################################################################################################################################################
streamlined_model (TSLA.iloc[:,4:5].values, bPar, epochs, 'TSLA')

streamlined_model (TSLA.iloc[:,8:9].values, bPar, epochs, 'TSLA_daily_change')

# using this method to get the right shape only works if we are not referencing the last column.  Fix in next version.
#print("GOT TO THE END!!!")