####################################################################################################################################################
####################################################################################################################################################
# tesla2.py
####################################################################################################################################################
# This is the second part of a Tesla stock price analysis program.  The first wrangled the data and fit two machine learning models to it.  As fitting
# those models leads to long runtimes, the program was split in two.
# This second part creates visualizations for the machine learning models and data in general.  It also fits and plots a regression model.
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
# IMPORTS
####################################################################################################################################################
from io import StringIO
import requests
import json
import pandas as pd
import types
from botocore.client import Config
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
from keras.layers import Dense
from keras.layers import Input, LSTM
from keras.models import Model, model_from_json
import pickle
import os
import h5py
from sklearn.externals import joblib # for loading the minmaxscalar
####################################################################################################################################################
####################################################################################################################################################
# FUNCTIONS #
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
# LOAD DATA: gathers up key variables from tesla.py and returns them
#            not a user-friendly function
####################################################################################################################################################
def load_data(stock_name, model_name, load_dataset = False):
    # could I just pickle everything?
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

####################################################################################################################################################
# ensure_dir_exists: checks a target director to make a file, creates it
#                    if it does not exist
####################################################################################################################################################
def ensure_dir_exists(dir):
    if not os.path.exists(dir):
            os.makedirs(dir)
####################################################################################################################################################
# data_division_plot: bar plot of how data is being divided between
#                     scraps, train, test and future
####################################################################################################################################################
def data_division_plot(bPar, target_dir, title = 'Sample Sizes of Data'):
    # Visualising the Batches
    plt.bar ( ['scraps','train','test','future'], [bPar['scrap_end'], bPar['train_end'] - bPar['train_start'], bPar['test_end'] - bPar['test_start'], bPar['future_end'] - bPar['future_start'] ]  ,  label = 'Sample Sizes')
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Cases')
    plt.legend()

    ensure_dir_exists(target_dir)
    target_dir = os.path.join(target_dir, 'divisions.png')
    plt.savefig(target_dir)
    plt.close()
#    plt.show()
####################################################################################################################################################
# pred_res: creates and saves visuals of predictions and residuals
#           calculates R^2
####################################################################################################################################################
def pred_res(regressor_mae, bPar, sc, X_all, y_all, target_dir, pre_name, points):
    y_hat = regressor_mae.predict(X_all, batch_size = bPar['batch_size'])
    regressor_mae.reset_states()

    #reshaping
    y_hat = np.reshape(y_hat,(y_hat.shape[0], y_hat.shape[1] ) )
    y_all = np.reshape(y_all,(y_all.shape[0], y_all.shape[1] ) )
    X_all = np.reshape(X_all,(X_all.shape[0], X_all.shape[1] ) )

    #inverse transform  ## removing didn't do anything
    y_hat = sc.inverse_transform(y_hat)
    y_all = sc.inverse_transform(y_all)
    X_all = sc.inverse_transform(X_all)


    # make linear y_hat
    y_hat_linear30 = []
    y_hat_linear15 = []
    y_hat_linear1 = []
    X_plot = []

    #y_hat_linear = [0] * bPar['timesteps']
    for j in range(0, len(y_hat) ):
        X_plot = np.append(X_plot, X_all[j, 0] )
        y_hat_linear30 = np.append(y_hat_linear30, y_hat[j, bPar['timesteps'] -3])
        y_hat_linear15 = np.append(y_hat_linear15, y_hat[j, bPar['timesteps'] - 18] )
        y_hat_linear1  = np.append(y_hat_linear1,  y_hat[j, bPar['timesteps'] - 32 ] ) # doing it this way for consistency...could be simpler for some of these though.

    y_hat_linear30_test = y_hat_linear30[ bPar['test_start'] - bPar['train_start'] : bPar['test_end'] - bPar['train_start'] ]
    y_test30 = X_plot[ bPar['test_start'] - bPar['train_start'] +30 : bPar['test_end'] - bPar['train_start'] + 30]

    res30 = y_test30 - y_hat_linear30_test
    R2_30 = 1 - sum( (res30) ** 2 ) / sum( (y_test30 - np.mean( y_test30 ) ) ** 2 )

    y_hat_linear15_test = y_hat_linear15[ bPar['test_start'] - bPar['train_start'] : bPar['test_end'] - bPar['train_start'] ]
    y_test15 = X_plot[ bPar['test_start'] - bPar['train_start'] +15 : bPar['test_end'] - bPar['train_start'] + 15]
# HERE, UNFINISHED, HERE HERE HERE
    res15 = y_test15 - y_hat_linear15_test
    R2_15 = 1 - sum( (res15) ** 2 ) / sum( (y_test15 - np.mean( y_test15 ) ) ** 2 )

    y_hat_linear1_test = y_hat_linear1[ bPar['test_start'] - bPar['train_start'] : bPar['test_end'] - bPar['train_start'] ]
    y_test1 = X_plot[ bPar['test_start'] - bPar['train_start'] +1: bPar['test_end'] - bPar['train_start'] +1 ]

    res1 = y_test1 - y_hat_linear1_test
    R2_1 = 1 - sum( (res1) ** 2 ) / sum( (y_test1 - np.mean(y_test1 ) ) ** 2 )
    # # this would work better, be more accurate, if we were predicting changes.
    print("R^2 for 30-day prediction")
    print(R2_30)
    print("R^2 for 15-day prediction")
    print(R2_15)
    print("R^2 for 1-day prediction")
    print(R2_1)

    # # Visualising the results
    if (points):
#        plt.scatter( range(15, 15+len(X_plot)), y_hat_linear15[0 : ].astype(float), color = 'blue', alpha = 0.1, label = '15-Day Predicted Change')
        plt.scatter( range(1, 1+len(X_plot)), np.cbrt(y_hat_linear1[0 : ].astype(float)), color = 'green', alpha = 0.1, label = '1-Day Predicted Change')
        plt.scatter( range(0, len(X_plot)), np.cbrt(X_plot), color = 'red', alpha = 0.1, label = 'Real Change')
        plt.title('Predicted vs Real Daily Price Change')
        plt.xlabel('Market Days After IPO')
        plt.ylabel('Closing Price Change (cube root)')
        plt.axvline(x = bPar['train_end'] - bPar['train_start'])
        plt.axvline(x = bPar['test_end'] - bPar['train_start'])

        plt.text(x = (bPar['train_start'] + bPar['train_end'])/2, y = .20, s = 'training')
        plt.text(x = bPar['test_start'], y = .20, s = 'testing')
        plt.text(x = (bPar['future_start'] *2 + bPar['future_end'])/3, y = .20, s = 'future')

        plt.title('Predicted vs Real Prices')
        plt.xlabel('Market Days After IPO')
        plt.ylabel('Daily Price Change (cube root)')
        plt.legend()

        ensure_dir_exists(target_dir)
        f_name0 = pre_name + 'Predictions.png'
        target_dir0 = os.path.join(target_dir, f_name0 )
        plt.savefig(target_dir0)
        #plt.show()
        plt.close()
    else:
#        plt.plot( range(15 + 750, 15+len(X_plot)), y_hat_linear15[750 : ].astype(float), color = 'blue',  label = '15-day prediction')
        plt.plot( range( 1 + 750,  1+len(X_plot)), y_hat_linear1[750  : ].astype(float), color = 'purple', label = '1-day prediction')
        plt.plot( range(30 + 750, 30+len(X_plot)), y_hat_linear30[750 : ].astype(float), color = 'green', label = '30-day prediction')
        plt.plot( range(0 + 750, len(X_plot)), X_plot[750:], color = 'red', label = 'Real Tesla Stock Price')
        plt.title('Predicted vs Real Prices')
        plt.xlabel('Market Days After IPO')
        plt.ylabel('Tesla Stock Price')
        plt.axvline(x = bPar['train_end'] - bPar['train_start'])
        plt.axvline(x = bPar['test_end'] - bPar['train_start'])

        plt.text(x = (bPar['train_start'] + bPar['train_end'])/2, y = 100, s = 'training')
        plt.text(x = bPar['test_start'], y = 100, s = 'testing')

        plt.title('Predicted vs Real Prices')
        plt.xlabel('Market Days After IPO')
        plt.ylabel('Tesla Stock Price')
        plt.legend()

        ensure_dir_exists(target_dir)
        f_name0 = pre_name + 'Predictions.png'
        target_dir0 = os.path.join(target_dir, f_name0 )
        plt.savefig(target_dir0)
        #plt.show()

        plt.close()

        plt.plot( range( 1 + bPar['test_start'], 1 +len(X_plot)), y_hat_linear1[bPar[ 'test_start'] : ].astype(float), color = 'purple',label = ' 1-day prediction')
        plt.plot( range(15 + bPar['test_start'], 15+len(X_plot)), y_hat_linear15[bPar['test_start'] : ].astype(float), color = 'blue',  label = '15-day prediction')
        plt.plot( range(30 + bPar['test_start'], 30+len(X_plot)), y_hat_linear30[bPar['test_start'] : ].astype(float), color = 'green', label = '30-day prediction')
        plt.plot( range( 0 + bPar['test_start'],  0+len(X_plot)), X_plot[bPar['test_start'] : ], color = 'red', label = 'Real Tesla Stock Price')
        plt.title('Predicted vs Real Prices: Closer Look')
        plt.xlabel('Market Days After IPO')
        plt.ylabel('Tesla Stock Price')

        i = bPar['test_start'] - bPar['scrap_end'] + 64
        while i < bPar['future_end']:
            plt.axvline(x = i, alpha = 0.5)
            i+=64

        plt.title('Predicted vs Real Prices')
        plt.xlabel('Market Days After IPO')
        plt.ylabel('Tesla Stock Price')
        plt.legend()



        ensure_dir_exists(target_dir)
        f_name0 = pre_name + 'Predictions_Zoomed.png'
        target_dir0 = os.path.join(target_dir, f_name0 )
        plt.savefig(target_dir0)
        #plt.show()

        plt.close()



    #Residuals
    plt.scatter( range(1  + bPar['test_start'], 1  + bPar['test_start'] +  len(res30) ), res30, alpha = 0.3, color = 'green', label = '30-Day Residuals')
    plt.scatter( range(16 + bPar['test_start'], 16 + bPar['test_start'] +  len(res15) ), res15, alpha = 0.3, color = 'blue',  label = '15-Day Residuals')
    plt.scatter( range(30 + bPar['test_start'], 30 + bPar['test_start'] +  len(res1 ) ), res1,  alpha = 0.5, color = 'purple' ,  label = ' 1-Day Residuals')
    plt.axvline(x = 2130 + 31)
    plt.axvline(x = 2246 + 31) 
#    plt.axvline(x = 2347)

    plt.legend()
    plt.title('Residuals for Stateful LSTM Predicting Tesla Stock Price')
    plt.xlabel('Market Days After IPO')
    plt.ylabel('Residuals')

    f_name0 = pre_name + 'Residuals.png'
    target_dir0 = os.path.join(target_dir, f_name0 )
    plt.savefig(target_dir0)
    #plt.show()
    plt.close()

    plt.scatter( y_test1, y_hat_linear1_test, alpha = 0.35, label = '1-Day Predicted vs Actual')
    plt.plot( y_test1, y_test1, alpha = 0.5, color = 'grey')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('1-Day Predicted vs Actual Values')
    f_name0 = pre_name + 'Pred_v_actual.png'
    target_dir0 = os.path.join(target_dir, f_name0)
    plt.savefig(target_dir0)
    plt.close()


    return(R2_30, R2_15, R2_1)
####################################################################################################################################################
####################################################################################################################################################
# RELOADING, EXPLORATOIN, ANALYSIS, VISUALIZATION #
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
# RELOADING DATA #
####################################################################################################################################################
# RELOADING
this_dir = os.path.dirname(os.path.realpath('__file__') )
target_dir = os.path.join(this_dir, 'TSLA/TSLA.csv')
TSLA = pd.read_csv(target_dir, delimiter=',') 
####################################################################################################################################################
# DESCRIBE DATA #
####################################################################################################################################################
print(TSLA.head())
print(TSLA.isnull().values.any())
print(TSLA.describe())
print(TSLA.dtypes)
print(TSLA.corr())
####################################################################################################################################################
# VISUALIZATION #
####################################################################################################################################################
target_dir = os.path.join(this_dir, 'TSLA_Visuals')
ensure_dir_exists(target_dir)
target_dir = os.path.join(this_dir, 'TSLA_Visuals/General_Stock_Price.png')

plt.plot_date(TSLA.date, TSLA.close, fmt = '-r', label = 'TESLA Stock Price')
plt.title('TESLA stock price over time')
plt.xticks(TSLA.date[0:len(TSLA):500])
plt.xlabel('Date')
plt.ylabel('Tesla price (open, high, low, close)')
plt.legend()
plt.savefig(target_dir)
plt.close()

# NEXT VISUAL #
target_dir = os.path.join(this_dir, 'TSLA_Visuals/General_Daily_Change.png')

plt.plot_date(TSLA.date, TSLA.daily_change, fmt = '.b', label = 'Tesla: Daily Change')
plt.title('Tesla: Daily Price Change')
plt.xticks(TSLA.date[0:len(TSLA):500])
plt.xlabel('Date')
plt.ylabel('Closing Price Change')
plt.legend()
plt.savefig(target_dir)
plt.close()

# HISTOGRAMS
target_dir = os.path.join(this_dir, 'TSLA_Visuals/close_histograms.png')
n_bins = 20

target_dir = os.path.join(this_dir, 'TSLA_Visuals/close_histograms.png')

plt.hist(TSLA.close, bins=n_bins)
plt.title('closing')
plt.savefig(target_dir)
plt.close()
target_dir = os.path.join(this_dir, 'TSLA_Visuals/daily_change_histograms.png')
plt.hist(TSLA.daily_change, bins=n_bins)
plt.title('daily_change')
plt.savefig(target_dir)
plt.close()
####################################################################################################################################################
# REGRESSION ANALYSIS & VISUALIZATION #
####################################################################################################################################################
x = np.array(TSLA['days_after_ipo'])
y = np.array(TSLA['close'])

x = np.reshape(x, (-1, 1) )
y = np.reshape(y, (-1, 1) )
reg = LinearRegression()

#x = TSLA.date.reshape(-1,1)
#y = TSLA.close.rehape(-1,1)
#reg = reg.fit(TSLA.iloc[:,[0,4] ])
reg.fit( x, y )

pred = reg.predict(x)
MSE = sum( (pred - y) ** 2) / len(x)
R2 = 1 - sum( (pred - y) ** 2 ) / sum( (y - sum(y) / len(y) ) **2 )
print('Regression Performance')
print(MSE)
print(R2)
plt.plot(x, y, color = 'red', label = 'Real Tesla Stock Price')
plt.plot(x, pred, color = 'green', label = 'Predicted Tesla Stock Price')
plt.title('Real vs Predicted Tesla Stock Price')
plt.xlabel('Market Days After IPO')
plt.ylabel('Stock Price')
plt.legend()
target_dir = os.path.join(this_dir, 'TSLA_Visuals/TSLA_Closing_Reg_Predictions.png')
plt.savefig(target_dir)
plt.close()
#plt.show()
####################################################################################################################################################
# MACHINE LEARNING ANALYISIS & VISUALIZATION #
####################################################################################################################################################
# RELOADING
target_dir = os.path.join(this_dir, 'TSLA_Visuals/')
regressor_mae, X_all, y_all, bPar, sc   =   load_data('TSLA', 'closing', load_dataset = False)
# DATA DIVISION PLOT
data_division_plot(bPar, target_dir, title = 'Sample Sizes of Data')
# CLOSING PRICE VISUALIZATION
pred_res(regressor_mae, bPar, sc, X_all, y_all, target_dir, 'TSLA_Closing_Price', points = False)
# REOADING
regressor_mae, X_all, y_all, bPar, sc   =   load_data('TSLA', 'daily_change', load_dataset = False)
# DAILY CHANGE VISUALIZATION
pred_res(regressor_mae, bPar, sc, X_all, y_all, target_dir, 'TSLA_Daily_Change', points = True)

# # export DISPLAY=localhost:0.0 (add to ~/.bashrc to make permanent)