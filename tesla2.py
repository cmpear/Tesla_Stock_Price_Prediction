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

from keras.layers import Dense
from keras.layers import Input, LSTM
from keras.models import Model
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
def load_data(df_name):
    this_dir = os.path.dirname(os.path.realpath('__file__') )
    rel_dir = df_name + '/'
    this_dir = os.path.join(this_dir, rel_dir)
    model_name = df_name + '_with_mae_32_ts.h5'

    target_dir = os.path.join(this_dir, model_name)
    regressor_mae = load_model(target_dir)

    target_dir = os.path.join(this_dir, 'X_all.csv')
    X_all = pickle.load(open (target_dir, 'rb') )

    target_dir = os.path.join(this_dir, 'y_all.csv')
    y_all = pickle.load(open (target_dir, 'rb'))

    target_dir = os.path.join(this_dir, 'bPar.json')
    bPar = pickle.load(open (target_dir, 'rb') )

    target_dir = os.path.join(this_dir, 'sc.save')
    sc = joblib.load(target_dir) 

    target_dir = os.path.dirname(os.path.realpath('__file__') )
    rel_dir = 'data/TSLA.csv'
    target_dir = os.path.join(target_dir, rel_dir)
    return (regressor_mae, X_all, y_all, bPar, sc)
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
    plt.bar ( ['scraps','test','train','future'], [bPar['scrap_end'], bPar['test_end'] - bPar['test_start'], bPar['train_end'] - bPar['train_start'], bPar['future_end'] - bPar['future_start'] ]  ,  label = 'Sample Sizes')
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
    res15 = y_test30 - y_hat_linear15_test
    R2_15 = 1 - sum( (res15) ** 2 ) / sum( (y_test15 - np.mean( y_test15 ) ) ** 2 )

    y_hat_linear1_test = y_hat_linear1[ bPar['test_start'] - bPar['train_start'] : bPar['test_end'] - bPar['train_start'] ]
    y_test1 = X_plot[ bPar['test_start'] - bPar['train_start'] +1: bPar['test_end'] - bPar['train_start'] +1 ]

    res1 = y_test1 - y_hat_linear1_test
    R2_1 = 1 - sum( (res30) ** 2 ) / sum( (y_test1 - np.mean(y_test1 ) ) ** 2 )
    # # this would work better, be more accurate, if we were predicting changes.
    print("R^2 for 30-day prediction")
    print(R2_30)
    print("R^2 for 15-day prediction")
    print(R2_15)
    print("R^2 for 1-day prediction")
    print(R2_1)

    # # Visualising the results
    if (points):
        print('SHOULD HAVE DOTS')
#        plt.scatter( range(15, 15+len(X_plot)), y_hat_linear15[0 : ].astype(float), color = 'blue', alpha = 0.1, label = '15-Day Predicted Change')
        plt.scatter( range(1, 1+len(X_plot)), y_hat_linear1[0 : ].astype(float), color = 'green', alpha = 0.1, label = '1-Day Predicted Change')
        plt.scatter( range(0, len(X_plot)), X_plot, color = 'red', alpha = 0.1, label = 'Real Change')
        plt.title('Predicted vs Real Daily Price Change')
        plt.xlabel('Market Days After IPO')
        plt.ylabel('Closing - Opening Price')
    else:
        plt.plot( range(15 + 750, 15+len(X_plot)), y_hat_linear15[750 : ].astype(float), color = 'blue', label = '15-day prediction')
        plt.plot( range(30 + 750, 30+len(X_plot)), y_hat_linear30[750 : ].astype(float), color = 'green', label = '30-day prediction')
        plt.plot( range(0 + 750, len(X_plot)), X_plot[750:], color = 'red', label = 'Real Tesla Stock Price')
        plt.title('Predicted vs Real Prices')
        plt.xlabel('Market Days After IPO')
        plt.ylabel('Tesla Stock Price')
    plt.axvline(x = bPar['train_end'] - bPar['train_start'])
    plt.axvline(x = bPar['test_end'] - bPar['train_start'])

    plt.text(x = (bPar['train_start'] + bPar['train_end'])/2, y = .20, s = 'training')
    plt.text(x = bPar['test_start'], y = .20, s = 'testing')
    plt.text(x = (bPar['future_start'] *2 + bPar['future_end'])/3, y = .20, s = 'future')

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

    #Residuals
    plt.scatter( range(0 + bPar['test_start'], bPar['test_start'] +  len(res30) ), res30, color = 'green', label = 'residuals for Stateful LSTM')
    plt.title('Residuals for Stateful LSTM Predicting Tesla Stock Price')
    plt.xlabel('Market Days After IPO')
    plt.ylabel('Residuals')

    f_name0 = pre_name + 'Residuals.png'
    target_dir0 = os.path.join(target_dir, f_name0 )
    plt.savefig(target_dir0)
    #plt.show()
    plt.close()

    return(R2_30, R2_15, R2_1)
####################################################################################################################################################
####################################################################################################################################################
# RELOADING, EXPLORATOIN, ANALYSIS, VISUALIZATION #
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
# RELOADING & EXPLORATORY DATA VISUALIZATION #
####################################################################################################################################################
# RELOADING
this_dir = os.path.dirname(os.path.realpath('__file__') )
target_dir = os.path.join(this_dir, 'TSLA/TSLA.csv')
TSLA = pd.read_csv(target_dir, delimiter=',') 
# VISUALIZATION
target_dir = os.path.join(this_dir, 'TSLA_Visuals/TSLA.csv')
ensure_dir_exists(target_dir)
target_dir = os.path.join(this_dir, 'TSLA_Visuals/General_Stock_Price.png')

plt.plot_date(TSLA.date, TSLA.close, fmt = '-r', label = 'TESLA Stock Price')
plt.title('TESLA stock price over time')
plt.xticks(TSLA.date[0:len(TSLA):500])
plt.xlabel('Time (Days)')
plt.ylabel('Tesla price (open, high, low, close)')
plt.legend()
plt.savefig(target_dir)
plt.close()

# next visual
target_dir = os.path.join(this_dir, 'TSLA_Visuals/General_Daily_Change')

plt.plot_date(TSLA.date, TSLA.day_range, fmt = '.b', label = 'Tesla: Daily Change')
plt.title('Tesla: Daily Price Change')
plt.xticks(TSLA.date[0:len(TSLA):500])
plt.xlabel('Date')
plt.ylabel('Closing Price - Opening Price')
plt.legend()
plt.savefig(target_dir)
plt.close()

####################################################################################################################################################
# REGRESSION ANALYSIS & VISUALIZATION #
####################################################################################################################################################
x = np.array(TSLA['days_after_ipo'])
y = np.array(TSLA['close'])

x = np.reshape(x, (-1, 1) )
y = np.reshape(y, (-1, 1) )
print(x.shape)
print(y.shape)
reg = LinearRegression()

#x = TSLA.date.reshape(-1,1)
#y = TSLA.close.rehape(-1,1)
#reg = reg.fit(TSLA.iloc[:,[0,4] ])
reg.fit( x, y )

pred = reg.predict(x)
MSE = sum( (pred - y) ** 2) / len(x)
R2 = 1 - sum( (pred - y) ** 2 ) / sum( (y - sum(y) / len(y) ) **2 )
print(MSE)
print(R2)
plt.plot(x, y, color = 'red', label = 'Real Tesla Stock Price')
plt.plot(x, pred, color = 'green', label = 'Predicted Tesla Stock Price')
plt.title('Real vs Predicted Tesla Stock Price')
plt.xlabel('Date')
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
regressor_mae, X_all, y_all, bPar, sc   =   load_data('TSLA')
# DATA DIVISION PLOT
data_division_plot(bPar, target_dir, title = 'Sample Sizes of Data')
# CLOSING PRICE VISUALIZATION
pred_res(regressor_mae, bPar, sc, X_all, y_all, target_dir, 'TSLA_Closing_Price', points = False)
# REOADING
regressor_mae, X_all, y_all, bPar, sc   =   load_data('TSLA_daily_change')
# DAILY CHANGE VISUALIZATION
pred_res(regressor_mae, bPar, sc, X_all, y_all, target_dir, 'TSLA_Daily_Change', points = True)

# # export DISPLAY=localhost:0.0 (add to ~/.bashrc to make permanent)
print(" GOT TO THE END!!!")