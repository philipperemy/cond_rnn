import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.python.framework.random_seed import set_seed

from cond_rnn import ConditionalRNN

tf.keras.backend.set_floatx('float64')

def read_data():
    '''Reads in a dataset with daily recorded temperatures for major cities of the world
        see https://www.kaggle.com/sudalairajkumar/daily-temperature-of-major-cities

        Returns:
            df: dataframe with columns region, country, city, date and temperature
    '''
    index_names = ['region', 'country', 'city', 'date']
    df =(pd.read_csv('city_temperature.csv', na_values=[-99], low_memory=False)
           .rename(str.lower, axis='columns')
           # day 0 and year 200 and 201 seem seem errors in set
           .loc[lambda x: (x.day!=0)&(x.year!=200)&(x.year!=201)]  
           .drop(['state'], axis=1)
           .assign(region=lambda x: x.region.astype('category'),
                   country=lambda x: x.country.astype('category'),
                   city=lambda x: x.city.astype('category'))
           .assign(avgtemperature=lambda x: x.avgtemperature.fillna(method='ffill'))
           # convert Fahrenheit to Celcius
           .assign(temperature=lambda x: (x.avgtemperature-32)*5/9) 
           .drop(['avgtemperature'], axis=1) 
           .assign(date=lambda x: pd.to_datetime(x[['year', 'month', 'day']], errors='coerce'))
           .drop(['year','month','day'], axis=1)
           #NOTE: you could also take the mean!s
           .drop_duplicates(subset=index_names)
           .dropna()
           .set_index(index_names)
           .sort_index(level=index_names) 
        )
    return df

def generate_set(df, window=100):
    '''generates lagged set and encoded values

       temperature is standard scaled to improve learning
       region, country, city is one-hot encoded

        Args:
            df (DataFrame): DataFrame which results from read_data method
            window (int): for each output a certain number values, here days, are taken into account

        Returns:
            df: dataframe with daily recorded temperatures for major cities of the world
    '''
    df = (df.assign(y = lambda x: StandardScaler().fit_transform(x.temperature.to_numpy().reshape(-1, 1)))
           .drop(['temperature'], axis=1))
    for lag in range(window): df[f'x-{lag+1}']=df.y.shift(lag+1)
    df.dropna(inplace=True)
    for label in ['city', 'country', 'region']: df=pd.concat([df, pd.get_dummies(df.index.to_frame()[label], prefix=label)], axis=1)
    return df

def deterministic():
    '''helper function to fix the seed and obtain reproducable results

       in principle one should repeat calculation and average over outcomes, this is kept for the time
    '''
    set_seed(123)
    np.random.seed(123)
    random.seed(123)

def fitmodel(df, conditional=False, test=True):
    '''fit cond_rnn model or tensorflow model without conditions

       Args:
            df (DataFrame): DataFrame which results from generate_set method after train_test split
            conditional (boolean): select between conditional_rnn or regular tensorflow model
            test (boolean): speed up calculation for testing
    '''
    if test: df = df.sample(10)
    cells = 200
    epochs = 40
    test_size = 0.2
    validation_split = 0.2
    batch_size = 2048
    x = df.filter(like='x-', axis=1).values[:, np.newaxis]
    y = df.y.values[:, np.newaxis]
    if conditional:
        model = Sequential(layers=[ConditionalRNN(cells, cell='GRU'),
                            Dense(units=1, activation='linear')])
        x = [x, df.filter(like='region', axis=1).values,
                df.filter(like='country', axis=1).values,
                df.filter(like='city', axis=1).values]
        fname = 'results_cond_rnn.csv'
    else:
        model = Sequential(layers=[GRU(cells), Dense(units=1, activation='linear')])
        fname = 'results_rnn.csv'
    model.compile(optimizer='adam', loss='mae')
    if test: deterministic()
    model.fit(x=x, y=y, epochs=epochs, batch_size=batch_size,
              validation_split=validation_split,
              callbacks=[CSVLogger(fname)])
    return model