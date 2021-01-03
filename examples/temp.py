import json
import os
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.python.framework.random_seed import set_seed
from tqdm import tqdm

# https://towardsdatascience.com/a-practical-guide-on-exploratory-data-analysis-historical-temperatures-of-cities-e4cb0ca03e07
from cond_rnn import ConditionalRNN

tf.keras.backend.set_floatx('float64')

if not os.path.exists('out.json'):
    print('generate data...')
    df = pd.read_csv('city_temperature.csv', na_values=[-99], low_memory=False)
    df = df[df.Year > 1900]
    df['AvgTemperature'].fillna(method='ffill', inplace=True)
    df['celcius'] = (df['AvgTemperature'] - 32) * 5 / 9
    df['date'] = df['Year'].apply(str) + df['Month'].apply(lambda x: str(x).zfill(2)) + df['Day'].apply(
        lambda x: str(x).zfill(2))
    df['name'] = df['Region'] + '|' + df['Country'] + '|' + df['City']
    values = {}
    for name in tqdm(set(df['name']), desc='build dataset', file=sys.stdout):
        dfn = df[df['name'] == name]
        dfn.sort_values('date')
        # dfn = dfn[['date', 'name', 'celcius']]
        values[name] = list([int(float(a) * 100) / 100 for a in dfn['celcius'].values])
    with open('out.json', 'w') as w:
        json.dump(values, w, indent=2)
else:
    with open('out.json') as r:
        values = json.load(r)

# generate model inputs.
print('generate model inputs...')
cond = np.array([a.split('|', 2) for a in list(values.keys())])
num_regions, num_countries, num_cities = len(set(cond[:, 0])), len(set(cond[:, 1])), len(set(cond[:, 2]))

regions_map = dict(zip(sorted(set(cond[:, 0])), range(num_regions)))
countries_map = dict(zip(sorted(set(cond[:, 1])), range(num_countries)))
cities_map = dict(zip(sorted(set(cond[:, 2])), range(num_cities)))

window = 100
num_samples = 100_000
loop_idx = 0
x = np.zeros((num_samples, window, 1))
c1 = np.zeros((num_samples, num_regions))
c2 = np.zeros((num_samples, num_countries))
c3 = np.zeros((num_samples, num_cities))
y = np.zeros((num_samples, 1))

with tqdm(total=num_samples, desc='build inputs', file=sys.stdout) as bar:
    for _ in range(int(num_samples / len(values))):
        shuffled_values = list(values.items())
        random.shuffle(shuffled_values)
        for name, ts in shuffled_values:
            region, country, city = name.split('|')
            region_cond = np.zeros(shape=num_regions)
            region_cond[regions_map[region]] = 1
            country_cond = np.zeros(shape=num_countries)
            country_cond[countries_map[country]] = 1
            city_cond = np.zeros(shape=num_cities)
            city_cond[cities_map[city]] = 1

            start_idx = np.random.randint(len(ts) - window)
            tsn = ts[start_idx:start_idx + window]
            x[loop_idx] = np.expand_dims(ts[start_idx:start_idx + window], axis=-1)
            y[loop_idx] = ts[start_idx + window]
            c1[loop_idx] = region_cond
            c2[loop_idx] = country_cond
            c3[loop_idx] = city_cond
            loop_idx += 1
            bar.update()

model = Sequential(layers=[
    ConditionalRNN(100, cell='GRU'),
    Dense(units=1, activation='linear')
])

model2 = Sequential(layers=[
    GRU(100),
    Dense(units=1, activation='linear')
])

model.compile(optimizer='adam', loss='mae')
model2.compile(optimizer='adam', loss='mae')

set_seed(123)
np.random.seed(123)
random.seed(123)
model.fit(x=[x, c1, c2, c3], y=y, epochs=40, batch_size=1024, validation_split=0.2,
          callbacks=[CSVLogger('results_cond_rnn.csv')])

set_seed(123)
np.random.seed(123)
random.seed(123)
model2.fit(x=x, y=y, epochs=40, batch_size=1024, validation_split=0.2,
           callbacks=[CSVLogger('results_rnn.csv')])

model.summary()
model2.summary()
