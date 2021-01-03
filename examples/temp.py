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

from cond_rnn import ConditionalRNN


# https://towardsdatascience.com/a-practical-guide-on-exploratory-data-analysis-historical-temperatures-of-cities-e4cb0ca03e07
# Results:
# -----------------------------------
# Total sum of the training losses:
# ConditionalRNN: 66.12524319 (std 0.0687)
# RNN: 67.35795097 (std 0.0785)
# Total sum of the testing losses:
# ConditionalRNN: 66.36418901 (std 0.0300)
# RNN: 66.65316922 (std 0.0328)
# -----------------------------------
# Best training loss:
# ConditionalRNN: 1.642415467
# RNN: 1.691594203
# Best testing loss:
# ConditionalRNN: 1.681800592
# RNN: 1.688295371

# Remarks: CondRNN 2.90% better than the RNN to lower the training loss. 0.4% better on the test set.
# The model has a suboptimal generalization power (a quite shallow model), and this is why the
# result on the test set is not as good as the training test. ON the training set, the model can
# reduce the epistemic uncertainty. The RNN exhibits a higher training loss, meaning that it can't
# explain more incertitude and consider the rest as noise. The conditionalRNN can use the categorical
# data to explain some of the noise and provides a lower loss. With a better model (a few more layers
# and more thoughts into this), we should lower the test set loss.


def deterministic():
    set_seed(123)
    np.random.seed(123)
    random.seed(123)


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
num_samples = 200_000
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
    ConditionalRNN(200, cell='GRU'),
    Dense(units=1, activation='linear')
])

model2 = Sequential(layers=[
    GRU(200),
    Dense(units=1, activation='linear')
])

model.compile(optimizer='adam', loss='mae')
model2.compile(optimizer='adam', loss='mae')

cutoff = int(len(x) * 0.8)

# Not exactly correct because we leak some future information (time-wise). But good enough.
x_train = x[:cutoff]
y_train = y[:cutoff]
c1_train = c1[:cutoff]
c2_train = c2[:cutoff]
c3_train = c3[:cutoff]
x_test = x[cutoff:]
y_test = y[cutoff:]
c1_test = c1[cutoff:]
c2_test = c2[cutoff:]
c3_test = c3[cutoff:]

deterministic()
model.fit(x=[x_train, c1_train, c2_train, c3_train], y=y, epochs=40,
          batch_size=2048, validation_split=0.2,
          callbacks=[CSVLogger('results_cond_rnn.csv')])

deterministic()
model2.fit(x=x_train, y=y_train, epochs=40, batch_size=2048, validation_split=0.2,
           callbacks=[CSVLogger('results_rnn.csv')])

# cat results_cond_rnn.csv
# epoch,loss,val_loss
# 0,5.9044762802124025,2.53386900138855
# 1,2.0451689500808716,1.8385024347305299
# 2,1.8185846309661866,1.7780692615509033
# 3,1.7762680377960205,1.7446418466567992
# 4,1.7519025249481202,1.738777319908142
# 5,1.7438574428558349,1.7169553089141845
# 6,1.7305130863189697,1.7153366613388061
# 7,1.72012220287323,1.717546311378479
# 8,1.7138757123947144,1.7268745737075806
# 9,1.709419322013855,1.6952379293441773
# 10,1.7010830554962157,1.6979627094268799
# 11,1.69859104347229,1.6902550802230836
# 12,1.696402850151062,1.6917604742050172
# 13,1.6943436574935913,1.6934981899261474
# 14,1.694669309616089,1.689155652999878
# 15,1.6916284599304199,1.6861914052963256
# 16,1.6867354545593263,1.6883070735931396
# 17,1.6850832424163817,1.688975700378418
# 18,1.6838756952285767,1.6858656063079833
# 19,1.6801656255722046,1.6879012060165406
# 20,1.6763159046173095,1.6818005924224853
# 21,1.6780798892974853,1.6861273279190063
# 22,1.6741125926971436,1.6842413787841797
# 23,1.6691744747161865,1.6905452280044555
# 24,1.6690498428344727,1.6861427402496338
# 25,1.6671749687194823,1.687508749961853
# 26,1.663203508377075,1.6908981418609619
# 27,1.667403314590454,1.6914220447540282
# 28,1.6647329444885255,1.6822782535552978
# 29,1.6585801000595093,1.6848696880340577
# 30,1.6594591941833496,1.6835118341445923
# 31,1.6571005964279175,1.7055641012191773
# 32,1.6548523530960082,1.6887998666763306
# 33,1.6546397333145142,1.6949167547225952
# 34,1.6534125833511353,1.6865443964004516
# 35,1.6496666011810304,1.6888751363754273
# 36,1.6512721786499023,1.6982809133529664
# 37,1.6469017171859741,1.7008602695465087
# 38,1.6454049243927003,1.686760389328003
# 39,1.6424154672622682,1.6924264583587647

# cat results_rnn.csv
# epoch,loss,val_loss
# 0,6.804965030670166,2.8608989562988283
# 1,2.171871382713318,1.869871148109436
# 2,1.8377241439819336,1.7880843715667725
# 3,1.790917474746704,1.7528343725204467
# 4,1.7620672359466554,1.7365421905517577
# 5,1.7509015369415284,1.7243329029083252
# 6,1.7415653715133668,1.7150184249877929
# 7,1.732437931060791,1.7163320989608766
# 8,1.7272775135040284,1.7195946054458617
# 9,1.7241863813400269,1.709030837059021
# 10,1.7198550090789795,1.708265691757202
# 11,1.715962296485901,1.7030303745269775
# 12,1.7143471603393554,1.6997407522201537
# 13,1.713460618019104,1.6975479221343994
# 14,1.7141553030014038,1.7008580303192138
# 15,1.7165812940597533,1.6973607559204102
# 16,1.7124508800506593,1.7041066131591798
# 17,1.7098140487670899,1.7067196521759034
# 18,1.7094310178756713,1.698859739303589
# 19,1.7070101194381715,1.7006327571868896
# 20,1.7059875831604003,1.6937656774520875
# 21,1.7064809379577637,1.6935932903289794
# 22,1.7037200889587403,1.6939699573516847
# 23,1.7029644117355347,1.7030137643814087
# 24,1.707309970855713,1.725887640953064
# 25,1.7050414266586305,1.6920332279205321
# 26,1.7012271432876587,1.6942542362213135
# 27,1.7041923789978026,1.695572205543518
# 28,1.7005678110122682,1.6915680179595947
# 29,1.700346218109131,1.695366147994995
# 30,1.6986202239990233,1.6909933776855468
# 31,1.697178050994873,1.6957357177734376
# 32,1.6956451320648194,1.693692931175232
# 33,1.6982271547317505,1.693710618019104
# 34,1.6942549505233764,1.6890140857696534
# 35,1.693182267189026,1.6924312076568604
# 36,1.6942980756759645,1.7009847049713134
# 37,1.69286688041687,1.688295371055603
# 38,1.6915942029953004,1.6900234594345094
# 39,1.6922293395996093,1.690500343322754
