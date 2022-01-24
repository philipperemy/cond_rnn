#!/usr/bin/env python
# coding: utf-8
#
# **Abstract:**
# A tensorflow LSTM network is used to predict the daily temperature
#  of Amsterdam. Without exogoneous components the best MAE is 1.46
#  degrees on the test set. Cond_rnn is able to get a MAE of 0.87
#  using the temperature in 30 neighbouring
#  cities. A GPU is recommended to speed up calculations,
#  I used a [p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/)
#  from AWS for computation.

import numpy as np
# +
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.python.framework.random_seed import set_seed

import temp
from cond_rnn import ConditionalRecurrent

# settings
cells = 200
epochs = 40
test_size = 0.2
validation_split = 0  # we set to 0 for fair comparison with armax
window = 100
cities = 30  # neighbouring cities to include in cond_rnn vector
random_state = 123  # random state  fixed for similar result,
# ideally one should average and report mean
df = temp.read_data(option='daily')

#
# -

# Again, we mainly look at the temperature in Amsterdam.

df_city = (df.droplevel(level=['region', 'country'])
           .unstack(level='date').T.sort_index()
           )
df_city.Amsterdam.head()

# See the notes on ARMA. Here the 30 most correlating
# temperatures are used as exogenous component. In my opinion,
# tensorflow should be better with multicollinearity.

df_cor = df_city.corr()
df_cor.head()
# One more is grabbed as the most correlating city is Amsterdam itself
top_cities = (df_cor[
                  df_cor.index == 'Amsterdam'].T
              .nlargest(cities + 1, ['Amsterdam'])
              .index[0:cities + 1].to_list()
              )
df_data = (df_city[top_cities[1:]].shift(1)
           .assign(Amsterdam=df_city.Amsterdam)
           .dropna()
           )
df_data.columns = df_data.columns.astype(str)

# The dataset is transformed for machine learning.
# Temperature is standard scaled and an input x is generated which contains
#  the previous 100 values for the city of Amsterdam.
# For the other cities, only the previous daily temperature is used.

ct = ColumnTransformer([
    ('Amsterdam', StandardScaler(), ['Amsterdam']),
    ('Neighbours', StandardScaler(), top_cities[1:])
], remainder='passthrough')
df_data = pd.DataFrame(ct.fit_transform(df_city[top_cities]),
                       columns=top_cities)
for lag in range(window):
    df_data.loc[:, f'x-{lag + 1}'] = df_data.Amsterdam.shift(lag + 1)
df_data = df_data.dropna().sort_index()

# The data is split in a train and test set. Shuffle is disabled to enable
#  comparison with ARMAX.

train, test = train_test_split(df_data, test_size=test_size, shuffle=False)


# Libraries are loaded and the data is reshaped.


def create_xy(data):
    'helper function to create x, c and y with proper shapes'
    x = data.filter(like='x-', axis=1).values[:, :, np.newaxis]
    c = data[top_cities[1:]].to_numpy()
    y = data.Amsterdam.values[:, np.newaxis]
    return x, c, y


# create correct shapes for tensorflow
x_train, c_train, y_train = create_xy(train)
x_test, c_test, y_test = create_xy(test)
# deterministic
set_seed(random_state)

#  As before, I start out by a pure autoregressive model.

model = Sequential(layers=[GRU(cells), Dense(units=1, activation='linear')])
model.compile(optimizer='adam', loss='mae')
history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=None,
                    shuffle=True,
                    validation_split=validation_split)


# The final test loss is;


def inverseAms(data):
    return (ct.named_transformers_['Amsterdam']
            .inverse_transform(data)
            )


modelmae = mean_absolute_error(inverseAms(model.predict(x_test)),
                               inverseAms(y_test))
print(f"The MAE is {modelmae:.2f}")

# The above test loss is very similar to ARMA. Let's try to improve on this
#  estimate with an exogenous model.

print("WARNING: Install latest version of cond_rnn via git and not pip!")
model_exog = Sequential(layers=[ConditionalRecurrent(GRU(cells)),
                                Dense(units=1, activation='linear')])
model_exog.compile(optimizer='adam', loss='mae')

# Let's fit a model;

history = model_exog.fit(x=[x_train, c_train], y=y_train, epochs=epochs,
                         batch_size=None, shuffle=True,
                         validation_split=validation_split)

# The test loss for the exogenous model is;

exomae1 = mean_absolute_error(inverseAms(model_exog.predict([x_train,
                                                             c_train])),
                              inverseAms(y_train))
exomae2 = mean_absolute_error(inverseAms(model_exog.predict([x_test,
                                                             c_test])),
                              inverseAms(y_test))

print(f"The train MAE is {exomae1:.2f}")
print(f"The test MAE is {exomae2:.2f}")
