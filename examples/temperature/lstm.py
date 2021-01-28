#!/usr/bin/env python
# coding: utf-8

# **Abstract:**    
# A tensorflow LSTM network is used to predict the daily temperature of Amsterdam. Without exogoneous components the best MAE is 1.46 degrees on the test set.
# Cond_rnn is able to get a MAE of 0.87 using the temperature in 30 neighbouring cities. A GPU is recommended to speed up calculations, I used a [p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) from AWS for computation.

# In[1]:


# settings
cells = 200
epochs = 40
test_size = 0.2
validation_split = 0  # we set to 0 for fair comparison with armax
window = 100
cities = 30           # neighbouring cities to include in cond_rnn vector
random_state = 123    # random state is kept fixed to create similar result, in principle one should average over multiple results and report mean

import temp
import pandas as pd
from sklearn.preprocessing import StandardScaler
df = temp.read_data(option='daily')


# Again, we mainly look at the temperature in Amsterdam.

# In[2]:


df_city = df.droplevel(level=['region', 'country']).unstack(level='date').T.sort_index()
df_city.Amsterdam.head()


# See the notes on ARMA. Here the 30 most correlating temperatures are used as exogenous component. In my opinion, tensorflow should be better with multicollinearity.

# In[3]:


df_cor = df_city.corr()
df_cor.head()
# One more is grabbed as the most correlating city is Amsterdam itself
top_cities = df_cor[df_cor.index == 'Amsterdam'].T.nlargest(cities+1, ['Amsterdam']).index[0:cities+1].to_list()
df_data = (df_city[top_cities[1:]].shift(1)
                                  .assign(Amsterdam=df_city.Amsterdam)
                                  .dropna()) 
df_data.columns = df_data.columns.astype(str)


# The dataset is transformed for machine learning. 
# Temperature is standard scaled and an input x is generated which contains the previous 100 values for the city of Amsterdam.
# For the other cities, only the previous daily temperature is used.

# In[4]:


from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([
        ('Amsterdam', StandardScaler(), ['Amsterdam']),
        ('Neighbours', StandardScaler(), top_cities[1:])
    ], remainder='passthrough')
df_data = pd.DataFrame(ct.fit_transform(df_city[top_cities]), columns = top_cities)
for lag in range(window): df_data.loc[:,f'x-{lag+1}']=df_data.Amsterdam.shift(lag+1)
df_data = df_data.dropna().sort_index()


# The data is split in a train and test set. Shuffle is disabled to enable comparison with ARMAX.

# In[5]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df_data, test_size=test_size, shuffle=False)


# Libraries are loaded and the data is reshaped.

# In[6]:


from cond_rnn import ConditionalRNN
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.python.framework.random_seed import set_seed
import numpy as np
def create_xy(data):
    'helper function to create x, c and y with proper shapes'
    x = data.filter(like='x-', axis=1).values[:,:,np.newaxis]
    c = data[top_cities[1:]].to_numpy()
    y = data.Amsterdam.values[:, np.newaxis]
    return x, c, y
# create correct shapes for tensorflow
x_train, c_train, y_train = create_xy(train)
x_test, c_test, y_test = create_xy(test)
# deterministic
set_seed(random_state)


# As before, I start out by a pure autoregressive model.

# In[7]:


model = Sequential(layers=[GRU(cells), Dense(units=1, activation='linear')])
model.compile(optimizer='adam', loss='mae')
history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=None, shuffle=True,
              validation_split=validation_split)


# The final test loss is;

# In[9]:


inverseAms = lambda data: ct.named_transformers_['Amsterdam'].inverse_transform(data)
from sklearn.metrics import mean_absolute_error
print(f"The MAE is {mean_absolute_error(inverseAms(model.predict(x_test)),inverseAms(y_test)):.2f}")


# The above test loss is very similar to ARMA. Let's try to improve on this estimate with an exogenous model.

# In[7]:


print("WARNING: Install latest version of cond_rnn via git and not pip!")
model_exog = Sequential(layers=[ConditionalRNN(cells, cell='GRU'),
                                Dense(units=1, activation='linear')])
model_exog.compile(optimizer='adam', loss='mae')


# Let's fit a model;

# In[8]:


history = model_exog.fit(x=[x_train, c_train], y=y_train, epochs=epochs, batch_size=None, shuffle=True,
              validation_split=validation_split)


# The test loss for the exogenous model is;

# In[1]:


inverseAms = lambda data: ct.named_transformers_['Amsterdam'].inverse_transform(data)
from sklearn.metrics import mean_absolute_error
print(f"The traub MAE is {mean_absolute_error(inverseAms(model_exog.predict([x_train, c_train])),inverseAms(y_train)):.2f}")
print(f"The test MAE is {mean_absolute_error(inverseAms(model_exog.predict([x_test, c_test])),inverseAms(y_test)):.2f}")


# In[12]:


get_ipython().system('jupyter nbconvert --to script lstm.ipynb')

