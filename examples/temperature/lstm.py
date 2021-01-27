#!/usr/bin/env python
# coding: utf-8

# **Abstract:**    
# A tensorflow LSTM network is used to predict the temperature per day. Without exogoneous components the best MAE is 1.48 degrees on the test set.

# In[9]:


# settings
cells = 200
epochs = 40
test_size = 0.2
validation_split = 0  # we set to 0 for fair comparison with armax
window = 100

import temp
import pandas as pd
from sklearn.preprocessing import StandardScaler
df = temp.read_data(option='daily')


# Again, we mainly look at the temperature in Amsterdam.

# In[2]:


df_city = df.droplevel(level=['region', 'country']).unstack(level='date').T.sort_index()
df_city.Amsterdam.head()


# See the notes on ARMA, we use the five most correlating temperatures as exogenous component.

# In[3]:


df_cor = df_city.corr()
df_cor.head()
# The five most correlating temperatures for the city of Amsterdam
top_six = df_cor[df_cor.index == 'Amsterdam'].T.nlargest(6, ['Amsterdam']).index[0:6].to_list()
df_data = (df_city[top_six[1:]].shift(1)
                               .assign(Amsterdam=df_city.Amsterdam)
                               .dropna()) 
df_data.columns = df_data.columns.astype(str)


# The dataset is transformed for machine learning. 
# Temperature is standard scaled and a input x is generated which contains the previous 100 values. 

# In[4]:


scaler = StandardScaler().fit(df_data.Amsterdam.to_numpy().reshape(-1, 1))
df_data['y'] = scaler.transform(df_data.Amsterdam.to_numpy().reshape(-1, 1))
for lag in range(window): df_data.loc[:,f'x-{lag+1}']=df_data.y.shift(lag+1)
df_data = df_data.dropna().sort_index()


# The data is split in a train and test set. Shuffle is disabled to enable comparison with ARMAX.

# In[7]:


from sklearn.model_selection import train_test_split
random_state = 123
train, test = train_test_split(df_data, test_size=test_size, shuffle=False)


# The data is reshaped and a model is fit on the data.

# In[10]:


import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.python.framework.random_seed import set_seed
import numpy as np
def create_xy(data):
    'helper function to create x and y with proper shapes'
    x = data.filter(like='x-', axis=1).values[:,:,np.newaxis]
    y = data.y.values[:, np.newaxis]
    return x, y
# create correct shapes for tensorflow
x_train, y_train = create_xy(train)
x_test, y_test = create_xy(test)
# deterministic
set_seed(123)
# fit model
fname = 'results_rnn.csv'
model = Sequential(layers=[GRU(cells), Dense(units=1, activation='linear')])
model.compile(optimizer='adam', loss='mae')
history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=None, shuffle=True,
              validation_split=validation_split,
              callbacks=[CSVLogger(fname)])


# The final test loss is;

# In[11]:


from sklearn.metrics import mean_absolute_error
print(f"The MAE is {mean_absolute_error(scaler.inverse_transform(model.predict(x_test)),scaler.inverse_transform(y_test)):.2f}")


# In[ ]:




