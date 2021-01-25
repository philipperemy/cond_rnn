#!/usr/bin/env python
# coding: utf-8

#NOTE: FILE CAN BE CONVERTED TO NOTEBOOK VIA P2J
# **Abstract:**    
# ARIMAX is used to benchmark the results obtained by LSTM. The best result for daily temperature prediction is a MAE of 0.66 degrees for the city of Amsterdam.
# As exogenous components, the temperatures in five neighbouring cities are used. Naturally, these are lagged with one day.

# Calculations can be sped up by resampling. ARIMAX is not really suited to work with patterns on a fine [time scale](https://stackoverflow.com/questions/63438979/python-pmdarima-autoarima-does-not-work-with-large-data). It is mainly used here as bench mark. Monthly calculations can be done on a laptop for others I used a [p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) from AWS.

# In[]:


import temp
import pmdarima as pm
import pandas as pd
import numpy as np
import statsmodels.api as sm
options = ['monthly', 'biweekly', 'daily']
print("Please choose:")
for idx, element in enumerate(options):
    print("{}) {}".format(idx,element))
choice = options[int(input("Enter number: "))]

df = temp.read_data().droplevel(level=['region', 'country'])
if choice == 'monthly':
    df = df.groupby(['city', pd.Grouper(level='date', freq='m')]).mean()
elif choice == 'biweekly':
    df = df.groupby(['city', pd.Grouper(level='date', freq='2W')]).mean()


# For simplicity, I start out by predicting the temperature in a single city; Amsterdam.

# In[]:


df_amsterdam = df.xs("Amsterdam", level="city")
df_amsterdam.head()


# Let's try to look for a weak overall trend. This trends only appears if sampled at a daily basis.<br>
# I arrive at a global warming rate of 0.35 degrees for the city of Amsterdam per decade. This seems to be in line with [literature](https://www.climate.gov/news-features/understanding-climate/climate-change-global-temperature), which predicts 0.18 degrees since 1980.

# In[]:


X = sm.add_constant(np.arange(len(df_amsterdam)))
model = sm.OLS(df_amsterdam.values, X)
results = model.fit()
print(results.summary())
if choice == 'daily':
    print(f"Per decade the temperature rises with {results.params[1]*365*10:.2f} degrees")


# As a simple model, I use an ARIMA model without exogenous components.<br>
# The parameters are explained in [tips and trick](https://alkaline-ml.com/pmdarima/tips_and_tricks.html) of the ARIMA package.<br>
# The AutoARIMA parameters are explained [here](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html).<br>
# It is possible to fit a weak trend to it, which could be interpreted as global warming.

# In[]:


defaults =  {
              'test':'adf',       # use adftest to find optimal 'd'
              'trend': 'c',      # linear trend does not add value
                                 # use 't' for linear trend with time (https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html#statsmodels.tsa.statespace.sarimax.SARIMAX)
              # seasonal settings
              'm': 1,             # m observations per seasonal cycle, do not use larger numbers than 52 (takes too long too compute)
              'max_P': 3,
              'max_Q': 2,
              'D': 0,
              'seasonal':False,      # Seasonality, temperature is highly seasonal, but in the models they do not add value
              # regular settings
              'd': 0,
              'max_p': 6,
              'max_q': 4,
              'trace': True,
              'scoring': 'mae',      # mae is used as this is also done for RNN network
              'error_action': 'ignore',  
              'suppress_warnings': True, 
              'n_jobs': 1,         # Number of jobs in parallel, multiple jobs can only be done with gridsearch, stepwise is False
              'stepwise': True
            }  
model = pm.auto_arima(df_amsterdam, **defaults)
model.summary()


# ARIMA optimizes the AIC term. Let's convert it back to the Mean Absolute Error (MAE) for comparison with the LSTM model. 

# In[]:

import numpy as np
print(f"The AIC equals {model.aic():.2f}")
print(f"The MAE equals {np.mean(np.abs(model.resid())):.2f}")


# Let's assume heat is mostly transported by diffusion (Fick's law). Under this assumption the most correlating temperatures should be neighbouring towns.<br>
# The temperature in these town can be used as exogenous component. I compute the cross correlation matrix and grab the 4 most correlating components.<br>
# The model is developed with respect to the errors so the coefficients have the usual interprestation see [Hyndman](https://robjhyndman.com/hyndsight/arimax/).

# In[91]:

# compute temperature per city

# In[91]:


import matplotlib
df_city = df.unstack(level='date').T
#df_city.head()
df_cor = df_city.corr()
# The five most correlating temperatures for the city of Amsterdam
top_six = df_cor[df_cor.index == 'Amsterdam'].T.nlargest(6, ['Amsterdam']).index[0:6].to_list()
print(top_six[1:])


# The results above make sense. Brussels is closer to Amsterdam than Paris. There is a sea between London and Amsterdam.  <br>
# The exogenous components is computed via a shift.

# In[92]:


df_data = (df_city[top_six[1:]].shift(1)
                               .assign(Amsterdam=df_city.Amsterdam)
                               .dropna()) 
df_data.head()


# In[ ]:

# If sampling is put at monthly, the best results are obtained with a constant trend.

# In[99]:



model = pm.auto_arima(df_city.Amsterdam, exogenous=df_city[top_six[1:]], **defaults)
model.summary()


# In[100]:



print(f"The AIC equals {model.aic():.2f}")
print(f"The MAE equals {np.mean(np.abs(model.resid())):.2f}")


# Convert this file to python

# In[97]:



get_ipython().system('jupyter nbconvert --to script arma.ipynb')

