#!/usr/bin/env python
# coding: utf-8

# Calculations can be sped up resampling to a monthly basis and setting the frequency on 'M'.

# In[ ]:


import temp
df = temp.read_data().groupby(['city', pd.Grouper(level='date', freq='M')]).mean()


# For simplicity, I start out by predicting the temperature in a single city; Amsterdam.

# In[48]:


import pandas as  pd
df_amsterdam = df.xs("Amsterdam", level="city")
df_amsterdam.head()


# Let's try to look for a weak overall trend. This trends only appears if sampled at a daily basis.
# I arrive at a global warming rate of 0.35 degrees for the city of Amsterdam per decade. This seems to be in line with [literature](https://www.climate.gov/news-features/understanding-climate/climate-change-global-temperature), which predicts 0.18 degrees since 1980.

# In[50]:


import statsmodels.api as sm
X = sm.add_constant(np.arange(len(df_amsterdam)))
model = sm.OLS(df_amsterdam.values, X)
results = model.fit()
print(results.summary())
#print(f"Per decade the temperature would rise with {results.params[1]*365*10:.2f} degrees")


# As a simple model, I use an ARIMA model without exogenous components.
# The parameters are explained in [tips and trick](https://alkaline-ml.com/pmdarima/tips_and_tricks.html) of the ARIMA package.
# The AutoARIMA parameters are explained [here](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html).
# It is possible to fit a weak trend to it, which could be interpreted as global warming.

# In[89]:


import pmdarima as pm
import numpy as np
model = pm.auto_arima(df_amsterdam, 
                      test='adf',       # use adftest to find optimal 'd'
                      # trend
                      trend = 't',      # we assumme a linear trend with time  (https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html#statsmodels.tsa.statespace.sarimax.SARIMAX)
                      # seasonal settings
                      m=12,             # m observations per seasonal cycle
                      max_P = 3,
                      max_Q = 2,
                      D = 0,
                      seasonal=True,      # Seasonality, temperature is highly seasonal!
                      n_jobs = 1,         # Number of jobs in parallel
                      # regular settings
                      #start_p=4, 
                      #start_q=0,
                      d = 0,
                      max_p = 6,
                      max_q = 2,
                      trace=True,
                      scoring='mae',      # mae is used as this is also done for RNN network
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)       # multiple jobs can only be done with gridsearch
model.summary()


# ARIMA optimizes the AIC term. Let's convert it back to the Mean Absolute Error (MAE) for comparison with the LSTM model. 

# In[90]:


import numpy as np
print(f"The AIC equals {model.aic():.2f}")
print(f"The MAE equals {np.mean(np.abs(model.resid())):.2f}")


# Let's assume heat is mostly transported by diffusion (Fick's law). Under this assumption the most correlating temperatures should be neighbouring towns.
# The temperature in these town can be used as exogenous component. I compute the cross correlation matrix and grab the 4 most correlating components.
# The model is developed with respect to the errors so the coefficients have the usual interprestation see [Hyndman](https://robjhyndman.com/hyndsight/arimax/).

# In[91]:


# compute temperature per city
import matplotlib
df_city = df.unstack(level='date').T
#df_city.head()
df_cor = df_city.corr()
# The five most correlating temperatures for the city of Amsterdam
top_six = df_cor[df_cor.index == 'Amsterdam'].T.nlargest(6, ['Amsterdam']).index[0:6].to_list()
print(top_six)


# The results above make sense. Brussels is closer to Amsterdam than Paris. There is a sea between London and Amsterdam.  
# The exogenous components is computed via a shift.

# In[92]:


df_data = (df_city[top_six[1:]].shift(1)
                               .assign(Amsterdam=df_city.Amsterdam)
                               .dropna()) 
df_data.head()


# In[ ]:


# If sampling is put at monthly, the best results are obtained with a constant trend.


# In[99]:


model = pm.auto_arima(df_city.Amsterdam, exogenous=df_city[top_six[1:]],
                      test='adf',       # use adftest to find optimal 'd'
                      # trend
                      trend = 'c',      # we assumme a linear trend with time  (https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html#statsmodels.tsa.statespace.sarimax.SARIMAX)
                      # seasonal settings
                      m=12,               # m observations per seasonal cycle
                      max_P = 3,
                      max_Q = 2,
                      D = 0,
                      seasonal=True,      # Seasonality, temperature is highly seasonal!
                      n_jobs = 1,         # Number of jobs in parallel
                      # regular settings
                      #start_p=4, 
                      #start_q=0,
                      d = 0,
                      max_p = 6,
                      max_q = 2,
                      trace=True,
                      scoring='mae',      # mae is used as this is also done for RNN network
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)       # multiple jobs can only be done with gridsearch
model.summary()


# In[100]:


print(f"The AIC equals {model.aic():.2f}")
print(f"The MAE equals {np.mean(np.abs(model.resid())):.2f}")


# Convert this file to python

# In[97]:


get_ipython().system('jupyter nbconvert --to script arma.ipynb')

