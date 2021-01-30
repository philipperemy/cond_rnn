# Temperature
Performance benchmark of cond_rnn for predicting the daily temperature in Amsterdam.  
Cond_rnn is compared against; an ARMA model, an ARMAX model and a pure autoregressive LSTM.
The results are as follows;
* ARMA model: Mean Absolute Error (MAE) of 1.47 degrees
* ARMAX model with 30 cities: MAE of 1.25 degrees
* Pure autoregressive LSTM model: MAE of 1.46 degrees
* Cond_rnn with 30 cities: MAE 0.87 degrees
## Setup notes
Extract city_temperature.csv.zip and convert python to notebook, 
e.g. ```jupytext --to notebook lstm.py```.