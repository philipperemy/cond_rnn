import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from cond_rnn import ConditionalRNN

# This is an example with dummy to explain how to use CondRNN.
# ______________________________________________________________________________________
# In this example, we have gathered data from 100 stations on a period of 365 days.
# Each station has a vector of 3 quantities (e.g. humidity, rainfall, temperature) per day.
# In addition, it has a variable that does not depend on time. It can be for example the temperature average
# over a period of 50 years. We call it a condition.
# We can also add the ID of the station in the data (as one-hot). This is another condition.

num_stations = 100  # number of stations.
time_steps = 365  # time dimensions (e.g. 365 days).
input_dim = 3  # number of variables depending on time (day axis), per station.
condition_dim_1 = num_stations  # one-hot vector to identify the station.
condition_dim_2 = 1  # average input for the station (data leakage in this example).

np.random.seed(123)

# generate the variables depending on time for each station.
continuous_data = np.cumsum(np.random.uniform(low=-1, high=1, size=(num_stations, time_steps, input_dim)), axis=1)
continuous_data /= np.max(np.abs(continuous_data))

plt.plot(continuous_data[0, ..., 0])
plt.plot(continuous_data[0, ..., 1])
plt.plot(continuous_data[0, ..., 2])
plt.legend(['Input 0', 'Input 1', 'Input 2'])
plt.title('Station 0 - continuous time-dependent inputs')
plt.xlabel('time (days)')
plt.ylabel('input (no unit)')
plt.show()

condition_data_1 = np.diag(num_stations * [1])
condition_data_2 = np.mean(continuous_data, axis=(1, 2))

window = 50  # we split series in 50 days (look-back window)

x, y, c1, c2 = [], [], [], []
for s in range(num_stations):
    for t in range(window, continuous_data.shape[1]):
        x.append(continuous_data[s][t - window:t])
        y.append(continuous_data[s][t][-1])
        c1.append(condition_data_1[s])
        c2.append(condition_data_2[s])

# now we have (batch_dim * station_dim, time_steps, input_dim).
x = np.array(x)
y = np.array(y)
c1 = np.array(c1)
c2 = np.array(c2)

print(x.shape, y.shape, c1.shape, c2.shape)

model = Sequential(layers=[
    ConditionalRNN(10, cell='GRU'),  # num_cells = 10
    Dense(units=1, activation='linear')  # regression problem.
])

model.compile(optimizer='adam', loss='mse')
model.fit(x=[x, c1, c2], y=y, epochs=20, validation_split=0.2)
