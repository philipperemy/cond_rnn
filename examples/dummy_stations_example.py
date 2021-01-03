# 10 stations
# 365 days
# 3 continuous variables A and B => C is target.
# 2 conditions dim=5 and dim=1
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from cond_rnn import ConditionalRNN

stations = 10  # 10 stations.
time_steps = 365  # 365 days.
continuous_variables_per_station = 3  # A,B,C where C is the target.
condition_variables_per_station = 2  # 2 variables of dim 5 and 1.
condition_dim_1 = 5
condition_dim_2 = 1

np.random.seed(123)
continuous_data = np.random.uniform(size=(stations, time_steps, continuous_variables_per_station))
condition_data_1 = np.zeros(shape=(stations, condition_dim_1))
condition_data_1[:, 0] = 1  # dummy.
condition_data_2 = np.random.uniform(size=(stations, condition_dim_2))

window = 50  # we split series in 50 days (look-back window)

x, y, c1, c2 = [], [], [], []
for i in range(window, continuous_data.shape[1]):
    x.append(continuous_data[:, i - window:i])
    y.append(continuous_data[:, i])
    c1.append(condition_data_1)  # just replicate.
    c2.append(condition_data_2)  # just replicate.

# now we have (batch_dim, station_dim, time_steps, input_dim).
x = np.array(x)
y = np.array(y)
c1 = np.array(c1)
c2 = np.array(c2)

print(x.shape, y.shape, c1.shape, c2.shape)

# let's collapse the station_dim in the batch_dim.
x = np.reshape(x, [-1, window, x.shape[-1]])
y = np.reshape(y, [-1, y.shape[-1]])
c1 = np.reshape(c1, [-1, c1.shape[-1]])
c2 = np.reshape(c2, [-1, c2.shape[-1]])

print(x.shape, y.shape, c1.shape, c2.shape)

model = Sequential(layers=[
    ConditionalRNN(10, cell='GRU'),  # num_cells = 10
    Dense(units=1, activation='linear')  # regression problem.
])

model.compile(optimizer='adam', loss='mse')
model.fit(x=[x, c1, c2], y=y, epochs=2, validation_split=0.2)
