import numpy as np
import tensorflow as tf

# https://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/

tf.enable_eager_execution()

batch_size = 3
time_steps = 10
input_dim = 1
hidden_size = 12
# inputs = tf.placeholder(dtype=tf.float32, shape=(None, time_steps, input_dim))
inputs = tf.constant(dtype=tf.float32, value=np.random.uniform(size=(batch_size, time_steps, input_dim)))
# init_state = tf.placeholder(tf.float32, [1, 2, batch_size, hidden_size])
init_state = tf.constant(dtype=tf.float32, value=np.random.uniform(size=[2, batch_size, hidden_size]))

cell = tf.keras.layers.LSTMCell(units=hidden_size)

rnn = tf.keras.layers.RNN(cell=cell, dtype=tf.float32, return_state=True, return_sequences=True)

init_state = tf.unstack(init_state, axis=0)
outputs, h, c = rnn(inputs, initial_state=init_state)
final_states = tf.stack([h, c])

outputs = tf.keras.layers.Dense(units=1)(outputs)

print(outputs)
