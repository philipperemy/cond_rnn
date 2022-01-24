import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, GRU

from cond_rnn import ConditionalRecurrent

model = Sequential()

forward_layer = ConditionalRecurrent(GRU(units=12, return_sequences=True))
backward_layer = ConditionalRecurrent(GRU(units=13, return_sequences=True, go_backwards=True))

# concat mode.
model.add(Bidirectional(
    layer=forward_layer,
    backward_layer=backward_layer
))
model.compile(loss='categorical_crossentropy')

NUM_SAMPLES = 100
TIME_STEPS = 10
INPUT_DIM = 3
NUM_CLASSES = 2
train_inputs = np.random.uniform(size=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM))
train_targets = np.zeros(shape=[NUM_SAMPLES, NUM_CLASSES])
assert model.predict(x=[train_inputs, train_targets]).shape == (NUM_SAMPLES, 10, 12 + 13)
