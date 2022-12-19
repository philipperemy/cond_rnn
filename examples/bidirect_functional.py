import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Bidirectional

from cond_rnn import ConditionalRecurrent

forward_layer = ConditionalRecurrent(LSTM(units=12, return_sequences=True))
backward_layer = ConditionalRecurrent(LSTM(units=13, return_sequences=True, go_backwards=True))

NUM_SAMPLES = 100
TIME_STEPS = 10
INPUT_DIM = 3
NUM_CLASSES = 2

inputs = (
    Input(shape=(TIME_STEPS, INPUT_DIM)),
    Input(shape=(NUM_CLASSES,))  # conditions.
)

outputs = Bidirectional(
    layer=forward_layer,
    backward_layer=backward_layer,
)(inputs=inputs)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy')

model.summary()

train_inputs = np.random.uniform(size=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM))
train_targets = np.zeros(shape=[NUM_SAMPLES, NUM_CLASSES])
assert model.predict(x=[train_inputs, train_targets]).shape == (NUM_SAMPLES, 10, 12 + 13)
