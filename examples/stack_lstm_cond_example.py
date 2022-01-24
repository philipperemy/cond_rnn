import numpy as np
from single_cond_example import create_conditions
from tensorflow.keras import Input, Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM

from cond_rnn import Conditional

NUM_SAMPLES = 10_000
INPUT_DIM = 1
NUM_CLASSES = 3
TIME_STEPS = 10
NUM_CELLS = 6


def main():
    # Cannot do stacked ConditionalRNN with Sequential. Have to rely on the functional API. See below.
    model = Sequential(layers=[
        Conditional(NUM_CELLS, cell='LSTM', return_sequences=True, name='cond_rnn_0'),
        LSTM(NUM_CELLS),
        Dense(units=NUM_CLASSES, activation='softmax')
    ])

    # Stacked ConditionalRNN with the functional API.
    i = Input(shape=[TIME_STEPS, INPUT_DIM], name='input_0')
    c = Input(shape=[NUM_CLASSES], name='input_1')
    # add the condition tensor here.
    x = Conditional(NUM_CELLS, cell='LSTM', return_sequences=True, name='cond_rnn_0')([i, c])
    # and here too.
    x = Conditional(NUM_CELLS, cell='LSTM', return_sequences=False, name='cond_rnn_1')([x, c])
    x = Dense(units=NUM_CLASSES, activation='softmax')(x)
    model2 = Model(inputs=[i, c], outputs=[x])

    # Define (dummy) data.
    train_inputs = np.random.uniform(size=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM))
    test_inputs = np.random.uniform(size=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM))
    test_targets = train_targets = create_conditions()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        x=[train_inputs, train_targets], y=train_targets,
        validation_data=([test_inputs, test_targets], test_targets),
        epochs=3
    )

    model2.fit(
        x=[train_inputs, train_targets], y=train_targets,
        validation_data=([test_inputs, test_targets], test_targets),
        epochs=3
    )

    assert abs(model.evaluate([test_inputs, test_targets], test_targets)[1] - 1) < 1e-5
    assert abs(model2.evaluate([test_inputs, test_targets], test_targets)[1] - 1) < 1e-5


if __name__ == '__main__':
    main()
