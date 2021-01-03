import numpy as np
from single_cond_example import create_conditions
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

from cond_rnn import ConditionalRNN

NUM_SAMPLES = 10_000
INPUT_DIM = 1
NUM_CLASSES = 3
TIME_STEPS = 10
NUM_CELLS = 6


def main():
    model = Sequential(layers=[
        ConditionalRNN(NUM_CELLS, cell='LSTM', return_sequences=True, name='cond_rnn_0'),
        LSTM(NUM_CELLS),
        Dense(units=NUM_CLASSES, activation='softmax')
    ])

    # Define (real) data.
    train_inputs = np.random.uniform(size=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM))
    test_inputs = np.random.uniform(size=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM))
    test_targets = train_targets = create_conditions()

    model.compile(optimizer=Adam(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        x=[train_inputs, train_targets], y=train_targets,
        validation_data=([test_inputs, test_targets], test_targets),
        epochs=10
    )

    te_loss, te_acc = model.evaluate([test_inputs, test_targets], test_targets)
    assert abs(te_acc - 1) < 1e-5


if __name__ == '__main__':
    main()
