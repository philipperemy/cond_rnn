import numpy as np
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from cond_rnn import Conditional

NUM_SAMPLES = 10_000
INPUT_DIM = 1
NUM_CLASSES = 3
TIME_STEPS = 10
NUM_CELLS = 12


def create_conditions():
    conditions = np.zeros(shape=[NUM_SAMPLES, NUM_CLASSES])
    for i, kk in enumerate(conditions):
        kk[i % NUM_CLASSES] = 1
    return conditions


def main():
    model = Sequential(layers=[
        Conditional(GRU(10)),
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
