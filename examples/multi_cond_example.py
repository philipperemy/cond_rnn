import numpy as np
import tensorflow as tf

from cond_rnn import ConditionalRNN

NUM_SAMPLES = 10000
TIME_STEPS = 10
INPUT_DIM = 1
NUM_CELLS = 24
COND_1_DIM = 2
COND_2_DIM = 3
NUM_CLASSES = COND_1_DIM + COND_2_DIM + 1


def create_conditions(input_dim):
    return np.array([np.random.choice([0, 1], size=input_dim, replace=True) for _ in range(NUM_SAMPLES)], dtype=float)


def main():
    class MySimpleModel(tf.keras.Model):
        def __init__(self):
            super(MySimpleModel, self).__init__()
            self.cond = ConditionalRNN(NUM_CELLS, cell='GRU', dtype=tf.float32)
            self.out = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax')

        def call(self, inputs, **kwargs):
            o = self.cond(inputs)
            o = self.out(o)
            return o

    model = MySimpleModel()

    # Define (real) data.
    train_inputs = np.random.uniform(size=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM))
    test_inputs = np.random.uniform(size=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM))
    test_cond_1 = train_cond_1 = create_conditions(input_dim=COND_1_DIM)
    test_cond_2 = train_cond_2 = create_conditions(input_dim=COND_2_DIM)

    test_targets = np.sum(np.hstack([test_cond_2, test_cond_1]), axis=1)
    train_targets = np.sum(np.hstack([train_cond_2, train_cond_1]), axis=1)
    train_targets = tf.keras.utils.to_categorical(train_targets, num_classes=NUM_CLASSES)
    test_targets = tf.keras.utils.to_categorical(test_targets, num_classes=NUM_CLASSES)

    model.call([train_inputs, train_cond_1, train_cond_2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=[train_inputs, train_cond_1, train_cond_2], y=train_targets,
              validation_data=([test_inputs, test_cond_1, test_cond_2], test_targets),
              epochs=10)

    te_loss, te_acc = model.evaluate([test_inputs, test_cond_1, test_cond_2], test_targets)
    assert abs(te_acc - 1) < 1e-5


if __name__ == '__main__':
    main()
