import numpy as np
import tensorflow as tf

from cond_rnn import ConditionalRNN

NUM_CLASSES = 3
NUM_SAMPLES = 1000
TIME_STEPS = 10
INPUT_DIM = 1
NUM_CELLS = 12
DTYPE = tf.float32


def create_conditions():
    conditions = np.zeros(shape=[NUM_SAMPLES, NUM_CLASSES])
    for i, kk in enumerate(conditions):
        kk[i % NUM_CLASSES] = 1
    return conditions


def acc(a: np.array, b: np.array):
    return np.mean(a.argmax(axis=1) == b.argmax(axis=1))


def main():
    sess = tf.Session()

    # Placeholders.
    inputs = tf.placeholder(name='inputs', dtype=DTYPE, shape=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM))
    targets = tf.placeholder(name='targets', dtype=DTYPE, shape=(NUM_SAMPLES, NUM_CLASSES))
    cond = tf.placeholder(name='conditions', dtype=DTYPE, shape=(NUM_SAMPLES, NUM_CLASSES))

    # Conditional RNN.
    outputs = ConditionalRNN(NUM_CELLS, cell='RNN', dtype=DTYPE)(inputs, cond)

    # Classification layer.
    outputs = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax')(outputs)

    # Loss + Optimizer.
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=targets))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Initialize variables (tensorflow)
    sess.run(tf.global_variables_initializer())

    # Define (real) data.
    train_inputs = np.random.uniform(size=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM))
    test_inputs = np.random.uniform(size=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM)) * 2
    test_targets = train_targets = create_conditions()

    # Define the binding between placeholders and real data.
    train_feed_dict = {inputs: train_inputs, targets: train_targets, cond: train_targets}
    test_feed_dict = {inputs: test_inputs, targets: test_targets, cond: test_targets}

    # Main loop. Optimize then evaluate.
    for epoch in range(1000):
        sess.run(optimizer, train_feed_dict)
        if epoch % 10 == 0:
            train_outputs, train_loss = sess.run([outputs, cost], train_feed_dict)
            test_outputs, test_loss = sess.run([outputs, cost], test_feed_dict)
            train_acc = acc(train_outputs, train_targets)
            test_acc = acc(test_outputs, test_targets)
            print(f'[{str(epoch).zfill(4)}] train cost = {train_loss:.4f}, '
                  f'train acc = {train_acc:.2f}, test cost = {test_loss:.4f}, '
                  f'test acc = {test_acc:.2f}.')


if __name__ == '__main__':
    main()
