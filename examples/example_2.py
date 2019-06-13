import numpy as np
import tensorflow as tf

from cond_rnn import ConditionalRNN

NUM_SAMPLES = 1000
TIME_STEPS = 10
INPUT_DIM = 1
NUM_CELLS = 24
DTYPE = tf.float32
COND_1_DIM = 2
COND_2_DIM = 3
NUM_CLASSES = COND_1_DIM + COND_2_DIM + 1


def create_conditions(input_dim):
    return np.array([np.random.choice([0, 1], size=input_dim, replace=True) for _ in range(NUM_SAMPLES)])
    # conditions = np.zeros(shape=[NUM_SAMPLES, input_dim])
    # for i, kk in enumerate(conditions):
    #     xx = np.random.randint(low=0, high=input_dim)
    #     kk[xx] = 1
    # return conditions


def acc(a: np.array, b: np.array):
    return np.mean(a.argmax(axis=1) == b.argmax(axis=1))


def main():
    sess = tf.Session()

    # Placeholders.
    inputs = tf.placeholder(name='inputs', dtype=DTYPE, shape=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM))
    targets = tf.placeholder(name='targets', dtype=DTYPE, shape=(NUM_SAMPLES, NUM_CLASSES))

    cond_1 = tf.placeholder(name='conditions_1', dtype=DTYPE, shape=(NUM_SAMPLES, COND_1_DIM))
    cond_2 = tf.placeholder(name='conditions_2', dtype=DTYPE, shape=(NUM_SAMPLES, COND_2_DIM))
    cond_list = [cond_1, cond_2]

    # Conditional RNN.
    outputs = ConditionalRNN(NUM_CELLS, cell='LSTM', cond=cond_list, dtype=DTYPE)(inputs)

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
    test_cond_1 = train_cond_1 = create_conditions(input_dim=COND_1_DIM)
    test_cond_2 = train_cond_2 = create_conditions(input_dim=COND_2_DIM)

    test_targets = np.sum(np.hstack([test_cond_2, test_cond_1]), axis=1)
    train_targets = np.sum(np.hstack([train_cond_2, train_cond_1]), axis=1)
    train_targets = tf.keras.utils.to_categorical(train_targets, num_classes=NUM_CLASSES)
    test_targets = tf.keras.utils.to_categorical(test_targets, num_classes=NUM_CLASSES)

    # Define the binding between placeholders and real data.
    train_feed_dict = {inputs: train_inputs, targets: train_targets, cond_1: train_cond_1, cond_2: train_cond_2}
    test_feed_dict = {inputs: test_inputs, targets: test_targets, cond_1: test_cond_1, cond_2: test_cond_2}

    # Main loop. Optimize then evaluate.
    for epoch in range(100000):
        sess.run(optimizer, train_feed_dict)
        if epoch % 100 == 0:
            train_outputs, train_loss = sess.run([outputs, cost], train_feed_dict)
            test_outputs, test_loss = sess.run([outputs, cost], test_feed_dict)
            train_acc = acc(train_outputs, train_targets)
            test_acc = acc(test_outputs, test_targets)
            print(f'[{str(epoch).zfill(4)}] train cost = {train_loss:.4f}, '
                  f'train acc = {train_acc:.2f}, test cost = {test_loss:.4f}, '
                  f'test acc = {test_acc:.2f}.')


if __name__ == '__main__':
    main()
