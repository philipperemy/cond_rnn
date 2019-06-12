import numpy as np
import tensorflow as tf

from rnn_cond import ConditionalRNN

NUM_CLASSES = 3
NUM_SAMPLES = 1000
TIME_STEPS = 10
INPUT_DIM = 1
NUM_CELLS = 12
DTYPE = tf.float32


# https://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/

def create_conditions():
    conditions = np.zeros(shape=[NUM_SAMPLES, NUM_CLASSES])
    for i, kk in enumerate(conditions):
        kk[i % NUM_CLASSES] = 1
    conditions = np.tile(conditions, [2, 1, 1])
    return conditions


def main():
    tf.Graph()
    sess = tf.Session()

    inputs = tf.placeholder(name='inputs', dtype=DTYPE, shape=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM))
    targets = tf.placeholder(name='targets', dtype=DTYPE, shape=(NUM_SAMPLES, NUM_CLASSES))
    cond = tf.placeholder(name='conditions', dtype=DTYPE, shape=[2, NUM_SAMPLES, NUM_CLASSES])

    rnn = ConditionalRNN(NUM_CELLS, initial_cond=cond)
    outputs, _ = rnn(inputs)

    outputs = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax')(outputs)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=targets))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    sess.run(tf.global_variables_initializer())

    conditions = create_conditions()

    train_inputs = np.random.uniform(size=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM))
    train_targets = conditions[0]

    test_inputs = np.random.uniform(size=(NUM_SAMPLES, TIME_STEPS, INPUT_DIM)) * 2
    # test_inputs = np.random.normal(loc=0, scale=2, size=(batch_size, time_steps, input_dim))
    test_targets = conditions[0]

    train_feed_dict = {inputs: train_inputs, targets: train_targets, cond: conditions}
    test_feed_dict = {inputs: test_inputs, targets: test_targets, cond: conditions}

    for epoch in range(1_000_000):

        sess.run(optimizer, train_feed_dict)
        if epoch % 10 == 0:
            o_, t_ = sess.run([outputs, targets], train_feed_dict)
            acc = np.mean(o_.argmax(axis=1) == t_.argmax(axis=1))
            print(f'train cost = {sess.run(cost, train_feed_dict):.4f}, train acc = {acc:.2f}.')
            # print(o_[0], t_[0])
            # print(o_[1], t_[1])
            # print(o_[2], t_[2])

            o_, t_ = sess.run([outputs, targets], test_feed_dict)
            acc = np.mean(o_.argmax(axis=1) == t_.argmax(axis=1))
            print(f'test cost = {sess.run(cost, train_feed_dict):.4f}, test acc = {acc:.2f}.')
            # print(o_[0], t_[0])
            # print(o_[1], t_[1])
            # print(o_[2], t_[2])


if __name__ == '__main__':
    main()
