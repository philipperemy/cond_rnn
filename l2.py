import numpy as np
import tensorflow as tf

# https://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/

num_classes = 3
batch_size = 1000
time_steps = 10
input_dim = 1
hidden_size = 12

tf.Graph()
sess = tf.Session()

init_state_cond_np = np.zeros(shape=[batch_size, num_classes])
for i, kk in enumerate(init_state_cond_np):
    kk[i % num_classes] = 1
init_state_cond_np = np.tile(init_state_cond_np, [2, 1, 1])
targets = tf.constant(dtype=tf.float32, value=init_state_cond_np[0])

# inputs = tf.placeholder(dtype=tf.float32, shape=(None, time_steps, input_dim))
inputs = tf.constant(dtype=tf.float32, value=np.random.uniform(size=(batch_size, time_steps, input_dim)))
# init_state = tf.placeholder(tf.float32, [1, 2, batch_size, hidden_size])

init_state_cond = tf.constant(dtype=tf.float32, value=init_state_cond_np)

# -> [2, batch_size, hidden_size]
init_state = tf.keras.layers.Dense(units=hidden_size)(init_state_cond)

cell = tf.keras.layers.LSTMCell(units=hidden_size)

rnn = tf.keras.layers.RNN(cell=cell, dtype=tf.float32, return_state=True, return_sequences=True)

init_state = tf.unstack(init_state, axis=0)
outputs, h, c = rnn(inputs, initial_state=init_state)
final_states = tf.stack([h, c])

outputs = outputs[:, -1, :]  # last step

outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(outputs)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=targets))
optimizer = tf.train.AdamOptimizer().minimize(cost)

sess.run(tf.global_variables_initializer())

for epoch in range(1_000_000):
    sess.run(optimizer)
    if epoch % 10 == 0:
        o_ = sess.run(outputs)
        t_ = sess.run(targets)
        acc = np.mean(o_.argmax(axis=1) == t_.argmax(axis=1))
        print(f'cost = {sess.run(cost):.4f}, acc = {acc:.2f}.')
        print(o_[0], t_[0])
        print(o_[1], t_[1])
        print(o_[2], t_[2])

