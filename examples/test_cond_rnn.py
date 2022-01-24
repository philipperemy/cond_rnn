import unittest

import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import GRU, SimpleRNN
from tensorflow.keras.models import load_model

from cond_rnn import ConditionalRecurrent
from single_cond_example import create_conditions


class Test(unittest.TestCase):

    def test_constructor(self):
        ConditionalRecurrent(GRU(5, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        ConditionalRecurrent(LSTM(5, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        ConditionalRecurrent(GRU(5, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        ConditionalRecurrent(SimpleRNN(5, return_sequences=True))
        ConditionalRecurrent(LSTM(5, return_sequences=True))
        ConditionalRecurrent(GRU(5, return_sequences=True))

    def test_serialize(self):
        num_samples = 10_000
        input_dim = 1
        num_classes = 3
        time_steps = 10
        num_cells = 6
        # Stacked ConditionalRNN with the functional API.
        i = Input(shape=[time_steps, input_dim], name='input_0')
        c = Input(shape=[num_classes], name='input_1')
        # add the condition tensor here.
        x = ConditionalRecurrent(LSTM(num_cells, return_sequences=True, name='cond_rnn_0'))([i, c])
        # and here too.
        x = ConditionalRecurrent(LSTM(num_cells, return_sequences=False, name='cond_rnn_1'))([x, c])
        x = Dense(units=num_classes, activation='softmax')(x)
        model = Model(inputs=[i, c], outputs=[x])

        # Define data.
        test_inputs = np.random.uniform(size=(num_samples, time_steps, input_dim))
        test_targets = create_conditions()

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # load_model will not work with Sequential. You have to use the Functional API!
        checkpoint_name = 'test_model.h5'
        pred1 = model.predict([test_inputs, test_targets])
        model.save(checkpoint_name)
        model_h5 = load_model(checkpoint_name, custom_objects={'ConditionalRecurrent': ConditionalRecurrent})
        pred2 = model_h5.predict([test_inputs, test_targets])
        np.testing.assert_almost_equal(pred1, pred2)
