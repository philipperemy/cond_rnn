import unittest

from tensorflow.keras import regularizers
from tensorflow.keras.layers import GRU, LSTM, SimpleRNN

from cond_rnn import Conditional


class Test(unittest.TestCase):

    def test_constructor(self):
        Conditional(GRU(5, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        Conditional(LSTM(5, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        Conditional(GRU(5, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        Conditional(SimpleRNN(5, return_sequences=True))
        Conditional(LSTM(5, return_sequences=True))
        Conditional(GRU(5, return_sequences=True))
