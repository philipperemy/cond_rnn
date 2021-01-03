import unittest

from tensorflow.keras import regularizers

from cond_rnn import ConditionalRNN


class Test(unittest.TestCase):

    def test_constructor(self):
        ConditionalRNN(5, cell='GRU', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))
        ConditionalRNN(5, cell='LSTM', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))
        ConditionalRNN(5, cell='RNN', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))
        ConditionalRNN(5, cell='RNN', return_sequences=True)
        ConditionalRNN(5, cell='LSTM', return_sequences=True)
        ConditionalRNN(5, cell='GRU', return_sequences=True)
