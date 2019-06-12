import tensorflow as tf


def _get_tensor_shape(t):
    return t.get_shape().as_list()


class ConditionalRNN:

    def __init__(self,
                 units,
                 cell=tf.keras.layers.LSTMCell,  # Or LSTMCell(units).
                 initial_cond=None,  # Condition [2, batch_size, hidden_size]
                 *args, **kwargs):  # Arguments to the RNN like return_sequences, return_state...
        if isinstance(cell, str):
            if cell.upper() == 'GRU':
                cell = tf.keras.layers.GRUCell
            elif cell.upper() == 'LSTM':
                cell = tf.keras.layers.LSTMCell
            elif cell.upper() == 'RNN':
                cell = tf.keras.layers.SimpleRNNCell
            else:
                raise Exception('Only GRU, LSTM and RNN are supported as cells.')
        self._cell = cell if hasattr(cell, 'units') else cell(units=units)
        initial_cond_shape = _get_tensor_shape(initial_cond)
        if len(initial_cond_shape) == 2:
            initial_cond = tf.expand_dims(initial_cond, axis=0)
        first_cond_dim = _get_tensor_shape(initial_cond)[0]
        if isinstance(self._cell, tf.keras.layers.LSTMCell):
            if first_cond_dim == 1:
                initial_cond = tf.tile(initial_cond, [2, 1, 1])
            elif first_cond_dim != 2:
                raise Exception('Initial cond should have shape: [2, batch_size, hidden_size]\n'
                                'or [batch_size, hidden_size]. Shapes do not match.', initial_cond_shape)
        elif isinstance(self._cell, tf.keras.layers.GRUCell) or isinstance(self._cell, tf.keras.layers.SimpleRNNCell):
            if first_cond_dim != 1:
                raise Exception('Initial cond should have shape: [1, batch_size, hidden_size]\n'
                                'or [batch_size, hidden_size]. Shapes do not match.', initial_cond_shape)
        else:
            raise Exception('Only GRU, LSTM and RNN are supported as cells.')
        self.rnn = tf.keras.layers.RNN(cell=self._cell, *args, **kwargs)
        self.final_states = None
        self.init_state = None
        if initial_cond is not None:
            self.init_state = tf.keras.layers.Dense(units=units)(initial_cond)
            self.init_state = tf.unstack(self.init_state, axis=0)

    def __call__(self, inputs, *args, **kwargs):
        out = self.rnn(inputs, initial_state=self.init_state, *args, **kwargs)
        if self.rnn.return_state:
            outputs, h, c = out
            final_states = tf.stack([h, c])
            return outputs, final_states
        else:
            return out
