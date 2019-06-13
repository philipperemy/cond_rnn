import tensorflow as tf


def _get_tensor_shape(t):
    return t.get_shape().as_list()


class ConditionalRNN:

    def __init__(self,
                 units,
                 cell=tf.keras.layers.LSTMCell,
                 cond=None,
                 *args, **kwargs):  # Arguments to the RNN like return_sequences, return_state...
        """
        Conditional RNN. Conditions time series on categorical data.
        :param units: int, The number of units in the RNN Cell
        :param cell: string, cell class or object (pre-instantiated). In the case of string, 'GRU',
        'LSTM' and 'RNN' are supported.
        :param cond (optional): Tensor or list of tensors with shape [batch_size, cond_dim].
        In the case of a list, the tensors can have a different cond_dim.
        :param args: Any parameters of the tf.keras.layers.RNN class, such as return_sequences,
        return_state, stateful, unroll...
        """
        self.final_states = None
        self.init_state = None
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
        if isinstance(cond, list):  # multiple conditions.
            cond = [self._standardize_condition(cond) for cond in cond]
            init_state_list = []
            for cond in cond:
                init_state_list.append(tf.keras.layers.Dense(units=units)(cond))
            multi_cond_projector = tf.layers.Dense(1, activation=None, use_bias=True)
            multi_cond_state = multi_cond_projector(tf.stack(init_state_list, axis=-1))
            multi_cond_state = tf.squeeze(multi_cond_state, axis=-1)
            self.init_state = tf.unstack(multi_cond_state, axis=0)
        else:
            cond = self._standardize_condition(cond)
            if cond is not None:
                self.init_state = tf.keras.layers.Dense(units=units)(cond)
                self.init_state = tf.unstack(self.init_state, axis=0)
        self.rnn = tf.keras.layers.RNN(cell=self._cell, *args, **kwargs)

    def _standardize_condition(self, initial_cond):
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
        return initial_cond

    def __call__(self, inputs, *args, **kwargs):
        """
        :param inputs: 3-D Tensor with shape [batch_size, time_steps, input_dim].
        :return: outputs, states or outputs (if return_state=False)
        """
        out = self.rnn(inputs, initial_state=self.init_state, *args, **kwargs)
        if self.rnn.return_state:
            outputs, h, c = out
            final_states = tf.stack([h, c])
            return outputs, final_states
        else:
            return out
