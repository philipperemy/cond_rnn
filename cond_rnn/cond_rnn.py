import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Dense, Layer, GRU, RNN, Lambda


class ConditionalRNN(Layer):

    # Arguments to the RNN like return_sequences, return_state...
    def __init__(self, rnn=LSTM, units=10, *args, **kwargs):
        """
        Conditional RNN. Conditions time series on categorical data.
        :param units: int, The number of units in the RNN Cell
        :param rnn_type: string, cell class or object (pre-instantiated). In the case of string, 'GRU',
        'LSTM' and 'RNN' are supported.
        :param args: Any parameters of the tf.keras.layers.RNN class, such as return_sequences,
        return_state, stateful, unroll...
        """
        super(ConditionalRNN, self).__init__()
        self.max_num_conditions = 10
        self.rnn = rnn(units=units, *args, **kwargs)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        # single cond
        self.dense_init_single = Dense(units=self.rnn.units)
        # multi cond
        self.dense_init_multi = []
        for i in range(self.max_num_conditions):
            self.dense_init_multi.append(Dense(units=self.rnn.units))
        self.multi_cond_p = Dense(1)

        self.expand_dims_1 = Lambda(lambda x: K.expand_dims(x, axis=0))
        self.tile_1 = Lambda(lambda x: K.tile(x, [2, 1, 1]))

        self.stack_1 = Lambda(lambda x: K.stack(x, axis=-1))
        self.squeeze_layer = Lambda(lambda x: K.squeeze(x, axis=-1))
        self.unstack_layer = Lambda(lambda x: tf.unstack(x, axis=0))

    @property
    def go_backwards(self):
        return self.rnn.go_backwards

    @property
    def return_sequences(self):
        return self.rnn.return_sequences

    @property
    def return_state(self):
        return self.rnn.return_state

    def _standardize_condition(self, initial_cond):
        initial_cond_shape = initial_cond.shape
        if len(initial_cond_shape) == 2:
            # initial_cond = tf.expand_dims(initial_cond, axis=0)
            initial_cond = self.expand_dims_1(initial_cond)
        first_cond_dim = initial_cond.shape[0]
        if isinstance(self.rnn, LSTM):
            if first_cond_dim == 1:
                # initial_cond = tf.tile(initial_cond, [2, 1, 1])
                initial_cond = self.tile_1(initial_cond)
            elif first_cond_dim != 2:
                raise Exception('Initial cond should have shape: [2, batch_size, hidden_size] '
                                'or [batch_size, hidden_size]. Shapes do not match.', initial_cond_shape)
        elif isinstance(self.rnn, (GRU, RNN)):
            if first_cond_dim != 1:
                raise Exception('Initial cond should have shape: [1, batch_size, hidden_size] '
                                'or [batch_size, hidden_size]. Shapes do not match.', initial_cond_shape)
        else:
            raise Exception('Only GRU, LSTM and RNN are supported as cells.')
        return initial_cond

    def call(self, inputs, trainable=None, **kwargs):
        """
        :param trainable:
        :param inputs: List of n elements:
                    - [0] 3-D Tensor with shape [batch_size, time_steps, input_dim]. The inputs.
                    - [1:] list of tensors with shape [batch_size, cond_dim]. The conditions.
        In the case of a list, the tensors can have a different cond_dim.
        :return: outputs, states or outputs (if return_state=False)
        """
        assert isinstance(inputs, (list, tuple)) and len(inputs) >= 2
        x = inputs[0]
        cond = inputs[1:]
        if len(cond) > 1:  # multiple conditions.
            init_state_list = [self.dense_init_multi[i](self._standardize_condition(c)) for i, c in enumerate(cond)]
            multi_cond_state = self.multi_cond_p(self.stack_1(init_state_list))
            multi_cond_state = self.squeeze_layer(multi_cond_state)
            init_state = self.unstack_layer(multi_cond_state)
        else:
            cond = self._standardize_condition(cond[0])
            if cond is not None:
                init_state = self.unstack_layer(self.dense_init_single(cond))
        return self.rnn(x, initial_state=init_state, **kwargs)

    def get_config(self):
        return self.rnn.get_config()
