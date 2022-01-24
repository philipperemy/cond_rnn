import tensorflow as tf
from keras.utils import generic_utils
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Dense, GRU, RNN, Lambda, Wrapper


class ConditionalRecurrent(Wrapper):

    def __init__(self, layer, **kwargs):
        """
        Conditional Recurrent Wrapper.
        """
        super(ConditionalRecurrent, self).__init__(layer, **kwargs)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape=None):
        # single cond
        self.dense_init_single = Dense(units=self.layer.units)
        # multi cond
        max_num_conditions = 10
        self.dense_init_multi = []
        for i in range(max_num_conditions):
            self.dense_init_multi.append(Dense(units=self.layer.units))
        self.multi_cond_p = Dense(1)

        self.expand_dims_1 = Lambda(lambda x: K.expand_dims(x, axis=0))
        self.tile_1 = Lambda(lambda x: K.tile(x, [2, 1, 1]))

        self.stack_1 = Lambda(lambda x: K.stack(x, axis=-1))
        self.squeeze_layer = Lambda(lambda x: K.squeeze(x, axis=-1))
        self.unstack_layer = Lambda(lambda x: tf.unstack(x, axis=0))

    @property
    def go_backwards(self):
        return self.layer.go_backwards

    @property
    def return_sequences(self):
        return self.layer.return_sequences

    @property
    def return_state(self):
        return self.layer.return_state

    def _standardize_condition(self, initial_cond):
        initial_cond_shape = initial_cond.shape
        if len(initial_cond_shape) == 2:
            initial_cond = self.expand_dims_1(initial_cond)
        first_cond_dim = initial_cond.shape[0]
        if isinstance(self.layer, LSTM):
            if first_cond_dim == 1:
                initial_cond = self.tile_1(initial_cond)
            elif first_cond_dim != 2:
                raise Exception('Initial cond should have shape: [2, batch_size, hidden_size] '
                                'or [batch_size, hidden_size]. Shapes do not match.', initial_cond_shape)
        elif isinstance(self.layer, (GRU, RNN)):
            if first_cond_dim != 1:
                raise Exception('Initial cond should have shape: [1, batch_size, hidden_size] '
                                'or [batch_size, hidden_size]. Shapes do not match.', initial_cond_shape)
        else:
            raise Exception('Only GRU, LSTM and RNN are supported as cells.')
        return initial_cond

    def call(self, inputs, training=None, **kwargs):
        """
        :param inputs: List of n elements:
                    - [0] 3-D Tensor with shape [batch_size, time_steps, input_dim]. The inputs.
                    - [1:] list of tensors with shape [batch_size, cond_dim]. The conditions.
        :param training: Python boolean indicating whether the layer should behave in training mode or
        in inference mode. This argument is passed to the wrapped layer (only if the layer supports this argument).
        In the case of a list, the tensors can have a different cond_dim.
        :return: outputs, states or outputs (if return_state=False)
        """
        assert isinstance(inputs, (list, tuple)) and len(inputs) >= 2
        if generic_utils.has_arg(self.layer.call, 'training'):
            kwargs['training'] = training
        x = inputs[0]
        cond = inputs[1:]
        init_state = None
        if len(cond) > 1:  # multiple conditions.
            init_state_list = [self.dense_init_multi[i](self._standardize_condition(c)) for i, c in enumerate(cond)]
            multi_cond_state = self.multi_cond_p(self.stack_1(init_state_list))
            multi_cond_state = self.squeeze_layer(multi_cond_state)
            init_state = self.unstack_layer(multi_cond_state)
        else:
            cond = self._standardize_condition(cond[0])
            if cond is not None:
                init_state = self.unstack_layer(self.dense_init_single(cond))
        return self.layer(x, initial_state=init_state, **kwargs)

    def get_config(self):
        return super(ConditionalRecurrent, self).get_config()
