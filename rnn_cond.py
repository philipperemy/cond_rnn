import tensorflow as tf


class ConditionalRNN:

    def __init__(self,
                 units,
                 cell=tf.keras.layers.LSTMCell,  # Or LSTMCell(units).
                 initial_cond=None,  # Condition [2, batch_size, hidden_size]
                 dtype=tf.float32,
                 return_sequences=False,
                 go_backwards=False,
                 unroll=False,
                 ):
        self._cell = cell if hasattr(cell, 'units') else cell(units=units)
        self.rnn = tf.keras.layers.RNN(cell=self._cell,
                                       dtype=dtype,
                                       return_state=True,
                                       return_sequences=return_sequences,
                                       go_backwards=go_backwards,
                                       unroll=unroll)
        self.final_states = None
        self.init_state = None
        if initial_cond is not None:
            self.init_state = tf.keras.layers.Dense(units=units)(initial_cond)
            self.init_state = tf.unstack(self.init_state, axis=0)

    def __call__(self, inputs, *args, **kwargs):
        outputs, h, c = self.rnn(inputs, initial_state=self.init_state, *args, **kwargs)
        final_states = tf.stack([h, c])
        return outputs, final_states
