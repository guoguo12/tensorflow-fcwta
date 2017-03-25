import tensorflow as tf


class FullyConnectedWTA:
    """Fully-connected winner-take-all autoencoder."""

    def __init__(self,
                 input_dim,
                 sparsity=0.05,
                 hidden_units=24,
                 encode_layers=3,
                 optimizer=tf.train.AdamOptimizer,
                 learning_rate=1e-2,
                 name='FCWTA'):
        """Create the model.

        Args:
          input_dim: the dimensionality of the input data.
          sparsity: the lifetime sparsity constraint to enforce.
          hidden_units: the number of units in each ReLU (encode) layer, and
            also the dimensionality of the encoded data.
          encode_layers: the number ReLU (encode) layers.
          optimizer: a TensorFlow optimizer op that takes only a learning rate.
          learning_rate: the learning rate to train with.
          name: the name of the variable scope to use.
        """
        self.input_dim = input_dim
        self.sparsity = sparsity
        self.hidden_units = hidden_units
        self.encode_layers = encode_layers
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.name = name
        self._initialize_vars()

    def _initialize_vars(self):
        """Sets up the training graph."""
        with tf.variable_scope(self.name):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.input = tf.placeholder(tf.float32, shape=[None, self.input_dim])

            encoded = self.input
            for i in range(self.encode_layers):
                encoded = tf.layers.dense(
                    encoded,
                    self.hidden_units,
                    activation=tf.nn.relu,
                    name='relu_{}'.format(i))
            self.encoded = encoded

            encoded_t = tf.transpose(encoded)
            enc_shape = tf.shape(encoded_t)

            k = tf.to_int32(self.sparsity * tf.to_float(enc_shape[1]))
            top_values, _ = tf.nn.top_k(encoded_t, k=k, sorted=False)
            mask = tf.where(encoded_t < tf.reduce_min(top_values, axis=1, keep_dims=True),
                            tf.zeros(enc_shape, tf.float32),
                            tf.ones(enc_shape, tf.float32))
            sparse_encoded = encoded * tf.transpose(mask)

            decoded = tf.layers.dense(sparse_encoded,
                                      self.input_dim,
                                      name='linear')
            self.decoded = decoded

            self.loss = tf.reduce_sum(tf.square(decoded - self.input))
            self.optimizer_op = self.optimizer(self.learning_rate).minimize(
                self.loss, self.global_step)

            self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, input, forward_only=False):
        """Run a step of the model, feeding the given inputs.

        Args:
          session: TensorFlow session to use.
          input: NumPy array to feed as input.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A tuple containing the reconstruction and the (summed) squared loss.

        Raises:
          ValueError: if dimensionality of input disagrees with the input_dim
          provided in the constructor.
        """
        if input.shape[1] != self.input_dim:
            raise ValueError('Dimensionality of input must equal the input_dim'
                             'provided in the constructor, {} != {}.'.format(
                                input.shape[1], self.input_dim))

        if forward_only:
            decoded, loss = session.run(
                [self.decoded, self.loss],
                feed_dict={self.input: input})
        else:
            decoded, loss, _ = session.run(
                [self.decoded, self.loss, self.optimizer_op],
                feed_dict={self.input: input})
        return decoded, loss

    def encode(self, session, input):
        """Encode the given inputs.

        Args:
          session: TensorFlow session to use.
          input: NumPy array to feed as input.

        Returns:
          The encoded data, with shape (input.shape[1], hidden_units).

        Raises:
          ValueError: if dimensionality of input disagrees with the input_dim
          provided in the constructor.
        """
        if input.shape[1] != self.input_dim:
            raise ValueError('Dimensionality of input must equal the input_dim'
                             'provided in the constructor, {} != {}.'.format(
                                input.shape[1], self.input_dim))
        return session.run(self.encoded, feed_dict={self.input: input})

    def get_dictionary(self, session):
        """Fetch the learned code dictionary.

        Args:
          session: TensorFlow session to use.

        Returns:
          The code dictionary, with shape (hidden_units, input_dim).
        """
        with tf.variable_scope(self.name, reuse=True):
            decoded = tf.layers.dense(tf.eye(self.hidden_units),
                                      self.input_dim,
                                      name='linear')
            return session.run(decoded)
