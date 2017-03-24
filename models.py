import tensorflow as tf


class FullyConnectedWTA:

    def __init__(self,
                 input_dim,
                 sparsity=0.05,
                 hidden_units=24,
                 encode_layers=3,
                 optimizer=tf.train.AdamOptimizer,
                 learning_rate=1e-2,
                 name='FCWTA'):
        self.input_dim = input_dim
        self.sparsity = sparsity
        self.hidden_units = hidden_units
        self.encode_layers = encode_layers
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.name = name
        self._initialize_vars()

    def _initialize_vars(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape=[None, self.input_dim])

            encoded = self.input
            for i in range(self.encode_layers):
                encoded = tf.layers.dense(encoded, self.hidden_units, activation=tf.nn.relu, name='relu_{}'.format(i))
            self.encoded = encoded

            encoded_t = tf.transpose(encoded)
            enc_shape = tf.shape(encoded_t)

            k = tf.cast(self.sparsity * tf.cast(enc_shape[1], tf.float32), tf.int32)
            top_values, _ = tf.nn.top_k(encoded_t, k=k, sorted=False)
            mask = tf.where(encoded_t < tf.reduce_min(top_values, axis=1, keep_dims=True),
                            tf.zeros(enc_shape, tf.float32),
                            tf.ones(enc_shape, tf.float32))
            sparse_encoded = encoded * tf.transpose(mask)

            decoded = tf.layers.dense(sparse_encoded, self.input_dim, use_bias=False, name='linear')
            self.decoded = decoded

            self.loss = tf.reduce_mean(tf.square(decoded - self.input))
            self.optimizer_op = self.optimizer(self.learning_rate).minimize(self.loss)

    def step(self, session, input, forward_only=False):
        if forward_only:
            return session.run(self.encoded, feed_dict={self.input: input})
        else:
            decoded, loss, _ = session.run([self.decoded, self.loss, self.optimizer_op], feed_dict={self.input: input})
            return decoded, loss

    def get_dictionary(self, session):
        with tf.variable_scope(self.name, reuse=True):
            decoded = tf.layers.dense(tf.eye(self.hidden_units), self.input_dim, use_bias=False, name='linear')
            return session.run(decoded)
