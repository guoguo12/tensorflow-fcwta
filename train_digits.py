"""
Trains a FC-WTA autoencoder on the scikit-learn digits dataset. Also plots
some visualizations and evaluates the learned featurization by training an SVM
on the encoded data.

The default settings should give 99.0% classification accuracy, which is better
than the 95.5% accuracy achieved by an SVM trained on the raw pixel values.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import tensorflow as tf

from models import FullyConnectedWTA
from util import plot_dictionary, plot_reconstruction, plot_tsne, svm_acc

tf.app.flags.DEFINE_float('learning_rate', 1e-2,
                          'learning rate to use during training')
tf.app.flags.DEFINE_float('sparsity', 0.07,
                          'lifetime sparsity constraint to enforce')
tf.app.flags.DEFINE_float('test_size', 0.35,
                          'classification test set size')
tf.app.flags.DEFINE_integer('batch_size', 256,
                            'batch size to use during training')
tf.app.flags.DEFINE_integer('hidden_units', 256,
                            'size of each ReLU (encode) layer')
tf.app.flags.DEFINE_integer('num_layers', 1,
                            'number of ReLU (encode) layers')
tf.app.flags.DEFINE_integer('train_steps', 10000,
                            'total minibatches to train')
tf.app.flags.DEFINE_integer('steps_per_display', 500,
                            'minibatches to train before printing loss')
tf.app.flags.DEFINE_boolean('use_seed', True,
                            'fix random seed to guarantee reproducibility')

FLAGS = tf.app.flags.FLAGS

QUEUE_CAPACITY = 512
QUEUE_MIN_AFTER_DEQUEUE = 128


def main():
    if FLAGS.use_seed:
        np.random.seed(0)
        tf.set_random_seed(0)

    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=FLAGS.test_size,
        random_state=0 if FLAGS.use_seed else None)

    batch = tf.train.shuffle_batch(
        [X_train],
        batch_size=FLAGS.batch_size,
        capacity=QUEUE_CAPACITY,
        min_after_dequeue=QUEUE_MIN_AFTER_DEQUEUE,
        seed=1 if FLAGS.use_seed else None,
        enqueue_many=True)

    fcwta = FullyConnectedWTA(64,
                              FLAGS.batch_size,
                              sparsity=FLAGS.sparsity,
                              hidden_units=FLAGS.hidden_units,
                              encode_layers=FLAGS.num_layers,
                              learning_rate=FLAGS.learning_rate)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for step in range(FLAGS.train_steps):
            curr_batch = batch.eval()
            _, loss = fcwta.step(sess, curr_batch)
            if step % FLAGS.steps_per_display == 0:
                print('step={}, loss={:.3f}'.format(step, loss))

        # Examine code dictionary
        dictionary = fcwta.get_dictionary(sess)
        plot_dictionary(dictionary, (8, 8), num_shown=128, row_length=16)

        # Examine reconstructions of first batch of images
        decoded, _ = fcwta.step(sess, digits.data[:FLAGS.batch_size], forward_only=True)
        plot_reconstruction(digits.data[:FLAGS.batch_size], decoded, (8, 8), 20)

        # Featurize data
        X_train_f = fcwta.encode(sess, X_train)
        X_test_f = fcwta.encode(sess, X_test)

        # Examine t-SNE visualizations
        plot_tsne(X_train, y_train)
        plot_tsne(X_train_f, y_train)

        # Evaluate classification accuracy
        for C in np.logspace(-5, 5, 11):
            raw_acc, _ = svm_acc(X_train, y_train, X_test, y_test, C)
            featurized_acc, _ = svm_acc(X_train_f, y_train, X_test_f, y_test, C)
            print('C={:.5f}, raw acc={:.3f}, featurized acc={:.3f}'.format(
                C, raw_acc, featurized_acc))

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
