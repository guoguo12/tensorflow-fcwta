"""
Trains a FC-WTA autoencoder on the MNIST digits dataset. Also plots
some visualizations and evaluates the learned featurization by training an SVM
on the encoded data.

The default settings should give 98.62% classification accuracy, which is better
than the 95.5% accuracy achieved by an SVM trained on the raw pixel values and
is close to the 98.80% accuracy reported in the original WTA paper.
"""

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models import FullyConnectedWTA
from util import plot_dictionary, plot_reconstruction, plot_tsne, svm_acc, timestamp, value_to_summary

default_dir_suffix = timestamp()

tf.app.flags.DEFINE_string('data_dir', 'MNIST_data/',
                           'where to load data from (or download data to)')
tf.app.flags.DEFINE_string('train_dir', 'train_%s' % default_dir_suffix,
                           'where to store checkpoints to (or load checkpoints from)')
tf.app.flags.DEFINE_string('log_dir', 'log_%s' % default_dir_suffix,
                           'where to store logs to (use with --write_logs)')
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                          'learning rate to use during training')
tf.app.flags.DEFINE_float('sparsity', 0.05,
                          'lifetime sparsity constraint to enforce')
tf.app.flags.DEFINE_integer('batch_size', 100,
                            'batch size to use during training')
tf.app.flags.DEFINE_integer('hidden_units', 2000,
                            'size of each ReLU (encode) layer')
tf.app.flags.DEFINE_integer('num_layers', 1,
                            'number of ReLU (encode) layers')
tf.app.flags.DEFINE_integer('train_steps', 30000,
                            'total minibatches to train')
tf.app.flags.DEFINE_integer('steps_per_display', 100,
                            'minibatches to train before printing loss')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 10000,
                            'minibatches to train before saving checkpoint')
tf.app.flags.DEFINE_integer('train_size', 60000,
                            'number of examples to use to train classifier')
tf.app.flags.DEFINE_integer('test_size', 10000,
                            'number of examples to use to test classifier')
tf.app.flags.DEFINE_boolean('use_seed', True,
                            'fix random seed to guarantee reproducibility')
tf.app.flags.DEFINE_boolean('write_logs', True,
                            'write log files')

FLAGS = tf.app.flags.FLAGS


def main():
    if FLAGS.use_seed:
        np.random.seed(0)
        tf.set_random_seed(0)

    mnist = input_data.read_data_sets(FLAGS.data_dir, validation_size=0)
    X_train = mnist.train.images[:FLAGS.train_size]
    y_train = mnist.train.labels[:FLAGS.train_size]
    X_test = mnist.test.images[:FLAGS.test_size]
    y_test = mnist.test.labels[:FLAGS.test_size]

    fcwta = FullyConnectedWTA(784,
                              FLAGS.batch_size,
                              sparsity=FLAGS.sparsity,
                              hidden_units=FLAGS.hidden_units,
                              encode_layers=FLAGS.num_layers,
                              learning_rate=FLAGS.learning_rate)

    with tf.Session() as sess:
        if FLAGS.write_logs:
            writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Restoring from %s' % ckpt.model_checkpoint_path)
            fcwta.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            if not os.path.exists(FLAGS.train_dir):
                os.makedirs(FLAGS.train_dir)

        avg_time = avg_loss = 0  # Averages over FLAGS.steps_per_display steps
        step = 0
        while step < FLAGS.train_steps:
            start_time = time.time()
            batch_x, _ = mnist.train.next_batch(FLAGS.batch_size)
            _, loss = fcwta.step(sess, batch_x)

            avg_time += (time.time() - start_time) / FLAGS.steps_per_display
            avg_loss += loss / FLAGS.steps_per_display
            step += 1

            if step % FLAGS.steps_per_display == 0:
                global_step = fcwta.global_step.eval()
                print('step={}, global step={}, loss={:.3f}, time={:.3f}'.format(
                    step, global_step, avg_loss, avg_time))
                if FLAGS.write_logs:
                    writer.add_summary(value_to_summary(avg_loss, 'loss'),
                                       global_step=global_step)
                avg_time = avg_loss = 0
            if step % FLAGS.steps_per_checkpoint == 0:
                checkpoint_path = FLAGS.train_dir + '/ckpt'
                fcwta.saver.save(sess,
                                 checkpoint_path,
                                 global_step=fcwta.global_step)
                print('Wrote checkpoint')

        # Examine code dictionary
        dictionary = fcwta.get_dictionary(sess)
        plot_dictionary(dictionary, (28, 28), num_shown=200, row_length=20)

        # Examine reconstructions of first 20 images
        decoded, _ = fcwta.step(sess, X_train[:100], forward_only=True)
        plot_reconstruction(X_train[:100], decoded, (28, 28), 20)

        # Featurize data
        X_train_f = fcwta.encode(sess, X_train)
        X_test_f = fcwta.encode(sess, X_test)

        # Examine t-SNE visualizations
        plot_tsne(X_train[:1000], y_train[:1000])
        plot_tsne(X_train_f[:1000], y_train[:1000])

        # Evaluate classification accuracy
        for C in np.logspace(-3, 2, 6):
            acc, _ = svm_acc(X_train_f, y_train, X_test_f, y_test, C)
            print('C={:.3f}, acc={:.4f}'.format(C, acc))

if __name__ == '__main__':
    main()
