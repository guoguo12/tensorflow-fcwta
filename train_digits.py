import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import sklearn.svm
import tensorflow as tf

from models import FullyConnectedWTA

tf.app.flags.DEFINE_float('learning_rate', 1e-2,
                          'learning rate to use during training')
tf.app.flags.DEFINE_float('sparsity', 0.05,
                          'lifetime sparsity constraint to enforce')
tf.app.flags.DEFINE_float('test_size', 0.2,
                          'classification test set size')
tf.app.flags.DEFINE_integer('batch_size', 256,
                            'batch size to use during training')
tf.app.flags.DEFINE_integer('hidden_units', 256,
                            'size of each ReLU (encode) layer')
tf.app.flags.DEFINE_integer('num_layers', 3,
                            'number of ReLU (encode) layers')
tf.app.flags.DEFINE_integer('train_steps', 200000,
                            'total minibatches to train')
tf.app.flags.DEFINE_integer('steps_per_display', 1000,
                            'minibatches to train before printing loss')
tf.app.flags.DEFINE_boolean('use_seed', True,
                            'fix random seed to guarantee reproducibility')

FLAGS = tf.app.flags.FLAGS

QUEUE_CAPACITY = 512
QUEUE_MIN_AFTER_DEQUEUE = 128


def visualize_dictionary(dictionary, shape, num_shown=20, row_length=10):
    rows = num_shown / row_length
    for i, image in enumerate(dictionary[:num_shown]):
        plt.subplot(rows, 10, i + 1)
        plt.axis('off')
        plt.imshow(image.reshape(shape), cmap=plt.cm.gray_r)
    plt.show()


def visualize_decode(truth, reconstructed, shape, num_shown=10):
    for i, image in enumerate(truth[:num_shown]):
        plt.subplot(2, num_shown, i + 1)
        plt.axis('off')
        plt.imshow(image.reshape(shape), cmap=plt.cm.gray_r)
    for i, image in enumerate(reconstructed[:num_shown]):
        plt.subplot(2, num_shown, i + num_shown + 1)
        plt.axis('off')
        plt.imshow(image.reshape(shape), cmap=plt.cm.gray_r)
    plt.show()


def svm_acc(X_train, y_train, X_test, y_test, C, kernel='linear'):
    """Trains and evaluates an SVM with the given hyperparameters and data."""
    clf = sklearn.svm.SVC(C=C, kernel=kernel)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred)


def main():
    if FLAGS.use_seed:
        np.random.seed(0)
        tf.set_random_seed(0)

    digits = load_digits()
    batch = tf.train.shuffle_batch(
        [digits.data],
        batch_size=FLAGS.batch_size,
        capacity=QUEUE_CAPACITY,
        min_after_dequeue=QUEUE_MIN_AFTER_DEQUEUE,
        seed=0,
        enqueue_many=True)

    fcwta = FullyConnectedWTA(64,
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
            if step % FLAGS.steps_per_display == 0:
                _, loss = fcwta.step(sess, curr_batch)
                print('step={}, loss={:.3f}'.format(step, loss))

        # Examine code dictionary
        dictionary = fcwta.get_dictionary(sess)
        visualize_dictionary(dictionary, (8, 8), num_shown=40)

        # Examine reconstructions
        decoded, _ = fcwta.step(sess, digits.data)
        visualize_decode(digits.data, decoded, (8, 8), 20)

        # Evaluate classification accuracy
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data,
            digits.target,
            test_size=FLAGS.test_size,
            random_state=0 if FLAGS.use_seed else None)
        X_train_f = fcwta.step(sess, X_train, forward_only=True)
        X_test_f = fcwta.step(sess, X_test, forward_only=True)
        for C in np.logspace(-2, 3, 6):
            raw_acc, _ = svm_acc(X_train, y_train, X_test, y_test, C)
            featurized_acc, _ = svm_acc(X_train_f, y_train, X_test_f, y_test, C)
            print('C={:.2f}, raw acc={:.3f}, featurized acc={:.3f}'.format(
                C, raw_acc, featurized_acc))

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
