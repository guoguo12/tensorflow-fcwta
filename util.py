"""
Shared training utilities.
"""

import datetime

import matplotlib.pyplot as plt
import sklearn.decomposition
import sklearn.manifold
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.svm
import tensorflow as tf


def timestamp(format='%Y_%m_%d_%H_%M_%S'):
    """Returns the current time as a string."""
    return datetime.datetime.now().strftime(format)


def plot_dictionary(dictionary, shape, num_shown=20, row_length=10):
    """Plots the code dictionary."""
    rows = num_shown / row_length
    for i, image in enumerate(dictionary[:num_shown]):
        plt.subplot(rows, row_length, i + 1)
        plt.axis('off')
        plt.imshow(image.reshape(shape), cmap=plt.cm.gray)
    plt.show()


def plot_reconstruction(truth, reconstructed, shape, num_shown=10):
    """Plots reconstructed images below the ground truth images."""
    for i, image in enumerate(truth[:num_shown]):
        plt.subplot(2, num_shown, i + 1)
        plt.axis('off')
        plt.imshow(image.reshape(shape), cmap=plt.cm.gray)
    for i, image in enumerate(reconstructed[:num_shown]):
        plt.subplot(2, num_shown, i + num_shown + 1)
        plt.axis('off')
        plt.imshow(image.reshape(shape), cmap=plt.cm.gray)
    plt.show()


def plot_tsne(X, labels):
    """Plots a t-SNE visualization of the given data."""
    if X.shape[1] > 50:
        X = sklearn.decomposition.PCA(50).fit_transform(X)
    X = sklearn.manifold.TSNE(learning_rate=200).fit_transform(X)
    plt.scatter(X[:,0], X[:,1], c=labels, cmap=plt.cm.viridis)
    plt.show()


def svm_acc(X_train, y_train, X_test, y_test, C):
    """Trains and evaluates a linear SVM with the given data and C value."""
    clf = sklearn.svm.LinearSVC(C=C)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred)


def value_to_summary(value, tag):
    """Converts a numerical value into a tf.Summary object."""
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
