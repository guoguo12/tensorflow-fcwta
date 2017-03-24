"""
Shared training utilities.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.svm


def plot_dictionary(dictionary, shape, num_shown=20, row_length=10):
    """Plots the code dictionary."""
    rows = num_shown / row_length
    for i, image in enumerate(dictionary[:num_shown]):
        plt.subplot(rows, 10, i + 1)
        plt.axis('off')
        plt.imshow(image.reshape(shape), cmap=plt.cm.gray_r)
    plt.show()


def plot_reconstruction(truth, reconstructed, shape, num_shown=10):
    """Plots reconstructed images below the ground truth images."""
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
