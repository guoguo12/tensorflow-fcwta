# tensorflow-fcwta

TensorFlow implementation of a **fully-connected winner-take-all (FC-WTA) autoencoder**,
as described in ["Winner-Take-All Autoencoders"](https://arxiv.org/pdf/1409.2752.pdf) (2015) by Alireza Makhzani and Brendan Frey
at the University of Toronto.

See `train_digits.py` and `train_mnist.py` for example code.

## Example images

The following images are created by `train_digits.py`, which trains a FC-WTA autoencoder on the [scikit-learn handwritten digits dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits) with 7% sparsity and 128 hidden units.

This plot compares the original images (top row) to the autoencoder's reconstructions (bottom row):
![Digit reconstruction visualization](images/digits_reconstruction.png?raw=true)

This one shows the autoencoder's learned code dictionary:
![Code dictionary visualization](images/digits_dictionary.png?raw=true)

Finally, here are t-SNE plots of the original data (left) and the featurized data (right):
![t-SNE visualizations of original and featurized images](images/digits_tsne_merged.png?raw=true)

## Credits

The code for enforcing lifetime sparsity is based on the implementation at [iwyoo/tf_ConvWTA](https://github.com/iwyoo/tf_ConvWTA).
