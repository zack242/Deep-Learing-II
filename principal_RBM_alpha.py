import os
import requests
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from RBM import RBM


def sigmoid(x):
    """Computes the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


# Function for loading the alphadigits dataset
def get_alphadigs():
    """Loads the alphadigits dataset from disk or downloads if not found."""
    if os.path.exists("data/alphadigs.mat"):
        return loadmat("data/alphadigs.mat")

    alphadigs_url = "https://cs.nyu.edu/~roweis/data/binaryalphadigs.mat"
    r = requests.get(alphadigs_url, allow_redirects=True)

    with open("data/alphadigs.mat", "wb") as f:
        f.write(r.content)

    return loadmat("data/alphadigs.mat")


# Function for reading selected alphadigits
def lire_alpha_digit(digits_list):
    """Reads the selected alphadigits from the dataset."""
    # Load the dataset
    dataset = get_alphadigs()

    # Filter the data according to the desired digits list
    digit2idx = {}
    for i, digit in enumerate(dataset["classlabels"][0]):
        digit2idx[digit[0]] = i
    idxs = []
    for digit in digits_list:
        idxs.append(digit2idx[digit])

    # Reshape the data into (n, p) format, where each row is an image and each column represents a pixel
    return np.stack(np.concatenate(dataset["dat"][idxs])).reshape(-1, 20 * 16)


def init_RBM(p, q):
    """Initialize et retourne un RBM"""
    return RBM(p, q)


def entree_sortie_RBM(rbm, x):
    p_h_v = sigmoid(np.dot(x, rbm.W) + rbm.b)
    data_h = np.random.binomial(1, p_h_v, size=None)
    return p_h_v, data_h


def sortie_entree_RBM(rbm, y):  # (n * q) -> (n * p)
    p_v_h = sigmoid(np.dot(y, rbm.W.T) + rbm.a)
    reconstructed_data = np.random.binomial(1, p_v_h, size=None)
    return p_v_h, reconstructed_data


# Algorithme vu en cours
def train_RBM(rbm, X, learning_rate, epochs, batch_size):
    """
    Trains a RBM using the contrastive divergence algorithm.
    """
    n = X.shape[0]
    p = X.shape[1]
    q = len(rbm.b)

    error = []
    for epoch in range(epochs):
        X_copy = X.copy()
        np.random.shuffle(X_copy)
        for batch in range(0, n, batch_size):
            X_batch = X_copy[batch : batch + batch_size]

            tb = len(X_batch)
            v_0 = X_batch  # (tb * p)
            p_h_v_0, _ = entree_sortie_RBM(rbm, v_0)  # (tb * q)
            h_0 = (p_h_v_0 > np.random.rand(tb, q)).astype(int)  # (tb * q)
            p_v_h_0, _ = sortie_entree_RBM(rbm, h_0)  # (tb * p)
            v_1 = (p_v_h_0 > np.random.rand(tb, p)).astype(int)  # (tb * p)
            p_h_v_1, _ = entree_sortie_RBM(rbm, v_1)  # (tb * q)
            norm_learning_rate = learning_rate / tb
            rbm.W += norm_learning_rate * (
                np.dot(v_0.T, p_h_v_0) - np.dot(v_1.T, p_h_v_1)
            )  # (p * q)

            rbm.a += norm_learning_rate * np.sum(v_0 - v_1, axis=0)  # (1 * p)
            rbm.b += norm_learning_rate * np.sum(p_h_v_0 - p_h_v_1, axis=0)  # (1 * q)

        H, _ = entree_sortie_RBM(rbm, X)
        X_reconstruit, _ = sortie_entree_RBM(rbm, H)
        loss = np.sum((X - X_reconstruit) ** 2) / (n * p)
        error.append(loss)
        if epoch % 10 == 0:
            print("Epoch : ", epoch, "Erreur : ", loss)

    return rbm, error


def generer_image_RBM(rbm, nb_iter, nb_images):
    """
    Generates and displays images from random signals using a trained RBM.
    """
    p = rbm.a.shape[1]
    q = len(rbm.b)
    for i in range(nb_images):
        image = np.random.uniform(0, 1, (1, p))
        for j in range(nb_iter):
            prob_h_given_v, _ = entree_sortie_RBM(rbm, image)
            h = (prob_h_given_v > np.random.rand(q)) * 1.0
            prob_v_given_h, _ = sortie_entree_RBM(rbm, h)
            image = (prob_v_given_h > np.random.rand(p)) * 1.0

        image = np.reshape(image, (20, 16))
        plt.imshow(image, cmap="gray")
        plt.show()
