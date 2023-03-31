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


def train_RBM(rbm, data, epochs, learning_rate, batch_size, cd_k=1, verbose=False):
    """
    Trains an RBM using contrastive divergence.
    """
    loss_history = []
    data_train = data.copy()

    for epoch in range(epochs):
        np.random.shuffle(data_train)
        loss = 0.0
        num_samples = data_train.shape[0]

        for start in range(0, num_samples - batch_size, batch_size):
            v_0 = data_train[start : start + batch_size, :]
            v_k = data_train[start : start + batch_size, :]
            batch_size = v_0.shape[0]
            p_h_0, h_0 = entree_sortie_RBM(rbm, v_0)

            for k in range(cd_k):
                _, h_k = entree_sortie_RBM(rbm, v_k)
                _, v_k = sortie_entree_RBM(rbm, h_k)

            p_h_k, _ = entree_sortie_RBM(rbm, v_k)
            rbm.W += (learning_rate / batch_size) * (
                np.dot(v_0.T, p_h_0) - np.dot(v_k.T, p_h_k)
            )
            rbm.a += (learning_rate / batch_size) * np.sum(v_0 - v_k, axis=0)
            rbm.b += (learning_rate / batch_size) * np.sum(p_h_0 - p_h_k, axis=0)

        v_0 = data_train
        v_k = data_train

        for k in range(cd_k):
            _, h_k = entree_sortie_RBM(rbm, v_k)
            _, v_k = sortie_entree_RBM(rbm, h_k)

        loss = np.mean((v_k - v_0) ** 2)
        loss_history.append(loss)

        if verbose:
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} -- Reconstruction error: {loss}")

    return rbm


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
