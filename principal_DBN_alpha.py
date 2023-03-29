import matplotlib.pyplot as plt
import numpy as np
from principal_RBM_alpha import *


def init_DBN(config):
    for i in range(len(config) - 1):
        if i == 0:
            DBN = [init_RBM(config[i], config[i + 1])]
        else:
            DBN.append(init_RBM(config[i], config[i + 1]))
    return DBN


def train_DBN(DBN, H, epsilon, epochs, batch_size):
    for i in range(len(DBN)):
        DBN[i], error = train_RBM(DBN[i], H, epsilon, epochs, batch_size)
        H = entree_sortie_RBM(DBN[i], H)
    return DBN


def generer_image_DBN(Dbm, config, nb_iter, nb_images):
    for i in range(nb_images):
        v_0 = np.random.uniform(0, 1, (1, config[0]))
        v_k = v_0
        for j in range(nb_iter):

            for y in range(0, len(Dbm)):
                prob_h_given_v = entree_sortie_RBM(Dbm[y], v_k)
                v_k = (prob_h_given_v > np.random.rand(config[y + 1])) * 1.0

            h_k = v_k
            for k in range(len(Dbm) - 1, -1, -1):
                prob_v_given_h = sortie_entree_RBM(Dbm[k], h_k)
                h_k = (prob_v_given_h > np.random.rand(config[k])) * 1.0

            v_k = h_k

        image = np.reshape(v_k, (20, 16))
        plt.imshow(image, cmap="gray")
        plt.show()
