import os 
import requests 
import scipy.io as sio
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from dnn import DNN
from rbm import RBM
from utils_projet import lire_alpha_digit, sigmoid
from principal_RBM_alpha import init_RBM, entree_sortie_RBM,\
                                sortie_entree_RBM, train_RBM
########## Partie 3.2 - Fonctions DBN 
def init_DNN(p, hidden_layers_units, nbr_classes):
    """ construit et d’initialise les poids et les biais d’un DNN"""
    return DNN(p, hidden_layers_units, nbr_classes)

def pretrain_DNN(dnn, X, epochs, learning_rate, batch_size):
    """
    Pre entrainement du DNN
    Return 
        - Le DNN pre-entrainé
    """
    v = X.copy()
    # Greedy Layer: on entraine chaque RBM 
    for layer in range(dnn.nbr_hidden_layers):
        dnn.DBN[layer] = train_RBM(dnn.DBN[layer], v, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, cd_k=1)
        _, v = entree_sortie_RBM(dnn.DBN[layer], v)
    return dnn 

def generer_image_DBN(dnn, nbr_iterations_gibbs, nbr_images, image_shape=(20, 16)):
    """
    Genere et affiche des images à partir des signaux random avec un DNN appris. 
    """
    for image_count in range(nbr_images):
        # Initialize image random
        v_0 = np.random.rand(1, dnn.p)
        v_k = v_0
        # échantillonnage de gibbs 
        for i in range(nbr_iterations_gibbs):
            # Entree -> sortie 
            for layer in range(0, dnn.nbr_hidden_layers):
                _, h_k = entree_sortie_RBM(dnn.DBN[layer], v_k)
                v_k = h_k
            # sortie -> entree
            for layer in range(dnn.nbr_hidden_layers-1, -1, -1):
                _, v_k = sortie_entree_RBM(dnn.DBN[layer], h_k)
                h_k = v_k

        # Redimensionner image
        image_arr = v_k.reshape(image_shape)
        # Afficher Image 
        #plt.subplot(image_count)
        plt.imshow(image_arr, cmap='Greys',  interpolation='nearest')
        plt.show()