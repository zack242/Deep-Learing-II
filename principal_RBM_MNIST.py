import os 
import requests 
import scipy.io as sio
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

def get_mnist(): 
    mnist = fetch_openml('mnist_784')
    X = mnist['data']
    X = (mnist['data'] > 128).astype(int)
    y = mnist['target']
    y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    return X, y

def sigmoid(x):
    return expit(x)
    
class RBM:
    """
    Restricted Boltzman Machine class
    Implementation d'un RBM en suivant les memes notations qu'en cours:
        -> p: dimension input (v)
        -> q: dimension output (h)
        -> E(v, h) = - sum_p(a_i*v_i) - sum_q(b_j*h_j) - sum_p(W_ij*v_i*h_j)
    Cette classe implemente des méthodes initialisation et entrainement du RBM. 
    """
    def __init__(self, p, q):
        """ Constructeur de la classe """
        self.p = p
        self.q = q
        # biais des unités d’entrée) -> dim (1xp)
        self.a = np.zeros((1, self.p))
        # biais des unités de sortie -> dim (1xq)
        self.b = np.zeros((1, self.q))
        # initialisés aléatoirement suivant une loi normale centrée, de variance égale à 0.01
        self.W = np.random.normal(loc=0, scale=0.1, size=(self.p, self.q))


def init_RBM(p, q):
    """ Initialize et retourne un RBM"""
    return RBM(p, q)
        
def entree_sortie_RBM(rbm_unit, data):
    """
    Sort data_h de dimension (nxq)
    """
    p_h_donne_v = sigmoid(data@rbm_unit.W + rbm_unit.b)
    # return bernoulli distribution for the sampled probabilities 
    data_h = np.random.binomial(1, p_h_donne_v, size=None)
    # Nous retournons les probabilités et les données de sortie
    # (n, q), (n, q)
    return p_h_donne_v, data_h

def sortie_entree_RBM(rbm_unit, data_h):
    """
    Sort entree data reconstruit de dimension (nxp)
    """
    p_v_donne_h = sigmoid(data_h@rbm_unit.W.T + rbm_unit.a)
    # return bernoulli distribution for the sampled probabilities 
    reconstructed_data = np.random.binomial(1, p_v_donne_h, size=None)
    # Nous retournons les probabilités et les données de sortie
    # (n, p), (n, p)
    return p_v_donne_h, reconstructed_data
    
def train_RBM(rbm_unit, X, epochs, learning_rate, batch_size, cd_k=1):
    """
    Entraine un RBM et retourne le même RBM entrainé
    """
    # Implementer un Constrative Divergence-1 
    loss_history = []
    # On crée une copie pour ne pas modifier le vecteur originel
    X_train = X.copy()
    for epoch in range(epochs):
        # shuffle data 
        np.random.shuffle(X_train)
        # Initialization du loss et taille des données 
        loss = 0.
        n = X_train.shape[0]
        # Raise erreur si la taille du batch est > que la quantité de données
        if batch_size > n:
            raise ValueError('Taille du batch est > que la quantité de données ! ')
        # Iteration des batches 
        for batch_start in range(0, n - batch_size, batch_size):
            # On initialise l'état initial en tant que les données 
            v_0 = X_train[batch_start:batch_start + batch_size, :]
            v_k = X_train[batch_start:batch_start + batch_size, :]
            # taille batch 
            taille_batch = v_0.shape[0]
            # On fait le forward pass 
            ph_0, h_0 = entree_sortie_RBM(rbm_unit, v_0)
            # Appliquons la Contrastive Divergence
            # En pratique nous allons appliquer un CD-1, mais 
            # ci-dessous nous écrivons le code pour un CD-K 
            # quelconque
            for k in range(cd_k):
                # Sample h
                _, h_k = entree_sortie_RBM(rbm_unit, v_k)
                _, v_k = sortie_entree_RBM(rbm_unit, h_k)
            # Prenons la probabilité de chaque composante de la sortie pour des mises
            # à jour
            ph_k, _ = entree_sortie_RBM(rbm_unit, v_k)
            # Mise à jour des parametres
            rbm_unit.W += (learning_rate/taille_batch)*(v_0.T @ ph_0 - v_k.T @ ph_k)
            rbm_unit.a += (learning_rate/taille_batch)*np.sum(v_0 - v_k, axis=0)
            rbm_unit.b += (learning_rate/taille_batch)*np.sum(ph_0 - ph_k, axis=0)
        # Calculer l'erreur de reconstruction
        v_0 = X_train
        v_k = X_train
        for k in range(cd_k):
            # Sample h
            _, h_k = entree_sortie_RBM(rbm_unit, v_k)
            _, v_k = sortie_entree_RBM(rbm_unit, h_k)

        loss = np.mean((v_k - v_0)**2)
        loss_history.append(loss)
        print(f'Epoch: {epoch} -- Erreur de reconstruction: {loss}')
    return rbm_unit

def generer_image_RBM(rbm_unit, nbr_iterations_gibbs, nbr_images, image_shape=(20, 16)):
    """
    Genere et affiche des images à partir des signaux random avec un RBM appris. 
    """
    for image_count in range(nbr_images):
        # Initialize image random
        v_0 = np.random.rand(1, rbm_unit.p)
        v_k = v_0
        # échantillonnage de gibbs 
        for i in range(nbr_iterations_gibbs):
            # Sample h
            _, h_k = entree_sortie_RBM(rbm_unit, v_k)
            _, v_k = sortie_entree_RBM(rbm_unit, h_k)
        
        # Redimensionner image
        image_arr = v_k.reshape(image_shape)
        # Afficher Image 
        #plt.subplot(image_count)
        plt.imshow(image_arr, cmap='Greys',  interpolation='nearest')
        plt.show()