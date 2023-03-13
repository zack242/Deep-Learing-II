import os
import requests
from scipy.io import loadmat
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import random

def relu(x):
    return np.maximum(x, 0)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_alphadigs():
    if os.path.exists('data/alphadigs.mat'):
        return loadmat('data/alphadigs.mat')
    
    alphadigs_url = 'https://cs.nyu.edu/~roweis/data/binaryalphadigs.mat'
    r = requests.get(alphadigs_url, allow_redirects=True)

    with open('data/alphadigs.mat', 'wb') as f:
        f.write(r.content)
                
    return loadmat('data/alphadigs.mat')
      
def lire_alpha_digit(digits_list):
    # Charger les données
    dataset = get_alphadigs()

    # Filtrages des données selon la liste de chiffres voulus
    digit2idx = {}
    for i, digit in enumerate(dataset['classlabels'][0]):
        digit2idx[digit[0]] = i
    
    idxs = []
    for digit in digits_list:
        idxs.append(digit2idx[digit])
    
    # Adaptation au format (n, p), chaque colonne designe un pixel et chaque ligne une image
    return np.stack(np.concatenate(dataset['dat'][idxs])).reshape(-1, 20*16)

class RBM:
    def __init__(self, p, q):
        self.W = 0.01 * np.random.rand(p,q)
        self.a = np.zeros((1,p))
        self.b = np.zeros((1,q))
        
def init_RBM(p, q):
    return RBM(p, q)

def entree_sortie_RBM(rbm, x):
    return sigmoid(np.dot(x, rbm.W) + rbm.b)

def sortie_entree_RBM(rbm, y): #(n * q) -> (n * p)
    return sigmoid(np.dot(y, rbm.W.T) + rbm.a)

def train_RBM(rbm,X,learning_rate, epochs, batch_size):
        
    n = X.shape[0]
    p = X.shape[1]
    q = len(rbm.b) 
    
    error = []
    for epoch in range(epochs):
        X_copy = X.copy()
    
        for batch  in range(0, n, batch_size):
            X_batch = X_copy[batch:batch + batch_size]
            
            tb = len(X_batch)
            
            v_0 = X_batch # (tb * p)
            #print(tb,p,v_0.shape)
            
            p_h_v_0 = entree_sortie_RBM(rbm,v_0) # (tb * q)
            #print(tb,q,p_h_v_0.shape)
            
            h_0 = (p_h_v_0 > np.random.rand(tb, q)).astype(int) # (tb * q)
            #print(tb,q,h_0.shape)
            
            p_v_h_0 = sortie_entree_RBM(rbm,h_0) # (tb * p)
            #print(tb,p,p_v_h_0.shape)
            v_1 = (p_v_h_0 > np.random.rand(tb, p)).astype(int) # (tb * p)
            #print(tb,p,v_1.shape)
            
            p_h_v_1 = entree_sortie_RBM(rbm,v_1) # (tb * q)
            #print(tb,q,p_h_v_1.shape)
            
            norm_learning_rate = learning_rate/tb
            
            rbm.W += norm_learning_rate * (np.dot(v_0.T, p_h_v_0) - np.dot(v_1.T, p_h_v_1)) # (p * q)   
            #print(p,q,rbm.W.shape)     
            rbm.a += norm_learning_rate * np.sum(v_0 - v_1, axis=0) # (1 * p)
            #print(p,rbm.a.shape)
            rbm.b += norm_learning_rate * np.sum(p_h_v_0 - p_h_v_1, axis=0) # (1 * q)
            #print(q,rbm.b.shape)
            
        H = entree_sortie_RBM(rbm,X)
        X_reconstruit = sortie_entree_RBM(rbm,H)
        loss = np.sum((X - X_reconstruit)**2)/(n*p)
        error.append(loss)
        if epoch % 10 == 0:
            print("Epoch : ", epoch, "Erreur : ", loss)
            
    return rbm,error


def generer_image_RBM(rbm, nb_iter, nb_images):
    p = rbm.a.shape[1]
    q = len(rbm.b)
    for i in range(nb_images):
        image = np.random.uniform(0, 1, (1, p))
        for j in range(nb_iter):
            prob_h_given_v = entree_sortie_RBM(rbm, image)
            h = (prob_h_given_v > np.random.rand(q)) * 1.0
            prob_v_given_h = sortie_entree_RBM(rbm, h)
            image = (prob_v_given_h > np.random.rand(p)) * 1.0

        image = np.reshape(image, (20, 16))
        plt.imshow(image, cmap='gray')
        plt.show()
               
def init_DBN(config):
    for i in range(len(config)-1):
        if i == 0:
            DBN = [init_RBM(config[i], config[i+1])]
        else:
            DBN.append(init_RBM(config[i], config[i+1]))
    return DBN

def train_DBN(DBN, H, epsilon, epochs, batch_size):
    for i in range(len(DBN)):
            DBN[i],error = train_RBM(DBN[i], H, epsilon, epochs, batch_size)
            H = entree_sortie_RBM(DBN[i],H)
    return DBN

def generer_image_DBN(Dbm, nb_iter, nb_images):
    for i in range(nb_images):
        v_0 = np.random.uniform(0, 1, (1, config[0]))
        v_k = v_0
        for j in range(nb_iter):
            
            for y in range(0,len(Dbm)): 
                prob_h_given_v = entree_sortie_RBM(Dbm[y], v_k)
                v_k = (prob_h_given_v > np.random.rand(config[y+1])) * 1.0

            h_k = v_k
            for k in range(len(Dbm)-1, -1, -1):
                prob_v_given_h = sortie_entree_RBM(Dbm[k], h_k)
                h_k = (prob_v_given_h > np.random.rand(config[k])) * 1.0

            v_k = h_k

        image = np.reshape(v_k, (20, 16))
        plt.imshow(image, cmap='gray')
        plt.show()
         