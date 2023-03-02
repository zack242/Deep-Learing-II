import os 
import requests 
import scipy.io as sio
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from dnn import DNN
from rbm import RBM
from utils_projet import lire_alpha_digit, sigmoid
from principal_RBM_alpha import init_RBM, entree_sortie_RBM

from sklearn.utils import shuffle
import pdb

########## Partie 3.3 - Fonctions DNN
def calcul_softmax(rbm_unit, data_entree):
    """
    Calculate stable softmax of a given rbm_unit output
    Return: 
        - Probabilités de chaque unité de sortie par softmax 
        - Les valeurs originales avant le softmax 
    """
    # raw_output, _ = entree_sortie_RBM(rbm_unit, data_entree)
    raw_output = data_entree@rbm_unit.W + rbm_unit.b
    # calculate a stable softmax 
    exps = np.exp(raw_output - np.max(raw_output, axis=1)[:, None])
    return exps/np.sum(exps, axis=1)[:, None], raw_output

def entree_sortie_reseau(dnn, data_entree):
    """
    Return: list
        - les sorties sur chaque couche cachées du réseau ainsi que les
        probabilités sur les unités de sortie
    """
    outputs_reseau = []
    v = data_entree.copy()
    for i in range(dnn.nbr_hidden_layers):
        v, _ = entree_sortie_RBM(dnn.DBN[i], v)
        outputs_reseau.append(v)
    class_probabilities, raw_classification_output = calcul_softmax(dnn.classification_layer, v)
    outputs_reseau.append(raw_classification_output)
    outputs_reseau.append(class_probabilities)
    return outputs_reseau

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def retropropagation(dnn, epochs, learning_rate, batch_size, X, y):
    """
    Return:
        - DNN fine-tuned
    """
    # Implementer un Constrative Divergence-1 
    loss_history = []
    # On crée une copie pour ne pas modifier le vecteur originel
    X_train = X.copy()
    y_train = y.copy()
    for epoch in range(epochs):
        # shuffle data 
        #np.random.shuffle(X_train)
        X_train, y_train = shuffle(X_train, y_train)
        # shuffle target accordingly
        # Initialization du loss et taille des données 
        loss = 0.
        n = X_train.shape[0]
        # Raise erreur si la taille du batch est > que la quantité de données
        if batch_size > n:
            raise ValueError('Taille du batch est > que la quantité de données ! ')
        # Iteration des batches 
        for batch_start in range(0, n - batch_size, batch_size):
            # On initialise l'état initial en tant que les données 
            x_batch = X_train[batch_start:batch_start + batch_size, :]
            y_batch = y_train[batch_start:batch_start + batch_size, :]
            taille_batch = x_batch.shape[0]
            ### Forward: On fait le forward pass 
            outputs_reseau = entree_sortie_reseau(dnn, x_batch)
            y_pred = outputs_reseau[-1]
            
            ### Backpropagation: update gradients 
            grads = {}
            # softmax layer
            # (n, k)
            c_p = y_pred - y_batch
            # (n, p)
            x_l_minus_1 = outputs_reseau[-3] # deux dernieres places pour la dernière couche (output du couche et probas par softmax)
            grads[f'dW_softmax'] = (x_l_minus_1.T @ c_p)/taille_batch
            grads[f'db_softmax'] = np.mean(c_p, axis=0)
            
            c_l_plus_1 = c_p
            for l in range(dnn.nbr_hidden_layers - 1, -1, -1):
                x_l = outputs_reseau[l]
                # Get edge case of first layer
                if l != 0:
                    x_l_minus_1 = outputs_reseau[l-1]
                else:
                    x_l_minus_1 = x_batch
                # edge case of last layer
                if l == dnn.nbr_hidden_layers - 1:
                    W_l_plus_1 = dnn.classification_layer.W
                else:
                    W_l_plus_1 = dnn.DBN[l+1].W
                c_l = (c_l_plus_1 @ W_l_plus_1.T)*(x_l*(1-x_l)) 
                c_l_plus_1 = c_l
                grads[f'dW_{l}'] = (x_l_minus_1.T @ c_l)/taille_batch
                grads[f'db_{l}'] = np.mean(c_l, axis=0)

            ### Mise à jour des parametres
            dnn.classification_layer.W -= learning_rate*grads['dW_softmax']
            dnn.classification_layer.b -= learning_rate*grads['db_softmax']
            for l in range(dnn.nbr_hidden_layers - 1, -1, -1):
                dnn.DBN[l].W -= learning_rate*grads[f'dW_{l}']
                dnn.DBN[l].b -= learning_rate*grads[f'db_{l}']
        # Calculons la loss
        # On fait le forward pass 
        outputs_reseau = entree_sortie_reseau(dnn, X_train)
        prob_y_pred = outputs_reseau[-1]
        # On calcule la loss 
        train_loss = cross_entropy(prob_y_pred, y_train)
        loss_history.append(train_loss)
        #print(f'Epoch: {epoch} -- Loss: {train_loss}')
    # Print picture of loss 
    f = plt.figure(figsize=(10, 7))
    plt.plot(range(epochs), loss_history)
    plt.legend(['Entropie croisée'])
    plt.title("Entropie croisée au cours des iterations")
    plt.xlabel("epochs")
    plt.ylabel('entropie croisée')
    f.savefig(f'retropropagation_epoch{epochs}_learningRate_{learning_rate}.png')
    plt.show()
    return [dnn, loss_history]

def test_DNN(dnn, X_test, y_test):
    """
    Return:
    - Loss
    - Accuracy 
    """
    outputs_reseau = entree_sortie_reseau(dnn, X_test)
    # dernier element est la probabilité de chaque classe 
    y_pred = outputs_reseau[-1]
    test_loss = cross_entropy(y_pred, y_test)
    # transformer dans 0/1's
    pred_labels = np.zeros_like(y_pred)
    pred_labels[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
    # calculate taux
    acc_test = np.sum((pred_labels == y_test).all(1))/pred_labels.shape[0]
    print(f'Test loss: {test_loss}, Acc. %: {100*acc_test}%')
    return test_loss, acc_test