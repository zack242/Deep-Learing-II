from sklearn.utils import shuffle
from principal_RBM_MNIST import init_RBM, entree_sortie_RBM,\
    sortie_entree_RBM, train_RBM
import numpy as np
import matplotlib.pyplot as plt
# Définition de la classe DNN


class DNN:
    # Initialisation de la classe DNN avec 3 arguments: p, hidden_layers_units, nbr_classes
    def __init__(self, p, hidden_layers_units, nbr_classes):
        # Vérifie si hidden_layers_units contient au moins un élément
        if len(hidden_layers_units) == 0:
            # Si hidden_layers_units est vide, une exception ValueError est levée
            raise ValueError('Pas de hidden layers...')
        # Stocke les arguments en tant qu'attributs de la classe
        self.p = p
        self.nbr_classes = nbr_classes
        # Initialise layer_p avec la valeur de p
        layer_p = p
        # Initialise une liste vide pour stocker les RBM
        self.DBN = []
        # Parcours tous les éléments de hidden_layers_units
        for l, q in enumerate(hidden_layers_units):
            # Initialise un RBM avec p=layer_p et q=q, puis ajoute ce RBM à la liste DBN
            self.DBN.append(init_RBM(p=layer_p, q=q))
            # Met à jour la valeur de layer_p avec la valeur de q
            layer_p = q
        # Initialise un RBM pour la couche de classification avec p=layer_p et q=nbr_classes
        self.classification_layer = init_RBM(p=layer_p, q=nbr_classes)
        # Stocke le nombre de couches cachées dans l'attribut nbr_hidden_layers
        self.nbr_hidden_layers = len(self.DBN)

# fonction qui initialise un objet DNN avec les arguments donnés


def init_DNN(p, hidden_layers_units, nbr_classes):
    return DNN(p, hidden_layers_units, nbr_classes)

# fonction qui effectue une phase de pré-entraînement sur l'objet DNN donné avec les données X,
# pendant un certain nombre d'époques, avec un taux d'apprentissage et une taille de batch donnés


def pretrain_DNN(dnn, X, epochs, learning_rate, batch_size):
    # Copie les données X dans v
    v = X.copy()
    # Parcours toutes les couches cachées de l'objet DNN
    for layer in range(dnn.nbr_hidden_layers):
        # Effectue une phase d'entraînement sur la couche RBM actuelle avec les données v,
        # pendant un certain nombre d'époques, avec un taux d'apprentissage et une taille de batch donnés
        dnn.DBN[layer] = train_RBM(dnn.DBN[layer], v, epochs=epochs,
                                   learning_rate=learning_rate, batch_size=batch_size, cd_k=1)
        # Calcule la sortie de la couche RBM actuelle et stocke cette sortie dans v
        _, v = entree_sortie_RBM(dnn.DBN[layer], v)
    # Retourne l'objet DNN avec les couches RBM pré-entraînées
    return dnn

# Cette fonction prend en entrée une RBM et une matrice d'entrée, et calcule la sortie
# softmax de la couche de classification.


def calcul_softmax(rbm_unit, data_entree):
    # Calcul de la sortie brute de la RBM.
    raw_output = data_entree @ rbm_unit.W + rbm_unit.b
    # Calcul des exponentielles de la sortie brute, avec une normalisation pour éviter les erreurs numériques.
    exps = np.exp(raw_output - np.max(raw_output, axis=1)[:, None])
    # Calcul de la sortie softmax.
    return exps / np.sum(exps, axis=1)[:, None], raw_output

# Cette fonction prend en entrée un réseau DNN et une matrice d'entrée, et calcule la sortie
# du réseau pour cette entrée.


def entree_sortie_reseau(dnn, data_entree):
    # Initialisation d'une liste pour stocker les sorties de chaque couche.
    outputs_reseau = []
    # Copie de la matrice d'entrée.
    v = data_entree.copy()
    # Boucle sur les couches cachées du DNN.
    for i in range(dnn.nbr_hidden_layers):
        # Calcul de la sortie de la couche cachée avec une RBM.
        v, _ = entree_sortie_RBM(dnn.DBN[i], v)
        # Stockage de la sortie de la couche cachée.
        outputs_reseau.append(v)
    # Calcul de la sortie de la couche de classification.
    class_probabilities, raw_classification_output = calcul_softmax(
        dnn.classification_layer, v)
    # Stockage de la sortie brute de la couche de classification et de la sortie softmax.
    outputs_reseau.append(raw_classification_output)
    outputs_reseau.append(class_probabilities)
    # Retourne la liste des sorties de chaque couche.
    return outputs_reseau

# Cette fonction calcule l'entropie croisée entre les prédictions et les cibles.


def cross_entropy(predictions, targets, epsilon=1e-12):
    # Limitation des prédictions pour éviter les erreurs de calcul.
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    # Calcul de la taille de l'échantillon.
    N = predictions.shape[0]
    # Calcul de l'entropie croisée.
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce


def retropropagation(dnn, epochs, learning_rate, batch_size, X, y, display=False):
    loss_history = []  # Liste vide pour stocker les valeurs de la fonction de coût
    X_train = X.copy()
    y_train = y.copy()
    for epoch in range(epochs):
        # On mélange les données
        X_train, y_train = shuffle(X_train, y_train)
        loss = 0.
        n = X_train.shape[0]
        if batch_size > n:
            raise ValueError('Batch size > Train size')
        for batch_start in range(0, n - batch_size, batch_size):
            # Extraction des exemples pour le batch
            x_batch = X_train[batch_start:batch_start + batch_size, :]
            # Extraction des labels pour le batch
            y_batch = y_train[batch_start:batch_start + batch_size, :]
            taille_batch = x_batch.shape[0]
            outputs_reseau = entree_sortie_reseau(dnn, x_batch)
            y_pred = outputs_reseau[-1]
            grads = {}
            # Calcul de la dérivée partielle de la fonction de coût par rapport aux sorties
            c_p = y_pred - y_batch
            # de la dernière couche
            # Données d'entrée pour la dernière couche cachée
            x_l_minus_1 = outputs_reseau[-3]
            # Calcul du gradient de la dernière couche
            grads[f'dW_softmax'] = (x_l_minus_1.T @ c_p)/taille_batch
            # de classification
            grads[f'db_softmax'] = np.mean(c_p, axis=0)
            c_l_plus_1 = c_p  # Initialisation de la dérivée partielle de la fonction de coût par rapport aux sorties
            # de la couche précédente
            # Parcourt sur les couches cachées du réseau de neurones partant de la dernière couche cachée à la première.
            for l in range(dnn.nbr_hidden_layers - 1, -1, -1):
                # récupèration de la sortie de la couche cachée actuelle.
                x_l = outputs_reseau[l]
                if l != 0:
                    x_l_minus_1 = outputs_reseau[l-1]
                else:
                    x_l_minus_1 = x_batch
                if l == dnn.nbr_hidden_layers - 1:
                    W_l_plus_1 = dnn.classification_layer.W
                else:
                    W_l_plus_1 = dnn.DBN[l+1].W
                c_l = (c_l_plus_1 @ W_l_plus_1.T)*(x_l*(1-x_l))
                # mise à jour l'erreur pour la couche suivante.
                c_l_plus_1 = c_l
                # calcul des gradients pour la couche actuelle.
                grads[f'dW_{l}'] = (x_l_minus_1.T @ c_l)/taille_batch
                grads[f'db_{l}'] = np.mean(c_l, axis=0)
            dnn.classification_layer.W -= learning_rate * \
                grads['dW_softmax']  # Mise à jour du gradient
            dnn.classification_layer.b -= learning_rate * \
                grads['db_softmax']  # Mise à jour du gradient
            for l in range(dnn.nbr_hidden_layers - 1, -1, -1):
                # Mise à jour des poids et biais de chaque couche cachée
                dnn.DBN[l].W -= learning_rate*grads[f'dW_{l}']
                dnn.DBN[l].b -= learning_rate*grads[f'db_{l}']
        # Calcul de la perte sur les données d'entraînement
        outputs_reseau = entree_sortie_reseau(dnn, X_train)
        prob_y_pred = outputs_reseau[-1]
        train_loss = cross_entropy(prob_y_pred, y_train)
        loss_history.append(train_loss)
    # Visualisation de la courbe de perte au fil des epochs
    if display:
        f = plt.figure(figsize=(10, 7))
        plt.plot(range(epochs), loss_history)
        plt.legend(['Cross entropy'])
        plt.title("Cross entropy per epochs")
        plt.xlabel("epochs")
        plt.ylabel('Cross entropy')
        plt.show()
    return [dnn, loss_history]


def test_DNN(dnn, X_test, y_test):
    # Compute the outputs of the network for the test set
    outputs_reseau = entree_sortie_reseau(dnn, X_test)
    # Get the predicted labels from the network's output
    y_pred = outputs_reseau[-1]
    # Compute the cross-entropy loss for the test set
    test_loss = cross_entropy(y_pred, y_test)
    # Convert the predicted labels to one-hot encoding
    pred_labels = np.zeros_like(y_pred)
    pred_labels[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
    # Compute the accuracy of the predictions
    acc_test = np.sum((pred_labels == y_test).all(1))/pred_labels.shape[0]
    # Print the test loss and accuracy
    print(f'Test loss: {test_loss}, Acc. %: {100*acc_test}%')
    # Return the test loss and accuracy
    return test_loss, acc_test
