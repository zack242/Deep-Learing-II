from sklearn.utils import shuffle
from principal_RBM_MNIST import (
    init_RBM,
    entree_sortie_RBM,
    train_RBM,
)
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


class DNN:
    def __init__(self, p, hidden_layers_units, nbr_classes):
        self.p = p
        self.nbr_classes = nbr_classes
        layer_p = p
        self.DBN = []
        # Initialize RBMs for each hidden layer
        for l, q in enumerate(hidden_layers_units):
            self.DBN.append(init_RBM(p=layer_p, q=q))
            layer_p = q
        # Initialize the classification layer
        self.classification_layer = init_RBM(p=layer_p, q=nbr_classes)
        self.nbr_hidden_layers = len(self.DBN)


# Initialize the DNN
def init_DNN(p, hidden_layers_units, nbr_classes):
    return DNN(p, hidden_layers_units, nbr_classes)


# Pretrain the DNN using the RBM approach
def pretrain_DNN(dnn, X, epochs, learning_rate, batch_size):
    X_copy = X.copy()
    for layer in range(dnn.nbr_hidden_layers):
        dnn.DBN[layer] = train_RBM(
            dnn.DBN[layer],
            X_copy,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            cd_k=1,
        )
        _, v = entree_sortie_RBM(dnn.DBN[layer], X_copy)
    return dnn


# Calculate network output given input data
def entree_sortie_reseau(dnn, data_entree):
    outputs_reseau = []
    v = data_entree.copy()
    for i in range(dnn.nbr_hidden_layers):
        v, _ = entree_sortie_RBM(dnn.DBN[i], v)
        outputs_reseau.append(v)
    class_probabilities, raw_classification_output = calcul_softmax(
        dnn.classification_layer, v
    )
    outputs_reseau.append(raw_classification_output)
    outputs_reseau.append(class_probabilities)
    return outputs_reseau


# Perform backpropagation on the DNN
def retropropagation(dnn, epochs, learning_rate, batch_size, X, y, display=False):
    loss_history = []  # Empty list to store the loss values
    X_train = X.copy()
    y_train = y.copy()

    pbar = tqdm(range(epochs), desc="Retropopagation")
    for epoch in pbar:
        # Shuffle the data
        X_train, y_train = shuffle(X_train, y_train)
        n = X_train.shape[0]
        for batch_start in range(0, n - batch_size, batch_size):
            x_batch = X_train[batch_start : batch_start + batch_size, :]

            y_batch = y_train[batch_start : batch_start + batch_size, :]
            taille_batch = x_batch.shape[0]
            outputs_reseau = entree_sortie_reseau(dnn, x_batch)
            y_pred = outputs_reseau[-1]
            grads = {}

            c_p = y_pred - y_batch
            x_l_minus_1 = outputs_reseau[-3]

            grads[f"dW_softmax"] = (x_l_minus_1.T @ c_p) / taille_batch
            grads[f"db_softmax"] = np.mean(c_p, axis=0)
            c_l_plus_1 = c_p
            for l in range(dnn.nbr_hidden_layers - 1, -1, -1):
                x_l = outputs_reseau[l]
                if l != 0:
                    x_l_minus_1 = outputs_reseau[l - 1]
                else:
                    x_l_minus_1 = x_batch
                if l == dnn.nbr_hidden_layers - 1:
                    W_l_plus_1 = dnn.classification_layer.W
                else:
                    W_l_plus_1 = dnn.DBN[l + 1].W
                c_l = (c_l_plus_1 @ W_l_plus_1.T) * (x_l * (1 - x_l))

                c_l_plus_1 = c_l

                grads[f"dW_{l}"] = (x_l_minus_1.T @ c_l) / taille_batch
                grads[f"db_{l}"] = np.mean(c_l, axis=0)
            dnn.classification_layer.W -= learning_rate * grads["dW_softmax"]
            dnn.classification_layer.b -= learning_rate * grads["db_softmax"]

            for l in range(dnn.nbr_hidden_layers - 1, -1, -1):
                dnn.DBN[l].W -= learning_rate * grads[f"dW_{l}"]
                dnn.DBN[l].b -= learning_rate * grads[f"db_{l}"]

        outputs_reseau = entree_sortie_reseau(dnn, X_train)
        prob_y_pred = outputs_reseau[-1]
        train_loss = cross_entropy(prob_y_pred, y_train)
        loss_history.append(train_loss)

        pbar.set_postfix({"loss retropagation": train_loss})
    if display:
        f = plt.figure(figsize=(10, 7))
        plt.plot(range(epochs), loss_history)
        plt.legend(["Cross entropy"])
        plt.title("Cross entropy per epochs")
        plt.xlabel("epochs")
        plt.ylabel("Cross entropy")
        plt.show()
    return [dnn, loss_history]


# Test the DNN
def test_DNN(dnn, X_test, y_test):
    network_outputs = entree_sortie_reseau(dnn, X_test)
    y_pred = network_outputs[-1]
    test_loss = cross_entropy(y_pred, y_test)

    pred_labels = np.zeros_like(y_pred)
    pred_labels[np.arange(len(y_pred)), y_pred.argmax(1)] = 1

    correct_predictions = (pred_labels == y_test).all(1)
    acc_test = np.sum(correct_predictions) / pred_labels.shape[0]

    print(f"Test loss: {test_loss}, Accuracy: {100 * acc_test}%")
    return test_loss, acc_test


# Utility functions
def calcul_softmax(rbm_unit, data_entree):
    raw_output = data_entree @ rbm_unit.W + rbm_unit.b
    exps = np.exp(raw_output - np.max(raw_output, axis=1)[:, None])
    return exps / np.sum(exps, axis=1)[:, None], raw_output


def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce
