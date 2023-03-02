import numpy as np
from rbm import RBM
from utils_projet import lire_alpha_digit, sigmoid
from principal_RBM_alpha import init_RBM, entree_sortie_RBM,\
                                sortie_entree_RBM, train_RBM
class DNN:
    def __init__(self, p, hidden_layers_units, nbr_classes):
        """
        Parameters:
            - p = dimension input
            - hidden_layers_units = [100, 200 ,100] <- 3 hiddens layers 
            - nbr_classes for output (final q)
        """
        if len(hidden_layers_units) == 0:
            raise ValueError('Pas de hidden layers...')
        self.p = p
        self.nbr_classes = nbr_classes
        layer_p = p
        self.DBN = []
        for l, q in enumerate(hidden_layers_units):
            self.DBN.append(init_RBM(p=layer_p, q=q))
            layer_p = q
        # Ajouter dernier RBM pour matcher le numéro de classes
        # La classification se fera par l'argmax de l'output de cette couche
        self.classification_layer = init_RBM(p=layer_p, q=nbr_classes)
        self.nbr_hidden_layers = len(self.DBN)
        