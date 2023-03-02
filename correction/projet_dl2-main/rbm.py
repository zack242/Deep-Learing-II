import numpy as np
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