import numpy as np

class RBM:
    """
    Restricted Boltzman Machine class
    """
    def __init__(self, p, q):
        """Constructeur de la classe"""
        self.p = p
        self.q = q
        self.a = np.zeros((1, self.p))
        self.b = np.zeros((1, self.q))
        self.W = 0.01 * np.random.rand(p, q)