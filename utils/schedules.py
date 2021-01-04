import numpy as np

# a schedule that starts out flat and has a logistic drop-off towards later epochs
class MultiplicativeAnnealing(object):
    def __init__(self, T=20):
        self.T = T
        self.sigma_2 = np.square(T / 4.0)
    def __call__(self, epoch):
        if epoch < self.T:
            return (1-np.exp(-np.square(epoch-self.T)/(2*self.sigma_2))) / (1-np.exp(-np.square(epoch-1-self.T)/(2*self.sigma_2)))
        else:
            return 1.0