import pandas as pd
import numpy as np
from tqdm import tqdm

class LinearGradient:
    """
        Linear Gradient Descent
        y = theta_0*1 + theta_1*x_1 + theta_2*x_2
    """


    def __init__(self, data: pd.DataFrame, x: list, y: str, alpha=0.01):
        
        self.data = data
        self.x = x
        self.y = y
        self.alpha = alpha
        np.random.seed(42)
        self.thetas = np.random.rand(len(x)+1)

        # Add theta_0 column
        self.data['joker'] = [1] * self.data.shape[0]
        self.x.append('joker')


    def function(self, row):
        
        ys = [row[c]*self.thetas[i] for i, c in enumerate(self.x)]
        ys = np.sum(ys)
        return (ys - row[self.y])


    def mse(self, i):
        
        summ = 0

        for indx, row in self.data.iterrows():
            summ += self.function(row) * row[self.x[i]]

        return summ / self.data.shape[0]


    def update_theta(self):
        
        tmp_thetas = []

        for i, theta in enumerate(self.thetas):

            tmp_theta = theta - self.alpha * self.mse(i)
            tmp_thetas.append(tmp_theta)

        self.thetas = tmp_thetas


    def fit(self, data):

        for steps in tqdm(range(100)):
            self.update_theta()


    def get_thetas(self):
        return self.thetas

    
    def predict(self, values: list):

        values.append(1)

        res = 0
        for i, theta in enumerate(self.thetas):
            res += theta * values[i]

        return res


class PolinomialGradient:
    """
        Polinomial Gradient Descent
        y = theta_0*1 + theta_1*(x_1**2) + theta_2*(x_2**2)
    """


    def __init__(self, data: pd.DataFrame, x: list, y: str, alpha=0.01):
        
        self.data = data
        self.x = x
        self.y = y
        self.alpha = alpha
        self.thetas = np.random.rand(len(x)+1)

        # Add theta_0 column
        self.data['joker'] = [1] * self.data.shape[0]
        self.x.append('joker')


    def function(self, row):
        
        ys = [row[c]*self.thetas[i] for i, c in enumerate(self.x)]
        ys = np.sum(ys)
        return (ys - row[self.y])


    def mse(self, i):
        
        summ = 0

        for indx, row in self.data.iterrows():
            summ += self.function(row) * row[self.x[i]]**2

        return summ / self.data.shape[0]


    def update_theta(self):
        
        tmp_thetas = []

        for i, theta in enumerate(self.thetas):

            tmp_theta = theta - self.alpha * self.mse(i)

            tmp_thetas.append(tmp_theta)

        self.thetas = tmp_thetas


    def fit(self, data):

        for steps in tqdm(range(100)):
            self.update_theta()


    def get_thetas(self):
        return self.thetas


    def predict(self, values: list):

        values.append(1)

        res = 0
        for i, theta in enumerate(self.thetas):
            res += theta * (values[i]**2)

        return res