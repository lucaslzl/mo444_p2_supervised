import pandas as pd
import numpy as np
from tqdm import tqdm

class LinearGradient:
    """
        Linear Gradient Descent
        y = theta_0*x_0 + theta_1*x_1
    """


    def __init__(self, data: pd.DataFrame, x: list, y: str, alpha=0.01):
        
        self.data = data
        self.x = x
        self.y = y
        self.alpha = alpha
        self.thetas = [np.random.rand() for i in range(len(x))]


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


class PolinomialGradient:
    """
        Polinomial Gradient Descent
        y = theta_0*(x_0**2) + theta_1*(x_1**2)
    """


    def __init__(self, data: pd.DataFrame, x: list, y: str, alpha=0.01):
        
        self.data = data
        self.x = x
        self.y = y
        self.alpha = alpha
        self.thetas = [np.random.rand() for i in range(len(x))]


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


def main():

    df1 = pd.read_csv('./datasets/kick1.dat', sep=' ')
    # df2 = pd.read_csv('./datasets/kick1.dat', sep=' ')

    print('Linear: x, y = z')
    lg = LinearGradient(df1, x=['x', 'y'], y='z')
    print(lg.get_thetas())

    lg.fit(df1)
    print(lg.get_thetas())

    print('Linear: z, y = x')
    lg = LinearGradient(df1, x=['z', 'y'], y='x')
    print(lg.get_thetas())

    lg.fit(df1)
    print(lg.get_thetas())

    print('Polinomial: x, y = z')
    pg = PolinomialGradient(df1, x=['x', 'y'], y='z')
    print(pg.get_thetas())

    pg.fit(df1)
    print(pg.get_thetas())

    print('Linear: z, y = x')
    pg = PolinomialGradient(df1, x=['z', 'y'], y='x')
    print(pg.get_thetas())

    pg.fit(df1)
    print(pg.get_thetas())


if __name__ == '__main__':
    main()