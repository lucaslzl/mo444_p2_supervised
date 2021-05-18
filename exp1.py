import pandas as pd

from gradient import LinearGradient, PolinomialGradient


def main():

    for dataset in ['./datasets/kick1.dat', './datasets/kick2.dat']:

        df = pd.read_csv(dataset, sep=' ')

        lg = LinearGradient(df, x=['y'], y='z')
        print(lg.get_thetas())

        lg.fit(df)
        print(lg.get_thetas())

        lg = LinearGradient(df, x=['y'], y='x')
        print(lg.get_thetas())

        lg.fit(df)
        print(lg.get_thetas())

        pg = PolinomialGradient(df, x=['y'], y='z')
        print(pg.get_thetas())

        pg.fit(df)
        print(pg.get_thetas())

        pg = PolinomialGradient(df, x=['y'], y='x')
        print(pg.get_thetas())

        pg.fit(df)
        print(pg.get_thetas())


if __name__ == '__main__':
    main()