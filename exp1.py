import pandas as pd
import pickle

from gradient import LinearGradient, PolinomialGradient


def populate(results, 
            dataset_name,
            Type,
            x,
            y,
            a,
            ta,
            tb,
            pred):

    results['Dataset'].append(dataset_name)
    results['Type'].append(Type)
    results['X'].append(x)
    results['Y'].append(y)
    results['Alpha'].append(a)
    results['Thetas A'].append(ta)
    results['Thetas B'].append(tb)
    results['Predicted'].append(pred)


def save_pickle(results, name='results.pickle'):

    with open(name, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name='results.pickle'):

    with open(name, 'rb') as handle:
        return pickle.load(handle)


def describe_results():

    results = load_pickle()
    print(pd.DataFrame(results))


def run_experiment():

    results = {'Dataset': [], 
                'Type': [], 
                'X': [], 
                'Y': [],
                'Alpha': [],
                'Thetas A': [], 
                'Thetas B': [],
                'Predicted': []}

    for dataset in ['./datasets/kick1.dat', './datasets/kick2.dat']:

        dataset_name = dataset.split('/')[-1]

        df = pd.read_csv(dataset, sep=' ')

        df['y'] = -df['y']

        for alpha in [0.01, 0.05, 0.001]:

            ###############################
            ### LinearGradient Y / Z    ###
            ###############################
            lg = LinearGradient(df, x=['y'], y='z', alpha=alpha)
            thetas_a = lg.get_thetas()

            lg.fit(df)
            thetas_b = lg.get_thetas()

            pred = lg.predict([0])

            populate(results,
                    dataset_name,
                    'LinearGradient',
                    'y',
                    'z',
                    alpha,
                    thetas_a,
                    thetas_b,
                    pred)

            ###############################
            ### LinearGradient Y / X    ###
            ###############################
            lg = LinearGradient(df, x=['y'], y='x', alpha=alpha)
            thetas_a = lg.get_thetas()

            lg.fit(df)
            thetas_b = lg.get_thetas()

            pred = lg.predict([0])

            populate(results,
                    dataset_name,
                    'LinearGradient',
                    'y',
                    'x',
                    alpha,
                    thetas_a,
                    thetas_b,
                    pred)

            ################################
            ### PolinomialGradient Y / Z ###
            ################################
            pg = PolinomialGradient(df, x=['y'], y='z', alpha=alpha)
            thetas_a = pg.get_thetas()

            pg.fit(df)
            thetas_b = pg.get_thetas()

            pred = pg.predict([0])

            populate(results,
                    dataset_name,
                    'PolinomialGradient',
                    'y',
                    'z',
                    alpha,
                    thetas_a,
                    thetas_b,
                    pred)

            ################################
            ### PolinomialGradient Y / X ###
            ################################
            pg = PolinomialGradient(df, x=['y'], y='x', alpha=alpha)
            thetas_a = pg.get_thetas()

            pg.fit(df)
            thetas_b = pg.get_thetas()

            pred = pg.predict([0])

            populate(results,
                    dataset_name,
                    'PolinomialGradient',
                    'y',
                    'x',
                    alpha,
                    thetas_a,
                    thetas_b,
                    pred)

    save_pickle(results)


def main():

    # run_experiment()
    describe_results()


if __name__ == '__main__':
    main()