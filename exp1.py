import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

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


def prepare_each(ys, point):

    typ = point['Type']
    t0 = point['Thetas B'][0]
    t1 = point['Thetas B'][1]

    xs = []

    for y in ys:

        if typ == 'LinearGradient':
            xs.append(t0*y + t1)
        else:            
            xs.append(t0*(y**2) + t1)

    return xs


def prepare_info(df, point_z, point_x):

    mini = min(df['y'])
    ys = np.linspace(mini, 0)

    xs = prepare_each(ys, point_x)
    zs = prepare_each(ys, point_z)

    return xs, ys, zs


def plot_df(df, ax):

    # Plot data
    ax.scatter(df['x'], df['y'], df['z'], c='#229389', alpha=0.5)

    # Customize
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.2, 0)
    ax.set_zlim(0, 0.5)

    ax.plot([-2.5, -2.5, 2.5, 2.5], [0, -1, -1, 0], zdir='z', zs=0, c='#005660')
    ax.plot([-1.25, -1.25, 1.25, 1.25], [0, 0.2, 0.2, 0], zdir='z', zs=0, c='#005660')
    ax.plot([-3, 3], [0, 0.0], zdir='z', zs=0, c='#005660')

    ax.xaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.fill = False
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.fill = False
    ax.zaxis.pane.set_edgecolor('gray')
    ax.grid(False)


def plot_line(df, xs, ys, zs, point):

    # Configure 3D
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(111, projection='3d')

    plot_df(df, ax)

    first_point = list(df.iloc[0][['x', 'y', 'z']])

    ax.plot(xs, ys, zs, c='#57C3AD')

    # plt.show()
    data = point['Dataset']
    typ = point['Type']
    alpha = point['Alpha']
    plt.savefig(f'plots/{data}_{typ}_{alpha}.png')



def plot_all(datasets, results):

    for i in range(0, results.shape[0], 2):

        dataset = datasets[results.iloc[i]['Dataset']]

        point_z = results.iloc[i]
        point_x = results.iloc[i+1]

        xs, ys, zs = prepare_info(dataset, point_z, point_x)

        plot_line(dataset, xs, ys, zs, point_z)


def describe_results():

    datasets = {}

    for dataset in ['./datasets/kick1.dat', './datasets/kick2.dat']:

        dataset_name = dataset.split('/')[-1]
        df = pd.read_csv(dataset, sep=' ')
        df['y'] = -df['y']

        datasets[dataset_name] = df

    results = load_pickle()
    results = pd.DataFrame(results)
    print(results)

    plot_all(datasets, results)


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