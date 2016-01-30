from __future__ import print_function
import random
import cPickle as pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import seaborn as sns
from deepx.nn import *
from deepx.loss import *
from deepx.optimize import *
sns.set_style("white")

def plot(model_name, ypred, y):
    for i, name in enumerate(y.columns):
        plt.figure();
        plt.plot(y[name].as_matrix(), label='actual %s' % name, alpha=0.5);
        plt.plot(ypred[:, i], label='predicted %s' % name, alpha=0.5);
        plt.legend(loc='best')
        plt.savefig("out/%s-%s-regression.png" % (model_name, name),bbox_inches='tight')

def plot_model(model_name, model, X, y, **kwargs):
    print("Plotting", model_name)
    if model_name == 'svr':
        ypred = []
        for i, name in enumerate(y.columns):
            print(i, name)
            m = model().fit(X, y[name])
            ypred.append(m.predict(X))
        ypred = np.array(ypred).T
    elif model_name == 'net':
        net = Vector(8) >> Elu(800) >> Elu(800) >> Full(5)
        with open('net.pkl', 'r') as fp:
            net.set_state(pickle.load(fp))
        ypred = net.predict(X)
    else:
        model = model(**kwargs).fit(X, y)
        ypred = model.predict(X)
    plot(model_name, ypred, y)
    y = y.as_matrix()
    print("MSE:", ((ypred - y) ** 2).sum(axis=1).mean())
    mse = ((ypred - y) ** 2).sum(axis=1)
    plt.figure()
    plt.scatter(y[:, 4], mse, alpha=0.5)
    plt.savefig('out/%s-temp.png' % model_name)
    return ypred

if __name__ == "__main__":
    input = pd.read_csv('data/data.csv', index_col=0, parse_dates=True,date_parser=lambda a: pd.to_datetime(a, unit='s'))
    input.columns = ['NO2-A', 'NO2-W', 'O3-A', 'O3-W', 'CO-A', 'CO-W', 'PT', 'NC']

    target = pd.read_csv('data/target.csv', index_col=0, parse_dates=True)
    target.columns = ['CO-ppm', 'O3-ppm', 'NO-ppb', 'NO2-ppb', 'TEMP']

    data = input.join(target).dropna()

    X, y = data[input.columns], data[target.columns]

    plot_model('linear', LinearRegression, X, y)
    plot_model('ridge', Ridge, X, y)
    # plot_model('krr', KernelRidge, X, y)
    # plot_model('svr', SVR, X, y, kernel='rbf')
    plot_model('elastic', ElasticNet, X, y)
    plot_model('net', None, X, y)

