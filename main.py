import argparse
from collections import OrderedDict

import numpy as np

from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_blobs, make_regression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from AdamRegressor import AdamRegressor
from LinearRegressor import LinearRegressor
from SDGRegressor import SDGRegressor as MySDGRegressor


def plot_loss(loss: list, title: str = ""):
    # construct a figure that plots the loss over time
    fig = plt.figure()
    plt.plot(np.arange(0, len(loss)), loss)
    fig.suptitle("Training Loss - " + title)
    plt.xlabel("iter #")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = vars(parser.parse_args())
    N = 400
    batch_size = min(N, 40)
    Nit = 500
    passo = 0.001
    (X, S) = make_blobs(n_samples=N, n_features=2, centers=2, cluster_std=2.5, random_state=95)
    #(X, S) = np.random.random((N, 2)), np.random.randint(0, 2, N)
    S = S * 2 - 1

    clf = LinearRegressor(gamma=0)
    clf.fit(X, S)
    W = clf.weights_
    print('Optimal solution')
    print(W)
    print("Obtained MSE = %.3f" % mean_squared_error(S, clf.predict(X)))
    # 0.5 é a media entre as duas classes !
    y_opt = (-W[0] - (W[1] * X)) / W[2]

    w1 = np.zeros((X.shape[1], 1))
    clf = SGDRegressor(loss='squared_loss', penalty=None, alpha=0.0,
                       learning_rate='invscaling', eta0=passo, power_t=0.5,
                       max_iter=Nit)
    clf.fit(X, S, coef_init=w1)
    W = np.append(clf.intercept_, clf.coef_)
    print('Sklearn SDG solution')
    print(W)
    print("Obtained MSE = %.3f after %d iterations" % (mean_squared_error(S, clf.predict(X)), clf.n_iter_))
    y_sdg = (- W[0] - (W[1] * X)) / W[2]

    w1 = np.zeros((X.shape[1], 1))
    clf = MySDGRegressor(eta0=passo, power_t=0.5, max_iter=Nit)
    clf.fit(X, S, batch_size=batch_size, coef_init=w1)
    W = clf.coef_
    print('My SDG solution')
    print(W)
    print("Obtained MSE = %.3f after %d iterations" % (mean_squared_error(S, clf.predict(X)), clf.n_iter_))
    y_msdg = (- W[0] - (W[1] * X)) / W[2]
    plot_loss(clf.loss_hist_, "My SDG, sis. sobredet.")

    w1 = np.zeros((X.shape[1], 1))
    clf = AdamRegressor(eta0=passo, power_t=0.5, max_iter=Nit)
    clf.fit(X, S, batch_size=batch_size, coef_init=w1)
    W = clf.coef_
    print('Adam solution')
    print(W)
    print("Obtained MSE = %.3f after %d iterations" % (mean_squared_error(S, clf.predict(X)), clf.n_iter_))
    y_adam = (- W[0] - (W[1] * X)) / W[2]
    plot_loss(clf.loss_hist_, "Adam, sis. sobredet.")

    # plot the original data along with our line of best fit
    plt.figure()
    plt.title("Sistema sobre sobredeterminado")
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=S)
    plt.plot(X, y_sdg, "b-", label='Sklearn SDG solution')
    plt.plot(X, y_adam, "g-", label='Adam solution')
    plt.plot(X, y_opt, "r-", label='Optimal solution')
    plt.plot(X, y_msdg, "c-", label='My SDG solution')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim((X[:, 1].min() - X[:, 1].std(), X[:, 1].max() + X[:, 1].std()))
    plt.grid()
    plt.show()

    passo = 0.0001
    (X, S) = make_blobs(n_samples=N, n_features=N*2, centers=2, cluster_std=2.5, random_state=95)
    # (X, S) = np.random.random((N, 2)), np.random.randint(0, 2, N)

    clf = LinearRegressor(gamma=0)
    clf.fit(X, S)
    print('Optimal solution')
    print("Obtained MSE = %.3f" % mean_squared_error(S, clf.predict(X)))
    # 0.5 é a media entre as duas classes !

    w1 = np.zeros((X.shape[1], 1))
    clf = SGDRegressor(loss='squared_loss', penalty=None, alpha=0.0,
                       learning_rate='invscaling', eta0=passo, power_t=0.5,
                       max_iter=Nit)
    clf.fit(X, S, coef_init=w1)
    print('Sklearn SDG solution')
    print("Obtained MSE = %.3f after %d iterations" % (mean_squared_error(S, clf.predict(X)), clf.n_iter_))

    w1 = np.zeros((X.shape[1], 1))
    clf = MySDGRegressor(eta0=passo, power_t=0.5, max_iter=Nit)
    clf.fit(X, S, batch_size=batch_size, coef_init=w1)
    print('My SDG solution')
    print("Obtained MSE = %.3f after %d iterations" % (mean_squared_error(S, clf.predict(X)), clf.n_iter_))
    plot_loss(clf.loss_hist_, "My SDG, sis. subdet.")

    w1 = np.zeros((X.shape[1], 1))
    clf = AdamRegressor(eta0=passo, power_t=0.5, max_iter=Nit)
    clf.fit(X, S, batch_size=batch_size, coef_init=w1)
    print('Adam solution')
    print("Obtained MSE = %.3f after %d iterations" % (mean_squared_error(S, clf.predict(X)), clf.n_iter_))
    plot_loss(clf.loss_hist_, "Adam, sis. subdet.")

