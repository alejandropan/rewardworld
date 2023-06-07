import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.io import loadmat
import scipy.ndimage

def halfgaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Half-Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(0, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    return phi_x

def halfgaussian_filter1d(input, sigma, axis=-1, output=None,
                      mode="constant", cval=0.0, truncate=4.0):
    """
    Convolves a 1-D Half-Gaussian convolution kernel.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = halfgaussian_kernel1d(sigma, lw)
    origin = -lw // 2
    return scipy.ndimage.convolve1d(input, weights, axis, output, mode, cval, origin)

m = 5
Ncrossval = 10
lam = 0

variableE = loadmat('/Users/alexpan/Downloads/raw_data.mat')['variableE']
variableP = loadmat('/Users/alexpan/Downloads/raw_data.mat')['variableP']
Y = loadmat('/Users/alexpan/Downloads/raw_data.mat')['Y']
X = loadmat('/Users/alexpan/Downloads/raw_data.mat')['X']

n, p = X.shape
n, q = Y.shape

def make_fold(X, Y, s, fold):
    flag = np.ones(len(X), dtype=bool)
    mlim = min( [np.floor(fold*s), len(X)])
    flag[ int(np.floor((fold-1)*s)) : int(mlim)] = False  # This is test data. "s" is the size of the fold

    Xtrain = X[flag,:]
    Xtest  = X[~flag,:]

    Ytrain = Y[flag,:]
    Ytest  = Y[~flag,:]

    return Xtrain, Ytrain, Xtest, Ytest, flag

for idx in [1, 2]:
    if idx == 1:
        variable = variableP[0]
    else:
        variable = variableE[0]

    error = np.zeros((10, 2))

    for fold in range(1, Ncrossval+1):

        Xtrain, Ytrain, Xtest, Ytest, fl = make_fold(X, Y, n/Ncrossval, fold)

        Bols = np.linalg.inv(Xtrain.T @ Xtrain + lam*np.eye(p)) @ (Xtrain.T @ Ytrain)  # Estimate for B with ordinary least squares
        Yestimate = Xtrain @ Bols
        pca = PCA(n_components=m)
        pca.fit(Yestimate)
        V = pca.components_.T
        Brrr = Bols @ V @ V.T

        Yestimate = Xtest @ Brrr

        model = LinearRegression().fit(Yestimate, variable[~fl])
        error[fold-1, 0] = 1 - np.var(model.predict(Yestimate) - variable[~fl]) / np.var(variable[~fl])

        model = LinearRegression().fit(Ytrain, variable[fl])
        error[fold-1, 1] = 1 - np.var(model.predict(Ytest) - variable[~fl]) / np.var(variable[~fl])

    me = np.mean(error, axis=0)
    ee = np.std(error, axis=0) / np.sqrt(10)



 1 - np.var(Yestimate - Ytest) / np.var(Ytest)



    plt.figure(idx)
    plt.bar(1, me[1])  # All
    plt.bar(2, me[0])  # Subspace
    plt.errorbar(1, me[1], ee[1], ee[1])
    plt.errorbar(2, me[0], ee[0], ee[0])
    if idx == 1:
        plt.title("Regressing Position")
    else:
        plt.title("Regressing Evidence")


