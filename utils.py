import numpy as np


def data_iter(X, y, batch_size):
    # loop over our dataset `X` in mini-batches of size `batchSize`
    #shuffle date
    xy = np.column_stack((X, y))
    np.random.shuffle(xy)
    X, y = np.hsplit(xy, [-1])
    y = y.reshape(y.shape[0],)
    for i in np.arange(0, X.shape[0], batch_size):
        # yield a tuple of the current batched data and labels
        yield (X[i:i + batch_size], y[i:i + batch_size])
