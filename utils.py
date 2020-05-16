import numpy as np


def data_iter(X, y, batch_size):
    # loop over our dataset `X` in mini-batches of size `batchSize`
    for i in np.arange(0, X.shape[0], batch_size):
        # yield a tuple of the current batched data and labels
        yield (X[i:i + batch_size], y[i:i + batch_size])
