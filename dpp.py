import numpy as np
from itertools import product
"""
Determinantal point process sampling procedures based
on  (Fast Determinantal Point Process Sampling with
     Application to Clustering, Byungkon Kang, NIPS 2013)
"""

def build_similary_matrix(cov_function, items):
    """
    build the similarity matrix from a covariance function
    cov_function and a set of items. each pair of items
    is given to cov_function, which computes the similarity
    between two items.
    """
    L = np.zeros((len(items), len(items)))
    for i in range(len(items)):
        for j in range(i, len(items)):
            L[i, j] = cov_function(items[i], items[j])
            L[j, i] = L[i, j]
    return L


def exp_quadratic(sigma):
    """
    exponential quadratic covariance function
    """
    def f(p1, p2):
        return np.exp(-0.5 * (((p1 - p2)**2).sum()) / sigma**2)
    return f


def sample(items, L, max_nb_iterations=1000, rng=np.random):
    """
    Sample a list of items from a DPP defined
    by the similarity matrix L. The algorithm
    is iterative and runs for max_nb_iterations.
    The algorithm used is from
    (Fast Determinantal Point Process Sampling with
    Application to Clustering, Byungkon Kang, NIPS 2013)
    """
    Y = rng.choice((True, False), size=len(items))
    L_Y = L[Y, :]
    L_Y = L_Y[:, Y]
    L_Y_inv = np.linalg.inv(L_Y)

    for i in range(max_nb_iterations):
        u = rng.randint(0, len(items))

        c_u = L[u:u+1, :]
        c_u = c_u[:, u:u+1]
        b_u = L[Y, :]
        b_u = b_u[:, u:u+1]
        if Y[u] == False:
            p_include_U = min(1, c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u))
            if rng.uniform() <= p_include_U:
                d_u = (c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u))
                upleft = (L_Y_inv +
                          np.dot(np.dot(np.dot(L_Y_inv, b_u), b_u.T),
                                 L_Y_inv) / d_u)
                upright = -np.dot(L_Y_inv, b_u) / d_u
                downleft = -np.dot(b_u.T, L_Y_inv) / d_u
                downright = d_u
                L_Y_inv = np.bmat([[upleft, upright], [downleft, downright]])
                Y[u] = True
                L_Y = L[Y, :]
                L_Y = L_Y[:, Y]
        else:
            p_remove_U = min(1, 1./(c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u)))
            if rng.uniform() <= p_remove_U:
                l = L_Y_inv.shape[0] - 1
                D = L_Y_inv[0:l, :]
                D = D[:, 0:l]
                e = L_Y_inv[0:l, :]
                e = e[:, l:l+1]
                f = L_Y_inv[l:l+1, :]
                f = f[:, l:l+1]
                L_Y_inv = D - np.dot(e, e.T) / f
                Y[u] = False
                L_Y = L[Y, :]
                L_Y = L_Y[:, Y]
    return np.array(items)[Y]


def sample_k(items, L, k, max_nb_iterations=1000, rng=np.random):
    """
    Sample a list of k items from a DPP defined
    by the similarity matrix L. The algorithm
    is iterative and runs for max_nb_iterations.
    The algorithm used is from
    (Fast Determinantal Point Process Sampling with
    Application to Clustering, Byungkon Kang, NIPS 2013)
    """
    initial = rng.choice(range(len(items)), size=k, replace=False)
    X = [False] * len(items)
    for i in initial:
        X[i] = True
    X = np.array(X)
    for i in range(max_nb_iterations):
        u = rng.choice(np.arange(len(items))[X])
        v = rng.choice(np.arange(len(items))[~X])
        Y = X.copy()
        Y[u] = False
        L_Y = L[Y, :]
        L_Y = L_Y[:, Y]
        L_Y_inv = np.linalg.inv(L_Y)

        c_v = L[v:v+1, :]
        c_v = c_v[:, v:v+1]
        b_v = L[Y, :]
        b_v = b_v[:, v:v+1]
        c_u = L[u:u+1, :]
        c_u = c_u[:, u:u+1]
        b_u = L[Y, :]
        b_u = b_u[:, u:u+1]

        p = min(1, c_v - np.dot(np.dot(b_v.T, L_Y_inv), b_v) /
                (c_u - np.dot(np.dot(b_u.T, L_Y_inv.T), b_u)))
        if rng.uniform() <= p:
            X = Y[:]
            X[v] = True
    return np.array(items)[X]


def test():
    x = np.arange(1, 100)
    L = build_similary_matrix(exp_quadratic(sigma=0.1),
                              x)
    for i in range(10):
        #print(sample_k(x, L, 10))
        print(sample(x, L))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #test()
    #sys.exit(0)
    x = np.arange(0, 1, 0.1)
    y = np.arange(0, 1, 0.1)
    z = np.array(list(product(x, y)))
    sigmas = [0.0001, 0.1, 1, 2, 10]
    k = 1
    for sigma in sigmas:
        plt.subplot(1, len(sigmas) + 1, k)
        L = build_similary_matrix(exp_quadratic(sigma=sigma), z)
        selected_by_dpp = sample(z, L)
        plt.scatter(selected_by_dpp[:, 0], selected_by_dpp[:, 1])
        plt.title("DPP(sigma={0})".format(sigma))
        k += 1

    plt.subplot(1, len(sigmas) + 1, k)
    selected_by_random = z[np.random.choice((True, False),
                           size=len(z))]
    plt.scatter(selected_by_random[:, 0], selected_by_random[:, 1])
    plt.title("random")
    plt.show()
