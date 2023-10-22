import numpy as np
import matplotlib.pyplot as plt

def scenario1(size=100, seed=2023):
    """
    data generation for hw2 problem 6, with sample size 2*n and seed=seed
    """
    mu1 = np.array([2, 1])
    mu2 = np.array([1, 2])
    sigma1 = np.eye(N=2, M=2)
    sigma2 = np.eye(N=2, M=2)
    rng = np.random.default_rng(seed=seed)
    green = rng.multivariate_normal(mean=mu1, cov=sigma1, size=size)
    red = rng.multivariate_normal(mean=mu2, cov=sigma2, size=size)
    fig, ax = plt.subplots()
    ax.scatter(green[:, 0], green[:, 1], c='g')
    ax.scatter(red[:, 0], red[:, 1], c='r')
    # fig.show()
    predictors = np.vstack((green, red))
    targets = np.concatenate(([1]*size, [0]*size), axis=0)
    # samples = np.concatenate((predictors, targets.reshape(size*2, 1)), axis=1)
    return predictors, targets


# define a function to generate data
def gendata2(size, mu1, mu2, sig1, sig2, seed):
    rng2 = np.random.default_rng(seed=seed)
    mean1 = rng2.choice(mu1, size=size, replace=True)
    mean2 = rng2.choice(mu2, size=size, replace=True)
    green = np.zeros([size, 2])
    red = np.zeros([size, 2])
    for i in range(size):
        green[i, ] = rng2.multivariate_normal(mean=mean1[i, ], cov=sig1)
        red[i, ] = rng2.multivariate_normal(mean=mean2[i, ], cov=sig2)
    x = np.vstack((green, red))
    return x


def scenario2(size=100, seed=2023):
    """
    data generation for hw2 problem 6, with sample size 2*n and seed=seed
    """
    seed_center = 16
    rng = np.random.default_rng(seed=seed_center)
    mean_mu = [1, 0]
    mean_nu = [0, 1]
    cov = np.array([[1, 0], [0, 1]])

    # center for green and red classes
    mu = rng.multivariate_normal(mean=mean_mu, cov=cov, size=10)
    nu = rng.multivariate_normal(mean=mean_nu, cov=cov, size=10)

    predictors = gendata2(size, mu, nu, cov / 5, cov / 5, seed=seed)
    targets = np.concatenate(([1]*size, [0]*size), axis=0)
    # samples = np.concatenate((predictors, targets.reshape(size*2, 1)), axis=1)

    fig, ax = plt.subplots()
    ax.scatter(predictors[0:size, 0], predictors[:size, 1], c='g')
    ax.scatter(predictors[size:, 0], predictors[size:, 1], c='r')
    # fig.show()
    return predictors, targets

# test
# x, y = scenario2(size=100, seed=2000)
# print(x.shape, y.shape)

