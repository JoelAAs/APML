from scipy.stats import truncnorm, multivariate_normal
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt


def P_tssy(s1, s2, y, sigma_t):
    mean = s1 - s2

    b_bound = (np.inf if y > 0 else -np.inf)
    t_new = truncnorm(
        a=0,
        b=b_bound,
        loc=mean,
        scale=sigma_t
    ).rvs(
        size=1
    )[0]
    return t_new


def P_ssty(t, mu_1, mu_2, sigma_1, sigma_2, sigma_t):
    cov_s = np.matrix([[sigma_1, 0], [0, sigma_2]])
    A = np.matrix([1, -1])
    cov_s1s2_t = lg.inv(lg.inv(cov_s) + 1 / sigma_t * A.T.dot(A))
    mu_s1s2_t = cov_s1s2_t * (
            lg.inv(cov_s) * np.matrix([mu_1, mu_2]).T +
            A.T * 1 / sigma_t * t
    )
    S = multivariate_normal(
        mean=mu_s1s2_t.T.tolist()[0],
        cov=cov_s1s2_t
    ).rvs(
        size=1
    )
    return S


def sample(s_10, s_20, t_0, y, mu_1, mu_2, sigma_1, sigma_2, sigma_t, N_samples):
    S_1 = np.zeros(N_samples)
    S_2 = np.zeros(N_samples)
    T = np.zeros(N_samples)

    return _sample(
        s_10, s_20, t_0, y,
        mu_1, mu_2, sigma_1, sigma_2,
        sigma_t, N_samples, S_1, S_2, T, 0)


def _sample(s_1, s_2, t, y, mu_1, mu_2, sigma_1, sigma_2, sigma_t, N_samples, S_1, S_2, T, k):
    if k == N_samples:
        return S_1, S_2, T
    else:
        tn = P_tssy(s_1, s_2, y, sigma_t)
        s_1n, s_2n = P_ssty(t, mu_1, mu_2, sigma_1, sigma_2, sigma_t)
        S_1[k] = s_1
        S_2[k] = s_2
        T[k] = t

        return _sample(
            s_1n, s_2n, tn, y,
            mu_1, mu_2, sigma_1, sigma_2,
            sigma_t, N_samples, S_1, S_2, T, k+1)


### RUN
s_10 = 1
s_20 = 1
t_0 = 0
y = 1
mu_1 = 15
mu_2 = 15
sigma_1 = 2
sigma_2 = 2
sigma_t = 2
N_samples = 1000

S_1, S_2, T = sample(s_10, s_20, t_0, y, mu_1, mu_2, sigma_1, sigma_2, sigma_t, N_samples)

gen = np.linspace(0, N_samples-1, N_samples)
plt.plot(gen, S_1, "g")
plt.plot(gen, S_2, "r")
plt.plot(gen, T, "b")

plt.show()