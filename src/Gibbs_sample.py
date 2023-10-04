import math

import scipy
from scipy.stats import truncnorm, multivariate_normal
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt


def Pt_ssy(s1, s2, y, sigma_t):
    """

    :param s1:
    :param s2:
    :param y:
    :param sigma_t:
    :return:
    float t_n+1
    """
    mean = s1 - s2
    a = -mean / sigma_t
    b_bound = (np.inf if y > 0 else -np.inf)
    t_new = truncnorm(
        a=a,
        b=b_bound,
        loc=mean,
        scale=sigma_t
    ).rvs(
        size=1
    )[0]
    return t_new


def Ps1_sty(t, mu_1, mu_2, sigma_1, sigma_2, sigma_t):
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
            sigma_t, N_samples, S_1, S_2, T, k + 1)


def gaussian_approx(s1_vec, s2_vec, n_burn=5):
    s_cov = np.cov(s1_vec[5:], s2_vec[5:])
    s_mean = [np.mean(s1_vec[5:]), np.mean(s2_vec[5:])]
    return multivariate_normal(mean=s_mean, cov=s_cov)


### RUN
s_10 = 0
s_20 = 0
t_0 = 0
y = 1
mu_1 = 25
mu_2 = 25
sigma_1 = 25 / 3
sigma_2 = 25 / 3
sigma_t = 25 / 6
N_samples = 2000
n_burn = 5
S_1, S_2, T = sample(s_10, s_20, t_0, y, mu_1, mu_2, sigma_1, sigma_2, sigma_t, N_samples)

# Q 4.1 like 5 values?
gen = np.linspace(0, N_samples - 1, N_samples)
plt.plot(gen, S_1, "g")
plt.plot(gen, S_2, "r")
plt.plot(gen, T, "b")
plt.show()

# Q 4.2 gaussian_approx
# Q 4.3
x1 = np.linspace(0, 40, 200)
x = np.matrix([x1, [mu_2]*len(x1)]).T

gm = gaussian_approx(S_1, S_2, n_burn)
p = gm.pdf(x)

bins = int(math.sqrt(N_samples))
plt.hist(S_1[n_burn:], bins=bins, color="r", density=True)
plt.hist(S_2[n_burn:], bins=bins, color="g", density=True)
plt.hist(T[n_burn:], bins=bins, color="b", density=True)
plt.legend(["S1", "S2", "T"])
plt.show()

xx, yy = np.meshgrid(x1, x1)

x, y = np.mgrid[0:30:1, 0:30:1]
pos = np.dstack((x, y))
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contourf(x, y, gm.pdf(pos))
