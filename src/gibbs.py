# %%
from scipy.stats import truncnorm, multivariate_normal, norm
from scipy.special import erfc
import numpy as np
import numpy.linalg as lg

# P(t|s1,s2,y)
def Pt_s1s2y(s1, s2, y, sigma_t):
    mean = s1 - s2
    if y > 0:
        lim1 = -mean / sigma_t
        lim2 = np.inf
    else:
        lim1 = -np.inf
        lim2 = -mean / sigma_t

    t_new = truncnorm(
        a=lim1,
        b=lim2,
        loc=mean,
        scale=sigma_t
    ).rvs(
        size=1
    )[0]
    return t_new

# P(s1,s2|t)
def Ps1s2_t(t, mu_1, mu_2, sigma_1, sigma_2, Sigma_t):
    cov_s = np.matrix([[sigma_1**2, 0], [0, sigma_2**2]])
    A = np.matrix([1, -1])
    cov_s1s2_t = lg.inv(lg.inv(cov_s) + 1 / Sigma_t * A.T.dot(A))
    mu_s1s2_t = cov_s1s2_t * (
            lg.inv(cov_s) * np.matrix([mu_1, mu_2]).T +
            A.T * 1 / Sigma_t * t
    )
    S = multivariate_normal(
        mean=mu_s1s2_t.T.tolist()[0],
        cov=cov_s1s2_t
    ).rvs(
        size=1
    )
    return S

# P(y=1)
def Py_s1s2(mu1, mu2, sigma1, sigma2, Sigma_t):
    mu = mu1 - mu2
    sigma = Sigma_t + (sigma1**2 + sigma2**2)
    return erfc(-mu/np.sqrt(2)/sigma)/2


# Sample from the posterior P(s1,s2|t,y)
def sample(y, mu_1, mu_2, sigma_1, sigma_2, Sigma_t, nSamples):
    S_1 = np.zeros(nSamples)
    S_2 = np.zeros(nSamples)
    T = np.zeros(nSamples)

    s_1, s_2 = 0, 0
    for k in range(nSamples):
        t = Pt_s1s2y(s_1, s_2, y, Sigma_t)
        s_1, s_2 = Ps1s2_t(t, mu_1, mu_2, sigma_1, sigma_2, Sigma_t)
        S_1[k] = s_1
        S_2[k] = s_2
        T[k] = t
    
    return S_1, S_2, T

# Finds the underlying normal distribution
def gaussian_approx(s_vec):
    s_var = np.sqrt(np.var(s_vec))
    s_mean = np.mean(s_vec)
    return norm(loc=s_mean, scale=s_var), s_mean, s_var

