# %%
from scipy.stats import truncnorm, norm
from scipy.special import erfc
import numpy as np
import numpy.linalg as lg

# P(t|s1,s2,y)
def Pt_s1s2y(s1, s2, y, Sigma_t):
    mean = s1 - s2
    sigma_t = np.sqrt(Sigma_t)
    if y > 0:
        lim1 = -mean / Sigma_t
        lim2 = np.inf
    else:
        lim1 = -np.inf
        lim2 = -mean / Sigma_t

    t_new = truncnorm.rvs(
        a=lim1,
        b=lim2,
        loc=mean,
        scale=sigma_t,
        size=1
    )[0]
    return t_new

# P(s1,s2|t)
def Ps1s2_t(t, mu_1, mu_2, sigma_1, sigma_2, Sigma_t):
    cov_s = np.array([[sigma_1**2, 0], [0, sigma_2**2]])
    A = np.array([[1, -1]])
    Sigma_s1s2_t = lg.inv(lg.inv(cov_s) + A.T@A / Sigma_t)
    mu_s1s2_t = Sigma_s1s2_t @ (
            lg.inv(cov_s) @ np.array([mu_1, mu_2]).T +
            A[0] / Sigma_t * t
    )
    s1,s2 = np.random.multivariate_normal(mu_s1s2_t,
                                      Sigma_s1s2_t,
                                      1)[0]
    return s1,s2

# P(y=1)
def Py_s1s2(mu1, mu2, sigma1, sigma2, Sigma_t):
    mu = mu1 - mu2
    sigma = Sigma_t + (sigma1**2 + sigma2**2)
    return erfc(-mu/np.sqrt(2)/sigma)/2 # P(t>0), t~N(mu,sigma^2)


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


# Creates a callable Bayesian updater that uses Gibbs sampling
def createGibbsUpdater(nSamples, nBurn):
    def updateGibbs(mu1,sigma1, mu2,sigma2, y, Sigma_t):
        # Sample from the posterior
        s1s, s2s, _ = sample(y,
                            mu1, mu2,
                            sigma1, sigma2,
                            Sigma_t,
                            nSamples)

        # Estimate the new parameters from the normal distributions
        _, mu1, sigma1 = gaussian_approx(s1s[nBurn:])
        _, mu2, sigma2 = gaussian_approx(s2s[nBurn:])
        return mu1, sigma1, mu2, sigma2
    return updateGibbs

