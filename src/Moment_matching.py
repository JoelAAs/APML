# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:55:55 2023

@author: mohko200
"""
import numpy as np
import scipy.stats
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

def py(mu, sigma, y):
    t_norm = scipy.stats.norm(loc=mu, scale=sigma)
    if y == 1:
        a = 1
        b = np.inf
    elif y == -1:
        a = -np.inf
        b = -1
    elif y == 0:
        a = -1
        b = 1
    else:
        raise ValueError(f"Illegal value of y:{y}")

    return t_norm.cdf(b) - t_norm.cdf(a)

def pt_y(mu, sigma, y):
    if y == 1:
        a = 1
        b = np.inf
    elif y == -1:
        a = -np.inf
        b = -1
    elif y == 0:
        a = -1
        b = 1
    else:
        raise ValueError(f"Illegal value of y:{y}")
    return scipy.stats.truncnorm(
        a=a,
        b=b,
        loc=mu,
        scale=sigma
    )


def gaussianMultiplication(m1, s1, m2, s2):
    v1 = s1 ** 2
    v2 = s2 ** 2
    v = 1 / (1 / v1 + 1 / v2)
    m = (m1 / v1 + m2 / v2) * v
    return m, np.sqrt(v)


def gaussianDivision(m1, s1, m2, s2):
    m, v = gaussianMultiplication(m1, s1, m2, -s2)
    return m, np.sqrt(v)


def approx_q(mu_c2t, sigma_c2t, y):
    trunc_q = pt_y(mu_c2t, sigma_c2t, y)
    mu_q = trunc_q.mean()
    sigma_q = np.sqrt(trunc_q.var())
    return mu_q, sigma_q, trunc_q


def step_t2c(mu_1, sigma_1, mu_2, sigma_2, sigma_t, y):
    mu_c2t = mu_1 - mu_2
    sigma_c2t = np.sqrt(sigma_1 ** 2 + sigma_2 ** 2 + sigma_t ** 2)
    mu_q, sigma_q, trunc_q = approx_q(mu_c2t, sigma_c2t, y)
    mu_t2c, sigma_t2c = gaussianDivision(mu_q, sigma_q, mu_c2t, sigma_c2t)

    return mu_t2c, sigma_t2c, trunc_q


def ps1_y(mu_1, sigma_1, mu_2, sigma_2, sigma_t, y):
    """
    :return:
    mu, sigma of player 2 given y
    """
    mu_t2c, sigma_t2c, _ = step_t2c(mu_1, sigma_1, mu_2, sigma_2, sigma_t, y)
    mu_c2s1 = mu_t2c + mu_2
    sigma_c2s1 = np.sqrt(sigma_t ** 2 + sigma_2 ** 2 + sigma_t2c ** 2)
    mu_s1y, sigma_s1y = gaussianMultiplication(mu_c2s1, sigma_c2s1, mu_1, sigma_1)
    return mu_s1y, sigma_s1y


def ps2_y(mu_1, sigma_1, mu_2, sigma_2, sigma_t, y):
    """
    :return:
    mu, sigma of player 2 given y
    """
    mu_t2c, sigma_t2c, _ = step_t2c(mu_1, sigma_1, mu_2, sigma_2, sigma_t, y)
    mu_c2s2 = mu_1 - mu_t2c
    sigma_c2s2 = np.sqrt(sigma_t ** 2 + sigma_1 ** 2 + sigma_t2c ** 2)
    mu_s2y, sigma_s2y = gaussianMultiplication(mu_c2s2, sigma_c2s2, mu_2, sigma_2)
    return mu_s2y, sigma_s2y


def createMomentMatching():
    def updateMomentMatching(mu1, sigma1, mu2, sigma2, y, Sigma_t):
        sigma_t = np.sqrt(Sigma_t)
        mu1n, sigma1n = ps1_y(mu1, sigma1, mu2, sigma2, sigma_t, y)
        mu2n, sigma2n = ps2_y(mu1, sigma1, mu2, sigma2, sigma_t, y)
        return mu1n, sigma1n, mu2n, sigma2n

    return updateMomentMatching


def test():
    mu0, sigma0 = 25, 25 / 3
    sigma_t = 25 / 6
    y = -1

    x = np.linspace(0, 40, 200)
    plt.plot(x, scipy.stats.norm(mu1, s1).pdf(x), "r")
    plt.plot(x, scipy.stats.norm(mu2, s2).pdf(x), "g")
    plt.show()
