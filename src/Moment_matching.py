# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:55:55 2023

@author: mohko200
"""
import numpy as np
import scipy.stats

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


def gaussianMultiplication(m1, v1, m2, v2):
    v = 1 / (1 / v1 + 1 / v2)
    m = (m1 / v1 + m2 / v2) * v
    return m, v


def gaussianDivision(m1, v1, m2, v2):
    return gaussianMultiplication(m1, v1, m2, -v2)


def approx_q(mu_c2t, sigma_c2t, y):
    trunc_q = pt_y(mu_c2t, sigma_c2t, y)
    mu_q = trunc_q.mean()
    var_q = trunc_q.var()
    return mu_q, var_q, trunc_q


def step_t2c(mu_1, var_1, mu_2, var_2, var_t, y):
    mu_c2t = mu_1 - mu_2
    var_c2t = var_1 + var_2 + var_t
    mu_q, var_q, trunc_q = approx_q(mu_c2t, np.sqrt(var_c2t), y)
    mu_t2c, var_t2c = gaussianDivision(mu_q, var_q, mu_c2t, var_c2t)

    return mu_t2c, var_t2c, trunc_q


def ps1_y(mu_1, var_1, mu_2, var_2, var_t, y):
    """
    :return:
    mu, var of player 2 given y
    """
    mu_t2c, var_t2c, _ = step_t2c(mu_1, var_1, mu_2, var_2, var_t, y)
    mu_c2s1 = mu_t2c + mu_2
    var_c2s1 = var_t + var_2 + var_t2c
    mu_s1y, var_s1y = gaussianMultiplication(mu_c2s1, var_c2s1, mu_1, var_1)
    return mu_s1y, var_s1y


def ps2_y(mu_1, var_1, mu_2, var_2, var_t, y):
    """
    :return:
    mu, var of player 2 given y
    """
    mu_t2c, var_t2c, _ = step_t2c(mu_1, var_1, mu_2, var_2, var_t, y)
    mu_c2s2 = mu_1 - mu_t2c
    var_c2s2 = var_t + var_1 + var_t2c
    mu_s2y, var_s2y = gaussianMultiplication(mu_c2s2, var_c2s2, mu_2, var_2)
    return mu_s2y, var_s2y


def createMomentMatching():
    def updateMomentMatching(mu1, var1, mu2, var2, var_t, y):
        mu1n, var1n = ps1_y(mu1, var1, mu2, var2, var_t, y)
        mu2n, var2n = ps2_y(mu1, var1, mu2, var2, var_t, y)
        return mu1n, var1n, mu2n, var2n

    return updateMomentMatching

