# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:55:55 2023

@author: mohko200
"""
import math
import numpy as np
import scipy.stats
from scipy.stats import truncnorm
import matplotlib.pyplot as plt


def pt_y(mu, sigma, y):
    if y == 1:
        a = 0
        b = np.inf
    elif y == -1:
        a = -np.inf
        b = 0
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
    return m, math.sqrt(v)


def gaussianDivision(m1, s1, m2, s2):
    v1 = s1 ** 2
    v2 = s2 ** 2
    m = (m1 * v2 - m2 * v1) / (v2 - v1)
    v = v1 * v2 / (v2 - v1)
    return m, math.sqrt(v)


def approx_q(mu_c2t, sigma_c2t, y):
    trunc_q = pt_y(mu_c2t, sigma_c2t, y)
    mu_q = trunc_q.mean()
    sigma_q = math.sqrt(trunc_q.var())
    return mu_q, sigma_q, trunc_q


def step_t2c(mu_1, sigma_1, mu_2, sigma_2, sigma_t, y):
    mu_c2t = mu_1 - mu_2
    sigma_c2t = math.sqrt(sigma_1 ** 2 + sigma_2 ** 2 + sigma_t ** 2)
    mu_q, sigma_q, trunc_q = approx_q(mu_c2t, sigma_c2t, y)
    mu_t2c, sigma_t2c = gaussianDivision(mu_q, sigma_q, mu_c2t, sigma_c2t)

    return mu_t2c, sigma_t2c, trunc_q


def ps1_y(mu_1, sigma_1, mu_2, sigma_2, sigma_t, y):
    mu_t2c, sigma_t2c, _ = step_t2c(mu_1, sigma_1, mu_2, sigma_2, sigma_t, y)
    mu_c2s1 = mu_t2c + mu_2
    sigma_c2s1 = math.sqrt(sigma_t ** 2 + sigma_2 ** 2 + sigma_t2c ** 2)
    mu_s1y, sigma_s1y = gaussianMultiplication(mu_c2s1, sigma_c2s1, mu_1, sigma_1)
    return mu_s1y, sigma_s1y


def ps2_y(mu_1, sigma_1, mu_2, sigma_2, sigma_t, y):
    mu_t2c, sigma_t2c, _ = step_t2c(mu_1, sigma_1, mu_2, sigma_2, sigma_t, y)
    mu_c2s2 = mu_1 - mu_t2c
    sigma_c2s2 = math.sqrt(sigma_t ** 2 + sigma_1 ** 2 + sigma_t2c ** 2)
    mu_s2y, sigma_s2y = gaussianMultiplication(mu_c2s2, sigma_c2s2, mu_1, sigma_1)
    return mu_s2y, sigma_s2y






mu_1 = 25  # The mean of the prior s1
sigma_1 = 3  # The variance of the prior s1
mu_2 = 25  # The mean of the prior s2
sigma_2 = 3  # The variance of the prior s1
sigma_t = 1.5  # The variance of p(t|y)
y = 1  # The measurement

mu_t2c, sigma_t2c, trunc_q = step_t2c(mu_1, sigma_1, mu_2, sigma_2, sigma_t, y)
mu_s1y, sigma_s1y = ps1_y(mu_1, sigma_1, mu_2, sigma_2, sigma_t, y)
mu_s2y, sigma_s2y = ps2_y(mu_1, sigma_1, mu_2, sigma_2, sigma_t, y)
x = np.linspace(-5, 40, 1000)
plt.plot(x, scipy.stats.norm(mu_t2c, sigma_t2c).pdf(x), color="r")
plt.plot(x, trunc_q.pdf(x), color="b")
plt.plot(x, scipy.stats.norm(mu_s1y, sigma_s1y).pdf(x), color="g")
plt.plot(x, scipy.stats.norm(mu_s2y, sigma_s2y).pdf(x), color="y")

plt.show()
