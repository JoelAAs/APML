import numpy as np
import scipy.stats


def getAlphaBeta(y, consider_draw, drawstd = 5):
    if y == 1:
        a = (drawstd if consider_draw else 0)
        b = np.inf
    elif y == -1:
        a = -np.inf
        b = (-drawstd if consider_draw else 0)
    elif y == 0 and consider_draw:
        a = -drawstd
        b = drawstd
    else:
        raise ValueError(f"Illegal value of y:{y}")
    return a, b

def py(mu, sigma, y, consider_draw):
    t_norm = scipy.stats.norm(loc=mu, scale=sigma)
    a, b = getAlphaBeta(y, consider_draw)
    return t_norm.cdf(b) - t_norm.cdf(a)


def pt_y(mu, sigma, y, consider_draw):
    a, b = getAlphaBeta(y, consider_draw)
    alpha, beta = (a - mu) / sigma, (b - mu) / sigma
    return scipy.stats.truncnorm(
        a=alpha,
        b=beta,
        loc=mu,
        scale=sigma
    )


def gaussianMultiplication(m1, v1, m2, v2):
    v = 1 / (1 / v1 + 1 / v2)
    m = (m1 / v1 + m2 / v2) * v
    return m, v


def gaussianDivision(m1, v1, m2, v2):
    return gaussianMultiplication(m1, v1, m2, -v2)


def approx_q(mu_c2t, sigma_c2t, y, consider_draw):
    trunc_q = pt_y(mu_c2t, sigma_c2t, y, consider_draw)
    mu_q = trunc_q.mean()
    var_q = trunc_q.var()
    #print(f"{mu_c2t}, {sigma_c2t}, {y} -> {mu_q}, {var_q}")
    return mu_q, var_q, trunc_q


def step_t2c(mu_1, var_1, mu_2, var_2, var_t, y, consider_draw):
    mu_c2t = mu_1 - mu_2
    var_c2t = var_1 + var_2 + var_t
    mu_q, var_q, trunc_q = approx_q(mu_c2t, np.sqrt(var_c2t), y, consider_draw)
    mu_t2c, var_t2c = gaussianDivision(mu_q, var_q, mu_c2t, var_c2t)

    return mu_t2c, var_t2c, trunc_q


def ps1_y(mu_1, var_1, mu_2, var_2, var_t, y, consider_draw):
    """
    :return:
    mu, var of player 2 given y
    """
    mu_t2c, var_t2c, _ = step_t2c(mu_1, var_1, mu_2, var_2, var_t, y, consider_draw)
    mu_c2s1 = mu_t2c + mu_2
    var_c2s1 = var_t + var_2 + var_t2c
    mu_s1y, var_s1y = gaussianMultiplication(mu_c2s1, var_c2s1, mu_1, var_1)
    return mu_s1y, var_s1y


def ps2_y(mu_1, var_1, mu_2, var_2, var_t, y, consider_draw):
    """
    :return:
    mu, var of player 2 given y
    """
    mu_t2c, var_t2c, _ = step_t2c(mu_1, var_1, mu_2, var_2, var_t, y, consider_draw)
    mu_c2s2 = mu_1 - mu_t2c
    var_c2s2 = var_t + var_1 + var_t2c
    mu_s2y, var_s2y = gaussianMultiplication(mu_c2s2, var_c2s2, mu_2, var_2)
    return mu_s2y, var_s2y


def createMomentMatching(var_t, consider_draw=False):
    def updateMomentMatching(mu1, var1, mu2, var2, y):
        mu1n, var1n = ps1_y(mu1, var1, mu2, var2, var_t, y, consider_draw)
        mu2n, var2n = ps2_y(mu1, var1, mu2, var2, var_t, y, consider_draw)
        return mu1n, var1n, mu2n, var2n

    return updateMomentMatching
