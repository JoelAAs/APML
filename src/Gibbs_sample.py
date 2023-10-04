import math
import pandas as pd
from scipy.stats import truncnorm, multivariate_normal, norm
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt


def P_tssy(s1, s2, y, sigma_t):
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
            sigma_t, N_samples, S_1, S_2, T, k + 1)


def gaussian_approx(s_vec, n_burn=5):
    s_var = math.sqrt(np.var(s_vec[n_burn:]))
    s_mean = np.mean(s_vec[n_burn:])
    return norm(loc=s_mean, scale=s_var), s_mean, s_var


"""
The Gibbs sampler from Q.4 processes the result of one match to give a posterior distribution of
the skills given the match. We can use this posterior distribution as a prior for the next match in
what is commonly known as assumed density filtering (ADF). In this way, we can process a stream
of different matches between the players, each time using the posterior distribution of the previous
match as the prior for the current match.
m
• Use ADF with Gibbs sampling to process the matches in the SerieA dataset and estimate
the skill of all the teams in the dataset (each team is one Player with an associated skill
s i ). Note that there are draws in the dataset! For now, skip these matches and suppose that
they leave the skill unchanged for both players. For now, also skip the information of goals
scored. Only consider how won or lost the game.
What is the final ranking? Present the results in a suitable way. How can you interpret the
variance of the final skills?

• Change the order of the matches in the SerieA dataset at random and re-run ADF. Does the
result change? Why?
"""


ADF(seriesA_df, s_10, s_20, t_0, mu_1, sigma_1, sigma_t, N_samples=200, n_burn=5)

def ADF(score_df, s_10, s_20, t_0, mu_1, sigma_1, sigma_t, N_samples=200, n_burn=5):
    mean_var_mat = np.matrix([score_df.shape[0], 4])
    mu_var_dict = {
        team_name: [mu_1, sigma_1]
        for team_name in set(score_df.team1.tolist() + score_df.team2.tolist())
    }
    for i, row in score_df.iterrows():
        mu_1, sigma_1 = mu_var_dict[row["team1"]]
        mu_2, sigma_2 = mu_var_dict[row["team2"]]
        y = row["winner"]

        S_1, S_2, _ = sample(s_10, s_20, t_0, y, mu_1, mu_2, sigma_1, sigma_2, sigma_t, N_samples)

        _, mu_1, sigma_1 = gaussian_approx(S_1, n_burn)
        _, mu_1, sigma_1 = gaussian_approx(S_2, n_burn)

        mu_var_dict[row["team1"]] = [mu_1, sigma_1]
        mu_var_dict[row["team2"]] = [mu_2, sigma_2]
        mean_var_mat[i, ] = [mu_1, sigma_1, mu_2, sigma_2]
    return mean_var_mat


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
N_samples = 50
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
x = np.linspace(0, 40, 200)

s1_approx, _, _ = gaussian_approx(S_1, n_burn)
s2_approx, _, _ = gaussian_approx(S_2, n_burn)
p_s1 = s1_approx.pdf(x)
p_s2 = s2_approx.pdf(x)

# Q.4
bins = int(math.sqrt(N_samples))
plt.hist(S_1[n_burn:], bins=bins, color="r", density=True)
plt.plot(x, p_s1, color="darkred")
plt.hist(S_2[n_burn:], bins=bins, color="g", density=True)
plt.plot(x, p_s2, color="darkgreen")
plt.hist(T[n_burn:], bins=bins, color="b", density=True)
plt.legend(["S1", "P1", "S2", "P2", "T"])
plt.show()
a = "hej"
b = 12
c = "då"
print(f"{a} {b} {c}")

# Q 5
seriesA_df = pd.read_csv("data/SerieA.csv", sep=",")
seriesA_df["diff"] = seriesA_df.score1 - seriesA_df.score2
winner = lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
seriesA_df["winner"] = seriesA_df["diff"].apply(winner)
seriesA_df = seriesA_df[seriesA_df.winner != 0]

seriesA_df.groupby(["team1", "team2"])

ADF(seriesA_df, s_10, s_20, mu_1, sigma_1, sigma_t, N_samples=200, n_burn=5)
