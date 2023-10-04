# %%
import math
import pandas as pd
from scipy.stats import truncnorm, multivariate_normal, norm
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt

# %%

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
def Ps1s2_t(t, mu_1, mu_2, sigma_1, sigma_2, sigma_t):
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


# Sample from the posterior P(s1,s2|t,y)
def sample(y, mu_1, mu_2, sigma_1, sigma_2, sigma_t, nSamples):
    S_1 = np.zeros(nSamples)
    S_2 = np.zeros(nSamples)
    T = np.zeros(nSamples)

    s_1, s_2 = 0, 0
    for k in range(nSamples):
        t = Pt_s1s2y(s_1, s_2, y, sigma_t)
        s_1, s_2 = Ps1s2_t(t, mu_1, mu_2, sigma_1, sigma_2, sigma_t)
        S_1[k] = s_1
        S_2[k] = s_2
        T[k] = t
    
    return S_1, S_2, T

# Finds the underlying normal distribution
def gaussian_approx(s_vec):
    s_var = math.sqrt(np.var(s_vec))
    s_mean = np.mean(s_vec)
    return norm(loc=s_mean, scale=s_var), s_mean, s_var


# Q4 - Run a single match
def singleMatch():

    # starting values
    y = 1
    mu_1 = 25
    mu_2 = 25
    sigma_1 = 3
    sigma_2 = 3
    sigma_t = 1.5
    N_samples = 1000
    n_burn = 5
    S_1, S_2, T = sample(y, mu_1, mu_2, sigma_1, sigma_2, sigma_t, N_samples)

    # Q4.1 burn-in is about 5 values
    gen = np.linspace(0, N_samples - 1, N_samples)
    plt.plot(gen, S_1, "g")
    plt.plot(gen, S_2, "r")
    plt.plot(gen, T, "b")
    plt.show()

    # Q 4.2 gaussian_approx
    # Q 4.3
    s1_approx, _, _ = gaussian_approx(S_1[n_burn:])
    s2_approx, _, _ = gaussian_approx(S_2[n_burn:])
    
    x = np.linspace(0, 40, 200)
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

singleMatch()

# %% 

# Assumed Density Filtering
def ADF(score_df, mu0, sigma0,
        sigma_t, N_samples, n_burn):
    mean_var_mat = np.zeros([score_df.shape[0], 6])
    mu_var_dict = {
        team_name: [mu0, sigma0, 0]
        for team_name in set(score_df.team1.tolist() + score_df.team2.tolist())
    }
    i = 0
    for _, row in score_df.iterrows():
        team1, team2 = row["team1"], row["team2"]
        mu_1, sigma_1, matchN_1 = mu_var_dict[team1]
        mu_2, sigma_2, matchN_2 = mu_var_dict[team2]
        y = row["winner"]

        msg = f"{row['team1']}: mu:{mu_1} sigma: {sigma_1}\n"
        msg += f"{row['team2']}: mu:{mu_2} sigma: {sigma_2}\n"
        print(msg)

        S_1, S_2, _ = sample(
                            y,
                            mu_1,
                            mu_2,
                            sigma_1,
                            sigma_2,
                            sigma_t,
                            N_samples)

        _, mu_1, sigma_1 = gaussian_approx(S_1[n_burn:])
        _, mu_1, sigma_1 = gaussian_approx(S_2[n_burn:])

        mu_var_dict[team1] = [mu_1, sigma_1, matchN_1+1]
        mu_var_dict[team2] = [mu_2, sigma_2, matchN_2+1]
        mean_var_mat[i,:] = [mu_1, sigma_1, matchN_1,
                             mu_2, sigma_2, matchN_2]
        i += 1
    return mean_var_mat


# Q5
def rankTeams():
    # Load dataframe from file
    seriesA_df = pd.read_csv("data/SerieA.csv", sep=",")

    # Adds a column for the winner
    seriesA_df["diff"] = seriesA_df.score1 - seriesA_df.score2
    winner = lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    seriesA_df["winner"] = seriesA_df["diff"].apply(winner)
    seriesA_df = seriesA_df[seriesA_df.winner != 0]


    mu0, sigma0 = 25,3
    sigma_t = 1.5
    scores = ADF(seriesA_df, mu0, sigma0, sigma_t, 100, 5)
    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.rename(
        {
            0: "mu1",
            1: "sigma_1",
            2: "matchN1",
            3: "mu2",
            4: "sigma_2",
            5: "matchN2"
        }, axis=1)

    series_A_scored = pd.concat([seriesA_df.reset_index(drop=True),
                                 scores_df], axis=1)

    part1 = series_A_scored[["team1", "mu1", "sigma_1", "matchN1"]]
    part1 = part1.rename({
        "team1": "team", "mu1": "mu", "sigma_1": "sigma", "matchN1": "matchN"
    }, axis=1)
    part2 = series_A_scored[["team2", "mu2", "sigma_2", "matchN2"]]
    part2 = part2.rename({
        "team2": "team", "mu2": "mu", "sigma_2": "sigma", "matchN2": "matchN"
    }, axis=1)

    matched_results = pd.concat([part1, part2])
    matched_results.to_csv("test.csv")

rankTeams()

# %%
