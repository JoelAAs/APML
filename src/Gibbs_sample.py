# %%
import math
import pandas as pd
from scipy.stats import truncnorm, multivariate_normal, norm
from scipy.special import erfc
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
from tabulate import tabulate

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
    return erfc(-mu/math.sqrt(2)/sigma)/2


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
    s_var = math.sqrt(np.var(s_vec))
    s_mean = np.mean(s_vec)
    return norm(loc=s_mean, scale=s_var), s_mean, s_var


# Q4 - Run a single match
def singleMatch():

    # starting values
    y = 1
    mu_1 = 25
    mu_2 = 25
    sigma_1 = 25/3
    sigma_2 = 25/3
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
def ADF(nPlayers:int, results:np.array,
        mu0, sigma0, Sigma_t, nSamples, n_burn):
    
    history = np.zeros((len(results), 6))
    playerSkills = np.array([[mu0, sigma0, 0]] * nPlayers, dtype=np.float32)

    i,nCorrect = 0,0
    for row in results:
        p1, p2, y = row
        mu_1, sigma_1, matchNum_1 = playerSkills[p1]
        mu_2, sigma_2, matchNum_2 = playerSkills[p2]
       
        # msg = f"{p1}: mu:{mu_1} sigma: {sigma_1}\n"
        # msg += f"{p2}: mu:{mu_2} sigma: {sigma_2}\n"
        # msg += f"pred = {Py_s1s2(mu_1, mu_2, sigma_1, sigma_2, Sigma_t)}, result = {y}\n"
        # print(msg)

        # Predict the outcome, count the hits
        nCorrect += round(Py_s1s2(mu_1, mu_2, sigma_1, sigma_2, Sigma_t))*2-1 == y

        # Bayesian update with y - sample from the posterior
        S_1, S_2, _ = sample(y,
                            mu_1, mu_2,
                            sigma_1, sigma_2,
                            Sigma_t,
                            nSamples)

        # Estimate the parameters of the new normal distributions
        _, mu_1, sigma_1 = gaussian_approx(S_1[n_burn:])
        _, mu_1, sigma_1 = gaussian_approx(S_2[n_burn:])

        playerSkills[p1,:] = [mu_1, sigma_1, matchNum_1+1]
        playerSkills[p2,:] = [mu_2, sigma_2, matchNum_2+1]
        history[i,:] = [mu_1, sigma_1, matchNum_1,
                        mu_2, sigma_2, matchNum_2]
        i += 1
    return playerSkills, nCorrect/i

# ADF on pandas dataframe
def ADFpd(results_df, mu0, sigma0, sigma_t, nSamples, n_burn):
    # Adds a column for the winner
    results_df["diff"] = results_df.score1 - results_df.score2
    winner = lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    results_df["winner"] = results_df["diff"].apply(winner)
    results_df = results_df[results_df.winner != 0]

    # Assign numbers to the players
    players = pd.concat([results_df['team1'], results_df['team2']]).unique()
    playerIDs = {players[i]:i for i in range(len(players))}
    
    # Convert to numpy array
    results = np.zeros((results_df.shape[0],3),dtype=np.int32) # p1,p2,result
    i = 0
    for _, row in results_df.iterrows():
        results[i,:] = np.array([playerIDs[row["team1"]],
                                 playerIDs[row["team2"]],
                                 row["winner"]])
        i+=1
    
    playerSkills, accuracy = ADF(len(players), results,
                                mu0, sigma0, sigma_t, nSamples, n_burn)
    return players, playerSkills, accuracy



# Q5
def rankTeams():
    # Load dataframe from file
    seriesA_df = pd.read_csv("data/SerieA.csv", sep=",")

    # Chose hyper parameters
    mu0, sigma0 = 25,3
    Sigma_t = 1.5
    nSamples = 50
    nBurn = 5 

    # Run ADF on the dataframe rows
    teams, skills, accuracy = ADFpd(seriesA_df, mu0, sigma0, Sigma_t, nSamples, nBurn)

    # Tabulate resulting posteriors
    idx = np.flip(np.argsort(skills[:,0]))
    skilltable = np.column_stack((np.arange(len(teams)), teams[idx], skills[idx]))
    print(tabulate(skilltable,
                    headers=["Rank", "Team", "mu", "sigma", "Games"]))
   
    print(f"Prediction accuray: {accuracy}")

    # print(tabulate(skilltable,
    #                 headers=["rank", "team", "mu", "sigma", "games"],
    #                 floatfmt=".2f",
    #                 tablefmt="latex_raw"))


rankTeams()

# %%

