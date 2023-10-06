# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from gibbs import Py_s1s2, sample, gaussian_approx, createGibbsUpdater
from adf import ADFdf
import scipy
from Moment_matching import *

# %%

# Q.4 - Run a single match
def singleMatch():

    # Choose hyper parameters
    y = 1
    mu_1, mu_2 = 25, 25
    sigma_1, sigma_2 = 25/3, 25/3
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

    # Q4.2 gaussian_approx
    # Q4.3
    s1_approx, _, _ = gaussian_approx(S_1[n_burn:])
    s2_approx, _, _ = gaussian_approx(S_2[n_burn:])
    
    x = np.linspace(0, 40, 200)
    p_s1 = s1_approx.pdf(x)
    p_s2 = s2_approx.pdf(x)

    # Q4.4
    bins = int(np.sqrt(N_samples))
    plt.hist(S_1[n_burn:], bins=bins, color="r", density=True)
    plt.plot(x, p_s1, color="darkred")
    plt.hist(S_2[n_burn:], bins=bins, color="g", density=True)
    plt.plot(x, p_s2, color="darkgreen")
    plt.hist(T[n_burn:], bins=bins, color="b", density=True)
    plt.legend(["S1", "P1", "S2", "P2", "T"])
    plt.show()

singleMatch()

# %%

# Q.5, Q.6
def rankFootballTeams():
    # Load dataframe from file
    seriesA_df = pd.read_csv("../data/SerieA.csv", sep=",")

    # Choose hyper parameters
    mu0, sigma0 = 25, 25/3
    Sigma_t = (25/6)**2
    nSamples = 100
    nBurn = 5

    # Run ADF on the dataframe rows
    update = createGibbsUpdater(nSamples, nBurn)

     # Predicts y
    def predict(mu1, mu2, sigma1, sigma2, Sigma_t):
        return round(Py_s1s2(mu1, mu2, sigma1, sigma2, Sigma_t))*2-1 
    
    teams, skills, accuracy, _ = ADFdf(seriesA_df, mu0, sigma0, Sigma_t,
                                    '','team1','team2', lambda row : np.sign(row["score1"] - row["score2"]),
                                    predict, update, False)

    
    # Tabulate resulting posteriors
    idx = np.flip(np.argsort(skills[:,0]))
    skilltable = np.column_stack((1+np.arange(len(teams)), teams[idx], skills[idx]))
    print(tabulate(skilltable,
                    headers=["Rank", "Team", "mu", "sigma", "Games"]))
    print(f"Prediction accuray: {accuracy}")

rankFootballTeams()

# %%


# Q.8
def momentMatchingVsGibbs():
    mu0, sigma0 = 25, 25 / 3
    sigma_t = 25 / 6
    y = 1
    N_samples = 1000

    # Gibbs
    S_1, S_2, _ = sample(y, mu0, mu0, sigma0, sigma0, sigma_t, N_samples)

    # Moment matching
    mu1mm, s1mm = ps1_y(mu0, sigma0, mu0, sigma0, sigma_t, y)
    mu2mm, s2mm = ps2_y(mu0, sigma0, mu0, sigma0, sigma_t, y)

    bins = int(np.sqrt(N_samples))
    x = np.linspace(0, 50, 200)
    plt.hist(S_1, bins=bins, color="darkred", density=True)
    plt.plot(x, scipy.stats.norm(mu1mm, s1mm).pdf(x), "r")
    plt.hist(S_2, bins=bins, color="darkgreen", density=True)
    plt.plot(x, scipy.stats.norm(mu2mm, s2mm).pdf(x), "g")
    plt.legend(["P(S1|y) MM",  "P(S2|y) MM", "P(S1|y) gibbs", "P(S2|y) gibbs"])
    plt.title("Moment Matching vs Gibbs sampling of S posterior")
    plt.show()

momentMatchingVsGibbs()

# %%

# Q.10
def rankFootballTeamsDraw():
    # Load dataframe from file
    seriesA_df = pd.read_csv("../data/SerieA.csv", sep=",")

    # Choose hyper parameters
    mu0, sigma0 = 25, 25/3
    Sigma_t = (25/6)**2
    
    # -------------------------
    # Without draws
    update = createMomentMatching()

    def predict(mu1, mu2, sigma1, sigma2, Sigma_t):
        return round(Py_s1s2(mu1, mu2, sigma1, sigma2, Sigma_t))*2-1 
    
    _, _, accuracy, _ = ADFdf(seriesA_df, mu0, sigma0, Sigma_t,
                              '','team1','team2', lambda row : np.sign(row["score1"] - row["score2"]),
                              predict, update, False)

    
    # -------------------------
    # With draws
    update = createMomentMatching()

    def predict_draws(mu1, mu2, sigma1, sigma2, Sigma_t):
        mu = mu1-mu2
        sigma = np.sqrt(sigma1**2 + sigma2**2 + Sigma_t)

        values = [py(mu, sigma, y) for y in range(-1, 2)]
        predicted_value = np.argmax(values) - 1
        return predicted_value

    teams, skills, accuracy_draw, _ = ADFdf(seriesA_df, mu0, sigma0, Sigma_t,
                                   '','team1', 'team2', lambda row: np.sign(row["score1"] - row["score2"]),
                                   predict_draws, update, False, consider_draw=True)

    # -------------------------
    # Tabulate resulting posteriors
    idx = np.flip(np.argsort(skills[:,0]))
    skilltable = np.column_stack((1+np.arange(len(teams)), teams[idx], skills[idx]))
    print(tabulate(skilltable,
                    headers=["Rank", "Team", "mu", "sigma", "Games"],
                    floatfmt=[".2f",".2f",".2f",".2f",".0f"],
                    tablefmt="latex_raw"))

    print(f"Prediction accuray without draws: {accuracy}")
    print(f"Prediction accuray with draws: {accuracy_draw}")


rankFootballTeamsDraw()



# %%


# # Q.9, Q.10
def trackTennisPlayers():

    # Load dataframe from file
    tennis_df = pd.read_csv("../data/tennis2.csv", sep=",")

    # Choose hyper parameters
    mu0, sigma0 = 25, 25/3
    Sigma_t = (25/6)**2

    # Choose update method
    update = createGibbsUpdater(nSamples=50, nBurn=5)
    update = createMomentMatching()

    # Predicts y
    def predict(mu1, mu2, sigma1, sigma2, Sigma_t):
        return round(Py_s1s2(mu1, mu2, sigma1, sigma2, Sigma_t))*2-1 

    decayRate = 1/365
    def decay(sigma,dt):
        dt = dt*decayRate
        sigma = sigma0 + (sigma-sigma0)*np.exp(-dt)
        return sigma

    players, skills, accuracy, history = ADFdf(tennis_df, mu0, sigma0, Sigma_t,
                                    'day','winner_name','loser_name', lambda row : 1,
                                    predict, update, False, False, decay)



    idx = np.flip(np.argsort(skills[:,0]))
    skilltable = np.column_stack((1+np.arange(len(players)), players[idx], skills[idx]))
    skilltable = skilltable[:20]

    print(tabulate(skilltable,
                    headers=["Rank", "Team", "mu", "sigma", "Games"],
                    floatfmt=[".2f",".2f",".2f",".2f",".0f"],
                    tablefmt="latex_raw"))
    print(f"Prediction accuray: {accuracy}")

    # Draw history of skill
    for i in idx[:5]:
        xs = history[i][:,0]
        ys = history[i][:,1]
        errs = history[i][:,2]*1.96 # 95% confidence
        plt.plot(xs,ys,label=players[i])
        plt.fill_between(xs, ys-errs, ys+errs, alpha=0.5)
        plt.legend()
                    
trackTennisPlayers()


 
# %%
