# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from gibbs import sample, gaussian_approx
from adf import ADFdf

# %%

# Q4 - Run a single match
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

# Q5, Q6
def rankTeams():
    # Load dataframe from file
    seriesA_df = pd.read_csv("data/SerieA.csv", sep=",")

    # Choose hyper parameters
    mu0, sigma0 = 25, 25/3
    Sigma_t = 1.5
    nSamples = 1000
    nBurn = 5 

    # Run ADF on the dataframe rows
    teams, skills, accuracy = ADFdf(seriesA_df, mu0, sigma0, Sigma_t, nSamples, nBurn)

    # Tabulate resulting posteriors
    idx = np.flip(np.argsort(skills[:,0]))
    skilltable = np.column_stack((1+np.arange(len(teams)), teams[idx], skills[idx]))
    print(tabulate(skilltable,
                    headers=["Rank", "Team", "mu", "sigma", "Games"]))
   
    print(f"Prediction accuray: {accuracy}")

    # print(tabulate(skilltable,
    #                 headers=["rank", "team", "mu", "sigma", "games"],
    #                 floatfmt=".2f",
    #                 tablefmt="latex_raw"))

rankTeams()

# %%
results_df = pd.read_csv("data/SerieA.csv", sep=",")
results_df["diff"] = results_df.score1 - results_df.score2

players = pd.concat([results_df['team1'], results_df['team2']]).unique()
playerIDs = {players[i]:i for i in range(len(players))}

# %%

winner = lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
results_df["winner"] = results_df["diff"].apply(winner)
results_df = results_df[results_df.winner != 0]
teamFilter = "Juventus"
print(results_df[results_df["team1"] == teamFilter]["winner"])
print(results_df[results_df["team2"] == teamFilter]["winner"])
