# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time
from tabulate import tabulate
from gibbs import Py_s1s2, sample, gaussian_approx, createGibbsUpdater
from adf import ADFdf
from Moment_matching import createMomentMatching, ps1_y, ps2_y, py


import Moment_matching
from Moment_matching import createMomentMatching, ps1_y, ps2_y, py
import importlib
importlib.reload(Moment_matching)



import trueskill

def createTrueSkillUpdater():
    trueskill.setup(draw_probability=0)
    def updater(mu1, var1, mu2, var2, y):
        widx = -int((y-1)/2)    
        rs = (trueskill.Rating(mu1,np.sqrt(var1)),
              trueskill.Rating(mu2,np.sqrt(var2)))
        rs = trueskill.rate_1vs1(rs[widx], rs[1-widx])
        return rs[widx].mu, rs[widx].sigma**2, rs[1-widx].mu, rs[1-widx].sigma**2

    return updater

def createComparisonUpdater(u1,u2,eps):
    def updater(mu1, var1, mu2, var2, y):
        r1 = u1(mu1, var1, mu2, var2, y)
        r2 = u2(mu1, var1, mu2, var2, y)
        bad = False
        for e1,e2 in zip(r1,r2):
            if abs(e1-e2)>eps:
                bad = True
        if bad:
            print(r1,"warning")
            print(r2,"warning")
        else:
            print(r1)
            print(r2)
        print("")
        return r1
    return updater

def compare():
    mu0, var0 = 25, (25/3)**2
    var_t = (25/6)**2
    update = createComparisonUpdater(createTrueSkillUpdater(),
                                     createMomentMatching(var_t),
                                     0.1)
    update(mu0,var0,mu0,var0,1)

compare()
# %%

# Q.4 - Run a single match
def singleMatch():

    # Choose hyper parameters
    y = 1
    mu1, var1 = 25, (25/3)**2
    mu2, var2 = 25, (25/3)**2
    var_t = (25/6)**2
    N_samples = 1000
    n_burn = 5
    S_1, S_2, T = sample(mu1, var1, mu2, var2, var_t, y, N_samples)

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
    
    x = np.linspace(0, 60, 200)
    p_s1 = s1_approx.pdf(x)
    p_s2 = s2_approx.pdf(x)

    # Q4.4
    bins = int(np.sqrt(N_samples))
    plt.hist(S_1[n_burn:], bins=bins, color="r", density=True)
    plt.plot(x, p_s1, color="darkred")
    plt.hist(S_2[n_burn:], bins=bins, color="g", density=True)
    plt.plot(x, p_s2, color="darkgreen")
    plt.hist(T[n_burn:], bins=bins, color="b", density=True)
    plt.legend(["s1", "p1", "s2", "p2", "t"])
    plt.show()

singleMatch()

# %%


# Q.5, Q.6
def rankFootballTeams():
    # Load dataframe from file
    seriesA_df = pd.read_csv("../data/SerieA.csv", sep=",")

    # Choose hyper parameters
    mu0, var0 = 25, (25/3)**2
    var_t = (25/6)**2
    nSamples = 500
    nBurn = 5

    # Run ADF on the dataframe rows
   # update = createGibbsUpdater(nSamples, nBurn)
    update = createMomentMatching(var_t) # tODO
    #update = createTrueSkillUpdater()
    update = createComparisonUpdater(createTrueSkillUpdater(),
                                     createMomentMatching(var_t),
                                     0.1)

     # Predicts y
    def predict(mu1, var1, mu2, var2, var_t):
        return round(Py_s1s2(mu1, var1, mu2, var2, var_t))*2-1
    
    t0 = time.time()
    teams, skills, accuracy, _ = ADFdf(seriesA_df, mu0, var0, var_t,
                                       '','team1','team2', lambda row : np.sign(row["score1"] - row["score2"]),
                                       predict, update, False)
    t1 = time.time()

    print(f"Took {t1-t0}")

    skills[:,2] = np.sqrt(skills[:,2])

    # Tabulate resulting posteriors
    idx = np.flip(np.argsort(skills[:,0]))
    skilltable = np.column_stack((1+np.arange(len(teams)), teams[idx], skills[idx,:3]))
    print(tabulate(skilltable,
                    headers=["Rank", "Team", "mu", "sigma", "Games"],
                    floatfmt=[".0f",".2f",".2f",".2f",".0f"],
                    tablefmt="latex_raw"))
    print(f"Prediction accuracy: {accuracy}")

rankFootballTeams()

# %%


# Q.8

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time
from tabulate import tabulate
from gibbs import Py_s1s2, sample, gaussian_approx, createGibbsUpdater
from adf import ADFdf

import Moment_matching
from Moment_matching import createMomentMatching, ps1_y, ps2_y, py
import importlib
importlib.reload(Moment_matching)

def momentMatchingVsGibbs():
    mu0, var0 = 25, (25 / 3)**2
    var_t = (25 / 6)**2

    y = 1
    nSamples = 1000

    # Gibbs
    S_1, S_2, _ = sample(mu0, var0, mu0, var0, var_t, y, nSamples)

    # Moment matching
    mu1mm, var1mm = ps1_y(mu0, var0, mu0, var0, var_t, y)
    mu2mm, var2mm = ps2_y(mu0, var0, mu0, var0, var_t, y)

    sigma1mm, sigma2mm = np.sqrt(var1mm), np.sqrt(var2mm)

    print(mu1mm, sigma1mm)

    bins = int(np.sqrt(nSamples))
    x = np.linspace(0, 50, 200)
    plt.hist(S_1, bins=bins, color="darkred", density=True, alpha=0.6)
    plt.plot(x, scipy.stats.norm(mu1mm, sigma1mm).pdf(x), "r")
    plt.hist(S_2, bins=bins, color="darkgreen", density=True, alpha=0.6)
    plt.plot(x, scipy.stats.norm(mu2mm, sigma2mm).pdf(x), "g")
    plt.legend(["p(s1|y) MM", "p(s2|y) MM", "p(s1|y) Gibbs", "p(s2|y) Gibbs"])
    plt.title("Moment Matching vs. Gibbs sampling of s posterior.")
    plt.show()


momentMatchingVsGibbs()



# %%

# Q.10
def rankFootballTeamsDraw():
    # Load dataframe from file
    seriesA_df = pd.read_csv("../data/SerieA.csv", sep=",")

    # Choose hyper parameters
    mu0, var0 = 25, (25/3)**2
    var_t = (25/6)**2
    
    # -------------------------
    # Without draws
    update = createMomentMatching()

    def predict(mu1, var1, mu2, var2, var_t):
        return round(Py_s1s2(mu1, var1, mu2, var2, var_t))*2-1
    
    _, _, accuracy, _ = ADFdf(seriesA_df, mu0, var0, var_t,
                              '','team1','team2', lambda row : np.sign(row["score1"] - row["score2"]),
                              predict, update, False)

    
    # -------------------------
    # With draws
    update = createMomentMatching()

    def predict_draws(mu1, var1, mu2, var2, var_t):
        mu = mu1-mu2
        sigma = np.sqrt(var1 + var2 + var_t)

        values = [py(mu, sigma, y) for y in range(-1, 2)]
        predicted_value = np.argmax(values) - 1
        return predicted_value

    teams, skills, accuracy_draw, _ = ADFdf(seriesA_df, mu0, var0, var_t,
                                            '','team1', 'team2',
                                            lambda row: np.sign(row["score1"] - row["score2"]),
                                            predict_draws, update, False, consider_draw=True)

    skills[:,2] = np.sqrt(skills[:,2])

    # -------------------------
    # Tabulate resulting posteriors
    idx = np.flip(np.argsort(skills[:,0]))
    skilltable = np.column_stack((1+np.arange(len(teams)), teams[idx], skills[idx,:3]))
    print(tabulate(skilltable,
                    headers=["Rank", "Team", "mu", "sigma", "Games"],
                    floatfmt=[".0f",".2f",".2f",".2f",".0f"],
                    tablefmt="latex_raw"))

    print(f"Prediction accuracy without draws: {accuracy}")
    print(f"Prediction accuracy with draws: {accuracy_draw}")


rankFootballTeamsDraw()



# %%


# # Q.9, Q.10
def trackTennisPlayers():

    # Load dataframe from file
    tennis_df = pd.read_csv("../data/tennis2.csv", sep=",")

    # Choose hyper parameters
    mu0, var0 = 25, (25/3)**2
    var_t = (25/6)**2

    # Choose update method
    update = createGibbsUpdater(nSamples=100, nBurn=5)
    #update = createMomentMatching()

    # Predicts y
    def predict(mu1, var1, mu2, var2, var_t):
        return round(Py_s1s2(mu1, var1, mu2, var2, var_t))*2-1 

    decayRate = 1/365
    def decay(var,dt):
        return var0 + (var-var0)*np.exp(-dt*decayRate)

    players, skills, accuracy, history = ADFdf(tennis_df, mu0, var0, var_t,
                                               'day','winner_name','loser_name',
                                               lambda row : 1,
                                               predict, update, False, False, decay)

    skills[:,2] = np.sqrt(skills[:,2])

    idx = np.flip(np.argsort(skills[:,0]))
    skilltable = np.column_stack((1+np.arange(len(players)), players[idx], skills[idx,:3]))
    skilltable = skilltable[:20]

    print(tabulate(skilltable,
                    headers=["Rank", "Team", "mu", "sigma", "Games"],
                    floatfmt=[".0f",".2f",".2f",".2f",".0f"],
                    tablefmt="latex_raw"))
    print(f"Prediction accuracy: {accuracy}")

    # Draw history of skill
    for i in idx[:5]:
        xs = history[i][:,0]
        ys = history[i][:,1]
        errs = np.sqrt(history[i][:,2])*3 # 99% confidence
        plt.plot(xs,ys,label=players[i])
        plt.fill_between(xs, ys-errs, ys+errs, alpha=0.5)
        plt.legend()
                    
trackTennisPlayers()


 
# %%
