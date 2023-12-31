# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time
from tabulate import tabulate
from gibbs import Py_s1s2, gibbsSample, gaussian_approx, createGibbsUpdater
from adf import ADFdf
from Moment_matching import createMomentMatching, ps1_y, ps2_y, py


# %%
# Import the trueskill package for comparison. It is not used in the code.
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

# This is to show that our mm is slightly different from trueskill
def testCompareTrueskill():
    mu0, var0 = 25, (25/3)**2
    var_t = (25/6)**2
    update = createComparisonUpdater(createTrueSkillUpdater(),
                                     createMomentMatching(var_t),
                                     0.1)
    update(mu0*2,var0,mu0,var0,1)
testCompareTrueskill()
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
    S_1, S_2, T = gibbsSample(mu1, var1, mu2, var2, var_t, y, N_samples)

    # Q4.1 burn-in is about 5 values
    gen = np.linspace(0, N_samples - 1, N_samples)
    plt.plot(gen, S_1, "r", alpha=0.5)
    plt.plot(gen, S_2, "g", alpha=0.5)
    plt.plot(gen, T, "b", alpha=0.5)
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
    plt.hist(S_1[n_burn:], bins=bins, color="r", alpha=0.5, density=True)
    plt.plot(x, p_s1, color="r")
    plt.hist(S_2[n_burn:], bins=bins, color="g", alpha=0.5, density=True)
    plt.plot(x, p_s2, color="g")
    plt.hist(T[n_burn:], bins=bins, color="b", alpha=0.5, density=True)
    plt.legend(["s1", "p1", "s2", "p2", "t"])
    plt.show()

singleMatch()

# %%

# Create prediction summary string
def format_and_present_accuracy(predictions):
    msg = "Predictions results:\n"
    all_correct, total = 0, 0
    for state in predictions:
        n_correct, n_total = predictions[state]
        msg += f"Result {state:2}: {n_correct/n_total*100:6.2f}% ({n_correct}/{n_total})\n"
        all_correct += n_correct
        total += n_total

    msg += f"Average accuracy: {all_correct/total*100}"
    return msg

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
    update = createGibbsUpdater(var_t, nSamples, nBurn)
    update = createMomentMatching(var_t)

     # Predicts y
    def predict(mu1, var1, mu2, var2, var_t):
        return round(Py_s1s2(mu1, var1, mu2, var2, var_t))*2-1
    

    t0 = time.time()
    teams, skills, predictions, _ = ADFdf(seriesA_df, mu0, var0, var_t,
                                          '','team1','team2',
                                          lambda row : np.sign(row["score1"] - row["score2"]),
                                          False, False,
                                          update, predict)
    t1 = time.time()
    print(f"Took {t1-t0}")

    skills[:,1] = np.sqrt(skills[:,1])

    # Tabulate resulting posteriors
    idx = np.flip(np.argsort(skills[:,0]))
    skilltable = np.column_stack((1+np.arange(len(teams)), teams[idx], skills[idx,:3]))
    print(tabulate(skilltable,
                    headers=["Rank", "Team", "mu", "sigma", "Games"],
                    floatfmt=[".0f",".2f",".2f",".2f",".0f"],
                    tablefmt="latex_raw"))
    predictions_accuracy_msg = format_and_present_accuracy(predictions)
    print(predictions_accuracy_msg)

rankFootballTeams()

# %%


# Q.8

# %%

def momentMatchingVsGibbs():
    mu0, var0 = 25, (25 / 3)**2
    var_t = (25 / 6)**2

    y = 1
    nSamples = 1000

    # Gibbs
    S_1, S_2, _ = gibbsSample(mu0, var0, mu0, var0, var_t, y, nSamples)

    # Moment matching
    mu1mm, var1mm = ps1_y(mu0, var0, mu0, var0, var_t, y, consider_draw=False)
    mu2mm, var2mm = ps2_y(mu0, var0, mu0, var0, var_t, y, consider_draw=False)

    sigma1mm, sigma2mm = np.sqrt(var1mm), np.sqrt(var2mm)

    print(mu1mm, sigma1mm)
    _,mean, var = gaussian_approx(S_1)
    print(mean, np.sqrt(var))

    bins = int(np.sqrt(nSamples))
    x = np.linspace(0, 50, 200)
    plt.hist(S_1, bins=bins, color="r", alpha=0.5, density=True)
    plt.plot(x, scipy.stats.norm(mu1mm, sigma1mm).pdf(x), "r")
    plt.hist(S_2, bins=bins, color="g", alpha=0.5, density=True)
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
    update = createMomentMatching(var_t, consider_draw=False)

    def predict(mu1, var1, mu2, var2, var_t):
        return round(Py_s1s2(mu1, var1, mu2, var2, var_t))*2-1
    
    _, _, predictions, _ = ADFdf(seriesA_df, mu0, var0, var_t,
                                 '','team1','team2',
                                 lambda row : np.sign(row["score1"] - row["score2"]),
                                 False, False,
                                 update, predict)

    
    # -------------------------
    # With draws
    update = createMomentMatching(var_t, consider_draw=True)
    def predict_draws(mu1, var1, mu2, var2, var_t):
        mu = mu1-mu2
        sigma = np.sqrt(var1 + var2 + var_t)

        values = [py(mu, sigma, y, consider_draw=True) for y in range(-1, 2)]
        predicted_value = np.argmax(values) - 1
        return predicted_value

    teams, skills, predictions_draw, _ = ADFdf(seriesA_df, mu0, var0, var_t,
                                               '','team1', 'team2',
                                               lambda row: np.sign(row["score1"] - row["score2"]),
                                               False, True,
                                               update, predict_draws)

    skills[:,1] = np.sqrt(skills[:,1])

    # -------------------------
    # Tabulate resulting posteriors
    idx = np.flip(np.argsort(skills[:,0]))
    skilltable = np.column_stack((1+np.arange(len(teams)), teams[idx], skills[idx,:3]))
    print(tabulate(skilltable,
                    headers=["Rank", "Team", "mu", "sigma", "Games"],
                    floatfmt=[".0f",".2f",".2f",".2f",".0f"],
                    tablefmt="latex_raw"))
    predictions_accuracy_msg = format_and_present_accuracy(predictions)
    print("Without draws:\n", predictions_accuracy_msg)

    predictions_accuracy_msg_draw = format_and_present_accuracy(predictions_draw)
    print("With draws:\n", predictions_accuracy_msg_draw)


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
    update = createMomentMatching(var_t)
    #update = createTrueSkillUpdater()

    # Predicts y
    def predict(mu1, var1, mu2, var2, var_t):
        return round(Py_s1s2(mu1, var1, mu2, var2, var_t))*2-1 

    # Multiply skill variance by e after 50 years of inactivity 
    # That's already enough to make a difference
    decayRate = 1/(365*50)
    def decay(var,dt):
        return var0 + (var-var0)*np.exp(-dt*decayRate)

    players, skills, predictions_tennis, history = ADFdf(tennis_df, mu0, var0, var_t,
                                               'day','winner_name','loser_name',
                                               lambda row : 1, False, False,
                                               update, predict #, decay
                                               )

    skills[:,1] = np.sqrt(skills[:,1])

    filter = skills[:,2] > 100
    skills = skills[filter]
    players = players[filter]
    history = [item for item, mask in zip(history, filter) if mask]
    idx = np.flip(np.argsort(skills[:,0]))
    
    skilltable = np.column_stack((1+np.arange(len(players)),
                                  players[idx],
                                  skills[idx,:3]))
    skilltable = skilltable[:20]

    print(tabulate(skilltable,
                    headers=["Rank", "Player", "mu", "sigma", "Games"],
                    floatfmt=[".0f",".2f",".2f",".2f",".0f",".0f"],
                    tablefmt="latex_raw"))
    predictions_tennis_msg = format_and_present_accuracy(predictions_tennis)
    print(predictions_tennis_msg)

    # Draw history of skill
    day0 = history[0][0,0]
    for i in idx[:5]:
        xs = (history[i][:,0]-day0)/365+1991
        ys = history[i][:,1]
        errs = np.sqrt(history[i][:,2])*2 # 95% confidence
        plt.plot(xs,ys,label=players[i])
        plt.fill_between(xs, ys-errs, ys+errs, alpha=0.5)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Skill mean")
        plt.xticks(list(range(1999,2018)),
           [x if x in [2000,2006,2010,2016] else "" for x in range(1999,2018)])
                

trackTennisPlayers()

