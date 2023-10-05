import numpy as np
import pandas as pd
from gibbs import Py_s1s2, sample, gaussian_approx


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
