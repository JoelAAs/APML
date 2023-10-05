import numpy as np
import pandas as pd
from gibbs import Py_s1s2, sample, gaussian_approx


# Assumed Density Filtering
def ADF(nPlayers:int, results:np.array,
        mu0, sigma0, Sigma_t, nSamples, nBurn):
    
    playerSkills = np.array([[mu0, sigma0, 0]] * nPlayers, dtype=np.float32)

    i,nCorrect = 0,0
    for row in results:
        p1, p2, y = row
        mu1, sigma1, nMatches1 = playerSkills[p1]
        mu2, sigma2, nMatches2 = playerSkills[p2]
       
        # Predict the outcome, count the hits
        predicted = Py_s1s2(mu1, mu2, sigma1, sigma2, Sigma_t)
        nCorrect += round(predicted)*2-1 == y

       # print(f"{mu_1} vs {mu_2} -> {predicted} but {y}")

        # Bayesian update - sample from the posterior
        s1s, s2s, _ = sample(y,
                            mu1, mu2,
                            sigma1, sigma2,
                            Sigma_t,
                            nSamples)

        # if 
        # print(f"")

        # Estimate the parameters of the new normal distributions
        _, mu1, sigma1 = gaussian_approx(s1s[nBurn:])
        _, mu2, sigma2 = gaussian_approx(s2s[nBurn:])

        # Update
        playerSkills[p1,:] = [mu1, sigma1, nMatches1+1]
        playerSkills[p2,:] = [mu2, sigma2, nMatches2+1]
        i += 1
        print(f"Done {i}/{len(results)}")

    return playerSkills, nCorrect/i

# ADF on pandas dataframe
def ADFdf(results_df, mu0, sigma0, Sigma_t, nSamples, nBurn):
    # Assign numbers to the players
    players = pd.concat([results_df['team1'], results_df['team2']]).unique()
    playerIDs = {players[i]:i for i in range(len(players))}
    
    # Convert to numpy array
    results = np.zeros((results_df.shape[0],3),dtype=np.int32) # p1,p2,result
    nDecisive = 0
    for _, row in results_df.iterrows():
        y = np.sign(row["score1"] - row["score2"])
        if y == 0:
            continue
        results[nDecisive,:] = np.array([playerIDs[row["team1"]],
                                         playerIDs[row["team2"]],
                                         y])
        nDecisive += 1
    
    playerSkills, accuracy = ADF(len(players), results[:nDecisive],
                                 mu0, sigma0, Sigma_t, nSamples, nBurn)
    return players, playerSkills, accuracy
