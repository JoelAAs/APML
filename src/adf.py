import numpy as np
import pandas as pd

# Assumed Density Filtering
def ADF(nPlayers:int, results:np.array,
        mu0, sigma0, Sigma_t, predict:callable, update:callable):
    
    playerSkills = np.array([[mu0, sigma0, 0]] * nPlayers, dtype=np.float32)

    i,nCorrect = 0,0
    for row in results:
        p1, p2, y = row
        mu1, sigma1, nMatches1 = playerSkills[p1]
        mu2, sigma2, nMatches2 = playerSkills[p2]
       
        # Predict the outcome, count the hits
        nCorrect += predict(mu1, mu2, sigma1, sigma2, Sigma_t)==y

       # print(f"{mu_1} vs {mu_2} -> {predicted} but {y}")

        # Bayesian update
        mu1,sigma1, mu2,sigma2 = update(mu1,sigma1, mu2,sigma2, y, Sigma_t)
        playerSkills[p1,:] = [mu1, sigma1, nMatches1+1]
        playerSkills[p2,:] = [mu2, sigma2, nMatches2+1]
        
        i += 1
        if i%10 == 0:
            print(f"Finished {i}/{len(results)}")

    return playerSkills, nCorrect/i

# ADF on pandas dataframe
def ADFdf(results_df, mu0, sigma0, Sigma_t,
          predict:callable, update:callable, shuffle:bool):
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
    
    results = results[:nDecisive]
    if shuffle:
        np.random.shuffle(results)

    playerSkills, accuracy = ADF(len(players), results,
                                 mu0, sigma0, Sigma_t, predict, update)
    return players, playerSkills, accuracy
