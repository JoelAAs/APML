import numpy as np
import pandas as pd

# Assumed Density Filtering
def ADF(nPlayers:int, results:np.array,
        mu0, sigma0, Sigma_t,
        predict:callable, update:callable):
    
    playerSkills = np.array([[mu0, sigma0, 0]] * nPlayers, dtype=np.float32)

    i,nCorrect = 0,0
    for row in results:
        p1, p2, y = row
        mu1, sigma1, nMatches1 = playerSkills[p1]
        mu2, sigma2, nMatches2 = playerSkills[p2]

        pred = predict(mu1, mu2, sigma1, sigma2, Sigma_t) == y
        print(f"pred {pred}, true {y}")
        # Predict the outcome, count the hits
        nCorrect += predict(mu1, mu2, sigma1, sigma2, Sigma_t)==y

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
          player1Column, player2Column, getWinner:callable,
          predict:callable, update:callable, shuffle:bool, consider_draw=False):
    
    # Assign numbers to the players
    players = pd.concat([results_df[player1Column], results_df[player2Column]]).unique()
    playerIDs = {players[i]:i for i in range(len(players))}
    
    # Convert to numpy array
    results = np.zeros((results_df.shape[0],3),dtype=np.int32) # p1,p2,result
    nDecisive = 0
    for _, row in results_df.iterrows():
        y = getWinner(row)
        if y == 0 and not consider_draw:
            continue
        results[nDecisive,:] = np.array([playerIDs[row[player1Column]],
                                         playerIDs[row[player2Column]],
                                         y])
        nDecisive += 1
    
    results = results[:nDecisive]
    if shuffle:
        np.random.shuffle(results)

    playerSkills, accuracy = ADF(len(players), results,
                                 mu0, sigma0, Sigma_t, predict, update)
    return players, playerSkills, accuracy
