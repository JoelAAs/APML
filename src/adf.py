import numpy as np
import pandas as pd

# Assumed Density Filtering
def ADF(nPlayers:int, results:np.array,
        mu0, var0, var_t,
        predict:callable, update:callable,
        decay:callable = None):
    
    playerSkills = np.array([[mu0, var0, 0, -1]] * nPlayers, dtype=np.float32)
    history = [[] for _ in range(nPlayers)]

    i,nCorrect = 0,0
    for row in results:
        time, p1, p2, y = row
        mu1, var1, nMatches1, last1 = playerSkills[p1]
        mu2, var2, nMatches2, last2 = playerSkills[p2]

        # Predict the outcome, count the hits
        nCorrect += predict(mu1, var1, mu2, var2, var_t)==y

        # Increate the standard deviation based on the time since their last game
        if decay != None:
            if last1 != -1:
                var1 = decay(var1,time-last1)
            if last2 != -1:
                var2 = decay(var2,time-last2)
        
        # Bayesian update
        mu1,var1, mu2,var2 = update(mu1,var1, mu2,var2, y)
        playerSkills[p1,:] = [mu1, var1, nMatches1+1, time]
        playerSkills[p2,:] = [mu2, var2, nMatches2+1, time]
        
        history[p1].append([time, mu1, var1])
        history[p2].append([time, mu2, var2])

        i += 1
        if i%10 == 0:
            print(f"Finished {i}/{len(results)}")

    return playerSkills, nCorrect/i, [np.array(h) for h in history]

# ADF on pandas dataframe
def ADFdf(results_df, mu0, var0, var_t,
          timeColumn:str, player1Column:str, player2Column:str, getWinner:callable,
          predict:callable, update:callable, shuffle:bool, consider_draw = False,
          decay:callable = None):
    
    # Assign numbers to the players
    players = pd.concat([results_df[player1Column], results_df[player2Column]]).unique()
    playerIDs = {players[i]:i for i in range(len(players))}
    
    # Convert to numpy array
    results = np.zeros((results_df.shape[0],4),dtype=np.int32) # time,p1,p2,result
    nDecisive = 0
    for _, row in results_df.iterrows():
        y = getWinner(row)
        if y == 0 and not consider_draw:
            continue
        results[nDecisive,:] = np.array([0 if timeColumn=="" else row[timeColumn],
                                         playerIDs[row[player1Column]],
                                         playerIDs[row[player2Column]],
                                         y])
        nDecisive += 1

    results = results[:nDecisive]
    if shuffle:
        np.random.shuffle(results)

    playerSkills, accuracy, history = ADF(len(players), results,
                                          mu0, var0, var_t,
                                          predict, update, decay)
    return players, playerSkills, accuracy, history
