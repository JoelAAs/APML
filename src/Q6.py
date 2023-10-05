# %%
# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import truncnorm
import scipy.stats as stats
from scipy.integrate import quad, trapezoid
import time

pd.options.display.max_columns = None
pd.set_option('display.expand_frame_repr', False)

PATH = os.path.dirname(os.path.realpath(__file__))


# %% Load data

df_orig = pd.read_csv(PATH+r"\SerieA.csv")
# Copy original in-data
df_orig.columns = ["date", "time", "team 1", "team 2",
                           "goals team 1", "goals team 2"]

# %% Work with data
df = df_orig.copy()

unique_teams = df["team 1"].unique()
unique_teams = df[~df.duplicated(subset=["team 1"])].copy()
unique_teams = unique_teams[["team 1"]].sort_values("team 1")
df['winning team']  = np.sign(df.loc[:,"goals team 1"] - df.loc[:,"goals team 2"])
indexDraw = df[ (df['winning team'] == 0) ].index
df.drop(indexDraw , inplace=True)
df=df.reset_index(drop=True)




#%% Q6

# Calculate the marginal distribution of t, and use it to predict who will win
# If 1 - P(-infty <= t <= 0) is <0.5 then prediction is that team 2 wins, otherwise team 1

# Choose initial values for skill of the teams and their variance + variance of t

A = np.array([[1, -1]]) # As mean of t = s_1 - s_2
K = 100 # Number of Gibbs iterations

df_team_skills_Q6 = pd.DataFrame(columns = ['team', 'mean skill', 'std'])
df_team_skills_Q6['team'] = unique_teams
df_team_skills_Q6 = df_team_skills_Q6.reset_index(drop=True)
df_team_skills_Q6['mean skill'] = 25
df_team_skills_Q6['std'] = 3

sigma_t_given_s1s2 = 1.5 # Same throughout the Assumed Density Filtering



df_loop_Q6 = df.copy()

numb_total_guesses = 0
numb_correct_guesses = 0

for i in range(len(df)):
    # Pick index and values in the updating team skills data frame in the current game
    index_s1 = df_team_skills.loc[df_team_skills['team'] == df_loop_Q6.loc[i]['team 1']].index[0]
    index_s2 = df_team_skills.loc[df_team_skills['team'] == df_loop_Q6.loc[i]['team 2']].index[0]
    
    # Get values for prediction
    mu = np.array( [ [ df_team_skills["mean skill"][index_s1] ] , [  df_team_skills["mean skill"][index_s2] ] ])
    sigma_s1s2 = np.array([[df_team_skills['std'][index_s1]**2, 0], [0, df_team_skills['std'][index_s2]**2]])
    
    mean_t = mu[0] - mu[1]
    sigma_t = sigma_t_given_s1s2 + A @ sigma_s1s2 @ np.transpose(A)
    x = np.linspace(-1000,0,10000)
    t_dist_negative_x_values_only = stats.norm.pdf(x, mean_t, sigma_t)

    res = trapezoid(t_dist_negative_x_values_only, x)
    if res < 0.5:
        pred_winner = -1
    else:
        pred_winner = 1
        
    if pred_winner == df_loop_Q6['winning team'][i]:
        numb_correct_guesses += 1
    numb_total_guesses += 1  
    
    
 
    
    # Calculate values needed for the first iteration
    sigma_s1s2_given_t = np.linalg.inv(np.linalg.inv(sigma_s1s2)+np.transpose(A)*(1/sigma_t_given_s1s2**2)@A)
    
         
    # Get the values for the truncated gaussian
    t_std = sigma_t_given_s1s2 
    
    
    if df_loop["winning team"][index_s1] > 0:
        myclip_a = 0
        myclip_b = 1000 
    else:
        myclip_a = -1000
        myclip_b = 0 
    
    a = (myclip_a - t_mean) / t_std
    b = (myclip_b - t_mean) / t_std
    
    
    # Initial values for gibbs sampler
    s1 = np.zeros(K)
    s2 = np.zeros(K)
    t = np.zeros(K)
    s1[0] = 0   
    s2[0] = 0    
    t[0] = 0
    
    
    # Iterate the Gibbs sampler
    for k in range(K-1):
        mu_s1s2_given_t = sigma_s1s2_given_t@(np.linalg.inv(sigma_s1s2)@mu+np.transpose(A)*(1/sigma_t_given_s1s2**2)*t[k])
           
        mu_s1s2_given_t_flattened = np.ravel(mu_s1s2_given_t) # Go from size (2,1) vector to (2,) to make Python work...
        
        s1[k+1], s2[k+1] = np.random.multivariate_normal(mu_s1s2_given_t_flattened, sigma_s1s2_given_t)
        t_mean = s1[k+1] - s2[k+1]
        a = (myclip_a - t_mean) / t_std
        b = (myclip_b - t_mean) / t_std
        t[k+1] = truncnorm.rvs(a, b, t_mean, t_std)

    # Get the resulting new values for the teams' mean value and variance and save them in the team skills data frame
    mu_s1_samp = np.mean(s1[5:])  # Don't take the first values during the burn in
    sigma_s1_samp = np.std(s1[5:])

    mu_s2_samp = np.mean(s2[5:])  # Don't take the first values during the burn in
    sigma_s2_samp = np.std(s2[5:])
    
    df_team_skills_Q6['mean skill'][index_s1] = mu_s1_samp
    df_team_skills_Q6['std'][index_s1] = sigma_s1_samp
    
    df_team_skills_Q6['mean skill'][index_s2] = mu_s2_samp
    df_team_skills_Q6['std'][index_s2] = sigma_s2_samp


    print('Running game ' + str(i) + ' out of ' + str(len(df_loop)))

model_accuracy_q6 = numb_correct_guesses/numb_total_guesses

print('Model accuracy is ' +str(np.round(model_accuracy_q6*100,2)) + ' %')
