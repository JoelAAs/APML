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

#%%
# Q5

# Choose initial values for skill of the teams and their variance + variance of t

A = np.array([[1, -1]]) # As mean of t = s_1 - s_2
K = 50 # Number of Gibbs iterations

df_team_skills = pd.DataFrame(columns = ['team', 'mean skill', 'std'])

df_team_skills['team'] = unique_teams
df_team_skills=df_team_skills.reset_index(drop=True)
df_team_skills['mean skill'] = 25
df_team_skills['std'] = 3

sigma_t_given_s1s2 = 1.5 # Same throughout the Assumed Density Filtering


# Input the relations from Q3 in a loop



##### Choose one of the folowing

# Q5.1 - Original order of games
df_loop = df.copy()


# # Q5.2 - Random order of games
# df_loop = df.copy()
# df_loop = df_loop.sample(frac=1) # Sample the data at random without replacements, i.e. every row is selected only once
# df_loop=df_loop.reset_index(drop=True)





for i in range(len(df)):
    # Pick index and values in the updating team skills data frame in the current game
    index_s1 = df_team_skills.loc[df_team_skills['team'] == df_loop.loc[i]['team 1']].index[0]
    index_s2 = df_team_skills.loc[df_team_skills['team'] == df_loop.loc[i]['team 2']].index[0]
    
    # Get input values for the Gibbs sampling
    mu = np.array( [ [ df_team_skills["mean skill"][index_s1] ] , [  df_team_skills["mean skill"][index_s2] ] ])
    sigma_s1s2 = np.array([[df_team_skills['std'][index_s1]**2, 0], [0, df_team_skills['std'][index_s2]**2]])
    t_mean = mu[0] - mu[1]
    t_std = sigma_t_given_s1s2  
    
    # Calculate values needed for the first iteration
    sigma_s1s2_given_t = np.linalg.inv(np.linalg.inv(sigma_s1s2)+np.transpose(A)*(1/sigma_t_given_s1s2**2)@A)
    
         
    # Get the values for the truncated gaussian
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
    
    df_team_skills['mean skill'][index_s1] = mu_s1_samp
    df_team_skills['std'][index_s1] = sigma_s1_samp
    
    df_team_skills['mean skill'][index_s2] = mu_s2_samp
    df_team_skills['std'][index_s2] = sigma_s2_samp

    print('Running game ' + str(i) + ' out of ' + str(len(df_loop)))

