import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Analysis.json_to_csv as j
from collections import defaultdict
import ENVIRONMENT_VARIABLES as EV
from Analysis.functions import match_detailed_functions as md
from Analysis.functions import player_functions as pf
import statsmodels.api as sm


#Get data
df = j.get_player_stats()
years = [2022, 2023, 2024, 2025]
match_detailed = j.get_detailed_match(years)
match_df = j.get_match_data(years)
odds = md.get_odds()

#Clean data
matches_full = md.clean_match_detailed(match_detailed, match_df)
matches_full = md.combine_odds(matches_full, odds)
matches_full = md.per_min_features(matches_full)
matches_full = matches_full.dropna(subset=['Referee'])
df = pf.player_data_cleaner(df, match_df, matches_full)

players_dict, averages, team, team_avgs, position_dfs = pf.create_player_dicts(df)

position_corr = {position: df.corr() for position, df in position_dfs.items()}

fp_odds_corr = {}

# Calculate correlation for each player (for simplicity, we assume correlation is 
for player, data in players_dict.items():
    total_points = data['Total Points']
    team_odds = data['Team Odds']
    
    if len(total_points) > 5:  # Correlation requires at least 2 data points
        correlation = np.corrcoef(total_points, team_odds)[0, 1]
    else:
        correlation = None
    
    # Store in correlation_dict
    fp_odds_corr[player] = correlation
        

backs_avgs = averages['backs']
backs_avgs['Fantasy Price'] = np.round(abs((backs_avgs['Total Points'] - backs_avgs['Total Points'].min()) / 
                                        (backs_avgs['Total Points'].max() - backs_avgs['Total Points'].min()))
                                        *(1000 - 250) + 250,2)

averages_corr = {position: avg_df.corr() for position, avg_df in averages.items()}


#Plots

halves = position_dfs['halves']

for team in halves['Team'].unique()[:17]:
    min_games = min(len(halves[(halves['Team'] == team) & (halves['Number'] == '7')]), len(halves[(halves['Team'] == team) & (halves['Number'] == '6')]))
    x = [*range(0, min_games, 1)]
    y1 = halves[(halves['Team'] == team) & (halves['Number'] == '7')]['All Runs'][0:min_games]
    n1 = halves[(halves['Team'] == team) & (halves['Number'] == '7')][0:min_games]['Name'].tolist()
    y2 = halves[(halves['Team'] == team) & (halves['Number'] == '6')]['All Runs'][0:min_games]
    n2 = halves[(halves['Team'] == team) & (halves['Number'] == '6')][0:min_games]['Name'].tolist()
    plt.figure(figsize =(10,4))
    plt.plot(x, y1, label='7')
    for i, txt in enumerate(n1):
        plt.annotate(txt, (x[i], y1.iloc[i]), rotation=45, rotation_mode='anchor')
    plt.plot(x, y2, label='6')
    for i, txt in enumerate(n2):
        plt.annotate(txt, (x[i], y2.iloc[i]), rotation=45, rotation_mode='anchor')
    plt.title(f'{team}')
    plt.xlabel('Game')
    plt.ylabel('All Runs')
    plt.legend()
    plt.show()

import seaborn as sns


#Fullbacks

fullbacks = position_dfs['fullbacks']
fullback_top = position_corr['fullbacks'].sort_values(by='Total Points', ascending=False)['Total Points'].head(10)
fullbacks_avgs = averages['fullbacks']

plt.style.use("dark_background")  # Set dark background
plt.figure(figsize=(8, 10))
sns.barplot(x=fullback_top.values[:15], y=fullback_top.index[:15], palette="plasma")
plt.xlabel("Correlation Coefficient", fontsize=16, color="white")
plt.ylabel("Stat", fontsize=16, color="white")
plt.title("Fullbacks Top 15 Stat Correlations with Fantasy Score", fontsize=20, color="white")
plt.xticks(fontsize=16, color="white")
plt.yticks(fontsize=16, color="white")
plt.show()

stats = fullback_top.index[:5]

for i in stats:
    plt.figure(figsize=(8, 10))
    sns.barplot(x=fullbacks_avgs.sort_values(by=i, ascending=False)[i][:15], 
                y=fullbacks_avgs.sort_values(by=i, ascending=False).index[:15], 
                palette="plasma")
    plt.xlabel(f"Average {i}")
    plt.ylabel("Player")
    plt.title(f"Fullbacks {i}")
    plt.show()




#Visualisations

import matplotlib.pyplot as plt
import numpy as np

# Extract the data
y = position_dfs['full']['Dummy Passes']
x = position_dfs['full']['Round']

# Scatter plot
plt.scatter(x, y, label='Data points')

# Line of best fit
m, b = np.polyfit(x, y, 1)  # 1 indicates linear
plt.plot(x, m*x + b, color='red', label='Best fit line')

# Labels and legend
plt.ylabel('Total Points')
plt.xlabel('Time In Possession')
plt.title('Edges: Team Time In Possession vs Fantasy Points')
plt.legend()

plt.show()




#Models

edges = position_dfs['edges']

X = edges[['Time In Possession'
                   ]]
y = edges['Total Points']

x = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())




