import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Analysis.json_to_csv as j
from collections import defaultdict
import ENVIRONMENT_VARIABLES as EV
from Analysis.functions import match_detailed_functions as md
from Analysis.functions import player_functions as pf

from bokeh.plotting import figure, show
from bokeh.io import output_notebook

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm


#Get data
years = [2022, 2023, 2024, 2025]
df = j.get_player_stats()

match_detailed = j.get_detailed_match(years)
match_df = j.get_match_data(years)
odds = md.get_odds()

matches_full = md.clean_match_detailed(match_detailed, match_df)
matches_full = md.combine_odds(matches_full, odds)
matches_full = md.per_min_features(matches_full)
matches_full = md.opposition_stats(matches_full)
matches_full = matches_full.dropna(subset=['Referee'])

df = pf.player_data_cleaner(df, match_df, matches_full)
df = df.drop_duplicates()
players_dict, averages, team, team_avgs, position_dfs = pf.create_player_dicts(df)
position_corr = {position: df.corr() for position, df in position_dfs.items()}

match_stats = md.create_model_features(matches_full, df)

df_test = md.opposition_stats(matches_full)



#-----------#
# Try Score Positions
#-----------#

    
grouped = df.groupby(['Round', 'Year', 'Team_Name', 'Position'])['Tries'].sum().reset_index()
pivoted = grouped.pivot_table(index=['Round', 'Year', 'Team_Name'],
                               columns='Position',
                               values='Tries',
                               fill_value=0)

pivoted.columns = [f'Tries_{col}' for col in pivoted.columns]
try_scorers = pivoted.reset_index()
try_scorers['Year'] = try_scorers['Year'].astype('int')
matches_full_try_scorers = pd.merge(matches_full, try_scorers, on=['Round', 'Year', 'Team_Name'])
matches_full_try_scorers_corr = matches_full_try_scorers.corr()


#-----------#
# Odds Analysis
#-----------#

matches_full['bet_return'] = matches_full['Team_Win']*matches_full['Team Odds']*10 - 10
team_return = matches_full.groupby(by=['Team_Name', 'Team'])['bet_return'].sum()
matches_full['Team Line Odds'] = np.where(matches_full['Home'] == 1, matches_full['Home Line Odds Open'],
                                      matches_full['Away Line Odds Open'])
matches_full['line_return'] = matches_full['Line Win']*matches_full['Team Line Odds']*10 - 10
line_return = matches_full.groupby(by=['Team_Name', 'Team'])['line_return'].sum()




#------------#
# Visualisations
#------------#

'''p = figure(title=", 
           y_axis_label='Danceability', 
           x_axis_label='Release Date', 
           width=700, 
           x_axis_type='datetime',
           height = 400)

p.line(x=grouped_sample['release_date'], 
    y=grouped_sample['danceability mean'], color = 'blue', alpha = .75)

p.line(x=grouped_sample['release_date'], 
    y=grouped_sample['danceability min'], color = 'red', alpha = .25)

p.line(x=grouped_sample['release_date'], 
    y=grouped_sample['danceability max'], color = 'green', alpha = .25)

show(p)'''

# Extract the data
y = matches_full['Margin']
x = matches_full['Kick Return Metres']

# Scatter plot
plt.scatter(x, y, label='Data points')

# Line of best fit
m, b = np.polyfit(x, y, 1)  # 1 indicates linear
plt.plot(x, m*x + b, color='red', label='Best fit line')

# Labels and legend
plt.ylabel('Margin')
plt.xlabel('Kick Return Metres')
plt.title('Margin vs Kick Return Metres')
plt.legend()

plt.show()


y = df['Tries']
x = df['Tackle Breaks']

# Scatter plot
plt.scatter(x, y, label='Data points')

# Line of best fit
m, b = np.polyfit(x, y, 1)  # 1 indicates linear
plt.plot(x, m*x + b, color='red', label='Best fit line')

# Labels and legend
plt.ylabel('Margin')
plt.xlabel('Kick Return Metres')
plt.title('Margin vs Kick Return Metres')
plt.legend()

plt.show()


#------------#
#Models
#------------#

sns.kdeplot(matches_full['Possession'].dropna(), fill=True)
plt.title('Density Plot of Possession')
plt.xlabel('Possession (%)')
plt.ylabel('Density')
plt.show()


X = matches_full[['Post Contact Metres', 'Kick Return Metres',
                 'Kicking Metres', 'Post Contact Metres Opp','Kick Return Metres Opp',
                 'Kicking Metres Opp'
                   ]]
y = matches_full['Margin']

x = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


y_pred = model.predict(X)
y_actual = y
plt.scatter(y_pred, y_actual, label='Data points')


comparison = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred})

print(comparison.head(60))


X = position_dfs['wingers'][['Tackle Breaks', 'Line Breaks', 'Offloads', 'All Run Metres', 'Kick Return Metres',
        'Post Contact Metres', 'Receipts'
                   ]]
y = position_dfs['wingers']['Margin']

x = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


y_pred = model.predict(X)
y_actual = y
plt.scatter(y_pred, y_actual, label='Data points')


comparison = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred})

print(comparison.head(60))


#------------#
# Analysis
#------------#

half_full = matches_full[['Half Time Score', 'Half Time Score Opp', 'Team_Score', 'Team_Score Opp']]
half_full['2nd Half Score'] = half_full['Team_Score'] - half_full['Half Time Score']
half_full['2nd Half Score Opp'] = half_full['Team_Score Opp'] - half_full['Half Time Score Opp']


X = half_full[['Half Time Score', 'Half Time Score Opp'
                   ]]
y = np.log(half_full['2nd Half Score'] + 15)

x = sm.add_constant(X)
model = sm.OLS(y, x).fit()
print(model.summary())


y_pred = np.exp(model.predict(X)) - 15
y_actual = np.exp(y) - 15
plt.scatter(y_pred, y_actual, label='Data points')


comparison = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred})

print(comparison.head(60))

plt.hist(np.log(half_full['2nd Half Score']+20), bins=10)


half_full['2nd Half Higher'] = np.where(half_full['2nd Half Score'] > half_full['Half Time Score'], 1, 0)
half_full['Winning ht'] = np.where(half_full['Half Time Score'] > half_full['Half Time Score Opp'], 1, 0)

second_half_higher_pct = sum(half_full['2nd Half Higher']) / len(half_full)
second_half_higher_pct_when_winning = sum(half_full[half_full['Winning ht'] == 1]['2nd Half Higher']) / len(half_full[half_full['Winning ht'] == 1])
second_half_higher_pct_when_losing = sum(half_full[half_full['Winning ht'] == 0]['2nd Half Higher']) / len(half_full[half_full['Winning ht'] == 0])





