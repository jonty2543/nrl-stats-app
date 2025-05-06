from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm
from itertools import combinations


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Analysis.json_to_csv as j
from collections import defaultdict
import ENVIRONMENT_VARIABLES as EV
from Analysis.functions import match_detailed_functions as md
from Analysis.functions import player_functions as pf

#Get data
years = [2022, 2023, 2024, 2025]
match_detailed = j.get_detailed_match(years)
match_df = j.get_match_data(years)
df = j.get_player_stats()


#Clean data
matches_full = md.clean_match_detailed(match_detailed, match_df)
odds = md.get_odds()
matches_full = md.combine_odds(matches_full, odds)
matches_full = md.per_min_features(matches_full)
matches_full = md.opposition_stats(matches_full)
matches_full = matches_full.dropna(subset=['Referee'])
df = pf.player_data_cleaner(df, match_df, matches_full)

#Create model features
match_stats = md.create_model_features(matches_full, df)
players_dict, averages, team, team_avgs, position_dfs = pf.create_player_dicts(df)


# ----------------------#
# ----------------------#
# Predictions
# ----------------------#
# ----------------------#



#Create fit data
#fit_data = match_stats.drop(columns=['Fantasy_Average']).dropna()
fit_data = match_stats.dropna()
fit_data = fit_data.sample(frac=1).reset_index(drop=True)

fit_data['Team Line Odds'] = np.where(fit_data['Home'] == 1, fit_data['Home Line Odds Open'],
                                      fit_data['Away Line Odds Open'])
fit_data['Home'] = (fit_data['Team'] == 'Home').astype(int)

fit_data_home = fit_data[fit_data['Team'] == 'Home']
fit_data_home = fit_data_home.sample(frac=1).reset_index(drop=True)
fit_data_away = fit_data[fit_data['Team'] == 'Away']
fit_data_away = fit_data_home.sample(frac=1).reset_index(drop=True)

#---------#
# Line Win
#---------#


# Define features and target
y = fit_data['Line Win']
X = fit_data[['Team Line Odds', 'Avg Diff Run Metres Per Min w', 
            'Avg Diff Kicking Metres Per Min w', 'Avg Diff pcm Per Min w', 'Avg Diff Line Breaks Per Min w',
            'Avg Diff Tackle Breaks Per Min w'
                         ]]

# Feature scaling
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)

# Define Logistic Regression model with balanced class weights
model = LogisticRegression(class_weight='balanced')

# Apply 5-fold cross-validation
rkf = RepeatedKFold(n_splits=5, n_repeats=100, random_state=8)
cv_scores = cross_val_score(model, X_scaled, y, cv=rkf, scoring='accuracy')

# Print cross-validation accuracy scores
print("Mean accuracy: {:.2f}%".format(np.mean(cv_scores) * 100))


#feature Selection
y = fit_data['Team_Win']
all_features = fit_data[['Fantasy_Average'
                         ]]

best_score = 0
best_features = None

for r in range(1, len(all_features) + 1):  # Try different numbers of predictors
    for feature_subset in combinations(all_features, r):
        X_subset = fit_data[list(feature_subset)]
        X_scaled = scaler.fit_transform(X_subset)

        cv_scores = cross_val_score(model, X_scaled, y, cv=rkf, scoring='accuracy')
        mean_score = np.mean(cv_scores)

        if mean_score > best_score:
            best_score = mean_score
            best_features = feature_subset

        #print(f"Features: {feature_subset}, Mean Accuracy: {mean_score:.4f}")

print("\nBest Feature Set:", best_features)
print("Best Accuracy: {:.2f}%".format(best_score * 100))



model.fit(X_scaled, y)

equation = f"Logit(P) = {model.intercept_[0]:.4f} "
for feature, coef in zip(X.columns, model.coef_[0]):
    equation += f"+ ({coef:.4f} * {feature}) "
print("Logistic Regression Equation:\n", equation)

def logit_to_probability(logit):
    return 1 / (1 + np.exp(-logit))

print(logit_to_probability(-4.6))


#---------#
# Team Win
#---------#



# Define features and target
y = fit_data['Team_Win']
X = fit_data[['Team Line Odds', 'Avg Diff Run Metres Per Min w', 
            'Avg Diff Kicking Metres Per Min w', 'Avg Diff pcm Per Min w', 'Avg Diff Line Breaks Per Min w',
            'Avg Diff Tackle Breaks Per Min w','Home']]

# Feature scaling
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)

# Define Logistic Regression model with balanced class weights
model = LogisticRegression(class_weight='balanced')

# Apply 5-fold cross-validation
rkf = RepeatedKFold(n_splits=5, n_repeats=100, random_state=8)
cv_scores = cross_val_score(model, X_scaled, y, cv=rkf, scoring='accuracy')

# Print cross-validation accuracy scores
print("Mean accuracy: {:.2f}%".format(np.mean(cv_scores) * 100))


#feature Selection
y = fit_data['Team_Win']
all_features = fit_data[['Team Odds', 'Home', 'Against_Avg_w', 'Avg Diff Run Metres w'
                         ]]

best_score = 0
best_features = None

for r in range(1, len(all_features) + 1):  # Try different numbers of predictors
    for feature_subset in combinations(all_features, r):
        X_subset = fit_data[list(feature_subset)]
        X_scaled = scaler.fit_transform(X_subset)

        cv_scores = cross_val_score(model, X_scaled, y, cv=rkf, scoring='accuracy')
        mean_score = np.mean(cv_scores)

        if mean_score > best_score:
            best_score = mean_score
            best_features = feature_subset

        #print(f"Features: {feature_subset}, Mean Accuracy: {mean_score:.4f}")

print("\nBest Feature Set:", best_features)
print("Best Accuracy: {:.2f}%".format(best_score * 100))



#---------#
#Linear Regression for margin
#---------#


X = fit_data[['Team Odds','Attack_Rating', 'Defense_Rating', 'Opp_Attack_Rating', 
                   'Opp_Defense_Rating', 'Avg Diff Run Metres w', 'Avg Diff Kicking Metres w'
                   ]]
y = fit_data['Margin']

x = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

y_pred = model.predict(x)
y_actual = y

comparison = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred})

print(comparison.head(60))

plt.figure(figsize=(8, 6))
plt.scatter(y_actual, y_pred, alpha=0.6, color='blue', label='Predicted vs Actual')
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], color='red', linestyle='--', label='Perfect Fit')
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()


#---------#
#Linear Regression for fantasy score
#---------#


mids = position_dfs['mids']

X = mids[[
                   ]]
y = mids['Total Points']

x = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())






# ----------------------#
# ----------------------#
# Insights
# ----------------------#
# ----------------------#

match_corr = matches_full.corr()

fit_data = matches_full
fit_data = fit_data.sample(frac=1).reset_index(drop=True)

fit_data['Team Line Odds'] = np.where(fit_data['Home'] == 1, fit_data['Home Line Odds Open'],
                                      fit_data['Away Line Odds Open'])
fit_data['Home'] = (fit_data['Team'] == 'Home').astype(int)

fit_data_home = fit_data[fit_data['Team'] == 'Home']
fit_data_home = fit_data_home.sample(frac=1).reset_index(drop=True)
fit_data_away = fit_data[fit_data['Team'] == 'Away']
fit_data_away = fit_data_home.sample(frac=1).reset_index(drop=True)


#---------#
# Margin
#---------#

X = fit_data[['All Run Metres Per Min_x', 'Post Contact Metres Per Min_x', 'Line Breaks Per Min_x', 'Kick Return Metres Per Min_x',
               'Tackle Breaks Per Min_x', 'Offloads Per Min_x', 'Completion Rate', 'Kicking Metres Per Min_x',
                   ]]
y = fit_data['Margin']

x = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


y_pred = model.predict(X)
y_actual = y

comparison = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred})

print(comparison.head(60))


X = fit_data[['All Run Metres', 'Post Contact Metres', 'Line Breaks', 'Kick Return Metres',
               'Tackle Breaks', 'Offloads', 'Completion Rate', 'Kicking Metres'
                   ]]
y = fit_data['Margin']

x = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


y_pred = model.predict(X)
y_actual = y

comparison = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred})

print(comparison.head(60))





