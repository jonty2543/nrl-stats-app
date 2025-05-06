import numpy as np
import pandas as pd

def clean_match_detailed(match_detailed, match_df):
        
    match_detailed['Team_Name'] = np.where(
        match_detailed['Team'] == 'Home',
        match_detailed['Match'].str.split(" v ").str[0],
        match_detailed['Match'].str.split(" v ").str[1]
    )
        
    match_detailed['Possession'] = match_detailed['Possession'].str.rstrip('%')  
    match_detailed['First Try Time'] = match_detailed['First Try Time'].str.rstrip("'")  
    
    def time_to_float(time_str):
        if not time_str or time_str == '0':
            return 0.0
        try:
            minutes, seconds = map(int, time_str.split(":"))
            return minutes + seconds / 60
        except Exception as e:
            print(f"Error processing {time_str}: {e}")
            return None
    
    match_detailed['Time In Possession'] = match_detailed['Time In Possession'].apply(time_to_float)
    
    categorical_cols = [
        "Round", "Match", "Ground Condition", "Weather Condition", "Referee",
        "First Try Scorer", "First Try Team", "Team", "Try Scorers", "Team_Name"
    ]
    match_detailed[categorical_cols] = match_detailed[categorical_cols].astype("category")
    
    # Convert numerical columns to appropriate types
    numeric_cols = [
        "Possession", "First Try Time", "Time In Possession", "Tries", "All Runs", "All Run Metres",
        "Post Contact Metres", "Line Breaks", "Tackle Breaks", "Kick Return Metres",
        "Offloads", "Receipts", "Total Passes", "Dummy Passes", "Kicks", "Kicking Metres",
        "Forced Drop Outs", "Bombs", "Grubbers", "Tackles Made", "Missed Tackles", 
        "Intercepts", "Ineffective Tackles", "Errors", "Penalties Conceded", 
        "Ruck Infringements", "Inside 10 Metres Penalties", "Interchanges Used",
        "Half Time Score"
    ]
    match_detailed[numeric_cols] = match_detailed[numeric_cols].replace(",", "", regex=True).apply(pd.to_numeric, errors="coerce")
    
    # Convert percentage columns to float
    percentage_cols = ["Completion Rate", "Kick Defusal %", "Effective Tackle %"]
    match_detailed[percentage_cols] = match_detailed[percentage_cols].astype("float") / 100
    
    
    matches_full = match_detailed.merge(
        match_df, 
        on=['Round', 'Year'],
        how='inner'
    ).query("Team_Name == Home or Team_Name == Away")
    
    matches_full = matches_full[~((matches_full['Round'] == '1') & (matches_full['Year'] == '2023'))]
    matches_full = matches_full[~((matches_full['Round'] == '26') & (matches_full['Year'] == '2022'))]
    matches_full = matches_full[~((matches_full['Round'] == '27') & (matches_full['Year'] == '2022'))]

    
    matches_full['Team_Score'] = np.where(
        matches_full['Team'] == 'Home',
        matches_full['Home_Score'],
        matches_full['Away_Score']
    )
    
    
    matches_full['Opp_Score'] = np.where(
        matches_full['Team'] == 'Home',
        matches_full['Away_Score'],
        matches_full['Home_Score']
    )
    
    matches_full['Margin'] = matches_full['Team_Score'] - matches_full['Opp_Score']
    matches_full['Team_Win'] = np.where(matches_full['Team_Score'] > matches_full['Opp_Score'], 
                                        1, 0)
    
    matches_full['Round'] = matches_full['Round'].astype(int)
    
    from dateutil import parser
    import re
    
    date_str = 'Thursday 1st August'
    
    def convert_date(date_str, year):
        # Remove the weekday (first word)
        cleaned_date = re.sub(r'^\w+ ', '', date_str)
        
        # Remove ordinal suffixes
        cleaned_date = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', cleaned_date)
        
        # Add the year and parse
        parsed_date = parser.parse(cleaned_date + f" {year}")
        
        return parsed_date.strftime("%Y-%m-%d")
    
    # Apply function to each row
    matches_full['Date_formatted'] = matches_full.apply(lambda row: convert_date(row['Date'], row['Year']), axis=1)
    
    return matches_full

def get_odds():
    odds = pd.read_csv('/Users/jontyandrew/Desktop/Nrl/nrl_odds.csv', header=1)
    
    odds['Date'] = pd.to_datetime(odds['Date']).dt.strftime("%Y-%m-%d")
    
    
    nrl_team_mapping = {
        "Brisbane Broncos": "Broncos",
        "Canberra Raiders": "Raiders",
        "Canterbury Bulldogs": "Bulldogs",
        "Canterbury-Bankstown Bulldogs": "Bulldogs",
        "Manly-Warringah Sea Eagles":"Sea Eagles",
        "Cronulla Sharks": "Sharks",
        "Cronulla-Sutherland Sharks": "Sharks",
        "Gold Coast Titans": "Titans",
        "Manly Sea Eagles": "Sea Eagles",
        "Melbourne Storm": "Storm",
        "Newcastle Knights": "Knights",
        "New Zealand Warriors": "Warriors",
        "North QLD Cowboys": "Cowboys",
        "North Queensland Cowboys": "Cowboys",
        "Parramatta Eels": "Eels",
        "Penrith Panthers": "Panthers",
        "South Sydney Rabbitohs": "Rabbitohs",
        "St George Illawarra Dragons": "Dragons",
        "St George Dragons": "Dragons",
        "St. George Illawarra Dragons": "Dragons",
        "Sydney Roosters": "Roosters",
        "Wests Tigers": "Wests Tigers",
        "Dolphins":"Dolphins",
        "Tigers": "Wests Tigers"
    }
    
    odds['Home Team'] = odds['Home Team'].replace(nrl_team_mapping)
    odds['Away Team'] = odds['Away Team'].replace(nrl_team_mapping)
    
    return odds


def combine_odds(matches_full, odds):
    matches_full = pd.merge(matches_full, odds[['Date', 'Home Team', 'Away Team', 'Home Odds', 'Away Odds', 
                                                'Home Line Open', 'Away Line Open','Home Line Odds Open', 'Away Line Odds Open']], 
                            left_on=['Date_formatted', 'Home'], right_on=['Date', 'Home Team'], how='left')

    matches_full['Team Odds'] = np.where(
        matches_full['Team'] == 'Home',
        matches_full['Home Odds'],
        matches_full['Away Odds'])

    matches_full['Line Diff'] = matches_full.apply(
        lambda row: row['Margin'] + row['Home Line Open'] if row['Team'] == 'Home'
                    else row['Margin'] + row['Away Line Open'],
        axis=1
    )

    matches_full['Line Win'] = matches_full.apply(
        lambda row: 1 if row['Line Diff'] > 0
                    else 0,
        axis=1
    )
    
    return matches_full




def create_model_features(matches_full, df):
    
    matches_full['Year'] = matches_full['Year'].astype('int')
    matches_full = matches_full.sort_values(['Year', 'Round'])
    
    # Assign a unique cumulative round number
    matches_full['Cumulative_Round'] = matches_full.groupby(['Year', 'Round']).ngroup() + 1
    matches_full['Year'] = matches_full['Year'].astype('string')


    def weighted_average(group, value_column, weight_column):
        return (group[value_column] * group[weight_column]).sum() / group[weight_column].sum()
    
    win_rates = matches_full[['Cumulative_Round', 'Team_Name', 'Team_Win', 'Team_Score', 'Opp_Score']]
    win_rates = win_rates.rename(columns={
        'Team_Name':'Team1'
    })

    w_rates = {}
    for i in range(1, (matches_full['Cumulative_Round'].max() + 1)):
        wr = win_rates[win_rates['Cumulative_Round'] < i].copy()
        wr['Weight'] = np.exp(-(i - wr['Cumulative_Round']) / 10)
        
        wr = wr.groupby('Team1').agg(
            Team_For=('Team_Score', 'sum'),
            Team_Against=('Opp_Score', 'sum'),
            For_Avg=('Team_Score', 'mean'),
            Against_Avg=('Opp_Score', 'mean'),
            For_Avg_w=('Team_Score', lambda x: weighted_average(wr.loc[x.index], 'Team_Score', 'Weight')),
            Against_Avg_w=('Opp_Score', lambda x: weighted_average(wr.loc[x.index], 'Opp_Score', 'Weight')),
            Win_Rate_w=('Team_Win', lambda x: weighted_average(wr.loc[x.index], 'Team_Win', 'Weight'))
        ).reset_index()

        wr['Pythagorean_Expectation'] = wr['Team_For']**2 / (wr['Team_For']**2 + wr['Team_Against']**2)
        wr['Cumulative Round Number'] = i
        w_rates[i] = wr

    win_percentages = pd.concat(w_rates.values(), ignore_index=True)


    match_stats = pd.merge(matches_full, win_percentages, left_on=['Team_Name','Cumulative_Round'], right_on=['Team1', 'Cumulative Round Number'])




    match_stats['Opposition'] = np.where(
        match_stats['Team_Name'] == match_stats['Home'],
        match_stats['Away'],
        match_stats['Home'])

    match_stats = pd.merge(match_stats, win_percentages, left_on=['Opposition','Cumulative_Round'], right_on=['Team1', 'Cumulative Round Number'])

    match_stats = match_stats.rename(columns={
        'For_Avg_y':'Opp_For_Avg',
        'For_Avg_w_y':'Opp_For_Avg_w',
        'Against_Avg_y':'Opp_Against_Avg',
        'Against_Avg_w_y':'Opp_Against_Avg_w',
        'For_Avg_x':'For_Avg',
        'For_Avg_w_x':'For_Avg_w',
        'Against_Avg_x':'Against_Avg',
        'Against_Avg_w_x':'Against_Avg_w',
        'Win_Rate_w_y':'Opp_Win_Rate',
        'Win_Rate_w_x':'Win_Rate',
        'Pythagorean_Expectation_y':'Opp_Pythag_Exp',
        'Pythagorean_Expectation_x':'Pythag_Exp'
    })


    match_stats['Opposition_Rating'] = ((match_stats['Opp_Win_Rate'] + match_stats['Opp_Pythag_Exp'])/2)
    match_stats['Rating'] = ((match_stats['Win_Rate'] + match_stats['Pythag_Exp'])/2)
    
    '''
    match_stats['Attack_Rating'] = (match_stats['For_Avg_w'] - match_stats['Min_Team_For_w'])/(match_stats['Max_Team_For_w']-match_stats['Min_Team_For_w'])
                                    
    match_stats['Defense_Rating'] = (match_stats['Against_Avg_w'] - match_stats['Min_Team_For_w'])/(match_stats['Max_Team_For_w']-match_stats['Min_Team_For_w']) 

    match_stats['Opp_Attack_Rating'] = (match_stats['Opp_For_Avg_w'] - match_stats['Min_Team_For_w'])/(match_stats['Max_Team_For_w']-match_stats['Min_Team_For_w']) 
    match_stats['Opp_Defense_Rating'] = (match_stats['Opp_Against_Avg_w'] - match_stats['Min_Team_For_w'])/(match_stats['Max_Team_For_w']-match_stats['Min_Team_For_w'])
    '''
    
    match_stats['Attack_Rating'] = match_stats.groupby("Round")['For_Avg_w'].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    match_stats['Defense_Rating'] = match_stats.groupby("Round")['Against_Avg_w'].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    match_stats['Opp_Attack_Rating'] = match_stats.groupby("Round")['Opp_For_Avg_w'].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    match_stats['Opp_Defense_Rating'] = match_stats.groupby("Round")['Opp_Against_Avg_w'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    '''
    cols1 = [
        'Opposition_Rating', 'Rating', 'Attack_Rating', 'Opp_Attack_Rating'
    ]

    cols2 = [
        'Defense_Rating', 'Opp_Defense_Rating'
    ]

    for col in cols1:
        match_stats[col] = (match_stats[col] - match_stats[col].median()) * np.exp(0.5 * np.log(match_stats['Cumulative_Round'] / 2))
        
    for col in cols2:
        match_stats[col] = -((match_stats[col] - match_stats[col].median()) * np.exp(0.5 * np.log(match_stats['Cumulative_Round'] / 2)))
    '''


    diffs = pd.merge(match_stats[['Cumulative_Round', 'Team_Name', 'All Run Metres Per Min', 'Post Contact Metres Per Min', 'Line Breaks Per Min', 'Kick Return Metres Per Min',
                                 'Tackles Made', 'Missed Tackles', 'Tackle Breaks Per Min', 'Offloads Per Min', 'Errors', 'Kicking Metres Per Min']]
                     , match_stats[['Cumulative_Round', 'Opposition', 'All Run Metres Per Min', 'Post Contact Metres Per Min', 'Line Breaks Per Min', 'Kick Return Metres Per Min',
                                                  'Tackles Made', 'Missed Tackles', 'Tackle Breaks Per Min', 'Offloads Per Min', 'Errors', 'Kicking Metres Per Min']],
                       left_on=['Cumulative_Round', 'Team_Name'], right_on=['Cumulative_Round', 'Opposition'])

    diff_columns = [
        "All Run Metres Per Min", "Post Contact Metres Per Min", "Kick Return Metres Per Min", "Line Breaks Per Min", "Tackles Made",
        "Missed Tackles", "Tackle Breaks Per Min", "Offloads Per Min", "Errors", "Kicking Metres Per Min"
    ]

    for col in diff_columns:
        diffs[f"Diff {col}"] = diffs[f"{col}_x"] - diffs[f"{col}_y"]

    match_stats = pd.merge(match_stats, diffs[['Cumulative_Round','Team_Name','Diff All Run Metres Per Min','Diff Post Contact Metres Per Min', 
                                               'Diff Kick Return Metres Per Min', 'Diff Line Breaks Per Min',
                                             'Diff Tackles Made', 'Diff Missed Tackles', 'Diff Tackle Breaks Per Min',
                                             'Diff Offloads Per Min','Diff Errors', 'Diff Kicking Metres Per Min' ]],
                           on=['Team_Name','Cumulative_Round'])

    match_stats = match_stats.drop_duplicates(['Team_Name', 'Cumulative_Round'])

    averages = match_stats[['Cumulative_Round', 'Team_Name', 'Diff All Run Metres Per Min', 'Line Breaks Per Min', 'Diff Kicking Metres Per Min',
                            'Errors','Total Passes', 'Penalties Conceded', 'Ruck Infringements', 'Opp_Defense_Rating', 
                            'Attack_Rating', 'Diff Line Breaks Per Min', 'Diff Post Contact Metres Per Min', 'Diff Tackle Breaks Per Min']]

    averages = averages.rename(columns={'Team_Name':'Team2'})
    avgs = {}

    for i in range(1, (matches_full['Cumulative_Round'].max() + 1)):
        average = averages[averages['Cumulative_Round'] < i]
        average['Weight'] = np.exp(-(i - average['Cumulative_Round']) / 20)
        
        average = average.groupby(by='Team2').agg(**{'Avg Diff Run Metres': ('Diff All Run Metres Per Min', 'mean'),
                                          'Avg Diff Run Metres Per Min w': ('Diff All Run Metres Per Min', lambda x: weighted_average(average.loc[x.index], 'Diff All Run Metres Per Min', 'Weight')),
                                          'Avg Diff pcm Per Min w': ('Diff Post Contact Metres Per Min', lambda x: weighted_average(average.loc[x.index], 'Diff Post Contact Metres Per Min', 'Weight')),
                                          'Avg Diff Line Breaks Per Min w': ('Diff Line Breaks Per Min', lambda x: weighted_average(average.loc[x.index], 'Diff Line Breaks Per Min', 'Weight')),
                                          'Avg Diff Tackle Breaks Per Min w': ('Diff Tackle Breaks Per Min', lambda x: weighted_average(average.loc[x.index], 'Diff Tackle Breaks Per Min', 'Weight')),
                                          'Avg Linebreaks Per Min': ('Line Breaks Per Min', 'mean'),
                                          'Avg Diff Kicking Metres Per Min': ('Diff Kicking Metres Per Min', 'mean'),
                                          'Avg Diff Kicking Metres Per Min w': ('Diff Kicking Metres Per Min', lambda x: weighted_average(average.loc[x.index], 'Diff Kicking Metres Per Min', 'Weight')),
                                          'Avg Errors': ('Errors', 'mean'),
                                          'Avg Total Passes': ('Total Passes', 'mean'),
                                          'Avg Penalties': ('Penalties Conceded', 'mean'),
                                          'Avg Ruck Infringements': ('Ruck Infringements', 'mean')
                                                    }).reset_index()
        average['Cumulative_Round_Number'] = i
        avgs[i] = average

        
    averages = pd.concat(avgs.values(), ignore_index=True)
        
    match_stats = pd.merge(match_stats, averages[['Cumulative_Round_Number', 'Team2', 'Avg Diff Run Metres', 'Avg Diff Run Metres Per Min w', 
                                                  'Avg Linebreaks Per Min', 'Avg Diff Kicking Metres Per Min','Avg Diff Kicking Metres Per Min w', 'Avg Errors', 
                                                  'Avg Total Passes', 'Avg Penalties', 'Avg Diff pcm Per Min w', 'Avg Diff Line Breaks Per Min w',
                                                  'Avg Diff Tackle Breaks Per Min w', 'Avg Ruck Infringements' ]], right_on=['Team2', 'Cumulative_Round_Number'],
                                                            left_on=['Team_Name', 'Cumulative_Round'], how='left')
    
    df['Year'] = df['Year'].astype('string')
    df['Round'] = df['Round'].astype('int')

    fantasy_avg_of_side = df.groupby(by=['Round', 'Team_Name', 'Year'])['Fantasy_Average'].mean().reset_index()
    match_stats = pd.merge(match_stats, fantasy_avg_of_side, how='left', on=['Round', 'Team_Name', 'Year'])

    return match_stats


def per_min_features(matches_full):
    
    matches_full['Tries Per Min'] = matches_full['Tries'] / matches_full['Time In Possession']
    matches_full['All Runs Per Min'] = matches_full['All Runs'] / matches_full['Time In Possession']
    matches_full['All Run Metres Per Min'] = matches_full['All Run Metres'] / matches_full['Time In Possession']
    matches_full['Post Contact Metres Per Min'] = matches_full['Post Contact Metres'] / matches_full['Time In Possession']
    matches_full['Line Breaks Per Min'] = matches_full['Line Breaks'] / matches_full['Time In Possession']
    matches_full['Tackle Breaks Per Min'] = matches_full['Tackle Breaks'] / matches_full['Time In Possession']
    matches_full['Kick Return Metres Per Min'] = matches_full['Kick Return Metres'] / matches_full['Time In Possession']
    matches_full['Offloads Per Min'] = matches_full['Offloads'] / matches_full['Time In Possession']
    matches_full['Receipts Per Min'] = matches_full['Receipts'] / matches_full['Time In Possession']
    matches_full['Total Passes Per Min'] = matches_full['Total Passes'] / matches_full['Time In Possession']
    matches_full['Dummy Passes Per Min'] = matches_full['Dummy Passes'] / matches_full['Time In Possession']
    matches_full['Kicks Per Min'] = matches_full['Kicks'] / matches_full['Time In Possession']
    matches_full['Kicking Metres Per Min'] = matches_full['Kicking Metres'] / matches_full['Time In Possession']
    matches_full['Forced Drop Outs Per Min'] = matches_full['Forced Drop Outs'] / matches_full['Time In Possession']
    matches_full['Bombs Per Min'] = matches_full['Bombs'] / matches_full['Time In Possession']
    matches_full['Grubbers Per Min'] = matches_full['Grubbers'] / matches_full['Time In Possession']
    matches_full['Errors Per Min'] = matches_full['Errors'] / matches_full['Time In Possession']
    matches_full['Interchanges Used Per Min'] = matches_full['Interchanges Used'] / matches_full['Time In Possession']
        
    return matches_full
        
        
    
def opposition_stats(matches_full):
    
    matches_full['Opposition'] = np.where(
        matches_full['Team_Name'] == matches_full['Home'],
        matches_full['Away'],
        matches_full['Home'])
    
    matches_full = pd.merge(matches_full, matches_full[['Round','Year', 'Team_Name', 'Team_Score', 'Half Time Score', 'All Run Metres Per Min', 'Post Contact Metres', 'Line Breaks', 'Kick Return Metres',
                                                  'Tackle Breaks', 'Offloads', 'Errors', 'Kicking Metres', 'All Run Metres Per Min', 'Post Contact Metres Per Min', 'Line Breaks Per Min', 'Kick Return Metres Per Min',
                                                  'Tackles Made', 'Missed Tackles', 'Tackle Breaks Per Min', 'Offloads Per Min', 'Errors', 'Kicking Metres Per Min']],
                       left_on=['Round','Year', 'Opposition'], right_on=['Round','Year', 'Team_Name'])
    
    rename_list = ['Team_Name', 'Team_Score', 'Half Time Score', 'All Run Metres', 'Post Contact Metres', 'Line Breaks', 'Kick Return Metres',
                'Tackle Breaks', 'Offloads', 'Kicking Metres', 'All Run Metres Per Min', 'Post Contact Metres Per Min', 'Line Breaks Per Min', 'Kick Return Metres Per Min',
                'Tackles Made', 'Missed Tackles', 'Tackle Breaks Per Min', 'Offloads Per Min', 'Errors', 'Kicking Metres Per Min']
    
    rename_list_x = [col + '_x' for col in rename_list]
    rename_list_y = [col + '_y' for col in rename_list]
    
    rename_list = rename_list_x + rename_list_y

    def generate_rename_mapping(columns):
        rename_map = {}
        for col in columns:
            if col.endswith('_x'):
                rename_map[col] = col[:-2]  # remove '_x'
            elif col.endswith('_y'):
                rename_map[col] = col[:-2] + ' Opp'  # replace '_y' with ' Opp'
        return rename_map
    
    rename_map = generate_rename_mapping(rename_list)
    
    matches_full = matches_full.rename(columns=rename_map)
    

    return matches_full

    