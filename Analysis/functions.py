import numpy as np
import pandas as pd
import ENVIRONMENT_VARIABLES as EV


def _strip_suffix(series, suffix):
    if series.dtype == object:
        return series.str.rstrip(suffix)
    return series


def _time_to_float(time_value):
    if pd.isna(time_value):
        return 0.0
    if isinstance(time_value, (int, float, np.integer, np.floating)):
        return float(time_value)
    time_str = str(time_value)
    if not time_str or time_str == '0':
        return 0.0
    try:
        minutes, seconds = map(int, time_str.split(":"))
        return minutes + seconds / 60
    except Exception as exc:
        print(f"Error processing {time_str}: {exc}")
        return None

def _normalize_round(series):
    values = series.astype(str).str.extract(r'(\d+)', expand=False)
    return pd.to_numeric(values, errors='coerce').fillna(0)



def clean_match_detailed(match_detailed, match_df):
    match_detailed['Team_Name'] = np.where(
        match_detailed['Team'] == 'Home',
        match_detailed['Match'].str.split(" v ").str[0],
        match_detailed['Match'].str.split(" v ").str[1]
    )

    match_detailed['Possession'] = match_detailed['Possession'].astype("string").str.rstrip('%')
    match_detailed['First Try Time'] = match_detailed['First Try Time'].astype("string").str.rstrip("'")

    match_detailed['Time In Possession'] = match_detailed['Time In Possession'].apply(_time_to_float)

    categorical_cols = [
        "Round", "Match", "Ground Condition", "Weather Condition", "Referee",
        "First Try Scorer", "First Try Team", "Team", "Try Scorers", "Team_Name"
    ]
    match_detailed[categorical_cols] = match_detailed[categorical_cols].astype("category")

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
    matches_full['Team_Win'] = np.where(matches_full['Team_Score'] > matches_full['Opp_Score'], 1, 0)

    matches_full['Round'] = matches_full['Round'].astype(int)

    if "Date" in matches_full.columns:
        matches_full['Date_formatted'] = pd.to_datetime(matches_full['Date']).dt.strftime("%Y-%m-%d")
    else:
        from dateutil import parser
        import re

        def convert_date(date_str, year):
            cleaned_date = re.sub(r'^\w+ ', '', date_str)
            cleaned_date = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', cleaned_date)
            parsed_date = parser.parse(cleaned_date + f" {year}")
            return parsed_date.strftime("%Y-%m-%d")

        matches_full['Date_formatted'] = matches_full.apply(
            lambda row: convert_date(row['Date'], row['Year']), axis=1
        )

    return matches_full


def get_odds():
    odds = pd.read_csv('/Users/jontyandrew/Desktop/Nrl/nrl_odds.csv', header=1)

    odds['Date'] = pd.to_datetime(odds['Date']).dt.strftime("%Y-%m-%d")

    nrl_team_mapping = {
        "Brisbane Broncos": "Broncos",
        "Canberra Raiders": "Raiders",
        "Canterbury Bulldogs": "Bulldogs",
        "Canterbury-Bankstown Bulldogs": "Bulldogs",
        "Manly-Warringah Sea Eagles": "Sea Eagles",
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
        "Dolphins": "Dolphins",
        "Tigers": "Wests Tigers"
    }

    odds['Home Team'] = odds['Home Team'].replace(nrl_team_mapping)
    odds['Away Team'] = odds['Away Team'].replace(nrl_team_mapping)

    return odds


def combine_odds(matches_full, odds):
    matches_full = pd.merge(
        matches_full,
        odds[['Date', 'Home Team', 'Away Team', 'Home Odds', 'Away Odds',
              'Home Line Open', 'Away Line Open', 'Home Line Odds Open', 'Away Line Odds Open']],
        left_on=['Date_formatted', 'Home'], right_on=['Date', 'Home Team'], how='left'
    )

    matches_full['Team Odds'] = np.where(
        matches_full['Team'] == 'Home',
        matches_full['Home Odds'],
        matches_full['Away Odds']
    )

    matches_full['Line Diff'] = matches_full.apply(
        lambda row: row['Margin'] + row['Home Line Open'] if row['Team'] == 'Home'
        else row['Margin'] + row['Away Line Open'],
        axis=1
    )

    matches_full['Line Win'] = matches_full.apply(
        lambda row: 1 if row['Line Diff'] > 0 else 0,
        axis=1
    )

    return matches_full


# Model helpers and per-minute stats logic are unchanged from the original helpers.
# They rely on normalized, numeric match data from clean_match_detailed().

def create_model_features(matches_full, df):
    matches_full['Year'] = matches_full['Year'].astype('int')
    matches_full = matches_full.sort_values(['Year', 'Round'])

    matches_full['Cumulative_Round'] = matches_full.groupby(['Year', 'Round']).ngroup() + 1
    matches_full['Year'] = matches_full['Year'].astype('string')

    def weighted_average(group, value_column, weight_column):
        return (group[value_column] * group[weight_column]).sum() / group[weight_column].sum()

    win_rates = matches_full[['Cumulative_Round', 'Team_Name', 'Team_Win', 'Team_Score', 'Opp_Score']]
    win_rates = win_rates.rename(columns={'Team_Name': 'Team1'})

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

        wr['Pythagorean_Expectation'] = wr['Team_For'] ** 2 / (wr['Team_For'] ** 2 + wr['Team_Against'] ** 2)
        wr['Cumulative Round Number'] = i
        w_rates[i] = wr

    win_percentages = pd.concat(w_rates.values(), ignore_index=True)

    match_stats = pd.merge(
        matches_full, win_percentages,
        left_on=['Team_Name', 'Cumulative_Round'], right_on=['Team1', 'Cumulative Round Number']
    )

    match_stats['Opposition'] = np.where(
        match_stats['Team_Name'] == match_stats['Home'],
        match_stats['Away'],
        match_stats['Home']
    )

    match_stats = pd.merge(
        match_stats, win_percentages,
        left_on=['Opposition', 'Cumulative_Round'], right_on=['Team1', 'Cumulative Round Number']
    )

    match_stats = match_stats.rename(columns={
        'For_Avg_y': 'Opp_For_Avg',
        'For_Avg_w_y': 'Opp_For_Avg_w',
        'Against_Avg_y': 'Opp_Against_Avg',
        'Against_Avg_w_y': 'Opp_Against_Avg_w',
        'For_Avg_x': 'For_Avg',
        'For_Avg_w_x': 'For_Avg_w',
        'Against_Avg_x': 'Against_Avg',
        'Against_Avg_w_x': 'Against_Avg_w',
        'Win_Rate_w_y': 'Opp_Win_Rate',
        'Win_Rate_w_x': 'Win_Rate',
        'Pythagorean_Expectation_y': 'Opp_Pythag_Exp',
        'Pythagorean_Expectation_x': 'Pythag_Exp'
    })

    match_stats['Opposition_Rating'] = ((match_stats['Opp_Win_Rate'] + match_stats['Opp_Pythag_Exp']) / 2)
    match_stats['Rating'] = ((match_stats['Win_Rate'] + match_stats['Pythag_Exp']) / 2)

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

    diffs = pd.merge(
        match_stats[['Cumulative_Round', 'Team_Name', 'All Run Metres Per Min', 'Post Contact Metres Per Min',
                     'Line Breaks Per Min', 'Kick Return Metres Per Min', 'Tackles Made', 'Missed Tackles',
                     'Tackle Breaks Per Min', 'Offloads Per Min', 'Errors', 'Kicking Metres Per Min']],
        match_stats[['Cumulative_Round', 'Opposition', 'All Run Metres Per Min', 'Post Contact Metres Per Min',
                     'Line Breaks Per Min', 'Kick Return Metres Per Min', 'Tackles Made', 'Missed Tackles',
                     'Tackle Breaks Per Min', 'Offloads Per Min', 'Errors', 'Kicking Metres Per Min']],
        left_on=['Cumulative_Round', 'Team_Name'], right_on=['Cumulative_Round', 'Opposition']
    )

    diff_columns = [
        "All Run Metres Per Min", "Post Contact Metres Per Min", "Kick Return Metres Per Min",
        "Line Breaks Per Min", "Tackles Made", "Missed Tackles", "Tackle Breaks Per Min",
        "Offloads Per Min", "Errors", "Kicking Metres Per Min"
    ]

    for col in diff_columns:
        diffs[f"Diff {col}"] = diffs[f"{col}_x"] - diffs[f"{col}_y"]

    match_stats = pd.merge(
        match_stats,
        diffs[['Cumulative_Round', 'Team_Name', 'Diff All Run Metres Per Min', 'Diff Post Contact Metres Per Min',
               'Diff Kick Return Metres Per Min', 'Diff Line Breaks Per Min', 'Diff Tackles Made',
               'Diff Missed Tackles', 'Diff Tackle Breaks Per Min', 'Diff Offloads Per Min',
               'Diff Errors', 'Diff Kicking Metres Per Min']],
        on=['Team_Name', 'Cumulative_Round']
    )

    match_stats = match_stats.drop_duplicates(['Team_Name', 'Cumulative_Round'])

    averages = match_stats[['Cumulative_Round', 'Team_Name', 'Diff All Run Metres Per Min', 'Line Breaks Per Min',
                            'Diff Kicking Metres Per Min', 'Errors', 'Total Passes', 'Penalties Conceded',
                            'Ruck Infringements', 'Opp_Defense_Rating', 'Attack_Rating',
                            'Diff Line Breaks Per Min', 'Diff Post Contact Metres Per Min',
                            'Diff Tackle Breaks Per Min']]

    averages = averages.rename(columns={'Team_Name': 'Team2'})
    avgs = {}

    for i in range(1, (matches_full['Cumulative_Round'].max() + 1)):
        average = averages[averages['Cumulative_Round'] < i]
        average['Weight'] = np.exp(-(i - average['Cumulative_Round']) / 20)

        average = average.groupby(by='Team2').agg(**{
            'Avg Diff Run Metres': ('Diff All Run Metres Per Min', 'mean'),
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

    match_stats = pd.merge(
        match_stats,
        averages[['Cumulative_Round_Number', 'Team2', 'Avg Diff Run Metres', 'Avg Diff Run Metres Per Min w',
                  'Avg Linebreaks Per Min', 'Avg Diff Kicking Metres Per Min', 'Avg Diff Kicking Metres Per Min w',
                  'Avg Errors', 'Avg Total Passes', 'Avg Penalties', 'Avg Diff pcm Per Min w',
                  'Avg Diff Line Breaks Per Min w', 'Avg Diff Tackle Breaks Per Min w', 'Avg Ruck Infringements']],
        right_on=['Team2', 'Cumulative_Round_Number'],
        left_on=['Team_Name', 'Cumulative_Round'],
        how='left'
    )

    df['Year'] = df['Year'].astype('string')
    df['Round'] = _normalize_round(df['Round']).astype('int')

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
        matches_full['Home']
    )

    matches_full = pd.merge(
        matches_full,
        matches_full[['Round', 'Year', 'Team_Name', 'Team_Score', 'Half Time Score', 'All Run Metres Per Min',
                      'Post Contact Metres', 'Line Breaks', 'Kick Return Metres', 'Tackle Breaks', 'Offloads',
                      'Errors', 'Kicking Metres', 'All Run Metres Per Min', 'Post Contact Metres Per Min',
                      'Line Breaks Per Min', 'Kick Return Metres Per Min', 'Tackles Made', 'Missed Tackles',
                      'Tackle Breaks Per Min', 'Offloads Per Min', 'Errors', 'Kicking Metres Per Min']],
        left_on=['Round', 'Year', 'Opposition'], right_on=['Round', 'Year', 'Team_Name']
    )

    rename_list = ['Team_Name', 'Team_Score', 'Half Time Score', 'All Run Metres', 'Post Contact Metres',
                   'Line Breaks', 'Kick Return Metres', 'Tackle Breaks', 'Offloads', 'Kicking Metres',
                   'All Run Metres Per Min', 'Post Contact Metres Per Min', 'Line Breaks Per Min',
                   'Kick Return Metres Per Min', 'Tackles Made', 'Missed Tackles', 'Tackle Breaks Per Min',
                   'Offloads Per Min', 'Errors', 'Kicking Metres Per Min']

    rename_list_x = [col + '_x' for col in rename_list]
    rename_list_y = [col + '_y' for col in rename_list]

    rename_list = rename_list_x + rename_list_y

    def generate_rename_mapping(columns):
        rename_map = {}
        for col in columns:
            if col.endswith('_x'):
                rename_map[col] = col[:-2]
            elif col.endswith('_y'):
                rename_map[col] = col[:-2] + ' Opp'
        return rename_map

    rename_map = generate_rename_mapping(rename_list)

    matches_full = matches_full.rename(columns=rename_map)

    return matches_full


def player_data_cleaner(df, match_data, matches_full):
    df = pd.merge(
        df,
        match_data[['Year', 'Round', 'Home', 'Home_Score', 'Away_Score']],
        left_on=['Year', 'Round', 'Home Team'],
        right_on=['Year', 'Round', 'Home']
    )

    df = df.drop_duplicates(subset=['Name', 'Round', 'Year'])

    n = 18
    rows = len(df)
    pattern = np.tile(np.concatenate([['Home Team'] * n, ['Away Team'] * n]), rows // (2 * n) + 1)[:rows]

    df['Team_Name'] = df.lookup(df.index, pattern)

    df = df.drop(columns=['Home'])

    df = df.replace('-', '0')
    df['Tackle Efficiency'] = _strip_suffix(df['Tackle Efficiency'], '%')
    df['Average Play The Ball Speed'] = _strip_suffix(df['Average Play The Ball Speed'], 's')

    df['Mins Played'] = df['Mins Played'].apply(_time_to_float)
    df['Stint One'] = df['Stint One'].apply(_time_to_float)
    df['Stint Two'] = df['Stint Two'].apply(_time_to_float)

    for col, col_type in EV.types_to_convert.items():
        if col not in df.columns:
            continue
        if col_type == 'drop':
            df = df.drop(columns=[col])
        elif col_type == 'string':
            df[col] = df[col].astype('string')
        elif col_type == 'category':
            df[col] = df[col].astype('category')
        elif col_type == 'int':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int')
        elif col_type == 'float':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.reset_index(drop=True)
    df['Round'] = _normalize_round(df['Round']).astype('int')

    df = pd.merge(df, matches_full, left_on=['Round', 'Team_Name', 'Year'], right_on=['Round', 'Team_Name', 'Year'])

    df.columns = df.columns.str.replace('_x', '', regex=True).str.replace('_y', '_team', regex=True)

    df['Year'] = df['Year'].astype('int')
    df['Round'] = _normalize_round(df['Round']).astype('int')

    def calculate_fantasy_average(player_data):
        player_data = player_data.sort_values(by=['Year', 'Round'])
        player_data['Fantasy_Average'] = None

        for i, row in player_data.iterrows():
            year, round_num = row['Year'], row['Round']

            past_scores = player_data[(player_data['Year'] == year) & (player_data['Round'] < round_num)]['Total Points'].dropna()

            if not past_scores.empty:
                avg = past_scores.mean()
            else:
                last_year_scores = player_data[player_data['Year'] == (year - 1)]['Total Points'].dropna()
                avg = last_year_scores.mean() if not last_year_scores.empty else None

            player_data.at[i, 'Fantasy_Average'] = avg

        return player_data

    fantasy_average = df.groupby(['Name', 'Team_Name']).apply(calculate_fantasy_average).reset_index(drop=True)[
        ['Name', 'Year', 'Team_Name', 'Round', 'Fantasy_Average']
    ]

    df = pd.merge(df, fantasy_average, on=['Name', 'Team_Name', 'Year', 'Round'])

    df['Date'] = pd.to_datetime(df['Date_team'])
    df = df.sort_values(by=['Name', 'Date'])

    df['days_since_last'] = df.groupby('Name')['Date'].diff().dt.days
    df['days_since_last'] = df['days_since_last'].fillna(10)

    df['days_since_last'] = df['days_since_last'].clip(upper=10)

    return df


def player_data_cleaner_simple(df, match_data):
    df = pd.merge(
        df,
        match_data[['Year', 'Round', 'Home', 'Home_Score', 'Away_Score']],
        left_on=['Year', 'Round', 'Home Team'],
        right_on=['Year', 'Round', 'Home']
    )

    df = df.drop_duplicates(subset=['Name', 'Round', 'Year'])

    n = 18
    rows = len(df)
    pattern = np.tile(np.concatenate([['Home Team'] * n, ['Away Team'] * n]), rows // (2 * n) + 1)[:rows]

    df['Team_Name'] = df.apply(lambda row: row[pattern[row.name]], axis=1)

    df['Opposition'] = df['Away Team']
    df.loc[df['Team_Name'] == df['Home Team'], 'Opposition'] = df['Away Team']
    df.loc[df['Team_Name'] != df['Home Team'], 'Opposition'] = df['Home Team']

    df = df.replace('-', '0')
    df['Tackle Efficiency'] = _strip_suffix(df['Tackle Efficiency'], '%')
    df['Average Play The Ball Speed'] = _strip_suffix(df['Average Play The Ball Speed'], 's')

    df['Mins Played'] = df['Mins Played'].apply(_time_to_float)
    df['Stint One'] = df['Stint One'].apply(_time_to_float)
    df['Stint Two'] = df['Stint Two'].apply(_time_to_float)

    for col, col_type in EV.types_to_convert.items():
        if col not in df.columns:
            continue
        if col_type == 'drop':
            df = df.drop(columns=[col])
        elif col_type == 'string':
            df[col] = df[col].astype(str)
        elif col_type == 'category':
            df[col] = df[col].astype('category')
        elif col_type == 'int':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int')
        elif col_type == 'float':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.reset_index(drop=True)
    df['Round'] = _normalize_round(df['Round']).astype('int')

    df.columns = df.columns.str.replace('_x', '', regex=True).str.replace('_y', '_team', regex=True)

    return df


def create_player_dicts(df):
    df = df[df['Mins Played'] > 0]
    full_game = df[df['Mins Played'] > 60]
    backs = full_game[full_game['Number'].isin(["1", "2", "3", "4", "5", "6", "7"])]
    forwards = df[df['Number'].isin(["8", "9", "10", "11", "12", "13"])]

    mids = df[df['Number'].isin(["8", "10", '13'])]
    hookers = df[df['Number'] == "9"]
    back_row = df[df['Number'].isin(["11", "12"])]
    halves = full_game[full_game['Number'].isin(["6", "7"])]
    centres = full_game[full_game['Number'].isin(["3", "4"])]
    wingers = full_game[full_game['Number'].isin(["2", "5"])]
    fullback = full_game[full_game['Number'] == "1"]
    edges = full_game[full_game['Number'].isin(["11", "12"])]

    position_dfs = {
        "full": df,
        "backs": backs,
        "fullbacks": fullback,
        "wingers": wingers,
        "centres": centres,
        "mids": mids,
        "hookers": hookers,
        "halves": halves,
        "edges": edges
    }

    team_avgs = {}
    for team in df['Team_Name'].unique():
        team_avgs[team] = df[df['Team_Name'] == f'{team}'].groupby(by="Name").mean()

    team = {}
    for team1 in df['Team_Name'].unique():
        team[team1] = df[df['Team_Name'] == f'{team1}']

    players_dict = {name: df for name, df in df.groupby("Name")}

    averages = {}
    for position, df in position_dfs.items():
        avg_df = df.groupby(by="Name").mean()
        avg_df["Games"] = df.groupby(by="Name")["Name"].count()
        averages[position] = avg_df[avg_df["Games"] > 5]

    return players_dict, averages, team, team_avgs, position_dfs
