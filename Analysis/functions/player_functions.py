import pandas as pd
import numpy as np
import ENVIRONMENT_VARIABLES as EV



def player_data_cleaner(df, match_data, matches_full):
    

    df = pd.merge(df, match_data[['Year', 'Round', 'Home', 'Home_Score', 'Away_Score']], left_on=['Year', 'Round', 'Home Team'],
                  right_on=['Year', 'Round', 'Home'])
    
    df = df.drop_duplicates(subset=['Name', 'Round', 'Year'])

    n = 18
    rows = len(df)
    pattern = np.tile(np.concatenate([['Home Team'] * n, ['Away Team'] * n]), rows // (2 * n) + 1)[:rows]
    
    # Assign values based on the pattern
    df['Team_Name'] = df.lookup(df.index, pattern)
    
    
    df = df.drop(columns=['Home'])
    
    df = df.replace('-', '0')  
    df['Tackle Efficiency'] = df['Tackle Efficiency'].str.rstrip('%')  
    df['Average Play The Ball Speed'] = df['Average Play The Ball Speed'].str.rstrip('s')  
    
    def time_to_float(time_str):
        if not time_str or time_str == '0':
            return 0.0
        try:
            minutes, seconds = map(int, time_str.split(":"))
            return minutes + seconds / 60
        except Exception as e:
            print(f"Error processing {time_str}: {e}")
            return None
    
    df['Mins Played'] = df['Mins Played'].apply(time_to_float)
    df['Stint One'] = df['Stint One'].apply(time_to_float)
    df['Stint Two'] = df['Stint Two'].apply(time_to_float)
    
    df.columns
    
    for col, col_type in EV.types_to_convert.items():
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
    
    df.dtypes.head(60)
    df = df.reset_index(drop=True)
    df['Round'] = df['Round'].astype('int')
            
    df = pd.merge(df, matches_full, left_on=['Round', 'Team_Name', 'Year'], right_on=['Round', 'Team_Name', 'Year'])
    
    df.columns = df.columns.str.replace('_x', '', regex=True).str.replace('_y', '_team', regex=True)
    
    df['Year'] = df['Year'].astype('int')
    df['Round'] = df['Round'].astype('int')
    
    def calculate_fantasy_average(player_data):
        player_data = player_data.sort_values(by=['Year', 'Round'])
        player_data['Fantasy_Average'] = None
    
        for i, row in player_data.iterrows():
            year, round_num = row['Year'], row['Round']
            
            # Get all scores before the current round in the same year
            past_scores = player_data[(player_data['Year'] == year) & (player_data['Round'] < round_num)]['Total Points'].dropna()
            
            if not past_scores.empty:
                avg = past_scores.mean()
            else:
                # If no past scores in the current year, use last year's average
                last_year_scores = player_data[player_data['Year'] == (year - 1)]['Total Points'].dropna()
                avg = last_year_scores.mean() if not last_year_scores.empty else None
            
            player_data.at[i, 'Fantasy_Average'] = avg
    
        return player_data
    
    # Apply function per player
    fantasy_average = df.groupby(['Name', 'Team_Name']).apply(calculate_fantasy_average).reset_index(drop=True)[['Name', 'Year', 'Team_Name', 'Round', 'Fantasy_Average']]

    df = pd.merge(df, fantasy_average, on=['Name', 'Team_Name', 'Year', 'Round'])
    
    df['Date'] = pd.to_datetime(df['Date_team'])
    df = df.sort_values(by=['Name', 'Date'])

    df['days_since_last'] = df.groupby('Name')['Date'].diff().dt.days
    df['days_since_last'] = df['days_since_last'].fillna(10)

    # Cap the value at 16
    df['days_since_last'] = df['days_since_last'].clip(upper=10)

    return df

def player_data_cleaner_simple(df, match_data):
        
    
    df = pd.merge(df, match_data[['Year', 'Round', 'Home', 'Home_Score', 'Away_Score']], left_on=['Year', 'Round', 'Home Team'],
                  right_on=['Year', 'Round', 'Home'])
    
    df = df.drop_duplicates(subset=['Name', 'Round', 'Year'])

    n = 18
    rows = len(df)
    pattern = np.tile(np.concatenate([['Home Team'] * n, ['Away Team'] * n]), rows // (2 * n) + 1)[:rows]
    
    # Assign values based on the pattern
    df['Team_Name'] = df.lookup(df.index, pattern)
    
    df['Opposition'] = df['Away Team']  # Default value
    df.loc[df['Team_Name'] == df['Home Team'], 'Opposition'] = df['Away Team']
    df.loc[df['Team_Name'] != df['Home Team'], 'Opposition'] = df['Home Team']
        
    df = df.replace('-', '0')  
    df['Tackle Efficiency'] = df['Tackle Efficiency'].str.rstrip('%')  
    df['Average Play The Ball Speed'] = df['Average Play The Ball Speed'].str.rstrip('s')  
    
    def time_to_float(time_str):
        if not time_str or time_str == '0':
            return 0.0
        try:
            minutes, seconds = map(int, time_str.split(":"))
            return minutes + seconds / 60
        except Exception as e:
            print(f"Error processing {time_str}: {e}")
            return None
    
    df['Mins Played'] = df['Mins Played'].apply(time_to_float)
    df['Stint One'] = df['Stint One'].apply(time_to_float)
    df['Stint Two'] = df['Stint Two'].apply(time_to_float)
    
    df.columns
    
    for col, col_type in EV.types_to_convert.items():
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
    
    df.dtypes.head(60)
    df = df.reset_index(drop=True)
    df['Round'] = df['Round'].astype('int')
                
    df.columns = df.columns.str.replace('_x', '', regex=True).str.replace('_y', '_team', regex=True)
        

    return df


def create_player_dicts(df):    
    
    #Create position datasets
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
