import json
import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict
import ENVIRONMENT_VARIABLES as EV
import numpy as np

def get_player_stats():
    
    #with open("data/player_statistics_2023.json", "r") as file:
        #data_2023 = json.load(file)
        
    with open("data/player_statistics_2024_r1-6.json", "r") as file:
        data_r1_6 = json.load(file)
        
    with open("data/player_statistics_2024_r7-r27.json", "r") as file:
        data_r7_27 = json.load(file)
        
    with open("data/player_statistics_2025.json", "r") as file:
        data_2025 = json.load(file)
    '''
    rounds = EV.ROUNDS
    data_25_rds = []
    
    for rd in rounds:
        with open(f"data/player_statistics_2025_r{rd}.json", "r") as file:
            data_25_rds[rd] = json.load(file)
    '''
    
    # Extract player statistics
    def extract_player_stats(data):
        player_stats = []
    
        for year_data in data["PlayerStats"]:
            for year, rounds in year_data.items():
                for round_data in rounds:
                    for round_num, games in round_data.items():
                        for game in games:
                            for game_key, players in game.items():
                                # Extract game details (e.g., year, round, teams)
                                game_parts = game_key.split("-")
                                year = game_parts[0]
                                round_num = game_parts[1]
                                teams_part = "-".join(game_parts[2:])  # Rejoin possible hyphenated team names
                                home_team, away_team = [team.strip() for team in teams_part.split("-v-")]
                                
                            # Flatten player data
                            for player in players:
                                player["Year"] = year
                                player["Round"] = round_num
                                player["Home Team"] = home_team
                                player["Away Team"] = away_team
                                player_stats.append(player)                            
        
        return player_stats
    # Extract player statistics from both datasets
    #player_stats_2023 = extract_player_stats(data_2023)
    player_stats_r1_6 = extract_player_stats(data_r1_6)
    player_stats_r7_27 = extract_player_stats(data_r7_27)
    player_stats_2025 = extract_player_stats(data_2025)
    
    player_stats_2025_rds = []
    '''
    for rd in rounds:
        player_stats_2025_rds[rd] = extract_player_stats(data_25_rds[rd])'''
    
    # Combine both datasets
    combined_player_stats =  player_stats_r1_6 + player_stats_r7_27 + player_stats_2025
    '''for rd in rounds:
        combined_player_stats = combined_player_stats + player_stats_2025_rds[rd]'''
    
    # Convert to DataFrame
    df = pd.DataFrame(combined_player_stats)
    
    #Remove hyphens
    df['Home Team'] = df['Home Team'].str.replace("-", " ", regex=False)
    df['Away Team'] = df['Away Team'].str.replace("-", " ", regex=False)
    
    
    #Remove duplicates
    df1 = df
    df = df.drop_duplicates(subset=['Name', 'Round', 'Year'], keep='first').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv("data/player_statistics.csv", index=False)
    
    return df   


        
def get_match_data(years):
    
    matches_dict = {}
    
    for year in years:
        with open(f"data/nrl_data_{year}.json", "r") as file:
            data = json.load(file)
        
        matches = []
        for season_data in data["NRL"]:
            for season_year, rounds in season_data.items():
                for round_data in rounds:
                    for round_num, games in round_data.items():
                        for game in games:
                            matches.append({
                                "Year": season_year,
                                "Round": round_num,
                                "Date": game["Date"],
                                "Home": game["Home"],
                                "Home_Score": int(game["Home_Score"]),
                                "Away": game["Away"],
                                "Away_Score": int(game["Away_Score"]),
                                "Venue": game["Venue"]
                            })
        
        matches_dict[year] = pd.DataFrame(matches)
    
    # Combine all years into a single DataFrame
    match_df = pd.concat(matches_dict.values(), ignore_index=True)
    
    return match_df


def get_detailed_match(years):
    
    matches_dict = {}

    for year in years:
        with open(f"data/nrl_detailed_match_data_{year}.json", "r") as file:
            data = json.load(file)
        
        matches = []
        for round_data in data["NRL"]:
            for match_id, games in round_data.items():
                for game in games:
                    for teams, details in game.items():
                        match_info = {
                            "Year": year,
                            "Round": match_id,
                            "Match": teams,
                            "Ground Condition": details["match"].get("ground_condition", ""),
                            "Weather Condition": details["match"].get("weather_condition", ""),
                            "Referee": details["match"].get("main_ref", ""),
                            "First Try Scorer": details["match"].get("overall_first_try_scorer", ""),
                            "First Try Time": details["match"].get("overall_first_try_minute", ""),
                            "First Try Team": details["match"].get("overall_first_try_round", ""),
                        }
    
                        # Home and Away Team Stats
                        for team_type in ["home", "away"]:
                            team_data = details[team_type]
                            team_stats = {
                                "Team": team_type.capitalize(),
                                "Possession": team_data.get("possession", ""),
                                "Time In Possession": team_data.get("Time In Possession", ""),
                                "Tries": team_data.get("tries", ""),
                                "Try Scorers": ", ".join(team_data.get("try_names", [])),
                                "Try Minutes": ", ".join(team_data.get("try_minutes", [])),
                                "Conversions": team_data.get("conversions", ""),
                                "Penalty Goals": team_data.get("penalty_goals", ""),
                                "Sin Bins": team_data.get("sin_bins", ""),
                                "1-Point Field Goals": team_data.get("1_point_field_goals", ""),
                                "2-Point Field Goals": team_data.get("2_point_field_goals", ""),
                                "Half Time Score": team_data.get("half_time", ""),
                                "All Runs": team_data.get("All Runs", ""),
                                "All Run Metres": team_data.get("All Run Metres", ""),
                                "Post Contact Metres": team_data.get("Post Contact Metres", ""),
                                "Line Breaks": team_data.get("Line Breaks", ""),
                                "Tackle Breaks": team_data.get("Tackle Breaks", ""),
                                "Kick Return Metres": team_data.get("Kick Return Metres", ""),
                                "Offloads": team_data.get("Offloads", ""),
                                "Receipts": team_data.get("Receipts", ""),
                                "Total Passes": team_data.get("Total Passes", ""),
                                "Dummy Passes": team_data.get("Dummy Passes", ""),
                                "Kicks": team_data.get("Kicks", ""),
                                "Kicking Metres": team_data.get("Kicking Metres", ""),
                                "Forced Drop Outs": team_data.get("Forced Drop Outs", ""),
                                "Bombs": team_data.get("Bombs", ""),
                                "Grubbers": team_data.get("Grubbers", ""),
                                "Tackles Made": team_data.get("Tackles Made", ""),
                                "Missed Tackles": team_data.get("Missed Tackles", ""),
                                "Intercepts": team_data.get("Intercepts", ""),
                                "Ineffective Tackles": team_data.get("Ineffective Tackles", ""),
                                "Errors": team_data.get("Errors", ""),
                                "Penalties Conceded": team_data.get("Penalties Conceded", ""),
                                "Ruck Infringements": team_data.get("Ruck Infringements", ""),
                                "Inside 10 Metres Penalties": team_data.get("Inside 10 Metres", ""),
                                "Interchanges Used": team_data.get("Used", ""),
                                "Completion Rate": team_data.get("Completion Rate", ""),
                                "Avg Play Ball Speed": team_data.get("Average_Play_Ball_Speed", ""),
                                "Kick Defusal %": team_data.get("Kick_Defusal", ""),
                                "Effective Tackle %": team_data.get("Effective_Tackle", ""),
                            }
    
                            # Append the combined match info and team stats
                            matches.append({**match_info, **team_stats})
        
        matches_dict[year] = pd.DataFrame(matches)
    
    # Combine all years into a single DataFrame
    match_detailed = pd.concat(matches_dict.values(), ignore_index=True)
    
    match_detailed['Year'] = match_detailed['Year'].astype(str)

    return match_detailed










