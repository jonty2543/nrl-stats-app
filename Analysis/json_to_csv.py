import os
import pandas as pd
from supabase import create_client
import ENVIRONMENT_VARIABLES as EV

try:
    import streamlit as st
except Exception:
    st = None

def _cache_data(**kwargs):
    if st is None:
        def decorator(func):
            return func
        return decorator
    return st.cache_data(**kwargs)


def _get_supabase_client():
    supabase_url = os.getenv("SUPABASE_URL", EV.SUPABASE_URL)
    supabase_key = os.getenv("SUPABASE_KEY", EV.SUPABASE_KEY)
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set.")
    return create_client(supabase_url, supabase_key)


@_cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def _fetch_all_rows(table_name, schema="nrl", page_size=1000):
    client = _get_supabase_client()
    all_rows = []
    start = 0

    while True:
        end = start + page_size - 1
        response = client.schema(schema).table(table_name).select("*").range(start, end).execute()
        rows = response.data or []
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size

    return all_rows


def _round_to_sort(round_value):
    if round_value is None or pd.isna(round_value):
        return None
    round_str = str(round_value)
    finals_map = {
        "Finals Week 1": 28,
        "Finals Week 2": 29,
        "Finals Week 3": 30,
        "Grand Final": 31,
    }
    for key, value in finals_map.items():
        if key.lower() in round_str.lower():
            return value
    digits = pd.to_numeric(pd.Series([round_str]).str.extract(r'(\d+)', expand=False), errors='coerce')
    if digits.notna().iloc[0]:
        return int(digits.iloc[0])
    return None


def _round_to_label(round_value):
    if round_value is None or pd.isna(round_value):
        return ""
    round_str = str(round_value)
    finals_map = {
        "Finals Week 1": "FW1",
        "Finals Week 2": "FW2",
        "Finals Week 3": "FW3",
        "Grand Final": "GF",
    }
    for key, value in finals_map.items():
        if key.lower() in round_str.lower():
            return value
    digits = pd.to_numeric(pd.Series([round_str]).str.extract(r'(\d+)', expand=False), errors='coerce')
    if digits.notna().iloc[0]:
        return str(int(digits.iloc[0]))
    return round_str


def get_player_stats():
    player_rows = _fetch_all_rows("player_stats")
    if not player_rows:
        return pd.DataFrame()

    df = pd.DataFrame(player_rows)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df["Year"] = df["match_date"].dt.year.astype(str)
    df["Round"] = df["round"].apply(_round_to_sort)
    df["Round_Label"] = df["round"].apply(_round_to_label)

    matches_rows = _fetch_all_rows("matches")
    if matches_rows:
        matches_df = pd.DataFrame(matches_rows)
        matches_df["match_date"] = pd.to_datetime(matches_df["match_date"])
        opponent_lookup = (
            matches_df[["match_date", "team", "opponent_team"]]
            .drop_duplicates()
            .rename(columns={"opponent_team": "opponent_team_match"})
        )
        df = df.merge(
            opponent_lookup,
            left_on=["match_date", "team"],
            right_on=["match_date", "team"],
            how="left",
        )
        df["Opponent"] = df["opponent_team_match"]

    match_df = get_match_data(sorted(df["Year"].unique().tolist()))
    if not match_df.empty:
        match_df["Date"] = pd.to_datetime(match_df["Date"])
        home_map = match_df[["Date", "Home", "Away"]].rename(
            columns={"Home": "team", "Away": "Away Team"}
        )
        home_map["Home Team"] = home_map["team"]
        away_map = match_df[["Date", "Home", "Away"]].rename(
            columns={"Away": "team", "Home": "Home Team"}
        )
        away_map["Away Team"] = away_map["team"]
        team_map = pd.concat([home_map, away_map], ignore_index=True)

        df = df.merge(
            team_map,
            left_on=["match_date", "team"],
            right_on=["Date", "team"],
            how="left",
        )
    else:
        df["Home Team"] = None
        df["Away Team"] = None

    rename_map = {
        "player": "Name",
        "team": "Team",
        "opponent_team": "Opponent",
        "number": "Number",
        "position": "Position",
        "mins_played": "Mins Played",
        "stint_one": "Stint One",
        "stint_two": "Stint Two",
        "points": "Points",
        "tries": "Tries",
        "conversions": "Conversions",
        "conversion_attempts": "Conversion Attempts",
        "penalty_goals": "Penalty Goals",
        "goal_conversion_rate": "Goal Conversion Rate",
        "one_point_field_goals": "1 Point Field Goals",
        "two_point_field_goals": "2 Point Field Goals",
        "total_points": "Total Points",
        "all_runs": "All Runs",
        "all_run_metres": "All Run Metres",
        "kick_return_metres": "Kick Return Metres",
        "post_contact_metres": "Post Contact Metres",
        "line_breaks": "Line Breaks",
        "line_break_assists": "Line Break Assists",
        "try_assists": "Try Assists",
        "line_engaged_runs": "Line Engaged Runs",
        "tackle_breaks": "Tackle Breaks",
        "hit_ups": "Hit Ups",
        "play_the_ball": "Play The Ball",
        "average_play_the_ball_speed": "Average Play The Ball Speed",
        "dummy_half_runs": "Dummy Half Runs",
        "dummy_half_run_metres": "Dummy Half Run Metres",
        "one_on_one_steal": "One on One Steal",
        "offloads": "Offloads",
        "dummy_passes": "Dummy Passes",
        "passes": "Passes",
        "receipts": "Receipts",
        "passes_to_run_ratio": "Passes To Run Ratio",
        "tackle_efficiency": "Tackle Efficiency",
        "tackles_made": "Tackles Made",
        "missed_tackles": "Missed Tackles",
        "ineffective_tackles": "Ineffective Tackles",
        "intercepts": "Intercepts",
        "kicks_defused": "Kicks Defused",
        "kicks": "Kicks",
        "kicking_metres": "Kicking Metres",
        "forced_drop_outs": "Forced Drop Outs",
        "bomb_kicks": "Bomb Kicks",
        "grubbers": "Grubbers",
        "forty_twenty": "40/20",
        "twenty_forty": "20/40",
        "cross_field_kicks": "Cross Field Kicks",
        "kicked_dead": "Kicked Dead",
        "errors": "Errors",
        "handling_errors": "Handling Errors",
        "one_on_one_lost": "One on One Lost",
        "penalties": "Penalties",
        "ruck_infringements": "Ruck Infringements",
        "on_report": "On Report",
        "sin_bins": "Sin Bins",
        "send_offs": "Send Offs",
    }

    df = df.rename(columns=rename_map)
    df = df.drop(columns=["Date", "opponent_team_match", "opponent_team"], errors="ignore")
    df = df.drop_duplicates(subset=["Name", "Round", "Year"], keep="first").reset_index(drop=True)

    df["Home Team"] = df["Home Team"].astype("string").str.replace("-", " ", regex=False)
    df["Away Team"] = df["Away Team"].astype("string").str.replace("-", " ", regex=False)

    return df



def get_match_data(years):
    matches_rows = _fetch_all_rows("matches")
    if not matches_rows:
        return pd.DataFrame()

    df = pd.DataFrame(matches_rows)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df["Year"] = df["match_date"].dt.year.astype(str)
    df["Round"] = df["round"].apply(_round_to_sort)
    df["Round_Label"] = df["round"].apply(_round_to_label)
    if years:
        df = df[df["Year"].isin([str(year) for year in years])]

    home_df = df[df["is_home"] == 1].copy()
    if home_df.empty:
        return pd.DataFrame()

    home_df = home_df.rename(
        columns={
            "match_date": "Date",
            "team": "Home",
            "score": "Home_Score",
            "opponent_team": "Away",
            "opponent_score": "Away_Score",
        }
    )

    home_df["Venue"] = None
    match_df = home_df[["Year", "Round", "Round_Label", "Date", "Home", "Home_Score", "Away", "Away_Score", "Venue"]]
    match_df = match_df.drop_duplicates(subset=["Date", "Home", "Away"]).reset_index(drop=True)

    return match_df



def get_detailed_match(years):
    matches_rows = _fetch_all_rows("matches")
    if not matches_rows:
        return pd.DataFrame()

    df = pd.DataFrame(matches_rows)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df["Year"] = df["match_date"].dt.year.astype(str)
    df["Round"] = df["round"].apply(_round_to_sort)
    df["Round_Label"] = df["round"].apply(_round_to_label)
    if years:
        df = df[df["Year"].isin([str(year) for year in years])]

    df["Team"] = df["is_home"].apply(lambda value: "Home" if value == 1 else "Away")
    df["Home Team"] = df.apply(
        lambda row: row["team"] if row["is_home"] == 1 else row["opponent_team"], axis=1
    )
    df["Away Team"] = df.apply(
        lambda row: row["opponent_team"] if row["is_home"] == 1 else row["team"], axis=1
    )
    df["Match"] = df["Home Team"].astype("string") + " v " + df["Away Team"].astype("string")

    match_detailed = pd.DataFrame(
        {
            "Year": df["Year"],
            "Round": df["Round"],
            "Round Label": df["Round_Label"],
            "Match": df["Match"],
            "Ground Condition": df.get("ground_conditions"),
            "Weather Condition": df.get("weather_conditions"),
            "Referee": df.get("referee"),
            "First Try Scorer": None,
            "First Try Time": None,
            "First Try Team": None,
            "Team": df["Team"],
            "Possession": df.get("possession_pct"),
            "Time In Possession": df.get("time_in_possession"),
            "Tries": df.get("tries"),
            "Try Scorers": df.get("tries_summary"),
            "Try Minutes": None,
            "Conversions": df.get("conversions"),
            "Penalty Goals": df.get("penalty_goals"),
            "Sin Bins": df.get("sin_bins"),
            "1-Point Field Goals": df.get("field_goals"),
            "2-Point Field Goals": None,
            "Half Time Score": df.get("halftime_score"),
            "All Runs": df.get("all_runs"),
            "All Run Metres": df.get("all_run_metres"),
            "Post Contact Metres": df.get("post_contact_metres"),
            "Line Breaks": df.get("line_breaks"),
            "Tackle Breaks": df.get("tackle_breaks"),
            "Kick Return Metres": df.get("kick_return_metres"),
            "Offloads": df.get("offloads"),
            "Receipts": df.get("receipts"),
            "Total Passes": df.get("total_passes"),
            "Dummy Passes": df.get("dummy_passes"),
            "Kicks": df.get("kicks"),
            "Kicking Metres": df.get("kicking_metres"),
            "Forced Drop Outs": df.get("forced_drop_outs"),
            "Bombs": df.get("bombs"),
            "Grubbers": df.get("grubbers"),
            "Tackles Made": df.get("tackles_made"),
            "Missed Tackles": df.get("missed_tackles"),
            "Intercepts": df.get("intercepts"),
            "Ineffective Tackles": df.get("ineffective_tackles"),
            "Errors": df.get("errors"),
            "Penalties Conceded": df.get("penalties_conceded"),
            "Ruck Infringements": df.get("ruck_infringements"),
            "Inside 10 Metres Penalties": df.get("inside_10_metres"),
            "Interchanges Used": df.get("used"),
            "Completion Rate": df.get("completion_rate"),
            "Avg Play Ball Speed": df.get("average_play_the_ball_speed"),
            "Kick Defusal %": df.get("kick_defusal_pct"),
            "Effective Tackle %": df.get("effective_tackle_pct"),
        }
    )

    match_detailed["Year"] = match_detailed["Year"].astype(str)

    return match_detailed
