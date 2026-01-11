import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import Analysis.json_to_csv as j
from collections import defaultdict
import ENVIRONMENT_VARIABLES as EV
from openai import OpenAI
import random
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr
import Analysis.functions as fn


st.set_page_config(page_title="NRL Stats", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #0b1020;
        --panel: #161c32;
        --panel-2: #1e2542;
        --accent: #c9f500;
        --text: #f5f7ff;
        --muted: #9aa4bf;
        --border: #2a3356;
    }
    .stApp {
        background: radial-gradient(1200px 600px at 15% 10%, #141a33 0%, #0b1020 55%, #070b16 100%);
        color: var(--text);
    }
    .page-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: var(--text);
        margin-bottom: 0.35rem;
    }
    .page-title span {
        color: var(--accent);
        margin-left: 0.35rem;
    }
    .title-divider {
        height: 2px;
        background: var(--accent);
        opacity: 0.85;
        margin-bottom: 1.25rem;
    }
    .filter-bar {
        background: var(--panel);
        border: 1px solid var(--border);
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stRadio [role="radiogroup"] {
        background: transparent;
        gap: 0.5rem;
    }
    .stRadio label {
        background: var(--panel-2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.4rem 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .stRadio label:hover {
        border-color: var(--accent);
        color: var(--accent);
    }
    .stButton>button {
        background: var(--panel-2);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 8px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .stButton>button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }
    .stSelectbox>div>div {
        background: var(--panel-2);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    .stDataFrame, .stTable {
        background: var(--panel);
        border-radius: 10px;
    }
    .stExpander {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="page-title">NRL <span>Stats</span></div>', unsafe_allow_html=True)
st.markdown('<div class="title-divider"></div>', unsafe_allow_html=True)

@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def load_data():
    match_df = j.get_match_data([])
    df = j.get_player_stats()
    df = fn.player_data_cleaner_simple(df, match_df)
    df = df[df['Mins Played'] > 0]
    df.rename(columns={"Total Points": "Fantasy"}, inplace=True)
    return df


df_all = load_data()
if df_all.empty:
    st.error("No data returned from Supabase.")
    st.stop()

df_all = df_all.loc[:, ~df_all.columns.duplicated()].copy()
if "Opponent" in df_all.columns and isinstance(df_all["Opponent"], pd.DataFrame):
    df_all["Opponent"] = df_all["Opponent"].iloc[:, 0]

available_years = sorted(df_all['Year'].dropna().unique().tolist(), reverse=True)
with st.container():
    year_choice = st.selectbox("Year", available_years, index=0)

df = df_all[df_all['Year'] == year_choice].copy()

if df.empty:
    st.warning("No data available for the selected year.")
    st.stop()

def plotly_chart_custom(fig):
    fig.update_layout(
        paper_bgcolor="#f5f6f8",
        plot_bgcolor="#ffffff",
        font=dict(color="#111827"),
        margin=dict(l=20, r=20, t=50, b=40),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
        title_font=dict(color="#111827"),
        tickfont=dict(color="#111827"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
        title_font=dict(color="#111827"),
        tickfont=dict(color="#111827"),
    )
    fig.update_layout(
        dragmode=False  # Disable the drag-to-zoom feature
    )
    
    plot_counter = st.session_state.get("_plot_counter", 0)
    st.session_state["_plot_counter"] = plot_counter + 1
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"plotly_chart_{plot_counter}",
        config={
            'displayModeBar': True,  # Show the mode bar
            'modeBarButtonsToRemove': ['Autoscale', 'zoom', 'pan', 'resetScale', 'zoomIn', 'zoomOut', 'select2d', 'lasso2d'],  # Remove zoom and pan buttons
            'modeBarButtonsToAdd': ['toImage', 'toggleFullscreen'],  # Add Fullscreen and Download only
            'scrollZoom': False,  # Disable zooming with mouse scroll
            'showTips': True  # Ensure that hover functionality works
        }
    )


def _set_corr_layout(fig, stat1, stat2, title, corr_text=None, corr_color=None):
    annotations = []
    if corr_text and corr_color:
        annotations = [
            dict(
                text=corr_text,
                xref='paper',
                yref='paper',
                x=0,
                y=1.05,
                showarrow=False,
                font=dict(size=12, color=corr_color),
                align='left',
                bgcolor="#f0f0f0",
                bordercolor=corr_color,
                borderwidth=1
            )
        ]
    fig.update_layout(
        title=dict(text=title, font=dict(color='black')),
        annotations=annotations,
        xaxis=dict(
            title=dict(text=stat1, font=dict(color='black')),
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            title=dict(text=stat2, font=dict(color='black')),
            tickfont=dict(color='black')
        ),
        legend=dict(font=dict(color='black'), y=1.15, x=0.6),
        plot_bgcolor='#99AEDE',
        paper_bgcolor='#99AEDE',
        font=dict(color='black'),
        hovermode='closest'
    )


def _add_trendline_and_correlation(fig, x, y, stat1, stat2, title):
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    x = data["x"]
    y = data["y"]

    if len(x) < 2 or len(y) < 2:
        st.info("Correlation unavailable: need at least two data points.")
        _set_corr_layout(fig, stat1, stat2, title)
        return
    if (x == 0).all() or (y == 0).all():
        st.info("Correlation unavailable: one of the stats is all 0.")
        _set_corr_layout(fig, stat1, stat2, title)
        return
    if x.nunique() < 2 or y.nunique() < 2:
        st.info("Correlation unavailable: one of the stats has no variation.")
        _set_corr_layout(fig, stat1, stat2, title)
        return

    m, b = np.polyfit(x, y, 1)
    fig.add_trace(go.Scatter(
        x=x,
        y=m * x + b,
        mode='lines',
        line=dict(dash='dash', color='black'),
        name='Trendline'
    ))

    corr_coef, p_value = pearsonr(x, y)
    abs_corr = abs(corr_coef)

    if abs_corr < 0.3:
        correlation = "Weak"
        color = "red"
    elif abs_corr < 0.7:
        correlation = "Medium"
        color = "orange"
    else:
        correlation = "Strong"
        color = "green"

    corr_text = f"<b>Correlation:</b> <span style='color:{color}'>{correlation}</span> (r = {corr_coef:.2f})"
    _set_corr_layout(fig, stat1, stat2, title, corr_text=corr_text, corr_color=color)


def _opposition_label(frame):
    def _col(series_or_df):
        if isinstance(series_or_df, pd.DataFrame):
            return series_or_df.iloc[:, 0]
        return series_or_df

    if "Opponent" in frame.columns:
        return _col(frame["Opponent"])
    if "Opposition" in frame.columns:
        return _col(frame["Opposition"])
    if {"Home Team", "Away Team", "Team"}.issubset(frame.columns):
        return np.where(
            _col(frame["Team"]) == _col(frame["Home Team"]),
            _col(frame["Away Team"]),
            _col(frame["Home Team"])
        )
    return ""


def _prepare_round_plot_df(frame):
    frame = frame.copy()
    if "Round" in frame.columns:
        frame = frame.sort_values("Round")
    if "Round_Label" in frame.columns:
        frame["Round_Display"] = frame["Round_Label"].astype(str)
    else:
        frame["Round_Display"] = frame["Round"].astype(str)
    finals_labels = {"FW1", "FW2", "FW3", "GF"}
    frame["Round_Hover"] = np.where(
        frame["Round_Display"].isin(finals_labels),
        frame["Round_Display"],
        "Rd " + frame["Round_Display"]
    )
    labels = np.asarray(_opposition_label(frame)).reshape(-1)
    frame["Opposition_Label"] = pd.Series(labels, index=frame.index).astype(str)
    return frame


team_list = sorted(df['Team'].dropna().unique().tolist())
player_list = sorted(df['Name'].dropna().unique().tolist())
player_stat_list = [stat for stat in EV.PLAYER_STATS if stat in df.columns]
team_stat_list = [stat for stat in EV.TEAM_STATS if stat in df.columns]

page = st.radio(
    "Views",
    ["Player Comparison", "Teams Comparison"],
    horizontal=True,
    label_visibility="collapsed",
)

if page == "Player Comparison":
    st.header("Player Comparison")
    left_col, right_col = st.columns([1.3, 2], gap="large")

    with left_col:
        player1_query = st.text_input("Search Player 1")
        player1_options = [p for p in player_list if player1_query.lower() in p.lower()] if player1_query else player_list
        player1 = st.selectbox("Select Player 1", player1_options)

        player2_query = st.text_input("Search Player 2 (Optional)")
        player2_options = [p for p in player_list if player2_query.lower() in p.lower()] if player2_query else player_list
        player2 = st.selectbox("Select Player 2 (Optional)", ["None"] + player2_options)
        stat1 = st.selectbox("Select Stat 1", player_stat_list)
        stat2 = st.selectbox("Select Stat 2 (Optional)", ["None"] + player_stat_list)

        summary_data = []

        # Player 1 Stat 1
        if player1 and stat1:
            df_p1_stat1 = df[df['Name'] == player1][stat1]
            summary_data.append({
                "Player": player1,
                "Stat": stat1,
                "Average": f"{df_p1_stat1.mean():.2f}",
                "Min": f"{df_p1_stat1.min():.2f}",
                "Max": f"{df_p1_stat1.max():.2f}"
            })

        # Player 2 Stat 1
        if player2 != "None" and stat1:
            df_p2_stat1 = df[df['Name'] == player2][stat1]
            summary_data.append({
                "Player": player2,
                "Stat": stat1,
                "Average": f"{df_p2_stat1.mean():.2f}",
                "Min": f"{df_p2_stat1.min():.2f}",
                "Max": f"{df_p2_stat1.max():.2f}"
            })

        # Player 1 Stat 2
        if player1 and stat2 != "None":
            df_p1_stat2 = df[df['Name'] == player1][stat2]
            summary_data.append({
                "Player": player1,
                "Stat": stat2,
                "Average": f"{df_p1_stat2.mean():.2f}",
                "Min": f"{df_p1_stat2.min():.2f}",
                "Max": f"{df_p1_stat2.max():.2f}"
            })
            
        # Player 2 Stat 2
        if player2 != "None" and stat2 != "None":
            df_p2_stat2 = df[df['Name'] == player2][stat2]
            summary_data.append({
                "Player": player2,
                "Stat": stat2,
                "Average": f"{df_p2_stat2.mean():.2f}",
                "Min": f"{df_p2_stat2.min():.2f}",
                "Max": f"{df_p2_stat2.max():.2f}"
            })

        # Display the summary table
        if summary_data:
            st.markdown("### 📊 Stat Summary")
            st.table(pd.DataFrame(summary_data))

    if player2 != "None" and stat2 != "None":
        plot_type = 4
    elif player2 != "None":
        plot_type = 2
    elif stat2 != "None":
        plot_type = 3
    else:
        plot_type = 1

    with right_col:
        st.subheader("📈 Stat Comparison by Round:")
        if plot_type == 4:

            # Filter the DataFrame for each team's data
            df_player1 = _prepare_round_plot_df(df[df['Name'] == player1])
            df_player2 = _prepare_round_plot_df(df[df['Name'] == player2])
            
            # --- Plot 1: stat1 over Round ---
            fig1 = go.Figure()
            
            # Team 1
            fig1.add_trace(go.Scatter(
                x=df_player1['Round_Display'],
                y=df_player1[stat1],
                mode='lines+markers',
                name=f"{player1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_player1['Opposition_Label'] + ', ' + df_player1['Round_Hover'].astype(str),
                hoverinfo='text'
                
            ))
            
            # Team 2
            fig1.add_trace(go.Scatter(
                x=df_player2['Round_Display'],
                y=df_player2[stat1],
                mode='lines+markers',
                name=f"{player2}",
                marker=dict(symbol='x', size=8),
                line=dict(color='green'),
                hovertext=df_player2['Opposition_Label'] + ', ' + df_player2['Round_Hover'].astype(str),
                hoverinfo='text'
            ))
            
            fig1.update_layout(
                title=dict(text=f"{stat1} Comparison: {player1} vs {player2}", font=dict(color='black')),
                xaxis=dict(title=dict(text="Round", font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(title=dict(text=f"{stat1}", font=dict(color='black')), tickfont=dict(color='black')),
                legend=dict(font=dict(color='black'), y=1.15, x=0.6),
                plot_bgcolor='#99AEDE',
                paper_bgcolor='#99AEDE'
            )
            
            plotly_chart_custom(fig1)
            
            # --- Plot 2: stat2 over Round ---
            fig2 = go.Figure()
            
            # Team 1
            fig2.add_trace(go.Scatter(
                x=df_player1['Round_Display'],
                y=df_player1[stat2],
                mode='lines+markers',
                name=f"{player1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_player1['Opposition_Label'] + ', ' + df_player1['Round_Hover'].astype(str),
                hoverinfo='text'
            ))
            
            # Team 2
            fig2.add_trace(go.Scatter(
                x=df_player2['Round_Display'],
                y=df_player2[stat2],
                mode='lines+markers',
                name=f"{player2}",
                marker=dict(symbol='x', size=8),
                line=dict(color='green'),
                hovertext=df_player2['Opposition_Label'] + ', ' + df_player2['Round_Hover'].astype(str),
                hoverinfo='text'
            ))
            
            fig2.update_layout(
                title=dict(text=f"{stat2} Comparison: {player1} vs {player2}", font=dict(color='black')),
                xaxis=dict(title=dict(text="Round", font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(title=dict(text=f"{stat2}", font=dict(color='black')), tickfont=dict(color='black')),
                legend=dict(font=dict(color='black'), y=1.15, x=0.6),
                plot_bgcolor='#99AEDE',
                paper_bgcolor='#99AEDE'
            )
            
            plotly_chart_custom(fig2)


        elif plot_type == 2:
        
            # Filter the DataFrame for each team's data
            df_player1 = _prepare_round_plot_df(df[df['Name'] == player1])
            df_player2 = _prepare_round_plot_df(df[df['Name'] == player2])
            
            # Create the figure for comparing stat1
            fig = go.Figure()
            
            # Team 1 for stat1
            fig.add_trace(go.Scatter(
                x=df_player1['Round_Display'],
                y=df_player1[stat1],
                mode='lines+markers',
                name=f"{player1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_player1['Opposition_Label'] + ', ' + df_player1['Round_Hover'].astype(str),
                hoverinfo='text'
            ))
            
            # Team 2 for stat1
            fig.add_trace(go.Scatter(
                x=df_player2['Round_Display'],
                y=df_player2[stat1],
                mode='lines+markers',
                name=f"{player2}",
                marker=dict(symbol='x', size=8),
                line=dict(color='green'),
                hovertext=df_player2['Opposition_Label'] + ', ' + df_player2['Round_Hover'].astype(str),
                hoverinfo='text'
            ))
            
            # Update the layout for the figure
            fig.update_layout(
                title=dict(text=f"{stat1} Comparison: {player1} vs {player2}", font=dict(color='black')),
                xaxis=dict(title=dict(text="Round", font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(title=dict(text=f"{stat1}", font=dict(color='black')), tickfont=dict(color='black')),
                legend=dict(font=dict(color='black'), y=1.15, x=0.6),
                plot_bgcolor='#99AEDE',
                paper_bgcolor='#99AEDE'
            )
            
            # Display the figure
            plotly_chart_custom(fig) 

        elif plot_type == 3:
            
            # Filter the DataFrame for each team's data
            df_player1 = _prepare_round_plot_df(df[df['Name'] == player1])

            
            # --- Plot 1: stat1 over Round ---
            fig1 = go.Figure()
            
            # Team 1
            fig1.add_trace(go.Scatter(
                x=df_player1['Round_Display'],
                y=df_player1[stat1],
                mode='lines+markers',
                name=f"{player1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_player1['Opposition_Label'] + ', ' + df_player1['Round_Hover'].astype(str),
                hoverinfo='text'
                
            ))
            
            fig1.update_layout(
                title=dict(text=f"{stat1}: {player1}", font=dict(color='black')),
                xaxis=dict(title=dict(text="Round", font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(title=dict(text=f"{stat1}", font=dict(color='black')), tickfont=dict(color='black')),
                legend=dict(font=dict(color='black'), y=1.15, x=0.6),
                plot_bgcolor='#99AEDE',
                paper_bgcolor='#99AEDE'
            )
            
            plotly_chart_custom(fig1)
            
            # --- Plot 2: stat2 ---
            fig2 = go.Figure()
            
            # Team 1
            fig2.add_trace(go.Scatter(
                x=df_player1['Round_Display'],
                y=df_player1[stat2],
                mode='lines+markers',
                name=f"{player1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_player1['Opposition_Label'] + ', ' + df_player1['Round_Hover'].astype(str),
                hoverinfo='text'
                
            ))
            
            fig2.update_layout(
                title=dict(text=f"{stat2}: {player1}", font=dict(color='black')),
                xaxis=dict(title=dict(text="Round", font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(title=dict(text=f"{stat2}", font=dict(color='black')), tickfont=dict(color='black')),
                legend=dict(font=dict(color='black'), y=1.15, x=0.6),
                plot_bgcolor='#99AEDE',
                paper_bgcolor='#99AEDE'
            )
            
            plotly_chart_custom(fig2)
                        
        else:
            
            # Filter the DataFrame for each team's data
            df_player1 = _prepare_round_plot_df(df[df['Name'] == player1])
            
            # --- Plot 1: stat1 over Round ---
            fig1 = go.Figure()
            
            # Team 1
            fig1.add_trace(go.Scatter(
                x=df_player1['Round_Display'],
                y=df_player1[stat1],
                mode='lines+markers',
                name=f"{player1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_player1['Opposition_Label'] + ', ' + df_player1['Round_Hover'].astype(str),
                hoverinfo='text'
                
            ))
            
            fig1.update_layout(
                title=dict(text=f"{stat1}: {player1}", font=dict(color='black')),
                xaxis=dict(title=dict(text="Round", font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(title=dict(text=f"{stat1}", font=dict(color='black')), tickfont=dict(color='black')),
                legend=dict(font=dict(color='black'), y=1.15, x=0.6),
                plot_bgcolor='#99AEDE',
                paper_bgcolor='#99AEDE'
            )
            
            plotly_chart_custom(fig1)


        st.subheader(f"📈 {stat1} vs {stat2}:")
        st.write("Hover for opposition")


        #if st.button("Create Stat Plot/s"):
        if plot_type == 4:
            # Filter the DataFrame for each team's data
            df_player1 = _prepare_round_plot_df(df[df['Name'] == player1])
            df_player2 = _prepare_round_plot_df(df[df['Name'] == player2])
            
            # --- Plot 1: Team 1 ---
            x1 = df_player1[stat1]
            y1 = df_player1[stat2]
            hovertext1 = df_player1['Opposition_Label'] + ', ' + df_player1['Round_Hover'].astype(str)
            
            fig1 = go.Figure()
            
            # Scatter points
            fig1.add_trace(go.Scatter(
                x=x1,
                y=y1,
                mode='markers',
                hovertext=hovertext1,
                hoverinfo='text',
                marker=dict(size=10, color='#1f77b4'),
                name=player1
            ))
            
            _add_trendline_and_correlation(
                fig1,
                x1,
                y1,
                stat1,
                stat2,
                f"{stat1} vs {stat2}: {player1}"
            )
            plotly_chart_custom(fig1)
                
            # --- Plot 2: Team 2 ---
            x2 = df_player2[stat1]
            y2 = df_player2[stat2]
            hovertext2 = df_player2['Opposition_Label'] + ', ' + df_player2['Round_Hover'].astype(str)
            
            fig2 = go.Figure()
            
            # Scatter points
            fig2.add_trace(go.Scatter(
                x=x2,
                y=y2,
                mode='markers',
                hovertext=hovertext2,
                hoverinfo='text',
                marker=dict(size=10, color='#ff7f0e'),
                name=player2
            ))
            
            _add_trendline_and_correlation(
                fig2,
                x2,
                y2,
                stat1,
                stat2,
                f"{stat1} vs {stat2}: {player2}"
            )
            plotly_chart_custom(fig2)
            

        elif plot_type == 3:
            
            df_player1 = _prepare_round_plot_df(df[df['Name'] == player1])


            if not df_player1.empty:
                x = df_player1[stat1]
                y = df_player1[stat2]
                
                fig = go.Figure()
            
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',  # No 'text' here since we only want hover
                hovertext=df_player1['Opposition_Label'] + ', ' + df_player1['Round_Hover'].astype(str),  # What shows on hover
                hoverinfo='text',                  # Use only the text above
                marker=dict(size=10, color='#1f77b4'),
                name='Opposition, Round'
                ))
            
                _add_trendline_and_correlation(
                    fig,
                    x,
                    y,
                    stat1,
                    stat2,
                    f"{stat1} vs {stat2}: {player1}"
                )
                plotly_chart_custom(fig)


        else:
            st.write("Please add a 2nd stat")

        
        
    
elif page == "Teams Comparison":
    st.header("Teams Comparison")

    if not team_stat_list:
        st.error("No team stats available for the selected year.")
        st.stop()

    left_col, right_col = st.columns([1.3, 2], gap="large")

    with left_col:
        team1 = st.selectbox("Select Team 1", team_list)
        team2 = st.selectbox("Select Team 2 (Optional)", ["None"] + team_list)
        stat1 = st.selectbox("Select Stat 1", team_stat_list)
        stat2 = st.selectbox("Select Stat 2 (Optional)", ["None"] + team_stat_list)
        
        group_cols = ['Team', 'Round', 'Opponent']
        if 'Round_Label' in df.columns:
            group_cols.append('Round_Label')
        team_df = df.groupby(group_cols, as_index=False)[team_stat_list].sum()

        summary_data = []

        # Player 1 Stat 1
        if team1 and stat1:
            df_p1_stat1 = team_df[team_df['Team'] == team1][stat1]
            summary_data.append({
                "team": team1,
                "Stat": stat1,
                "Average": f"{df_p1_stat1.mean():.2f}",
                "Min": f"{df_p1_stat1.min():.2f}",
                "Max": f"{df_p1_stat1.max():.2f}"
            })

        # team 2 Stat 1
        if team2 != "None" and stat1:
            df_p2_stat1 = team_df[team_df['Team'] == team2][stat1]
            summary_data.append({
                "team": team2,
                "Stat": stat1,
                "Average": f"{df_p2_stat1.mean():.2f}",
                "Min": f"{df_p2_stat1.min():.2f}",
                "Max": f"{df_p2_stat1.max():.2f}"
            })

        # team 1 Stat 2
        if team1 and stat2 != "None":
            df_p1_stat2 = team_df[team_df['Team'] == team1][stat2]
            summary_data.append({
                "team": team1,
                "Stat": stat2,
                "Average": f"{df_p1_stat2.mean():.2f}",
                "Min": f"{df_p1_stat2.min():.2f}",
                "Max": f"{df_p1_stat2.max():.2f}"
            })
            
        # team 2 Stat 2
        if team2 != "None" and stat2 != "None":
            df_p2_stat2 = team_df[team_df['Team'] == team2][stat2]
            summary_data.append({
                "team": team2,
                "Stat": stat2,
                "Average": f"{df_p2_stat2.mean():.2f}",
                "Min": f"{df_p2_stat2.min():.2f}",
                "Max": f"{df_p2_stat2.max():.2f}"
            })

        # Display the summary table
        if summary_data:
            st.markdown("### 📊 Stat Summary")
            st.table(pd.DataFrame(summary_data))

    if team2 != "None" and stat2 != "None":
        plot_type = 4
    elif team2 != "None":
        plot_type = 2
    elif stat2 != "None":
        plot_type = 3
    else:
        plot_type = 1
        
    with right_col:
        st.subheader("📈 Stat Comparison by Round:")
        st.write("Hover for opposition")

        if plot_type == 4:
            df_team1 = _prepare_round_plot_df(team_df[team_df['Team'] == team1])
            df_team2 = _prepare_round_plot_df(team_df[team_df['Team'] == team2])
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_team1['Round_Display'],
                y=df_team1[stat1],
                mode='lines+markers',
                name=f"{team1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_team1['Opposition_Label'] + ', ' + df_team1['Round_Hover'].astype(str),
                hoverinfo='text'
            ))
            fig1.add_trace(go.Scatter(
                x=df_team2['Round_Display'],
                y=df_team2[stat1],
                mode='lines+markers',
                name=f"{team2}",
                marker=dict(symbol='x', size=8),
                line=dict(color='green'),
                hovertext=df_team2['Opposition_Label'] + ', ' + df_team2['Round_Hover'].astype(str),
                hoverinfo='text'
            ))
            fig1.update_layout(
                title=dict(text=f"{stat1} Comparison: {team1} vs {team2}", font=dict(color='black')),
                xaxis=dict(title=dict(text="Round", font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(title=dict(text=f"{stat1}", font=dict(color='black')), tickfont=dict(color='black')),
                legend=dict(font=dict(color='black'), y=1.15, x=0.6),
                plot_bgcolor='#99AEDE',
                paper_bgcolor='#99AEDE'
            )
            plotly_chart_custom(fig1)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_team1['Round_Display'],
                y=df_team1[stat2],
                mode='lines+markers',
                name=f"{team1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_team1['Opposition_Label'] + ', ' + df_team1['Round_Hover'].astype(str),
                hoverinfo='text'
            ))
            fig2.add_trace(go.Scatter(
                x=df_team2['Round_Display'],
                y=df_team2[stat2],
                mode='lines+markers',
                name=f"{team2}",
                marker=dict(symbol='x', size=8),
                line=dict(color='green'),
                hovertext=df_team2['Opposition_Label'] + ', ' + df_team2['Round_Hover'].astype(str),
                hoverinfo='text'
            ))
            fig2.update_layout(
                title=dict(text=f"{stat2} Comparison: {team1} vs {team2}", font=dict(color='black')),
                xaxis=dict(title=dict(text="Round", font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(title=dict(text=f"{stat2}", font=dict(color='black')), tickfont=dict(color='black')),
                legend=dict(font=dict(color='black'), y=1.15, x=0.6),
                plot_bgcolor='#99AEDE',
                paper_bgcolor='#99AEDE'
            )
            plotly_chart_custom(fig2)
        elif plot_type == 2:
            df_team1 = _prepare_round_plot_df(team_df[team_df['Team'] == team1])
            df_team2 = _prepare_round_plot_df(team_df[team_df['Team'] == team2])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_team1['Round_Display'],
                y=df_team1[stat1],
                mode='lines+markers',
                name=f"{team1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_team1['Opposition_Label'] + ', ' + df_team1['Round_Hover'].astype(str),
                hoverinfo='text'
            ))
            fig.add_trace(go.Scatter(
                x=df_team2['Round_Display'],
                y=df_team2[stat1],
                mode='lines+markers',
                name=f"{team2}",
                marker=dict(symbol='x', size=8),
                line=dict(color='green'),
                hovertext=df_team2['Opposition_Label'] + ', ' + df_team2['Round_Hover'].astype(str),
                hoverinfo='text'
            ))
            fig.update_layout(
                title=dict(text=f"{stat1} Comparison: {team1} vs {team2}", font=dict(color='black')),
                xaxis=dict(title=dict(text="Round", font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(title=dict(text=f"{stat1}", font=dict(color='black')), tickfont=dict(color='black')),
                legend=dict(font=dict(color='black'), y=1.15, x=0.6),
                plot_bgcolor='#99AEDE',
                paper_bgcolor='#99AEDE'
            )
            plotly_chart_custom(fig)
        elif plot_type == 3:
            df_team1 = _prepare_round_plot_df(team_df[team_df['Team'] == team1])
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_team1['Round_Display'],
                y=df_team1[stat1],
                mode='lines+markers',
                name=f"{team1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_team1['Opposition_Label'] + ', ' + df_team1['Round_Hover'].astype(str),
                hoverinfo='text'
            ))
            fig1.update_layout(
                title=dict(text=f"{stat1}: {team1}", font=dict(color='black')),
                xaxis=dict(title=dict(text="Round", font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(title=dict(text=f"{stat1}", font=dict(color='black')), tickfont=dict(color='black')),
                legend=dict(font=dict(color='black'), y=1.15, x=0.6),
                plot_bgcolor='#99AEDE',
                paper_bgcolor='#99AEDE'
            )
            plotly_chart_custom(fig1)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_team1['Round_Display'],
                y=df_team1[stat2],
                mode='lines+markers',
                name=f"{team1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_team1['Opposition_Label'] + ', ' + df_team1['Round_Hover'].astype(str),
                hoverinfo='text'
            ))
            fig2.update_layout(
                title=dict(text=f"{stat2}: {team1}", font=dict(color='black')),
                xaxis=dict(title=dict(text="Round", font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(title=dict(text=f"{stat2}", font=dict(color='black')), tickfont=dict(color='black')),
                legend=dict(font=dict(color='black'), y=1.15, x=0.6),
                plot_bgcolor='#99AEDE',
                paper_bgcolor='#99AEDE'
            )
            plotly_chart_custom(fig2)
        else:
            df_team1 = _prepare_round_plot_df(team_df[team_df['Team'] == team1])
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_team1['Round_Display'],
                y=df_team1[stat1],
                mode='lines+markers',
                name=f"{team1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_team1['Opposition_Label'] + ', ' + df_team1['Round_Hover'].astype(str),
                hoverinfo='text'
            ))
            fig1.update_layout(
                title=dict(text=f"{stat1}: {team1}", font=dict(color='black')),
                xaxis=dict(title=dict(text="Round", font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(title=dict(text=f"{stat1}", font=dict(color='black')), tickfont=dict(color='black')),
                legend=dict(font=dict(color='black'), y=1.15, x=0.6),
                plot_bgcolor='#99AEDE',
                paper_bgcolor='#99AEDE'
            )
            plotly_chart_custom(fig1)

        st.subheader(f"📈 {stat1} vs {stat2}:")
        st.write("Hover for opposition")

        if plot_type == 4:
            df_team1 = _prepare_round_plot_df(team_df[team_df['Team'] == team1])
            df_team2 = _prepare_round_plot_df(team_df[team_df['Team'] == team2])
            
            x1 = df_team1[stat1]
            y1 = df_team1[stat2]
            hovertext1 = df_team1['Opposition_Label'] + ', ' + df_team1['Round_Hover'].astype(str)
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=x1,
                y=y1,
                mode='markers',
                hovertext=hovertext1,
                hoverinfo='text',
                marker=dict(size=10, color='#1f77b4'),
                name=team1
            ))
            
            _add_trendline_and_correlation(
                fig1,
                x1,
                y1,
                stat1,
                stat2,
                f"{stat1} vs {stat2}: {team1}"
            )
            plotly_chart_custom(fig1)
                
            x2 = df_team2[stat1]
            y2 = df_team2[stat2]
            hovertext2 = df_team2['Opposition_Label'] + ', ' + df_team2['Round_Hover'].astype(str)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=x2,
                y=y2,
                mode='markers',
                hovertext=hovertext2,
                hoverinfo='text',
                marker=dict(size=10, color='#ff7f0e'),
                name=team2
            ))
            
            _add_trendline_and_correlation(
                fig2,
                x2,
                y2,
                stat1,
                stat2,
                f"{stat1} vs {stat2}: {team2}"
            )
            plotly_chart_custom(fig2)
        elif plot_type == 3:
            df_team1 = _prepare_round_plot_df(team_df[team_df['Team'] == team1])

            if not df_team1.empty:
                x = df_team1[stat1]
                y = df_team1[stat2]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    hovertext=df_team1['Opposition_Label'] + ', ' + df_team1['Round_Hover'].astype(str),
                    hoverinfo='text',
                    marker=dict(size=10, color='#1f77b4'),
                    name='Opposition, Round'
                ))
                _add_trendline_and_correlation(
                    fig,
                    x,
                    y,
                    stat1,
                    stat2,
                    f"{stat1} vs {stat2}: {team1}"
                )
                plotly_chart_custom(fig)
        else:
            st.write("Please add a 2nd stat")



else:
    st.header("Home")
    st.write("Please select a view.")
    
