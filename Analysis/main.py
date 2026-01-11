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
    
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            'displayModeBar': True,  # Show the mode bar
            'modeBarButtonsToRemove': ['Autoscale', 'zoom', 'pan', 'resetScale', 'zoomIn', 'zoomOut', 'select2d', 'lasso2d'],  # Remove zoom and pan buttons
            'modeBarButtonsToAdd': ['toImage', 'toggleFullscreen'],  # Add Fullscreen and Download only
            'scrollZoom': False,  # Disable zooming with mouse scroll
            'showTips': True  # Ensure that hover functionality works
        }
    )


team_list = sorted(df['Team_Name'].dropna().unique().tolist())
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
    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        player1 = st.selectbox("Select Player 1", player_list)
        player2 = st.selectbox("Select Player 2 (Optional)", ["None"] + player_list)
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
                "Min": df_p1_stat1.min(),
                "Max": df_p1_stat1.max()
            })

        # Player 2 Stat 1
        if player2 != "None" and stat1:
            df_p2_stat1 = df[df['Name'] == player2][stat1]
            summary_data.append({
                "Player": player2,
                "Stat": stat1,
                "Average": f"{df_p2_stat1.mean():.2f}",
                "Min": df_p2_stat1.min(),
                "Max": df_p2_stat1.max()
            })

        # Player 1 Stat 2
        if player1 and stat2 != "None":
            df_p1_stat2 = df[df['Name'] == player1][stat2]
            summary_data.append({
                "Player": player1,
                "Stat": stat2,
                "Average": f"{df_p1_stat2.mean():.2f}",
                "Min": df_p1_stat2.min(),
                "Max": df_p1_stat2.max()
            })
            
        # Player 2 Stat 2
        if player2 != "None" and stat2 != "None":
            df_p2_stat2 = df[df['Name'] == player2][stat2]
            summary_data.append({
                "Player": player2,
                "Stat": stat2,
                "Average": f"{df_p2_stat2.mean():.2f}",
                "Min": df_p2_stat2.min(),
                "Max": df_p2_stat2.max()
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
            df_player1 = df[df['Name'] == player1]
            df_player2 = df[df['Name'] == player2]
            
            # --- Plot 1: stat1 over Round ---
            fig1 = go.Figure()
            
            # Team 1
            fig1.add_trace(go.Scatter(
                x=df_player1['Round'],
                y=df_player1[stat1],
                mode='lines+markers',
                name=f"{player1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_player1['Opposition'],
                hoverinfo='text'
                
            ))
            
            # Team 2
            fig1.add_trace(go.Scatter(
                x=df_player2['Round'],
                y=df_player2[stat1],
                mode='lines+markers',
                name=f"{player2}",
                marker=dict(symbol='x', size=8),
                line=dict(color='green'),
                hovertext=df_player2['Opposition'],
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
                x=df_player1['Round'],
                y=df_player1[stat2],
                mode='lines+markers',
                name=f"{player1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_player1['Opposition'],
                hoverinfo='text'
            ))
            
            # Team 2
            fig2.add_trace(go.Scatter(
                x=df_player2['Round'],
                y=df_player2[stat2],
                mode='lines+markers',
                name=f"{player2}",
                marker=dict(symbol='x', size=8),
                line=dict(color='green'),
                hovertext=df_player2['Opposition'],
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
            df_player1 = df[df['Name'] == player1]
            df_player2 = df[df['Name'] == player2]
            
            # Create the figure for comparing stat1
            fig = go.Figure()
            
            # Team 1 for stat1
            fig.add_trace(go.Scatter(
                x=df_player1['Round'],
                y=df_player1[stat1],
                mode='lines+markers',
                name=f"{player1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_player1['Opposition'],
                hoverinfo='text'
            ))
            
            # Team 2 for stat1
            fig.add_trace(go.Scatter(
                x=df_player2['Round'],
                y=df_player2[stat1],
                mode='lines+markers',
                name=f"{player2}",
                marker=dict(symbol='x', size=8),
                line=dict(color='green'),
                hovertext=df_player2['Opposition'],
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
            df_player1 = df[df['Name'] == player1]

            
            # --- Plot 1: stat1 over Round ---
            fig1 = go.Figure()
            
            # Team 1
            fig1.add_trace(go.Scatter(
                x=df_player1['Round'],
                y=df_player1[stat1],
                mode='lines+markers',
                name=f"{player1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_player1['Opposition'],
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
                x=df_player1['Round'],
                y=df_player1[stat2],
                mode='lines+markers',
                name=f"{player1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_player1['Opposition'],
                hoverinfo='text'
                
            ))
            
            fig2.update_layout(
                title=dict(text=f"{stat2}: {player1}", font=dict(color='black')),
                xaxis=dict(title=dict(text="Round", font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(title=dict(text=f"{stat1}", font=dict(color='black')), tickfont=dict(color='black')),
                legend=dict(font=dict(color='black'), y=1.15, x=0.6),
                plot_bgcolor='#99AEDE',
                paper_bgcolor='#99AEDE'
            )
            
            plotly_chart_custom(fig2)
                        
        else:
            
            # Filter the DataFrame for each team's data
            df_player1 = df[df['Name'] == player1]
            
            # --- Plot 1: stat1 over Round ---
            fig1 = go.Figure()
            
            # Team 1
            fig1.add_trace(go.Scatter(
                x=df_player1['Round'],
                y=df_player1[stat1],
                mode='lines+markers',
                name=f"{player1}",
                marker=dict(symbol='circle', size=8),
                line=dict(color='#1f77b4'),
                hovertext=df_player1['Opposition'],
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
            df_player1 = df[df['Name'] == player1]
            df_player2 = df[df['Name'] == player2]
            
            # --- Plot 1: Team 1 ---
            x1 = df_player1[stat1]
            y1 = df_player1[stat2]
            hovertext1 = df_player1['Opposition'] + ', Rd ' + df_player1['Round'].astype(str)
            
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
            
            # Trendline
            if not x1.empty and not y1.empty:
                m1, b1 = np.polyfit(x1, y1, 1)
                fig1.add_trace(go.Scatter(
                    x=x1,
                    y=m1 * x1 + b1,
                    mode='lines',
                    line=dict(dash='dash', color='black'),
                    name='Trendline'
                ))
                
                # Calculate correlation
                corr_coef, p_value = pearsonr(x1, y1)
                abs_corr = abs(corr_coef)
            
                # Determine trendline strength
                if abs_corr < 0.3:
                    correlation = "Weak"
                    color = "red"
                elif abs_corr < 0.7:
                    correlation = "Medium"
                    color = "orange"
                else:
                    correlation = "Strong"
                    color = "green"
            
                fig1.update_layout(
                    title=dict(text=f"{stat1} vs {stat2}: {player1}", font=dict(color='black')),
                    annotations=[
                        dict(
                            text=f"<b>Correlation:</b> <span style='color:{color}'>{correlation}</span> (r = {corr_coef:.2f})",
                            xref='paper',
                            yref='paper',
                            x=0,
                            y=1.05,
                            showarrow=False,
                            font=dict(size=12, color=color),
                            align='left',
                            bgcolor="#f0f0f0",
                            bordercolor=color,
                            borderwidth=1
                        )
                    ],
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
                    font=dict(color='black'),  # Optional fallback
                    hovermode='closest'
                )
            
            plotly_chart_custom(fig1)
                
            # --- Plot 2: Team 2 ---
            x2 = df_player2[stat1]
            y2 = df_player2[stat2]
            hovertext2 = df_player2['Opposition'] + ', Rd ' + df_player2['Round'].astype(str)
            
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
            
            # Trendline
            if not x2.empty and not y2.empty:
                m2, b2 = np.polyfit(x2, y2, 1)
                fig2.add_trace(go.Scatter(
                    x=x2,
                    y=m2 * x2 + b2,
                    mode='lines',
                    line=dict(dash='dash', color='black'),
                    name='Trendline'
                ))
                
                # Calculate correlation
                corr_coef, p_value = pearsonr(x2, y2)
                abs_corr = abs(corr_coef)
            
                # Determine trendline strength
                if abs_corr < 0.3:
                    correlation = "Weak"
                    color = "red"
                elif abs_corr < 0.7:
                    correlation = "Medium"
                    color = "orange"
                else:
                    correlation = "Strong"
                    color = "green"
            
                fig2.update_layout(
                    title=dict(text=f"{stat1} vs {stat2}: {player2}", font=dict(color='black')),
                    annotations=[
                        dict(
                            text=f"<b>Correlation:</b> <span style='color:{color}'>{correlation}</span> (r = {corr_coef:.2f})",
                            xref='paper',
                            yref='paper',
                            x=0,
                            y=1.05,
                            showarrow=False,
                            font=dict(size=12, color=color),
                            align='left',
                            bgcolor="#f0f0f0",
                            bordercolor=color,
                            borderwidth=1
                        )
                    ],
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
                    font=dict(color='black'),  # Optional fallback
                    hovermode='closest'
                )
            
            plotly_chart_custom(fig2)
            

        elif plot_type == 3:
            
            df_player1 = df[df['Name'] == player1]


            if not df_player1.empty:
                x = df_player1[stat1]
                y = df_player1[stat2]
                
                # Fit linear trendline
                m, b = np.polyfit(x, y, 1)
                trend_y = m * x + b

                # Calculate correlation
                corr_coef, p_value = pearsonr(x, y)
                abs_corr = abs(corr_coef)
            
                # Determine trendline strength
                if abs_corr < 0.3:
                    correlation = "Weak"
                    color = "red"
                elif abs_corr < 0.7:
                    correlation = "Medium"
                    color = "orange"
                else:
                    correlation = "Strong"
                    color = "green"
                    
                fig = go.Figure()
            
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',  # No 'text' here since we only want hover
                    hovertext=df_player1['Opposition'] + ', Rd ' + df_player1['Round'].astype(str),  # What shows on hover
                    hoverinfo='text',                  # Use only the text above
                    marker=dict(size=10, color='#1f77b4'),
                    name='Opposition, Round'
                ))
            
                # Trendline
                fig.add_trace(go.Scatter(
                    x=x, y=trend_y,
                    mode='lines',
                    line=dict(dash='dash', color='black'),
                    name='Trendline'
                ))
            
                fig.update_layout(
                    title=dict(text=f"{stat1} vs {stat2}: {player1}", font=dict(color='black')),
                    annotations=[
                        dict(
                            text=f"<b>Correlation:</b> <span style='color:{color}'>{correlation}</span> (r = {corr_coef:.2f})",
                            xref='paper',
                            yref='paper',
                            x=0,
                            y=1.05,
                            showarrow=False,
                            font=dict(size=12, color=color),
                            align='left',
                            bgcolor="#f0f0f0",
                            bordercolor=color,
                            borderwidth=1
                        )
                    ],
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
                    font=dict(color='black'),  # Optional fallback
                    hovermode='closest'
                )
                
                plotly_chart_custom(fig)


        else:
            st.write("Please add a 2nd stat")

        
        
    
elif page == "Teams Comparison":
    st.header("Teams Comparison")

    if not team_stat_list:
        st.error("No team stats available for the selected year.")
        st.stop()

    team1 = st.selectbox("Select Team 1", team_list)
    team2 = st.selectbox("Select Team 2 (Optional)", ["None"] + team_list)
    stat1 = st.selectbox("Select Stat 1", team_stat_list)
    stat2 = st.selectbox("Select Stat 2 (Optional)", ["None"] + team_stat_list)
    
    team_df = df.set_index(['Team_Name', 'Round', 'Opposition']).groupby(['Team_Name', 'Round', 'Opposition'])[team_stat_list].sum().reset_index()

    summary_data = []

    # Player 1 Stat 1
    if team1 and stat1:
        df_p1_stat1 = team_df[team_df['Team_Name'] == team1][stat1]
        summary_data.append({
            "team": team1,
            "Stat": stat1,
            "Average": f"{df_p1_stat1.mean():.2f}",
            "Min": df_p1_stat1.min(),
            "Max": df_p1_stat1.max()
        })

    # team 2 Stat 1
    if team2 != "None" and stat1:
        df_p2_stat1 = team_df[team_df['Team_Name'] == team2][stat1]
        summary_data.append({
            "team": team2,
            "Stat": stat1,
            "Average": f"{df_p2_stat1.mean():.2f}",
            "Min": df_p2_stat1.min(),
            "Max": df_p2_stat1.max()
        })

    # team 1 Stat 2
    if team1 and stat2 != "None":
        df_p1_stat2 = team_df[team_df['Team_Name'] == team1][stat2]
        summary_data.append({
            "team": team1,
            "Stat": stat2,
            "Average": f"{df_p1_stat2.mean():.2f}",
            "Min": df_p1_stat2.min(),
            "Max": df_p1_stat2.max()
        })
        
    # team 2 Stat 2
    if team2 != "None" and stat2 != "None":
        df_p2_stat2 = team_df[team_df['Team_Name'] == team2][stat2]
        summary_data.append({
            "team": team2,
            "Stat": stat2,
            "Average": f"{df_p2_stat2.mean():.2f}",
            "Min": df_p2_stat2.min(),
            "Max": df_p2_stat2.max()
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
        
    st.subheader("📈 Stat Comparison by Round:")
    st.write("Hover for opposition")


    #if st.button("Create Round Plot/s"):
    if plot_type == 4:

        # Filter the DataFrame for each team's data
        df_team1 = team_df[team_df['Team_Name'] == team1]
        df_team2 = team_df[team_df['Team_Name'] == team2]
        
        # --- Plot 1: stat1 over Round ---
        fig1 = go.Figure()
        
        # Team 1
        fig1.add_trace(go.Scatter(
            x=df_team1['Round'],
            y=df_team1[stat1],
            mode='lines+markers',
            name=f"{team1}",
            marker=dict(symbol='circle', size=8),
            line=dict(color='#1f77b4'),
            hovertext=df_team1['Opposition'],
            hoverinfo='text'
            
        ))
        
        # Team 2
        fig1.add_trace(go.Scatter(
            x=df_team2['Round'],
            y=df_team2[stat1],
            mode='lines+markers',
            name=f"{team2}",
            marker=dict(symbol='x', size=8),
            line=dict(color='green'),
            hovertext=df_team2['Opposition'],
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
        
        # --- Plot 2: stat2 over Round ---
        fig2 = go.Figure()
        
        # Team 1
        fig2.add_trace(go.Scatter(
            x=df_team1['Round'],
            y=df_team1[stat2],
            mode='lines+markers',
            name=f"{team1}",
            marker=dict(symbol='circle', size=8),
            line=dict(color='#1f77b4'),
            hovertext=df_team1['Opposition'],
            hoverinfo='text'
        ))
        
        # Team 2
        fig2.add_trace(go.Scatter(
            x=df_team2['Round'],
            y=df_team2[stat2],
            mode='lines+markers',
            name=f"{team2}",
            marker=dict(symbol='x', size=8),
            line=dict(color='green'),
            hovertext=df_team2['Opposition'],
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
     
        # Filter the DataFrame for each team's data
        df_team1 = team_df[team_df['Team_Name'] == team1]
        df_team2 = team_df[team_df['Team_Name'] == team2]
        
        # Create the figure for comparing stat1
        fig = go.Figure()
        
        # Team 1 for stat1
        fig.add_trace(go.Scatter(
            x=df_team1['Round'],
            y=df_team1[stat1],
            mode='lines+markers',
            name=f"{team1}",
            marker=dict(symbol='circle', size=8),
            line=dict(color='#1f77b4'),
            hovertext=df_team1['Opposition'],
            hoverinfo='text'
        ))
        
        # Team 2 for stat1
        fig.add_trace(go.Scatter(
            x=df_team2['Round'],
            y=df_team2[stat1],
            mode='lines+markers',
            name=f"{team2}",
            marker=dict(symbol='x', size=8),
            line=dict(color='green'),
            hovertext=df_team2['Opposition'],
            hoverinfo='text'
        ))
        
        # Update the layout for the figure
        fig.update_layout(
            title=dict(text=f"{stat1} Comparison: {team1} vs {team2}", font=dict(color='black')),
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
        df_team1 = team_df[team_df['Team_Name'] == team1]

        
        # --- Plot 1: stat1 over Round ---
        fig1 = go.Figure()
        
        # Team 1
        fig1.add_trace(go.Scatter(
            x=df_team1['Round'],
            y=df_team1[stat1],
            mode='lines+markers',
            name=f"{team1}",
            marker=dict(symbol='circle', size=8),
            line=dict(color='#1f77b4'),
            hovertext=df_team1['Opposition'],
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
        
        # --- Plot 2: stat2 ---
        fig2 = go.Figure()
        
        # Team 1
        fig2.add_trace(go.Scatter(
            x=df_team1['Round'],
            y=df_team1[stat2],
            mode='lines+markers',
            name=f"{team1}",
            marker=dict(symbol='circle', size=8),
            line=dict(color='#1f77b4'),
            hovertext=df_team1['Opposition'],
            hoverinfo='text'
            
        ))
        
        fig2.update_layout(
            title=dict(text=f"{stat2}: {team1}", font=dict(color='black')),
            xaxis=dict(title=dict(text="Round", font=dict(color='black')), tickfont=dict(color='black')),
            yaxis=dict(title=dict(text=f"{stat1}", font=dict(color='black')), tickfont=dict(color='black')),
            legend=dict(font=dict(color='black'), y=1.15, x=0.6),
            plot_bgcolor='#99AEDE',
            paper_bgcolor='#99AEDE'
        )
        
        plotly_chart_custom(fig2)
                     
    else:
        
        # Filter the DataFrame for each team's data
        df_team1 = team_df[team_df['Team_Name'] == team1]
        
        # --- Plot 1: stat1 over Round ---
        fig1 = go.Figure()
        
        # Team 1
        fig1.add_trace(go.Scatter(
            x=df_team1['Round'],
            y=df_team1[stat1],
            mode='lines+markers',
            name=f"{team1}",
            marker=dict(symbol='circle', size=8),
            line=dict(color='#1f77b4'),
            hovertext=df_team1['Opposition'],
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


    #if st.button("Create Stat Plot/s"):
    if plot_type == 4:
        # Filter the DataFrame for each team's data
        df_team1 = team_df[team_df['Team_Name'] == team1]
        df_team2 = team_df[team_df['Team_Name'] == team2]
        
        # --- Plot 1: Team 1 ---
        x1 = df_team1[stat1]
        y1 = df_team1[stat2]
        hovertext1 = df_team1['Opposition'] + ', Rd ' + df_team1['Round'].astype(str)
        
        fig1 = go.Figure()
        
        # Scatter points
        fig1.add_trace(go.Scatter(
            x=x1,
            y=y1,
            mode='markers',
            hovertext=hovertext1,
            hoverinfo='text',
            marker=dict(size=10, color='#1f77b4'),
            name=team1
        ))
        
        # Trendline
        if not x1.empty and not y1.empty:
            m1, b1 = np.polyfit(x1, y1, 1)
            fig1.add_trace(go.Scatter(
                x=x1,
                y=m1 * x1 + b1,
                mode='lines',
                line=dict(dash='dash', color='black'),
                name='Trendline'
            ))
            
            # Calculate correlation
            corr_coef, p_value = pearsonr(x1, y1)
            abs_corr = abs(corr_coef)
        
            # Determine trendline strength
            if abs_corr < 0.3:
                correlation = "Weak"
                color = "red"
            elif abs_corr < 0.7:
                correlation = "Medium"
                color = "orange"
            else:
                correlation = "Strong"
                color = "green"
        
        
            fig1.update_layout(
                title=dict(text=f"{stat1} vs {stat2}: {team1}", font=dict(color='black')),
                annotations=[
                    dict(
                        text=f"<b>Correlation:</b> <span style='color:{color}'>{correlation}</span> (r = {corr_coef:.2f})",
                        xref='paper',
                        yref='paper',
                        x=0,
                        y=1.05,
                        showarrow=False,
                        font=dict(size=12, color=color),
                        align='left',
                        bgcolor="#f0f0f0",
                        bordercolor=color,
                        borderwidth=1
                    )
                ],
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
                font=dict(color='black'),  # Optional fallback
                hovermode='closest'
            )
        
        plotly_chart_custom(fig1)
            
        # --- Plot 2: Team 2 ---
        x2 = df_team2[stat1]
        y2 = df_team2[stat2]
        hovertext2 = df_team2['Opposition'] + ', Rd ' + df_team2['Round'].astype(str)
        
        fig2 = go.Figure()
        
        # Scatter points
        fig2.add_trace(go.Scatter(
            x=x2,
            y=y2,
            mode='markers',
            hovertext=hovertext2,
            hoverinfo='text',
            marker=dict(size=10, color='#ff7f0e'),
            name=team2
        ))
        
        # Trendline
        if not x2.empty and not y2.empty:
            m2, b2 = np.polyfit(x2, y2, 1)
            fig2.add_trace(go.Scatter(
                x=x2,
                y=m2 * x2 + b2,
                mode='lines',
                line=dict(dash='dash', color='black'),
                name='Trendline'
            ))
            
            # Calculate correlation
            corr_coef, p_value = pearsonr(x2, y2)
            abs_corr = abs(corr_coef)
        
            # Determine trendline strength
            if abs_corr < 0.3:
                correlation = "Weak"
                color = "red"
            elif abs_corr < 0.7:
                correlation = "Medium"
                color = "orange"
            else:
                correlation = "Strong"
                color = "green"
        
            fig2.update_layout(
                title=dict(text=f"{stat1} vs {stat2}: {team2}", font=dict(color='black')),
                annotations=[
                    dict(
                        text=f"<b>Correlation:</b> <span style='color:{color}'>{correlation}</span> (r = {corr_coef:.2f})",
                        xref='paper',
                        yref='paper',
                        x=0,
                        y=1.05,
                        showarrow=False,
                        font=dict(size=12, color=color),
                        align='left',
                        bgcolor="#f0f0f0",
                        bordercolor=color,
                        borderwidth=1
                    )
                ],
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
                font=dict(color='black'),  # Optional fallback
                hovermode='closest'
            )
        
        plotly_chart_custom(fig2)
         

    elif plot_type == 3:
        
        df_team1 = team_df[team_df['Team_Name'] == team1]


        if not df_team1.empty:
            x = df_team1[stat1]
            y = df_team1[stat2]
            
            # Fit linear trendline
            m, b = np.polyfit(x, y, 1)
            trend_y = m * x + b

            # Calculate correlation
            corr_coef, p_value = pearsonr(x, y)
            abs_corr = abs(corr_coef)
        
            # Determine trendline strength
            if abs_corr < 0.3:
                correlation = "Weak"
                color = "red"
            elif abs_corr < 0.7:
                correlation = "Medium"
                color = "orange"
            else:
                correlation = "Strong"
                color = "green"
        
            fig = go.Figure()
        
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',  # No 'text' here since we only want hover
                hovertext=df_team1['Opposition'] + ', Rd ' + df_team1['Round'].astype(str),  # What shows on hover
                hoverinfo='text',                  # Use only the text above
                marker=dict(size=10, color='#1f77b4'),
                name='Opposition, Round'
            ))
        
            # Trendline
            fig.add_trace(go.Scatter(
                x=x, y=trend_y,
                mode='lines',
                line=dict(dash='dash', color='black'),
                name='Trendline'
            ))
        
            fig.update_layout(
                title=dict(text=f"{stat1} vs {stat2}: {team1}", font=dict(color='black')),
                annotations=[
                    dict(
                        text=f"<b>Correlation:</b> <span style='color:{color}'>{correlation}</span> (r = {corr_coef:.2f})",
                        xref='paper',
                        yref='paper',
                        x=0,
                        y=1.05,
                        showarrow=False,
                        font=dict(size=12, color=color),
                        align='left',
                        bgcolor="#f0f0f0",
                        bordercolor=color,
                        borderwidth=1
                    )
                ],
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
                font=dict(color='black'),  # Optional fallback
                hovermode='closest'
            )
            
            plotly_chart_custom(fig)


    else: st.write("Please add a 2nd stat") 



else:
    st.header("Home")
    st.write("Please select a view.")
    
