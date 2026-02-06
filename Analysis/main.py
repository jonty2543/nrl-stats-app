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
from scipy.stats import pearsonr, gaussian_kde
import Analysis.functions as fn


st.set_page_config(page_title="NRL Stats", layout="wide")

plt.rcParams.update({
    "figure.facecolor": "#0A1020",
    "axes.facecolor": "#0F1830",
    "axes.edgecolor": "#2A3356",
    "axes.labelcolor": "#E5E7EB",
    "text.color": "#E5E7EB",
    "xtick.color": "#C7D2FE",
    "ytick.color": "#C7D2FE",
    "grid.color": "#2A3356",
    "grid.alpha": 0.35,
    "axes.titleweight": "bold",
    "font.size": 11,
})

NEON = "#A8FF00"
PINK = "#FF4D7D"

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
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        display: inline-flex;
    }
    .stRadio label {
        background: transparent;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        cursor: pointer;
        transition: all 0.15s ease;
        color: var(--muted);
    }
    .stRadio label[data-checked="true"],
    .stRadio label:has(input:checked) {
        background: linear-gradient(135deg, var(--accent), #a0e800);
        color: #000000;
        font-weight: 800;
    }
    .stRadio label:hover {
        background: var(--panel-2);
        color: var(--text);
    }
    .stRadio label[data-checked="true"]:hover,
    .stRadio label:has(input:checked):hover {
        background: linear-gradient(135deg, var(--accent), #a0e800);
        color: #000000;
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
    .stMultiSelect>div>div {
        background: var(--panel-2);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 10px;
        min-height: 48px;
    }
    .stMultiSelect [data-baseweb="tag"] {
        background: #2a3a6b;
        color: #e6eeff;
        border: 1px solid #3c4d85;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 2px 8px;
    }
    .stMultiSelect [data-baseweb="tag"] svg {
        fill: #e6eeff;
    }
    .stMultiSelect [data-baseweb="tag"] [data-baseweb="tag__action"] {
        background: rgba(255,255,255,0.08);
        border-radius: 999px;
        margin-left: 6px;
    }
    .stMultiSelect [data-baseweb="tag"] [data-baseweb="tag__action"]:hover {
        background: rgba(255,255,255,0.2);
    }
    .stDataFrame, .stTable {
        background: var(--panel);
        border-radius: 10px;
    }
    .stExpander {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        margin-bottom: 0.75rem;
    }
    .stExpander [data-testid="stExpanderToggleIcon"] {
        color: var(--accent);
    }
    .stExpander summary span {
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
    }
    .stPlotlyChart > div {
        border-radius: 10px;
        overflow: hidden;
    }
    /* Teammate toggle styling */
    div[data-testid="stToggle"] label span[data-testid="stToggleLabel"] {
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.85rem;
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

def _ensure_selectbox_state(key, options, default=None):
    if not options:
        return None
    if default is None:
        default = options[0]
    if key not in st.session_state:
        st.session_state[key] = default
    elif st.session_state[key] not in options:
        st.session_state[key] = default
    return st.session_state[key]

def _prefer_default(options, preferred):
    return preferred if preferred in options else (options[0] if options else None)

available_years = sorted(df_all['Year'].dropna().unique().tolist(), reverse=True)
position_options = sorted(df_all.get("Position", pd.Series(dtype=str)).dropna().unique().tolist())

def _render_filters_and_prepare_data():
    """Render filter widgets and return filtered data + computed lists."""
    filter_cols = st.columns([1.1, 1, 1, 1], gap="medium")
    with filter_cols[0]:
        year_choice = st.multiselect("Year", available_years, default=available_years[:1])
    with filter_cols[1]:
        position_choice = st.selectbox("Position", ["All"] + position_options)
    with filter_cols[2]:
        minutes_threshold = st.number_input(
            "Minutes Threshold",
            min_value=0.0,
            max_value=80.0,
            value=0.0,
            step=5.0,
        )
    with filter_cols[3]:
        minutes_mode = st.selectbox("Minutes Filter", ["All", "Over", "Under"])

    year_choice = year_choice or available_years[:1]
    df_year = df_all[df_all['Year'].isin(year_choice)].copy()
    df_position = df_year.copy()
    if "Position" in df_position.columns and position_choice != "All":
        df_position = df_position[df_position["Position"] == position_choice]
    df = df_position.copy()
    if "Mins Played" in df.columns and minutes_mode != "All":
        if minutes_mode == "Over":
            df = df[df["Mins Played"] >= minutes_threshold]
        else:
            df = df[df["Mins Played"] <= minutes_threshold]

    team_list = sorted(df_position['Team'].dropna().unique().tolist())
    player_list = df_position['Name'].dropna().unique().tolist()
    fantasy_rank = {}
    if "Fantasy" in df_all.columns and "Year" in df_all.columns:
        if (df_all["Year"] == 2025).any():
            rank_source = df_all[df_all["Year"] == 2025]
        else:
            rank_source = df_all
        fantasy_rank = rank_source.groupby("Name")["Fantasy"].mean().to_dict()
    player_list = sorted(
        player_list,
        key=lambda name: (-(fantasy_rank.get(name, float("-inf"))), name),
    )
    player_stat_list = [stat for stat in EV.PLAYER_STATS if stat in df_position.columns]
    team_stat_list = [stat for stat in EV.TEAM_STATS if stat in df_position.columns]

    return df, df_year, df_position, year_choice, team_list, player_list, fantasy_rank, player_stat_list, team_stat_list

def plotly_chart_custom(fig):
    fig.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        font=dict(color="#e5e7eb"),
        margin=dict(l=20, r=20, t=50, b=40),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#1f2a44",
        zeroline=False,
        title_font=dict(color="#cbd5e1"),
        tickfont=dict(color="#cbd5e1"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#1f2a44",
        zeroline=False,
        title_font=dict(color="#cbd5e1"),
        tickfont=dict(color="#cbd5e1"),
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
                x=0.02,
                y=0.98,
                showarrow=False,
                font=dict(size=14, color=corr_color),
                align='left',
                bgcolor="#0b1020",
                bordercolor=corr_color,
                borderwidth=2
            )
        ]
    fig.update_layout(
        title=dict(text=title, font=dict(color='#e5e7eb')),
        annotations=annotations,
        xaxis=dict(
            title=dict(text=stat1, font=dict(color='#cbd5e1')),
            tickfont=dict(color='#cbd5e1'),
            gridcolor="#1f2a44"
        ),
        yaxis=dict(
            title=dict(text=stat2, font=dict(color='#cbd5e1')),
            tickfont=dict(color='#cbd5e1'),
            gridcolor="#1f2a44"
        ),
        legend=dict(font=dict(color='#e5e7eb'), y=1.15, x=0.6),
        plot_bgcolor='#0b1020',
        paper_bgcolor='#0b1020',
        font=dict(color='#e5e7eb'),
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
        line=dict(dash='dash', color='#ff4d87', width=3),
        name='Trendline'
    ))

    corr_coef, _ = pearsonr(x, y)
    corr_text = f"r = {corr_coef:+.2f}  |  games = {len(x)}"
    _set_corr_layout(fig, stat1, stat2, title, corr_text=corr_text, corr_color="#c6ff00")


def _corr_matplotlib_plot(frame, stat_x, stat_y, title):
    cols = [stat_x, stat_y]
    if "Round" in frame.columns:
        cols.append("Round")
    if "Year" in frame.columns:
        cols.append("Year")
    data = frame[cols].dropna()
    if data.empty or len(data) < 2:
        st.info("Correlation unavailable: need at least two data points.")
        return
    x = data[stat_x].to_numpy()
    y = data[stat_y].to_numpy()
    if np.all(x == 0) or np.all(y == 0) or np.unique(x).size < 2 or np.unique(y).size < 2:
        st.info("Correlation unavailable: one of the stats has no variation.")
        return

    if "Year" in data.columns and "Round" in data.columns:
        data = data.sort_values(["Year", "Round"])
    elif "Year" in data.columns:
        data = data.sort_values(["Year"])
    x = data[stat_x].to_numpy()
    y = data[stat_y].to_numpy()
    rn = np.arange(len(data))
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(
        data[stat_x].to_numpy(),
        data[stat_y].to_numpy(),
        c=rn,
        cmap="viridis",
        s=70,
        alpha=0.9,
        edgecolor="#111827",
        linewidth=0.6,
        vmin=0,
        vmax=max(len(data) - 1, 1),
    )
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, color=PINK, linestyle="--", linewidth=2.2)
    ax.set_title(title)
    ax.set_xlabel(stat_x)
    ax.set_ylabel(stat_y)
    ax.grid(True)

    r, _ = pearsonr(x, y)
    label = f"r = {r:+.2f}  |  games = {len(data)}"
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#0A1020",
            edgecolor=NEON,
            alpha=0.9,
        ),
        color=NEON,
        fontsize=11,
        fontweight="bold",
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, len(data) - 1])
    cbar.set_ticklabels(["First", "Last"])
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _round_label_order(*frames):
    labels = []
    seen = set()
    for frame in frames:
        if frame is None or frame.empty or "Round_Display" not in frame.columns:
            continue
        for label in frame["Round_Display"].astype(str).tolist():
            if label.isdigit():
                if int(label) > 27:
                    continue
            else:
                continue
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return sorted(labels, key=lambda v: int(v))


def _round_values(frame, stat, labels):
    if frame is None or frame.empty:
        return [np.nan for _ in labels]
    grouped = frame.groupby("Round_Display", dropna=False)[stat].mean()
    return [grouped.get(label, np.nan) for label in labels]


def _round_xy(labels, values):
    x_all = np.arange(len(labels))
    y_all = np.array(values, dtype=float)
    mask = ~np.isnan(y_all)
    return x_all, y_all, mask


def _round_matplotlib_with_without(df_all_player, stat, player_name, teammate_name, lookup_df):
    df_with = _filter_by_teammate(df_all_player, teammate_name, True, lookup_df)
    df_without = _filter_by_teammate(df_all_player, teammate_name, False, lookup_df)
    df_with = _prepare_round_plot_df(df_with)
    df_without = _prepare_round_plot_df(df_without)

    labels = _round_label_order(df_with, df_without)
    if not labels:
        return
    y_w = _round_values(df_with, stat, labels)
    y_wo = _round_values(df_without, stat, labels)
    x, y_w_all, m_w = _round_xy(labels, y_w)
    _, y_wo_all, m_wo = _round_xy(labels, y_wo)

    avg_w = np.nanmean(y_w_all) if np.any(m_w) else 0
    avg_wo = np.nanmean(y_wo_all) if np.any(m_wo) else 0

    games_w = int(np.sum(m_w))
    games_wo = int(np.sum(m_wo))

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x[m_w], y_w_all[m_w], color=NEON, linewidth=2.4, marker="o", markersize=6,
            label=f"With {teammate_name} ({games_w}g)")
    ax.plot(x[m_wo], y_wo_all[m_wo], color="#FFB347", linewidth=2.2, marker="s", markersize=5,
            label=f"Without {teammate_name} ({games_wo}g)")
    ax.axhline(y=avg_w, color=NEON, linestyle="--", linewidth=1.4, alpha=0.7,
               label=f"Avg with: {avg_w:.1f}")
    ax.axhline(y=avg_wo, color="#FFB347", linestyle="--", linewidth=1.4, alpha=0.7,
               label=f"Avg without: {avg_wo:.1f}")
    ax.set_title(f"{player_name} — {stat}: With vs Without {teammate_name}")
    ax.set_xlabel("Round")
    ax.set_ylabel(stat)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(True)
    ax.legend(loc="upper left")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _kde_curve(ax, vals, color, label, x_range):
    if len(vals) < 2 or vals.std() == 0:
        return
    kde = gaussian_kde(vals, bw_method=0.3)
    x = np.linspace(x_range[0], x_range[1], 300)
    y = kde(x)
    ax.plot(x, y, color=color, linewidth=2.4, label=label)
    ax.fill_between(x, y, color=color, alpha=0.15)


def _distribution_plot(frame1, frame2, stat, label1, label2=None):
    vals1 = frame1[stat].dropna() if frame1 is not None and not frame1.empty else pd.Series(dtype=float)
    vals2 = frame2[stat].dropna() if frame2 is not None and not frame2.empty else pd.Series(dtype=float)
    if vals1.empty and vals2.empty:
        return

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)

    all_vals = pd.concat([vals1, vals2])
    x_range = (all_vals.min() - all_vals.std() * 0.5, all_vals.max() + all_vals.std() * 0.5)

    _kde_curve(ax, vals1, NEON, f"{label1} (n={len(vals1)})", x_range)
    if label2 and not vals2.empty:
        _kde_curve(ax, vals2, "#41d6f4", f"{label2} (n={len(vals2)})", x_range)

    if not vals1.empty:
        ax.axvline(vals1.mean(), color=NEON, linestyle="--", linewidth=1.6, alpha=0.85,
                   label=f"{label1} avg: {vals1.mean():.1f}")
    if label2 and not vals2.empty:
        ax.axvline(vals2.mean(), color="#41d6f4", linestyle="--", linewidth=1.6, alpha=0.85,
                   label=f"{label2} avg: {vals2.mean():.1f}")

    ax.set_title(f"{stat} Distribution")
    ax.set_xlabel(stat)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _distribution_with_without(df_all_player, stat, player_name, teammate_name, lookup_df):
    df_with = _filter_by_teammate(df_all_player, teammate_name, True, lookup_df)
    df_without = _filter_by_teammate(df_all_player, teammate_name, False, lookup_df)
    vals_w = df_with[stat].dropna() if not df_with.empty else pd.Series(dtype=float)
    vals_wo = df_without[stat].dropna() if not df_without.empty else pd.Series(dtype=float)
    if vals_w.empty and vals_wo.empty:
        return

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)

    all_vals = pd.concat([vals_w, vals_wo])
    x_range = (all_vals.min() - all_vals.std() * 0.5, all_vals.max() + all_vals.std() * 0.5)

    _kde_curve(ax, vals_w, NEON, f"With {teammate_name} (n={len(vals_w)})", x_range)
    _kde_curve(ax, vals_wo, "#FFB347", f"Without {teammate_name} (n={len(vals_wo)})", x_range)

    if not vals_w.empty:
        ax.axvline(vals_w.mean(), color=NEON, linestyle="--", linewidth=1.6, alpha=0.85,
                   label=f"Avg with: {vals_w.mean():.1f}")
    if not vals_wo.empty:
        ax.axvline(vals_wo.mean(), color="#FFB347", linestyle="--", linewidth=1.6, alpha=0.85,
                   label=f"Avg without: {vals_wo.mean():.1f}")

    ax.set_title(f"{player_name} — {stat}: With vs Without {teammate_name}")
    ax.set_xlabel(stat)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _round_matplotlib_compare(frame1, frame2, stat, title, label1, label2):
    labels = _round_label_order(frame1, frame2)
    if not labels:
        return
    y1 = _round_values(frame1, stat, labels)
    y2 = _round_values(frame2, stat, labels)
    x, y1_all, m1 = _round_xy(labels, y1)
    _, y2_all, m2 = _round_xy(labels, y2)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x[m1], y1_all[m1], color=NEON, linewidth=2.4, marker="o", markersize=6, label=label1)
    ax.plot(x[m2], y2_all[m2], color="#41d6f4", linewidth=2.2, marker="s", markersize=5, label=label2)
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(stat)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(True)
    ax.legend(loc="upper left")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _round_matplotlib_single(frame, stat, title):
    labels = _round_label_order(frame)
    if not labels:
        return
    y = _round_values(frame, stat, labels)
    x, y_all, mask = _round_xy(labels, y)
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x[mask], y_all[mask], color=NEON, linewidth=2.4, marker="o", markersize=6)
    ax.fill_between(x[mask], y_all[mask], color=NEON, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(stat)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _round_matplotlib_dual(frame, stat1, stat2, title):
    labels = _round_label_order(frame)
    if not labels:
        return
    y1 = _round_values(frame, stat1, labels)
    y2 = _round_values(frame, stat2, labels)
    x, y1_all, m1 = _round_xy(labels, y1)
    _, y2_all, m2 = _round_xy(labels, y2)

    fig = plt.figure(figsize=(9, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x[m1], y1_all[m1], color=NEON, linewidth=2.6, marker="o", markersize=6)
    ax1.fill_between(x[m1], y1_all[m1], color=NEON, alpha=0.15)
    ax1.set_title(title)
    ax1.set_xlabel("Round")
    ax1.set_ylabel(stat1, color=NEON)
    ax1.tick_params(axis="y", colors=NEON)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(x[m2], y2_all[m2], color="#41d6f4", linewidth=2.4, marker="s", markersize=5)
    ax2.set_ylabel(stat2, color="#41d6f4")
    ax2.tick_params(axis="y", colors="#41d6f4")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _apply_round_layout(fig, title, y_title, y2_title=None, show_legend=True):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e5e7eb")),
        plot_bgcolor="#0b1020",
        paper_bgcolor="#0b1020",
        font=dict(color="#e5e7eb"),
        legend=dict(font=dict(color="#e5e7eb"), y=1.1, x=0.5, orientation="h"),
        showlegend=show_legend,
        hovermode="x unified",
    )
    fig.update_xaxes(
        title=dict(text="Round", font=dict(color="#cbd5e1")),
        tickfont=dict(color="#cbd5e1"),
        gridcolor="#1f2a44",
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(color="#cbd5e1")),
        tickfont=dict(color="#cbd5e1"),
        gridcolor="#1f2a44",
        zeroline=False,
    )
    if y2_title:
        fig.update_layout(
            yaxis2=dict(
                title=dict(text=y2_title, font=dict(color="#41d6f4")),
                tickfont=dict(color="#41d6f4"),
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False,
            )
        )


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


def _filter_by_teammate(player_df, teammate_name, with_teammate, lookup_df):
    if teammate_name == "None":
        return player_df
    teammate_games = lookup_df[lookup_df['Name'] == teammate_name][
        ['Team', 'Round', 'Year']
    ].drop_duplicates()
    if with_teammate:
        return player_df.merge(teammate_games, on=['Team', 'Round', 'Year'], how='inner')
    else:
        merged = player_df.merge(
            teammate_games, on=['Team', 'Round', 'Year'], how='left', indicator=True
        )
        return merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])


def _get_teammate_options(player_name, source_df, fantasy_rank):
    teams = source_df[source_df['Name'] == player_name]['Team'].unique()
    teammates = source_df[
        (source_df['Team'].isin(teams)) & (source_df['Name'] != player_name)
    ]['Name'].unique().tolist()
    return sorted(
        teammates,
        key=lambda name: (-(fantasy_rank.get(name, float("-inf"))), name),
    )


def _render_summary_table(data):
    if not data:
        return
    df_table = pd.DataFrame(data)
    # Drop entity column when all rows have the same value (single player/team)
    for col in ("Player", "Team"):
        if col in df_table.columns and df_table[col].nunique() <= 1:
            df_table = df_table.drop(columns=[col])
    header = "".join(f'<th style="padding:3px 6px;font-size:0.75rem;text-align:left;border-bottom:1px solid #2a3356;color:#9aa4bf;">{c}</th>' for c in df_table.columns)
    rows = ""
    for _, row in df_table.iterrows():
        cells = "".join(f'<td style="padding:3px 6px;font-size:0.75rem;color:#e5e7eb;">{v}</td>' for v in row)
        rows += f"<tr>{cells}</tr>"
    html = f'<table style="width:100%;border-collapse:collapse;"><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)


def _section_divider():
    st.markdown('<hr style="margin:6px 0;border:none;border-top:1px solid #2a3356;opacity:0.5;">', unsafe_allow_html=True)


def _section_header(title):
    st.markdown(
        f'<div style="font-size:0.7rem;font-weight:700;color:#9aa4bf;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px;">{title}</div>',
        unsafe_allow_html=True,
    )


def _render_profile(name, data_df, entity="player"):
    if data_df.empty:
        return
    games = len(data_df)
    if entity == "player":
        team = data_df['Team'].mode().iloc[0] if 'Team' in data_df.columns and not data_df['Team'].mode().empty else "\u2014"
        position = data_df['Position'].mode().iloc[0] if 'Position' in data_df.columns and not data_df['Position'].mode().empty else "\u2014"
        avg_mins = data_df['Mins Played'].mean() if 'Mins Played' in data_df.columns else 0
        detail = f'{team} &bull; {position} &bull; {games} games &bull; {avg_mins:.0f} avg mins'
    else:
        avg_pts = data_df['Points'].mean() if 'Points' in data_df.columns else 0
        detail = f'{games} matches &bull; {avg_pts:.1f} avg pts'
    st.markdown(
        f'<div style="margin-bottom:4px;">'
        f'<div style="font-size:0.85rem;font-weight:700;color:#A8FF00;text-transform:uppercase;letter-spacing:0.03em;">{name}</div>'
        f'<div style="font-size:0.72rem;color:#9aa4bf;margin-top:1px;">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def _render_key_metrics(items_data, stat_cols=None, all_df=None, group_col='Name'):
    if stat_cols is None:
        stat_cols = [
            ("Fantasy", "Fantasy Avg"),
            ("Tackles Made", "Tackles/Game"),
            ("All Run Metres", "Run Metres/Game"),
            ("Post Contact Metres", "PCM/Game"),
            ("Errors", "Errors/Game"),
            ("Offloads", "Offloads/Game"),
            ("Try Assists", "Try Assists/Game"),
            ("Line Breaks", "Line Breaks/Game"),
            ("Tackle Breaks", "Tackle Breaks/Game"),
        ]
    show_pct = len(items_data) == 1 and all_df is not None and not all_df.empty
    _section_header("Key Metrics")
    th = 'padding:2px 6px;font-size:0.7rem;border-bottom:1px solid #2a3356;color:#9aa4bf;'
    header = f'<th style="{th}text-align:left;">Metric</th>'
    if len(items_data) == 1:
        header += f'<th style="{th}text-align:right;">Value</th>'
        if show_pct:
            header += f'<th style="{th}text-align:right;">Pctl</th>'
    else:
        for name, _ in items_data:
            short = name.split()[-1] if len(name) > 14 else name
            header += f'<th style="{th}text-align:right;">{short}</th>'
    all_avgs = all_df.groupby(group_col).mean(numeric_only=True) if show_pct else None
    rows = ""
    for col, label in stat_cols:
        cells = f'<td style="padding:2px 6px;font-size:0.72rem;color:#9aa4bf;">{label}</td>'
        for _, pdf in items_data:
            if col in pdf.columns and not pdf[col].empty:
                val = pdf[col].mean()
                cells += f'<td style="padding:2px 6px;font-size:0.72rem;color:#e5e7eb;text-align:right;">{val:.1f}</td>'
                if show_pct and col in all_avgs.columns:
                    stat_vals = all_avgs[col].dropna()
                    pct = (stat_vals < val).mean() * 100 if not stat_vals.empty else 0
                    pct_color = "#A8FF00" if pct >= 75 else "#FFD700" if pct >= 50 else "#FF8C00" if pct >= 25 else "#FF4D7D"
                    cells += f'<td style="padding:2px 6px;font-size:0.72rem;color:{pct_color};font-weight:700;text-align:right;">{pct:.0f}th</td>'
                elif show_pct:
                    cells += '<td style="padding:2px 6px;font-size:0.72rem;color:#4a5070;text-align:right;">\u2014</td>'
            else:
                cells += '<td style="padding:2px 6px;font-size:0.72rem;color:#4a5070;text-align:right;">\u2014</td>'
                if show_pct:
                    cells += '<td style="padding:2px 6px;font-size:0.72rem;color:#4a5070;text-align:right;">\u2014</td>'
        rows += f"<tr>{cells}</tr>"
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;"><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )


def _render_percentile_ranks(stat_names, items_data, all_df, group_col='Name'):
    if all_df.empty:
        return
    _section_header("Percentile Rank")
    single = len(items_data) == 1
    all_avgs = all_df.groupby(group_col).mean(numeric_only=True)
    for stat_name in stat_names:
        if stat_name not in all_avgs.columns:
            continue
        stat_vals = all_avgs[stat_name].dropna()
        if stat_vals.empty:
            continue
        for name, pdf in items_data:
            if pdf.empty or stat_name not in pdf.columns:
                continue
            player_avg = pdf[stat_name].mean()
            pct = (stat_vals < player_avg).mean() * 100
            color = "#A8FF00" if pct >= 75 else "#FFD700" if pct >= 50 else "#FF8C00" if pct >= 25 else "#FF4D7D"
            label = stat_name if single else f'{name.split()[-1] if len(name) > 14 else name} \u2014 {stat_name}'
            st.markdown(
                f'<div style="margin-bottom:5px;">'
                f'<div style="display:flex;justify-content:space-between;font-size:0.68rem;color:#9aa4bf;margin-bottom:1px;">'
                f'<span>{label}</span>'
                f'<span style="color:{color};font-weight:700;">{pct:.0f}th</span></div>'
                f'<div style="background:#1e2542;border-radius:3px;height:6px;overflow:hidden;">'
                f'<div style="width:{pct}%;height:100%;background:{color};border-radius:3px;"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )


def _render_recent_form(stat_names, items_data):
    _section_header("Recent Form (Last 5)")
    single = len(items_data) == 1
    for stat_name in stat_names:
        for name, pdf in items_data:
            if pdf.empty or stat_name not in pdf.columns or len(pdf) < 2:
                continue
            sorted_df = pdf.copy()
            sort_cols = []
            if 'Year' in sorted_df.columns:
                sort_cols.append('Year')
            if 'Round' in sorted_df.columns:
                sorted_df['_rnd'] = pd.to_numeric(sorted_df['Round'], errors='coerce')
                sort_cols.append('_rnd')
            if sort_cols:
                sorted_df = sorted_df.sort_values(sort_cols, ascending=False)
            last_5_avg = sorted_df.head(5)[stat_name].mean()
            overall_avg = pdf[stat_name].mean()
            pct_chg = ((last_5_avg - overall_avg) / abs(overall_avg)) * 100 if overall_avg != 0 else 0
            arrow = "\u25b2" if pct_chg > 0 else "\u25bc" if pct_chg < 0 else "\u2014"
            color = "#A8FF00" if pct_chg > 0 else "#FF4D7D" if pct_chg < 0 else "#9aa4bf"
            label = stat_name if single else f'{name.split()[-1] if len(name) > 14 else name} \u2014 {stat_name}'
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:0.72rem;margin-bottom:3px;">'
                f'<span style="color:#9aa4bf;">{label}</span>'
                f'<span><span style="color:#e5e7eb;">{last_5_avg:.1f}</span>'
                f'<span style="color:{color};font-size:0.65rem;margin-left:4px;">{arrow} {abs(pct_chg):.0f}%</span></span></div>',
                unsafe_allow_html=True,
            )


page = st.radio(
    "Views",
    ["Player Comparison", "Teams Comparison"],
    horizontal=True,
    label_visibility="collapsed",
)

if page == "Player Comparison":
    with st.container(border=True):
        df, df_year, df_position, year_choice, team_list, player_list, fantasy_rank, player_stat_list, team_stat_list = _render_filters_and_prepare_data()
        if df.empty:
            st.warning("No data available for the selected filters.")
            st.stop()

        player1_options = player_list
        player1_value = _ensure_selectbox_state(
            "player1",
            player1_options,
            default=_prefer_default(player1_options, "Terrell May"),
        )
        player1_index = player1_options.index(player1_value) if player1_value in player1_options else 0

        p1_col, p1_tm_col = st.columns(2)
        with p1_col:
            player1 = st.selectbox("Select Player 1", player1_options, index=player1_index, key="player1")
        with p1_tm_col:
            p1_teammate_list = _get_teammate_options(player1, df_year, fantasy_rank)
            p1_teammate_full = ["None"] + p1_teammate_list
            p1_tm_value = _ensure_selectbox_state("p1_teammate", p1_teammate_full, default="None")
            p1_tm_index = p1_teammate_full.index(p1_tm_value) if p1_tm_value in p1_teammate_full else 0
            _tm1_sel, _tm1_tog = st.columns([3, 1])
            with _tm1_sel:
                p1_teammate = st.selectbox("Player 1 Teammate", p1_teammate_full, index=p1_tm_index, key="p1_teammate")
            with _tm1_tog:
                if p1_teammate != "None":
                    st.markdown('<div style="height:23px;"></div>', unsafe_allow_html=True)
                    p1_teammate_with = st.toggle("With", value=True, key="p1_teammate_with")
                else:
                    p1_teammate_with = True

        player2_options = player_list
        player2_full_options = ["None"] + player2_options
        player2_value = _ensure_selectbox_state(
            "player2",
            player2_full_options,
            default=_prefer_default(player2_full_options, "Payne Haas"),
        )
        player2_index = player2_full_options.index(player2_value) if player2_value in player2_full_options else 0

        p2_col, p2_tm_col = st.columns(2)
        with p2_col:
            player2 = st.selectbox("Select Player 2 (Optional)", player2_full_options, index=player2_index, key="player2")
        with p2_tm_col:
            if player2 != "None":
                p2_teammate_list = _get_teammate_options(player2, df_year, fantasy_rank)
                p2_teammate_full = ["None"] + p2_teammate_list
                p2_tm_value = _ensure_selectbox_state("p2_teammate", p2_teammate_full, default="None")
                p2_tm_index = p2_teammate_full.index(p2_tm_value) if p2_tm_value in p2_teammate_full else 0
                _tm2_sel, _tm2_tog = st.columns([3, 1])
                with _tm2_sel:
                    p2_teammate = st.selectbox("Player 2 Teammate", p2_teammate_full, index=p2_tm_index, key="p2_teammate")
                with _tm2_tog:
                    if p2_teammate != "None":
                        st.markdown('<div style="height:23px;"></div>', unsafe_allow_html=True)
                        p2_teammate_with = st.toggle("With", value=True, key="p2_teammate_with")
                    else:
                        p2_teammate_with = True
            else:
                p2_teammate = "None"
                p2_teammate_with = True

        _stat_l, _stat_r = st.columns(2)
        with _stat_l:
            stat1_value = _ensure_selectbox_state(
                "player_stat1",
                player_stat_list,
                default=_prefer_default(player_stat_list, "Tackles Made"),
            )
            stat1_index = player_stat_list.index(stat1_value) if stat1_value in player_stat_list else 0
            stat1 = st.selectbox("Select Stat 1", player_stat_list, index=stat1_index, key="player_stat1")
        with _stat_r:
            stat2_options = ["None"] + player_stat_list
            stat2_value = _ensure_selectbox_state(
                "player_stat2",
                stat2_options,
                default=_prefer_default(stat2_options, "All Runs"),
            )
            stat2_index = stat2_options.index(stat2_value) if stat2_value in stat2_options else 0
            stat2 = st.selectbox("Select Stat 2 (Optional)", stat2_options, index=stat2_index, key="player_stat2")

    # Apply teammate filters
    df_p1_filtered = _filter_by_teammate(df[df['Name'] == player1], p1_teammate, p1_teammate_with, df_year)
    if player2 != "None":
        df_p2_filtered = _filter_by_teammate(df[df['Name'] == player2], p2_teammate, p2_teammate_with, df_year)
    else:
        df_p2_filtered = pd.DataFrame()

    summary_data = []
    if player1 and stat1:
        s = df_p1_filtered[stat1]
        summary_data.append({
            "Player": player1, "Stat": stat1,
            "Average": f"{s.mean():.2f}", "Median": f"{s.median():.2f}",
            "Min": f"{s.min():.2f}", "Max": f"{s.max():.2f}",
        })
    if player2 != "None" and stat1:
        s = df_p2_filtered[stat1]
        summary_data.append({
            "Player": player2, "Stat": stat1,
            "Average": f"{s.mean():.2f}", "Median": f"{s.median():.2f}",
            "Min": f"{s.min():.2f}", "Max": f"{s.max():.2f}",
        })
    if player1 and stat2 != "None":
        s = df_p1_filtered[stat2]
        summary_data.append({
            "Player": player1, "Stat": stat2,
            "Average": f"{s.mean():.2f}", "Median": f"{s.median():.2f}",
            "Min": f"{s.min():.2f}", "Max": f"{s.max():.2f}",
        })
    if player2 != "None" and stat2 != "None":
        s = df_p2_filtered[stat2]
        summary_data.append({
            "Player": player2, "Stat": stat2,
            "Average": f"{s.mean():.2f}", "Median": f"{s.median():.2f}",
            "Min": f"{s.min():.2f}", "Max": f"{s.max():.2f}",
        })

    _p_items = [(player1, df_p1_filtered)]
    if player2 != "None":
        _p_items.append((player2, df_p2_filtered))
    _p_stats = [stat1] + ([stat2] if stat2 != "None" else [])

    with st.container(border=True):
        _sum_l, _sum_m, _sum_r = st.columns(3)
        with _sum_l:
            _render_profile(player1, df_p1_filtered, entity="player")
            if player2 != "None":
                _section_divider()
                _render_profile(player2, df_p2_filtered, entity="player")
            _section_divider()
            if summary_data:
                _render_summary_table(summary_data)
        with _sum_m:
            _render_percentile_ranks(_p_stats, _p_items, df)
            _section_divider()
            _render_recent_form(_p_stats, _p_items)
        with _sum_r:
            _render_key_metrics(_p_items, all_df=df, group_col='Name')

    if player2 != "None" and stat2 != "None":
        plot_type = 4
    elif player2 != "None":
        plot_type = 2
    elif stat2 != "None":
        plot_type = 3
    else:
        plot_type = 1

    # -- Panel rendering function --
    def _render_player_panel(pid, wide=False):
        if pid == "corr":
            with st.expander(f"{stat1} vs {stat2}", expanded=True):
                if plot_type == 4:
                    _df1 = _prepare_round_plot_df(df_p1_filtered)
                    _df2 = _prepare_round_plot_df(df_p2_filtered)
                    if wide:
                        _wl, _wr = st.columns(2)
                        with _wl:
                            _corr_matplotlib_plot(_df1, stat1, stat2, f"{player1} — {stat1} vs {stat2}")
                        with _wr:
                            _corr_matplotlib_plot(_df2, stat1, stat2, f"{player2} — {stat1} vs {stat2}")
                    else:
                        _corr_matplotlib_plot(_df1, stat1, stat2, f"{player1} — {stat1} vs {stat2}")
                        _corr_matplotlib_plot(_df2, stat1, stat2, f"{player2} — {stat1} vs {stat2}")
                elif plot_type == 3:
                    _df1 = _prepare_round_plot_df(df_p1_filtered)
                    if not _df1.empty:
                        _corr_matplotlib_plot(_df1, stat1, stat2, f"{player1} — {stat1} vs {stat2}")

        elif pid == "round":
            with st.expander("Stat Comparison by Round", expanded=True):
                _ry_opts = [str(y) for y in year_choice]
                _ry = st.radio("Year", _ry_opts, horizontal=True, key="round_year") if len(_ry_opts) > 1 else _ry_opts[0]
                _rp1 = df_p1_filtered[df_p1_filtered["Year"] == _ry]
                _rp2 = df_p2_filtered[df_p2_filtered["Year"] == _ry] if not df_p2_filtered.empty else df_p2_filtered
                if plot_type == 4:
                    _df1 = _prepare_round_plot_df(_rp1)
                    _df2 = _prepare_round_plot_df(_rp2)
                    if wide:
                        _wl, _wr = st.columns(2)
                        with _wl:
                            _round_matplotlib_compare(_df1, _df2, stat1, f"{stat1} Comparison: {player1} vs {player2}", player1, player2)
                        with _wr:
                            _round_matplotlib_compare(_df1, _df2, stat2, f"{stat2} Comparison: {player1} vs {player2}", player1, player2)
                    else:
                        _round_matplotlib_compare(_df1, _df2, stat1, f"{stat1} Comparison: {player1} vs {player2}", player1, player2)
                        _round_matplotlib_compare(_df1, _df2, stat2, f"{stat2} Comparison: {player1} vs {player2}", player1, player2)
                elif plot_type == 2:
                    _df1 = _prepare_round_plot_df(_rp1)
                    _df2 = _prepare_round_plot_df(_rp2)
                    _round_matplotlib_compare(_df1, _df2, stat1, f"{stat1} Comparison: {player1} vs {player2}", player1, player2)
                elif plot_type == 3:
                    _df1 = _prepare_round_plot_df(_rp1)
                    _round_matplotlib_dual(_df1, stat1, stat2, f"{player1} — {stat1} + {stat2} by round")
                else:
                    _df1 = _prepare_round_plot_df(_rp1)
                    _round_matplotlib_single(_df1, stat1, f"{stat1}: {player1}")

        elif pid == "dist":
            with st.expander("Distribution", expanded=True):
                _p2d = df_p2_filtered if player2 != "None" else pd.DataFrame()
                _l2 = player2 if player2 != "None" else None
                if wide and stat2 != "None":
                    _wl, _wr = st.columns(2)
                    with _wl:
                        _distribution_plot(df_p1_filtered, _p2d, stat1, player1, _l2)
                    with _wr:
                        _distribution_plot(df_p1_filtered, _p2d, stat2, player1, _l2)
                else:
                    _distribution_plot(df_p1_filtered, _p2d, stat1, player1, _l2)
                    if stat2 != "None":
                        _distribution_plot(df_p1_filtered, _p2d, stat2, player1, _l2)

        elif pid == "p1ww":
            _base = df[df['Name'] == player1]
            with st.expander(f"{player1}: With vs Without {p1_teammate}", expanded=True):
                _ww_opts = [str(y) for y in year_choice]
                _wwy = st.radio("Year", _ww_opts, horizontal=True, key="ww_year_p1") if len(_ww_opts) > 1 else _ww_opts[0]
                _ww_df = _base[_base["Year"] == _wwy]
                _ww_lk = df_year[df_year["Year"] == _wwy]
                if wide and stat2 != "None":
                    _wl, _wr = st.columns(2)
                    with _wl:
                        _round_matplotlib_with_without(_ww_df, stat1, player1, p1_teammate, _ww_lk)
                    with _wr:
                        _round_matplotlib_with_without(_ww_df, stat2, player1, p1_teammate, _ww_lk)
                else:
                    _round_matplotlib_with_without(_ww_df, stat1, player1, p1_teammate, _ww_lk)
                    if stat2 != "None":
                        _round_matplotlib_with_without(_ww_df, stat2, player1, p1_teammate, _ww_lk)

        elif pid == "p1ww_dist":
            _base = df[df['Name'] == player1]
            with st.expander(f"{player1}: With vs Without {p1_teammate} — Distribution", expanded=True):
                if wide and stat2 != "None":
                    _wl, _wr = st.columns(2)
                    with _wl:
                        _distribution_with_without(_base, stat1, player1, p1_teammate, df_year)
                    with _wr:
                        _distribution_with_without(_base, stat2, player1, p1_teammate, df_year)
                else:
                    _distribution_with_without(_base, stat1, player1, p1_teammate, df_year)
                    if stat2 != "None":
                        _distribution_with_without(_base, stat2, player1, p1_teammate, df_year)

        elif pid == "p2ww":
            _base = df[df['Name'] == player2]
            with st.expander(f"{player2}: With vs Without {p2_teammate}", expanded=True):
                _ww_opts = [str(y) for y in year_choice]
                _wwy = st.radio("Year", _ww_opts, horizontal=True, key="ww_year_p2") if len(_ww_opts) > 1 else _ww_opts[0]
                _ww_df = _base[_base["Year"] == _wwy]
                _ww_lk = df_year[df_year["Year"] == _wwy]
                if wide and stat2 != "None":
                    _wl, _wr = st.columns(2)
                    with _wl:
                        _round_matplotlib_with_without(_ww_df, stat1, player2, p2_teammate, _ww_lk)
                    with _wr:
                        _round_matplotlib_with_without(_ww_df, stat2, player2, p2_teammate, _ww_lk)
                else:
                    _round_matplotlib_with_without(_ww_df, stat1, player2, p2_teammate, _ww_lk)
                    if stat2 != "None":
                        _round_matplotlib_with_without(_ww_df, stat2, player2, p2_teammate, _ww_lk)

        elif pid == "p2ww_dist":
            _base = df[df['Name'] == player2]
            with st.expander(f"{player2}: With vs Without {p2_teammate} — Distribution", expanded=True):
                if wide and stat2 != "None":
                    _wl, _wr = st.columns(2)
                    with _wl:
                        _distribution_with_without(_base, stat1, player2, p2_teammate, df_year)
                    with _wr:
                        _distribution_with_without(_base, stat2, player2, p2_teammate, df_year)
                else:
                    _distribution_with_without(_base, stat1, player2, p2_teammate, df_year)
                    if stat2 != "None":
                        _distribution_with_without(_base, stat2, player2, p2_teammate, df_year)

    # -- Paired grid panels (round rendered separately when stat2 selected) --
    _grid_panels = []
    if stat2 != "None":
        _grid_panels.append("corr")
    _grid_panels.append("dist")
    if stat2 == "None":
        _grid_panels.append("round")
    if p1_teammate != "None":
        _grid_panels.append("p1ww")
        _grid_panels.append("p1ww_dist")
    if player2 != "None" and p2_teammate != "None":
        _grid_panels.append("p2ww")
        _grid_panels.append("p2ww_dist")

    for _i in range(0, len(_grid_panels) - (len(_grid_panels) % 2), 2):
        _pl, _pr = st.columns(2, gap="small")
        with _pl:
            _render_player_panel(_grid_panels[_i])
        with _pr:
            _render_player_panel(_grid_panels[_i + 1])
    if len(_grid_panels) % 2 == 1:
        _render_player_panel(_grid_panels[-1], wide=True)

    # Round panel: full-width with side-by-side charts when 2 stats selected
    if stat2 != "None":
        _render_player_panel("round", wide=True)

elif page == "Teams Comparison":
    with st.container(border=True):
        df, df_year, df_position, year_choice, team_list, player_list, fantasy_rank, player_stat_list, team_stat_list = _render_filters_and_prepare_data()
        if df.empty:
            st.warning("No data available for the selected filters.")
            st.stop()
        if not team_stat_list:
            st.error("No team stats available for the selected year.")
            st.stop()

        _t_sel_l, _t_sel_r = st.columns(2)
        with _t_sel_l:
            team1_value = _ensure_selectbox_state("team1", team_list)
            team1_index = team_list.index(team1_value) if team1_value in team_list else 0
            team1 = st.selectbox("Select Team 1", team_list, index=team1_index, key="team1")
        with _t_sel_r:
            team2_options = ["None"] + team_list
            team2_value = _ensure_selectbox_state("team2", team2_options, default="None")
            team2_index = team2_options.index(team2_value) if team2_value in team2_options else 0
            team2 = st.selectbox("Select Team 2 (Optional)", team2_options, index=team2_index, key="team2")
        _ts_sel_l, _ts_sel_r = st.columns(2)
        with _ts_sel_l:
            team_stat1_value = _ensure_selectbox_state("team_stat1", team_stat_list)
            team_stat1_index = team_stat_list.index(team_stat1_value) if team_stat1_value in team_stat_list else 0
            stat1 = st.selectbox("Select Stat 1", team_stat_list, index=team_stat1_index, key="team_stat1")
        with _ts_sel_r:
            team_stat2_options = ["None"] + team_stat_list
            team_stat2_value = _ensure_selectbox_state("team_stat2", team_stat2_options, default="None")
            team_stat2_index = team_stat2_options.index(team_stat2_value) if team_stat2_value in team_stat2_options else 0
            stat2 = st.selectbox("Select Stat 2 (Optional)", team_stat2_options, index=team_stat2_index, key="team_stat2")

    group_cols = ['Team', 'Round', 'Year', 'Opponent']
    if 'Round_Label' in df.columns:
        group_cols.append('Round_Label')
    team_df = df.groupby(group_cols, as_index=False)[team_stat_list].sum()

    df_t1 = team_df[team_df['Team'] == team1]
    df_t2 = team_df[team_df['Team'] == team2] if team2 != "None" else pd.DataFrame()

    summary_data = []
    if team1 and stat1:
        s = df_t1[stat1]
        summary_data.append({
            "Team": team1, "Stat": stat1,
            "Average": f"{s.mean():.2f}", "Median": f"{s.median():.2f}",
            "Min": f"{s.min():.2f}", "Max": f"{s.max():.2f}",
        })
    if team2 != "None" and stat1:
        s = df_t2[stat1]
        summary_data.append({
            "Team": team2, "Stat": stat1,
            "Average": f"{s.mean():.2f}", "Median": f"{s.median():.2f}",
            "Min": f"{s.min():.2f}", "Max": f"{s.max():.2f}",
        })
    if team1 and stat2 != "None":
        s = df_t1[stat2]
        summary_data.append({
            "Team": team1, "Stat": stat2,
            "Average": f"{s.mean():.2f}", "Median": f"{s.median():.2f}",
            "Min": f"{s.min():.2f}", "Max": f"{s.max():.2f}",
        })
    if team2 != "None" and stat2 != "None":
        s = df_t2[stat2]
        summary_data.append({
            "Team": team2, "Stat": stat2,
            "Average": f"{s.mean():.2f}", "Median": f"{s.median():.2f}",
            "Min": f"{s.min():.2f}", "Max": f"{s.max():.2f}",
        })

    _t_items = [(team1, df_t1)]
    if team2 != "None":
        _t_items.append((team2, df_t2))
    _t_stats = [stat1] + ([stat2] if stat2 != "None" else [])
    _team_key_stats = [
        ("Points", "Points/Game"),
        ("Tries", "Tries/Game"),
        ("All Run Metres", "Run Metres/Game"),
        ("Tackles Made", "Tackles/Game"),
        ("Errors", "Errors/Game"),
        ("Line Breaks", "Line Breaks/Game"),
        ("Post Contact Metres", "PCM/Game"),
        ("Offloads", "Offloads/Game"),
        ("Missed Tackles", "Missed Tackles/Game"),
    ]

    with st.container(border=True):
        _tsum_l, _tsum_m, _tsum_r = st.columns(3)
        with _tsum_l:
            _render_profile(team1, df_t1, entity="team")
            if team2 != "None":
                _section_divider()
                _render_profile(team2, df_t2, entity="team")
            _section_divider()
            if summary_data:
                _render_summary_table(summary_data)
        with _tsum_m:
            _render_percentile_ranks(_t_stats, _t_items, team_df, group_col='Team')
            _section_divider()
            _render_recent_form(_t_stats, _t_items)
        with _tsum_r:
            _render_key_metrics(_t_items, stat_cols=_team_key_stats, all_df=team_df, group_col='Team')

    if team2 != "None" and stat2 != "None":
        plot_type = 4
    elif team2 != "None":
        plot_type = 2
    elif stat2 != "None":
        plot_type = 3
    else:
        plot_type = 1

    # -- Build ordered panel list (round separate when stat2 selected) ---
    _t_panels = []
    if stat2 != "None":
        _t_panels.append("corr")
    _t_panels.append("dist")
    if stat2 == "None":
        _t_panels.append("round")

    def _render_team_panel(pid, wide=False):
        if pid == "corr":
            with st.expander(f"{stat1} vs {stat2}", expanded=True):
                if plot_type == 4:
                    _df1 = _prepare_round_plot_df(team_df[team_df['Team'] == team1])
                    _df2 = _prepare_round_plot_df(team_df[team_df['Team'] == team2])
                    if wide:
                        _wl, _wr = st.columns(2)
                        with _wl:
                            _corr_matplotlib_plot(_df1, stat1, stat2, f"{team1} — {stat1} vs {stat2}")
                        with _wr:
                            _corr_matplotlib_plot(_df2, stat1, stat2, f"{team2} — {stat1} vs {stat2}")
                    else:
                        _corr_matplotlib_plot(_df1, stat1, stat2, f"{team1} — {stat1} vs {stat2}")
                        _corr_matplotlib_plot(_df2, stat1, stat2, f"{team2} — {stat1} vs {stat2}")
                elif plot_type == 3:
                    _df1 = _prepare_round_plot_df(team_df[team_df['Team'] == team1])
                    if not _df1.empty:
                        _corr_matplotlib_plot(_df1, stat1, stat2, f"{team1} — {stat1} vs {stat2}")

        elif pid == "round":
            with st.expander("Stat Comparison by Round", expanded=True):
                _ry_opts = [str(y) for y in year_choice]
                _ry = st.radio("Year", _ry_opts, horizontal=True, key="team_round_year") if len(_ry_opts) > 1 else _ry_opts[0]
                _rtdf = team_df[team_df["Year"] == _ry] if "Year" in team_df.columns else team_df
                if plot_type == 4:
                    _df1 = _prepare_round_plot_df(_rtdf[_rtdf['Team'] == team1])
                    _df2 = _prepare_round_plot_df(_rtdf[_rtdf['Team'] == team2])
                    if wide:
                        _wl, _wr = st.columns(2)
                        with _wl:
                            _round_matplotlib_compare(_df1, _df2, stat1, f"{stat1} Comparison: {team1} vs {team2}", team1, team2)
                        with _wr:
                            _round_matplotlib_compare(_df1, _df2, stat2, f"{stat2} Comparison: {team1} vs {team2}", team1, team2)
                    else:
                        _round_matplotlib_compare(_df1, _df2, stat1, f"{stat1} Comparison: {team1} vs {team2}", team1, team2)
                        _round_matplotlib_compare(_df1, _df2, stat2, f"{stat2} Comparison: {team1} vs {team2}", team1, team2)
                elif plot_type == 2:
                    _df1 = _prepare_round_plot_df(_rtdf[_rtdf['Team'] == team1])
                    _df2 = _prepare_round_plot_df(_rtdf[_rtdf['Team'] == team2])
                    _round_matplotlib_compare(_df1, _df2, stat1, f"{stat1} Comparison: {team1} vs {team2}", team1, team2)
                elif plot_type == 3:
                    _df1 = _prepare_round_plot_df(_rtdf[_rtdf['Team'] == team1])
                    _round_matplotlib_dual(_df1, stat1, stat2, f"{team1} — {stat1} + {stat2} by round")
                else:
                    _df1 = _prepare_round_plot_df(_rtdf[_rtdf['Team'] == team1])
                    _round_matplotlib_single(_df1, stat1, f"{stat1}: {team1}")

        elif pid == "dist":
            _d1 = team_df[team_df['Team'] == team1]
            _d2 = team_df[team_df['Team'] == team2] if team2 != "None" else pd.DataFrame()
            _dl = team2 if team2 != "None" else None
            with st.expander("Distribution", expanded=True):
                if wide and stat2 != "None":
                    _wl, _wr = st.columns(2)
                    with _wl:
                        _distribution_plot(_d1, _d2, stat1, team1, _dl)
                    with _wr:
                        _distribution_plot(_d1, _d2, stat2, team1, _dl)
                else:
                    _distribution_plot(_d1, _d2, stat1, team1, _dl)
                    if stat2 != "None":
                        _distribution_plot(_d1, _d2, stat2, team1, _dl)

    # Render paired panels in rows, odd panel full-width with side-by-side charts
    _n_paired = len(_t_panels) - (len(_t_panels) % 2)
    for _i in range(0, _n_paired, 2):
        _pl, _pr = st.columns(2, gap="small")
        with _pl:
            _render_team_panel(_t_panels[_i])
        with _pr:
            _render_team_panel(_t_panels[_i + 1])
    if len(_t_panels) % 2 == 1:
        _render_team_panel(_t_panels[-1], wide=True)

    # Round panel: full-width with side-by-side charts when 2 stats selected
    if stat2 != "None":
        _render_team_panel("round", wide=True)

else:
    st.header("Home")
    st.write("Please select a view.")
    
