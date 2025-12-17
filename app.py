import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import io

st.set_page_config(page_title="Club Radar Comparison", layout="wide")
st.title("Club Comparison — Radar")

# ----------------- Styling (tabs optional) -----------------
st.markdown(
    """
    <style>
      section[data-testid="stTabs"] div[role="tablist"] { gap: 10px; }
      section[data-testid="stTabs"] button[role="tab"] {
        padding: 10px 16px; border-radius: 999px; font-weight: 700;
        border: 1px solid #E5E7EB; background: #F8FAFC;
      }
      section[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: #E5F0FF; border-color: #93C5FD; color: #1D4ED8;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Defaults -----------------
DEFAULT_METRICS = [
    {"Metric": "xG",               "Bottom": 0.0,  "Top": 3.0,  "Lower is better": False},
    {"Metric": "Goals",            "Bottom": 0.0,  "Top": 5.0,  "Lower is better": False},
    {"Metric": "Touches in Box",   "Bottom": 5.0,  "Top": 60.0, "Lower is better": False},
    {"Metric": "Shots",            "Bottom": 5.0,  "Top": 25.0, "Lower is better": False},
    {"Metric": "xG Against",       "Bottom": 0.0,  "Top": 3.0,  "Lower is better": True},
    {"Metric": "Goals Against",    "Bottom": 0.0,  "Top": 5.0,  "Lower is better": True},
    {"Metric": "Shots Against",    "Bottom": 5.0,  "Top": 25.0, "Lower is better": True},
    {"Metric": "Pressing (PPDA)",  "Bottom": 5.0,  "Top": 25.0, "Lower is better": True},
    {"Metric": "Possession",       "Bottom": 35.0, "Top": 70.0, "Lower is better": False},
    {"Metric": "Passes",           "Bottom": 200,  "Top": 800,  "Lower is better": False},
    {"Metric": "Passes Final 3rd", "Bottom": 40,   "Top": 250,  "Lower is better": False},
    {"Metric": "Long Passes",      "Bottom": 10,   "Top": 120,  "Lower is better": False},
]

# team colors
COL_A = "#C81E1E"
COL_B = "#1D4ED8"
FILL_A = (200/255, 30/255, 30/255, 0.55)
FILL_B = (29/255, 78/255, 216/255, 0.55)

# radar geometry
NUM_RINGS = 11
INNER_HOLE = 10
ring_edges = np.linspace(INNER_HOLE, 100, NUM_RINGS)

def to_pct(value, vmin, vmax, lower_is_better=False):
    if value is None or value == "" or pd.isna(value):
        return np.nan
    value = float(value)
    if vmax == vmin:
        pct = 50.0
    else:
        value = max(min(value, vmax), vmin)  # clamp
        pct = (value - vmin) / (vmax - vmin) * 100.0
    return (100.0 - pct) if lower_is_better else pct

def draw_radar(labels, A_r, B_r, teamA_name, teamA_sub, teamB_name, teamB_sub, theme="Light"):
    # ---- theme ----
    if theme == "Dark":
        PAGE_BG = "#0a0f1c"
        AX_BG = "#0a0f1c"
        GRID_BAND_OUTER = "#162235"
        GRID_BAND_INNER = "#0d1524"
        RING_COLOR = "#3a4050"
        LABEL_COLOR = "#f5f5f5"
        SUB_COLOR = "#cbd5e1"
    else:
        PAGE_BG = "#ffffff"
        AX_BG = "#ffffff"
        GRID_BAND_OUTER = "#e5e7eb"
        GRID_BAND_INNER = "#ffffff"
        RING_COLOR = "#d1d5db"
        LABEL_COLOR = "#0f172a"
        SUB_COLOR = "#475569"

    N = len(labels)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])
    Ar = np.concatenate([A_r, A_r[:1]])
    Br = np.concatenate([B_r, B_r[:1]])

    fig = plt.figure(figsize=(12.6, 7.6), dpi=220)
    fig.patch.set_facecolor(PAGE_BG)

    ax = plt.subplot(111, polar=True)
    ax.set_facecolor(AX_BG)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    ax.set_xticks(theta)
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)

    # Alternating ring bands (outer starts as OUTER color)
    for i in range(NUM_RINGS - 1):
        r0, r1 = ring_edges[i], ring_edges[i + 1]
        steps_from_outer = (NUM_RINGS - 2) - i
        band = GRID_BAND_OUTER if (steps_from_outer % 2 == 0) else GRID_BAND_INNER
        ax.add_artist(Wedge((0, 0), r1, 0, 360, width=(r1 - r0),
                            transform=ax.transData._b, facecolor=band,
                            edgecolor="none", zorder=0.8))

    # Ring outlines
    ring_t = np.linspace(0, 2*np.pi, 361)
    for r in ring_edges:
        ax.plot(ring_t, np.full_like(ring_t, r), color=RING_COLOR, lw=1.0, zorder=0.9)

    # Labels outside
    OUTER_LABEL_R = 106
    for ang, lab in zip(theta, labels):
        rot = np.degrees(-ang + np.pi/2) - 90
        rot_norm = ((rot + 180) % 360) - 180
        if rot_norm > 90 or rot_norm < -90:
            rot += 180
        ax.text(ang, OUTER_LABEL_R, lab, rotation=rot, rotation_mode="anchor",
                ha="center", va="center", fontsize=10, fontweight=650,
                color=LABEL_COLOR, clip_on=False)

    # Center hole
    ax.add_artist(Circle((0, 0), radius=INNER_HOLE - 0.6,
                         transform=ax.transData._b, color=PAGE_BG, ec="none", zorder=1.2))

    # Polygons
    ax.plot(theta_closed, Ar, color=COL_A, lw=2.4, zorder=3)
    ax.fill(theta_closed, Ar, color=FILL_A, zorder=2.5)

    ax.plot(theta_closed, Br, color=COL_B, lw=2.4, zorder=3)
    ax.fill(theta_closed, Br, color=FILL_B, zorder=2.5)

    ax.set_rlim(0, 100)

    # ---- NO MIDDLE TITLE ----
    # Left/Right team headers (bigger)
    fig.text(0.10, 0.95, teamA_name, ha="left", va="top",
             fontsize=26, fontweight="bold", color=COL_A)
    fig.text(0.10, 0.90, teamA_sub, ha="left", va="top",
             fontsize=11, color=SUB_COLOR)

    fig.text(0.90, 0.95, teamB_name, ha="right", va="top",
             fontsize=26, fontweight="bold", color=COL_B)
    fig.text(0.90, 0.90, teamB_sub, ha="right", va="top",
             fontsize=11, color=SUB_COLOR)

    return fig

# ----------------- UI -----------------
with st.sidebar:
    st.header("Setup")

    theme = st.radio("Theme", ["Light", "Dark"], index=0, horizontal=True)

    # team header text inputs
    teamA_display = st.text_input("Team A name (red)", "Swansea")
    teamA_sub = st.text_input("Team A subheading", "Championship • 2024/25")

    teamB_display = st.text_input("Team B name (blue)", "Team 2")
    teamB_sub = st.text_input("Team B subheading", "Championship • 2024/25")

    n_teams = st.number_input("How many teams will you enter?", min_value=2, max_value=50, value=6, step=1)

    st.divider()
    st.subheader("Metrics + Bottom/Top")
    st.caption("Edit Bottom/Top and whether lower values should score higher on the radar.")

# Initialize session state tables
if "metrics_df" not in st.session_state:
    st.session_state.metrics_df = pd.DataFrame(DEFAULT_METRICS)

metrics_df = st.data_editor(
    st.session_state.metrics_df,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "Metric": st.column_config.TextColumn(required=True),
        "Bottom": st.column_config.NumberColumn(required=True),
        "Top": st.column_config.NumberColumn(required=True),
        "Lower is better": st.column_config.CheckboxColumn(required=True),
    },
    key="metrics_editor",
)
st.session_state.metrics_df = metrics_df

# Create team input table
metric_names = metrics_df["Metric"].tolist()
team_cols = ["Team"] + metric_names

if "teams_df" not in st.session_state:
    st.session_state.teams_df = pd.DataFrame([{"Team": f"Team {i+1}"} for i in range(int(n_teams))])
    for m in metric_names:
        st.session_state.teams_df[m] = np.nan

# Keep row count synced with n_teams
teams_df = st.session_state.teams_df.copy()
if len(teams_df) < int(n_teams):
    add = int(n_teams) - len(teams_df)
    new_rows = pd.DataFrame([{"Team": f"Team {len(teams_df)+i+1}"} for i in range(add)])
    for m in metric_names:
        new_rows[m] = np.nan
    teams_df = pd.concat([teams_df, new_rows], ignore_index=True)
elif len(teams_df) > int(n_teams):
    teams_df = teams_df.iloc[: int(n_teams)].copy()

# Ensure metric columns match (if user edits metrics list)
for m in metric_names:
    if m not in teams_df.columns:
        teams_df[m] = np.nan
drop_cols = [c for c in teams_df.columns if c not in team_cols]
teams_df = teams_df.drop(columns=drop_cols)

st.subheader("Enter team values")
st.caption("Enter raw values. They will be normalized using your Bottom/Top ranges.")
teams_df = st.data_editor(
    teams_df,
    use_container_width=True,
    num_rows="fixed",
    column_config={"Team": st.column_config.TextColumn(required=True)},
    key="teams_editor",
)
st.session_state.teams_df = teams_df

# Team selectors (data source)
team_names = teams_df["Team"].fillna("").tolist()
c1, c2 = st.columns([1, 1])
with c1:
    teamA_pick = st.selectbox("Data Team A (red polygon)", team_names, index=0)
with c2:
    teamB_pick = st.selectbox("Data Team B (blue polygon)", team_names, index=1 if len(team_names) > 1 else 0)

# Build radar
rowA = teams_df[teams_df["Team"] == teamA_pick].iloc[0]
rowB = teams_df[teams_df["Team"] == teamB_pick].iloc[0]

labels = metric_names
A_r, B_r = [], []

bad_ranges = []
for _, r in metrics_df.iterrows():
    m = r["Metric"]
    vmin, vmax = float(r["Bottom"]), float(r["Top"])
    if vmax <= vmin:
        bad_ranges.append(m)
        continue
    lower_better = bool(r["Lower is better"])
    A_r.append(to_pct(rowA.get(m), vmin, vmax, lower_better))
    B_r.append(to_pct(rowB.get(m), vmin, vmax, lower_better))

if bad_ranges:
    st.error(f"These metrics have invalid ranges (Top must be > Bottom): {bad_ranges}")
    st.stop()

A_r = np.array(A_r, dtype=float)
B_r = np.array(B_r, dtype=float)

fig = draw_radar(labels, A_r, B_r, teamA_display, teamA_sub, teamB_display, teamB_sub, theme=theme)
st.pyplot(fig, use_container_width=True)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
st.download_button(
    "⬇️ Download PNG",
    data=buf.getvalue(),
    file_name=f"{teamA_display.replace(' ','_')}_vs_{teamB_display.replace(' ','_')}_radar.png",
    mime="image/png",
)
