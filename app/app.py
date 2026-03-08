import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="ATP Player Similarity (2000–2025)", layout="wide")

st.title("ATP Player Similarity Dashboard (Cosine Similarity)")
st.caption("Player-season-surface performance vectors computed from Kaggle ATP match data.")

# ----------------------------
# Load data (CSV exports from your notebook)
# ----------------------------
BASE = Path(__file__).resolve().parent.parent
TABLES = BASE / "outputs" / "tables"

@st.cache_data
def load_tables():
    matches_raw = pd.read_csv(TABLES / "matches_raw.csv")
    player_matches = pd.read_csv(TABLES / "player_matches.csv")
    player_features = pd.read_csv(TABLES / "player_features.csv")
    return matches_raw, player_matches, player_features

try:
    matches_raw, player_matches, player_features = load_tables()
except Exception as e:
    st.error(
        "Could not load CSV tables. Make sure you exported:\n"
        "- outputs/tables/matches_raw.csv\n"
        "- outputs/tables/player_matches.csv\n"
        "- outputs/tables/player_features.csv\n"
    )
    st.exception(e)
    st.stop()

# Basic cleaning / typing
for c in ["season"]:
    if c in player_features.columns:
        player_features[c] = pd.to_numeric(player_features[c], errors="coerce").astype("Int64")

# ----------------------------
# Feature columns used for similarity
# ----------------------------
FEATURE_COLS = [
    "win_rate",
    "avg_player_rank",
    "avg_opponent_rank",
    "avg_rank_diff",
    "avg_player_points",
    "avg_opponent_points",
    "avg_player_odds",
    "avg_opponent_odds",
    "underdog_win_rate",
    "best_of_5_rate",
]

missing = [c for c in FEATURE_COLS if c not in player_features.columns]
if missing:
    st.error(f"Missing required feature columns in player_features.csv: {missing}")
    st.stop()

# Fill NA in features (median)
X = player_features[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

# Standardize + similarity matrix (cached)
@st.cache_data
def compute_similarity_matrix(X_df: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    return cosine_similarity(X_scaled)

sim_matrix = compute_similarity_matrix(X)

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Filters")

players_list = sorted(player_features["player"].dropna().unique().tolist())
default_player = "Alcaraz C." if "Alcaraz C." in players_list else players_list[0]
player = st.sidebar.selectbox("Player", players_list, index=players_list.index(default_player))

seasons = sorted(player_features["season"].dropna().unique().tolist())
default_season = 2025 if 2025 in seasons else seasons[-1]
season = st.sidebar.selectbox("Season", seasons, index=seasons.index(default_season))

surfaces = sorted(player_features["surface"].dropna().unique().tolist())
default_surface = "Hard" if "Hard" in surfaces else surfaces[0]
surface = st.sidebar.selectbox("Surface", surfaces, index=surfaces.index(default_surface))

top_n = st.sidebar.slider("Top-N similar players", min_value=5, max_value=20, value=10, step=1)

# ----------------------------
# Helper: top similar
# ----------------------------
def top_similar(player_name: str, season_val: int, surface_val: str, top_n_val: int = 10) -> pd.DataFrame:
    mask = (
        (player_features["player"] == player_name) &
        (player_features["season"] == season_val) &
        (player_features["surface"] == surface_val)
    )
    idx = player_features.index[mask]
    if len(idx) == 0:
        return pd.DataFrame()

    i = idx[0]
    scores = pd.Series(sim_matrix[i], index=player_features.index).sort_values(ascending=False)

    out = player_features.loc[
        scores.index, ["player", "season", "surface", "matches", "win_rate"]
    ].copy()
    out["similarity"] = scores.values

    # Same season & surface only (prevents missing peer rows)
    out = out[
        (out["season"] == season_val) &
        (out["surface"] == surface_val)
    ]

    # Remove itself + Top-N
    out = out[out["player"] != player_name].head(top_n_val).reset_index(drop=True)
    return out

# ✅ IMPORTANT: compute topN BEFORE using it
topN = top_similar(player, int(season), surface, top_n)

# ----------------------------
# Layout
# ----------------------------
col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.subheader("Top Similar Players")
    if topN.empty:
        st.warning("No row found for this player-season-surface. Try another season/surface.")
    else:
        st.dataframe(topN, use_container_width=True)

with col2:
    st.subheader("Similarity Chart")
    if not topN.empty:
        fig = plt.figure()
        plt.barh(topN["player"][::-1], topN["similarity"][::-1])
        plt.xlabel("Cosine Similarity")
        plt.title(f"Top {top_n} Similar to {player} ({season}, {surface})")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

st.divider()

# ----------------------------
# Feature comparison vs top-1 similar
# ----------------------------
st.subheader("Feature Comparison (Selected vs Most Similar)")

if topN.empty:
    st.info("Select a scenario with available similarity results to see the feature comparison.")
else:
    peer = topN.loc[0, "player"]

    target_df = player_features[
        (player_features["player"] == player) &
        (player_features["season"] == int(season)) &
        (player_features["surface"] == surface)
    ]

    peer_df = player_features[
        (player_features["player"] == peer) &
        (player_features["season"] == int(season)) &
        (player_features["surface"] == surface)
    ]

    if target_df.empty:
        st.warning("No feature row found for the selected player in this season/surface.")
    elif peer_df.empty:
        st.warning(
            f"Most similar player '{peer}' has no feature row for ({season}, {surface}). "
            "Try increasing Top-N, changing season/surface, or lowering the match threshold in feature engineering."
        )
    else:
        target_row = target_df.iloc[0]
        peer_row = peer_df.iloc[0]

        comp = pd.DataFrame({
            "feature": FEATURE_COLS,
            player: [target_row[f] for f in FEATURE_COLS],
            peer: [peer_row[f] for f in FEATURE_COLS],
        })

        fig2 = plt.figure()
        x = np.arange(len(comp["feature"]))
        plt.bar(x - 0.2, comp[player], width=0.4, label=player)
        plt.bar(x + 0.2, comp[peer], width=0.4, label=peer)
        plt.xticks(x, comp["feature"], rotation=75, ha="right")
        plt.title(f"{player} vs {peer} ({season}, {surface})")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)

        with st.expander("Show comparison table"):
            st.dataframe(comp, use_container_width=True)

st.divider()

# ----------------------------
# Drill-down: recent matches for selected scenario (optional)
# ----------------------------
st.subheader("Recent Matches (Drill-down)")

pm = player_matches.copy()

# Normalize date typing if it was saved as string
if "date" in pm.columns:
    pm["date"] = pd.to_datetime(pm["date"], errors="coerce")

# Ensure season type alignment
if "season" in pm.columns:
    pm["season"] = pd.to_numeric(pm["season"], errors="coerce").astype("Int64")

scenario_matches = pm[
    (pm["player"] == player) &
    (pm["season"] == int(season)) &
    (pm["surface"] == surface)
].sort_values("date", ascending=False).head(25)

cols_to_show = [c for c in ["date", "tournament", "round", "surface", "opponent", "is_win", "score"] if c in scenario_matches.columns]
st.dataframe(scenario_matches[cols_to_show], use_container_width=True)
