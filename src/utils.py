"""
utils.py
India Tourist Crowd Forecasting System
Helper functions used across notebooks and the app
"""

import pandas as pd
import numpy as np


# ── Season from Month ──────────────────────────────
def get_season(month):
    """Return Indian tourist season for a given month number."""
    if month in [12, 1, 2]:  return 'Winter'
    elif month in [3, 4, 5]: return 'Summer'
    elif month in [6, 7, 8, 9]: return 'Monsoon'
    else: return 'Post-Monsoon'


# ── Month name ─────────────────────────────────────
MONTH_NAMES = {
    1:'January', 2:'February', 3:'March', 4:'April',
    5:'May', 6:'June', 7:'July', 8:'August',
    9:'September', 10:'October', 11:'November', 12:'December'
}

def month_name(month_num):
    return MONTH_NAMES.get(month_num, str(month_num))


# ── Crowd label color ──────────────────────────────
CROWD_COLORS = {'Low': '#2ECC71', 'Medium': '#F39C12', 'High': '#E74C3C'}

def crowd_color(label):
    """Return hex color for a crowd label (for dashboard display)."""
    return CROWD_COLORS.get(label, '#95A5A6')


# ── Crowd emoji ────────────────────────────────────
CROWD_EMOJI = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}

def crowd_emoji(label):
    return CROWD_EMOJI.get(label, '⚪')


# ── Best months for a place ────────────────────────
def get_best_months(place_id, df, n=3):
    """
    Return the n months with lowest predicted crowd for a place.

    Args:
        place_id : str — place identifier
        df       : pd.DataFrame — full dataset (must have relative_crowd_label)
        n        : int — number of best months to return

    Returns:
        list of dicts with month, year, label, season
    """
    place_df = df[df['place_id'] == place_id].copy()
    if place_df.empty:
        return []

    low_months = place_df[
        place_df['relative_crowd_label'] == 'Low'
    ].sort_values('monthly_crowd_score').head(n)

    return [{
        'month'  : int(row['month']),
        'year'   : int(row['year']),
        'label'  : row['relative_crowd_label'],
        'season' : get_season(int(row['month'])),
        'month_name': month_name(int(row['month']))
    } for _, row in low_months.iterrows()]


# ── Place type detection ───────────────────────────
PLACE_TYPE_FLAGS = {
    'Heritage'      : 'is_heritage',
    'Religious'     : 'is_religious',
    'Beach'         : 'is_beach',
    'Museum'        : 'is_museum',
    'Park'          : 'is_park',
    'Nature'        : 'is_nature',
    'Wildlife'      : 'is_wildlife',
    'Hill Station'  : 'is_hill_station',
    'Market'        : 'is_market',
    'Cave'          : 'is_cave',
    'Amusement Park': 'is_amusement_park',
    'Viewpoint'     : 'is_viewpoint',
    'Tourist Spot'  : 'is_tourist_spot',
}

def get_place_type(row):
    """Infer place type from is_* flag columns."""
    for ptype, flag in PLACE_TYPE_FLAGS.items():
        if flag in row and row[flag] == 1:
            return ptype
    return 'Tourist Spot'


# ── Indian zone name ───────────────────────────────
ZONE_COLS = {
    'zone_Northern'    : 'Northern India',
    'zone_Southern'    : 'Southern India',
    'zone_Eastern'     : 'Eastern India',
    'zone_Western'     : 'Western India',
    'zone_Northeastern': 'Northeastern India',
    'zone_Central'     : 'Central India',
}

def get_zone_name(row):
    """Get human-readable zone name from one-hot encoded zone columns."""
    for col, name in ZONE_COLS.items():
        if col in row and row[col] == 1:
            return name
    return 'India'


# ── Normalize place name for matching ─────────────
def normalize_name(s):
    """Normalize a place name for fuzzy matching."""
    return (str(s).lower().strip()
            .replace("'s", "s")
            .replace(" ", "")
            .replace("-", "")
            .replace(".", ""))


# ── Log transform with safe handling ─────────────
def safe_log1p(series):
    """Apply log1p transform, handling negatives and NaN safely."""
    return np.log1p(series.clip(lower=0).fillna(0))


# ── Memory optimization ────────────────────────────
def optimize_memory(df):
    """Downcast numeric columns to reduce memory usage."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df


# ── Quick dataset info ─────────────────────────────
def dataset_info(df):
    """Print a quick summary of a dataset."""
    print(f'Shape     : {df.shape}')
    print(f'Memory    : {df.memory_usage(deep=True).sum()/1e6:.1f} MB')
    print(f'Nulls     : {df.isnull().sum().sum()}')
    print(f'Dtypes    : {df.dtypes.value_counts().to_dict()}')
