"""
feature_engineering.py
India Tourist Crowd Forecasting System
All feature creation functions used in the data pipeline
"""

import pandas as pd
import numpy as np
import json


# ── Cyclical Encoding ──────────────────────────────
def add_cyclical_features(df):
    """
    Encode month cyclically so Dec and Jan are numerically close.
    Without this, month 12 and month 1 would be 11 apart numerically.
    """
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df


# ── Interaction Features ───────────────────────────
def add_interaction_features(df):
    """
    Add place-type × season interaction features.
    These help the model understand that different place types
    respond differently to seasons and festivals.

    Examples:
    - Beaches are busy in WINTER, not just festivals
    - Heritage sites peak during festivals
    - Museums are an indoor refuge during monsoon
    """
    # Beach interactions
    df['beach_x_winter']           = df['is_beach'] * df['season_Winter']
    df['beach_x_school_vacation']  = df['is_beach'] * df['is_school_vacation']
    df['beach_x_monsoon']          = df['is_beach'] * df.get('real_is_monsoon_month', 0)

    # Wildlife / nature
    df['wildlife_x_post_monsoon']  = df.get('is_wildlife', 0) * df['season_Post-Monsoon']
    df['nature_x_post_monsoon']    = df.get('is_nature', 0) * df['season_Post-Monsoon']

    # Heritage
    df['heritage_x_festival']      = df['is_heritage'] * df['has_major_festival']
    df['heritage_x_winter']        = df['is_heritage'] * df['season_Winter']
    df['heritage_x_long_weekend']  = df['is_heritage'] * df['is_long_weekend_month']

    # Religious
    df['religious_x_festival']     = df['is_religious'] * df['has_major_festival']

    # Museum (indoor → monsoon refuge)
    df['museum_x_monsoon']         = df['is_museum'] * df['season_Monsoon']

    # Park / Hill station
    df['park_x_school_vacation']           = df['is_park'] * df['is_school_vacation']
    df['hillstation_x_school_vacation']    = df.get('is_hill_station', 0) * df['is_school_vacation']

    return df


# ── Weather Suitability Score ──────────────────────
PLACE_TYPE_WEATHER_PREF = {
    'Beach'         : {'Pleasant': 1.3, 'Hot': 0.9, 'Hot/Rainy': 0.5, 'Rainy': 0.3, 'Cold': 0.7},
    'Hill Station'  : {'Pleasant': 1.2, 'Hot': 1.3, 'Hot/Rainy': 0.8, 'Rainy': 0.6, 'Cold': 0.7},
    'Wildlife'      : {'Pleasant': 1.3, 'Hot': 0.8, 'Hot/Rainy': 0.6, 'Rainy': 0.4, 'Cold': 1.1},
    'Heritage'      : {'Pleasant': 1.3, 'Hot': 0.8, 'Hot/Rainy': 0.6, 'Rainy': 0.5, 'Cold': 1.1},
    'Religious'     : {'Pleasant': 1.1, 'Hot': 1.0, 'Hot/Rainy': 0.9, 'Rainy': 0.8, 'Cold': 1.0},
    'Museum'        : {'Pleasant': 1.0, 'Hot': 1.1, 'Hot/Rainy': 1.0, 'Rainy': 1.1, 'Cold': 0.9},
    'Market'        : {'Pleasant': 1.1, 'Hot': 0.9, 'Hot/Rainy': 0.8, 'Rainy': 0.7, 'Cold': 1.0},
    'Park'          : {'Pleasant': 1.3, 'Hot': 0.8, 'Hot/Rainy': 0.5, 'Rainy': 0.4, 'Cold': 0.9},
    'Nature'        : {'Pleasant': 1.3, 'Hot': 0.8, 'Hot/Rainy': 0.6, 'Rainy': 0.5, 'Cold': 0.9},
    'Viewpoint'     : {'Pleasant': 1.3, 'Hot': 0.9, 'Hot/Rainy': 0.5, 'Rainy': 0.4, 'Cold': 0.8},
    'Cave'          : {'Pleasant': 1.0, 'Hot': 1.1, 'Hot/Rainy': 1.0, 'Rainy': 1.0, 'Cold': 0.9},
    'Amusement Park': {'Pleasant': 1.3, 'Hot': 0.9, 'Hot/Rainy': 0.6, 'Rainy': 0.5, 'Cold': 0.8},
    'Tourist Spot'  : {'Pleasant': 1.2, 'Hot': 0.9, 'Hot/Rainy': 0.7, 'Rainy': 0.6, 'Cold': 0.9},
}
DEFAULT_PREF = PLACE_TYPE_WEATHER_PREF['Tourist Spot']

def compute_weather_score(df):
    """
    Compute weather suitability score (0-100) for each (place_type, month) combination.
    Uses real ERA5 per-city weather category from Open-Meteo API.
    """
    def weather_mult(row):
        pref = PLACE_TYPE_WEATHER_PREF.get(row['place_type'], DEFAULT_PREF)
        cat  = row['real_weather_category'] if pd.notna(row.get('real_weather_category')) else 'Pleasant'
        return pref.get(cat, 1.0)

    df['weather_score'] = (df.apply(weather_mult, axis=1) * 50).clip(0, 100)
    return df


# ── Crowd Score Formula ────────────────────────────
def compute_crowd_score(df):
    """
    Composite monthly crowd score from real signals.

    Weights (sum to 1.0):
    - busyness_avg (real Google Popular Times)     : 25%
    - weather_score (real ERA5 per-city weather)   : 20%
    - festival_boost_score (real festival calendar): 20%
    - search_trend_this_month (real Google Trends) : 20%
    - review_velocity_score (real review data)     : 10%
    - calendar_boost (school vacations/holidays)   : 5%
    """
    df['busyness_avg'] = df['busyness_avg'].fillna(30)

    df['festival_boost_score'] = (
        df['num_festivals_in_month'].fillna(0) * 10 +
        df['has_major_festival'].fillna(0) * 20
    ).clip(0, 50)

    df['calendar_boost'] = (
        df['is_school_vacation'].fillna(0) * 10 +
        df['is_long_weekend_month'].fillna(0) * 5
    )

    trend_norm = df['search_trend_this_month'].clip(0, 100)

    df['monthly_crowd_score'] = (
        df['busyness_avg']          * 0.25 +
        df['weather_score']         * 0.20 +
        df['festival_boost_score']  * 0.20 +
        trend_norm                  * 0.20 +
        df['review_velocity_score'] * 0.10 +
        df['calendar_boost']        * 0.05
    ).round(2)

    return df


# ── Relative Labels ────────────────────────────────
def compute_relative_labels(df):
    """
    Assign crowd labels relative to each place's own 12-month pattern.
    This answers: "Is October the best/worst month to visit THIS specific place?"

    Uses .transform() (not .apply()) to avoid pandas 3.0 column-drop bug.
    """
    np.random.seed(42)
    scores = df['monthly_crowd_score']

    p33 = scores.groupby(df['place_id']).transform(lambda x: x.quantile(0.33))
    p66 = scores.groupby(df['place_id']).transform(lambda x: x.quantile(0.66))

    flat = (p33 == p66)
    if flat.any():
        noisy = scores + np.random.normal(0, 0.5, len(scores))
        p33_n = noisy.groupby(df['place_id']).transform(lambda x: x.quantile(0.33))
        p66_n = noisy.groupby(df['place_id']).transform(lambda x: x.quantile(0.66))
        scores = scores.where(~flat, noisy)
        p33    = p33.where(~flat, p33_n)
        p66    = p66.where(~flat, p66_n)

    df['relative_crowd_label'] = np.select(
        [scores <= p33, scores <= p66], ['Low', 'Medium'], default='High')

    return df


# ── Frequency Encoding ─────────────────────────────
def frequency_encode(df, col):
    """Replace a categorical column with its frequency proportion."""
    freq = df[col].value_counts(normalize=True)
    df[f'{col}_freq'] = df[col].map(freq)
    return df


# ── Parse Google Trends JSON ───────────────────────
def parse_trend_json(x):
    """
    Parse per-month Google Trends JSON blob.

    Input : '{"1": 17.14, "2": 22.15, ..., "12": 57.55}'
    Output: list of 12 float values [17.14, 22.15, ..., 57.55]
    """
    if pd.isna(x):
        return [np.nan] * 12
    try:
        d = json.loads(x)
        return [d.get(str(m), np.nan) for m in range(1, 13)]
    except (json.JSONDecodeError, TypeError):
        return [np.nan] * 12


def extract_monthly_trend(df):
    """
    Parse per-place Google Trends JSON and extract month-specific value.
    Must be called BEFORE expanding to timeseries.
    """
    import pandas as pd
    trend_parsed = df['search_trend_month_avg'].apply(parse_trend_json)
    trend_wide   = pd.DataFrame(
        trend_parsed.tolist(),
        columns=[f'trend_m{m}' for m in range(1, 13)],
        index=df.index)
    df = pd.concat([df, trend_wide], axis=1)
    df = df.drop(columns=['search_trend_month_avg'])

    parsed_ok = trend_wide.notna().any(axis=1).sum()
    print(f'Parsed real monthly trends for {parsed_ok:,} places ({parsed_ok/len(df)*100:.1f}%)')
    return df
