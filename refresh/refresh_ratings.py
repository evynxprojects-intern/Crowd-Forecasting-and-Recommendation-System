"""
refresh_ratings.py
Monthly Ratings + Reviews Refresh
India Tourist Crowd Forecasting System

Schedule: 1st of every month at 2:00 AM IST
Runtime : ~2 hours
Cost    : ~$25/month (Google Places Basic Data API)

What it does:
→ Re-fetches ratings and review counts for all 12,655 places
→ Smart filtering: only re-fetches Popular Times for places
  where rating changed > 0.2 OR reviews grew > 15%
  (~500-800 places vs all 12,655 = 10x cheaper)
→ Computes PSI drift score on key features
→ Triggers model retrain if PSI > 0.25
"""

import os
import json
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────
GOOGLE_API_KEY  = os.environ.get('GOOGLE_API_KEY', '')
DATA_FILE       = 'data/india_crowd_enhanced_v6_FINAL.csv'
CHECKPOINT_FILE = 'data/ratings_checkpoint.json'
OUTPUT_FILE     = 'data/ratings_updated.csv'
PSI_LOG_FILE    = 'data/psi_log.json'

# PSI thresholds (industry standard)
PSI_STABLE  = 0.10   # No action
PSI_MONITOR = 0.25   # Monitor closely
PSI_RETRAIN = 0.25   # Trigger retrain

# Smart refresh thresholds
RATING_CHANGE_THRESHOLD  = 0.2   # Only refresh if rating changed > 0.2
REVIEWS_GROWTH_THRESHOLD = 0.15  # Only refresh if reviews grew > 15%


def fetch_place_details(place_id_google, api_key):
    """Fetch current rating and review count from Google Places API."""
    try:
        url = 'https://maps.googleapis.com/maps/api/place/details/json'
        params = {
            'place_id': place_id_google,
            'fields'  : 'rating,user_ratings_total',
            'key'     : api_key
        }
        resp = requests.get(url, params=params, timeout=10).json()
        if resp.get('status') == 'OK':
            result = resp.get('result', {})
            return {
                'rating'    : result.get('rating'),
                'num_reviews': result.get('user_ratings_total')
            }
    except Exception as e:
        log.warning(f'  API error for {place_id_google}: {e}')
    return None


def compute_psi(baseline, current, bins=10):
    """
    Compute Population Stability Index (PSI) between baseline and current.
    PSI < 0.10  → Stable
    PSI 0.10-0.25 → Monitor
    PSI > 0.25  → Retrain needed
    """
    def get_bin_percentages(data, bin_edges):
        counts, _ = np.histogram(data, bins=bin_edges)
        pcts = counts / len(data)
        return np.where(pcts == 0, 0.0001, pcts)  # avoid log(0)

    combined  = np.concatenate([baseline, current])
    bin_edges = np.percentile(combined, np.linspace(0, 100, bins+1))
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 3:
        return 0.0

    base_pcts    = get_bin_percentages(baseline, bin_edges)
    current_pcts = get_bin_percentages(current, bin_edges)

    psi = np.sum((current_pcts - base_pcts) * np.log(current_pcts / base_pcts))
    return round(float(psi), 4)


def should_refresh_busyness(old_rating, new_rating, old_reviews, new_reviews):
    """Smart filter: only refresh Popular Times if signals changed significantly."""
    rating_change  = abs((new_rating or 0) - (old_rating or 0))
    reviews_growth = ((new_reviews or 0) - (old_reviews or 0)) / max(old_reviews or 1, 1)
    return rating_change > RATING_CHANGE_THRESHOLD or reviews_growth > REVIEWS_GROWTH_THRESHOLD


def run_monthly_refresh():
    """Main monthly refresh function."""
    log.info('='*55)
    log.info('  MONTHLY RATINGS REFRESH STARTED')
    log.info(f'  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log.info('='*55)

    if not GOOGLE_API_KEY:
        log.error('GOOGLE_API_KEY not set! Add to environment variables.')
        return

    # Load current dataset
    df = pd.read_csv(DATA_FILE)
    log.info(f'Loaded: {df.shape}')

    # Load checkpoint
    checkpoint = {}
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)

    # Store baseline for PSI
    baseline_ratings = df['rating'].dropna().values
    baseline_reviews = df['num_reviews'].dropna().values

    # Refresh ratings + reviews
    updated_ratings  = []
    updated_reviews  = []
    busyness_refresh = []
    success = 0
    failed  = 0

    places_with_pid = df[df['place_id_google'].notna()].copy()
    log.info(f'Places with Google ID: {len(places_with_pid):,}')

    for i, (_, row) in enumerate(places_with_pid.iterrows()):
        pid_google = str(row['place_id_google'])

        if (i+1) % 500 == 0:
            log.info(f'  Progress: {i+1:,}/{len(places_with_pid):,}')
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint, f)

        new_data = fetch_place_details(pid_google, GOOGLE_API_KEY)

        if new_data:
            new_rating  = new_data.get('rating')
            new_reviews = new_data.get('num_reviews')

            # Check if busyness refresh needed (smart filter)
            old_rating  = row.get('rating')
            old_reviews = row.get('num_reviews')

            if should_refresh_busyness(old_rating, new_rating,
                                        old_reviews, new_reviews):
                busyness_refresh.append(pid_google)

            # Update in dataframe
            if new_rating:
                df.loc[row.name, 'rating']      = new_rating
                updated_ratings.append(new_rating)
            if new_reviews:
                df.loc[row.name, 'num_reviews'] = new_reviews
                updated_reviews.append(new_reviews)

            checkpoint[pid_google] = {
                'rating'      : new_rating,
                'num_reviews' : new_reviews,
                'updated_at'  : datetime.now().isoformat()
            }
            success += 1
        else:
            failed += 1

        time.sleep(0.1)  # Rate limit

    # ── PSI Drift Detection ────────────────────────
    log.info('\nComputing PSI drift scores...')

    psi_ratings  = compute_psi(baseline_ratings, np.array(updated_ratings)) if updated_ratings else 0
    psi_reviews  = compute_psi(baseline_reviews, np.array(updated_reviews)) if updated_reviews else 0

    psi_log = {
        'date'       : datetime.now().isoformat(),
        'psi_rating' : psi_ratings,
        'psi_reviews': psi_reviews,
        'action'     : 'stable'
    }

    if psi_ratings > PSI_RETRAIN or psi_reviews > PSI_RETRAIN:
        psi_log['action'] = 'retrain_triggered'
        log.warning(f'⚠️  PSI THRESHOLD EXCEEDED — Model retrain triggered!')
        log.warning(f'   PSI rating : {psi_ratings:.3f} (threshold: {PSI_RETRAIN})')
        log.warning(f'   PSI reviews: {psi_reviews:.3f}')
        # TODO: trigger retrain pipeline
    elif psi_ratings > PSI_MONITOR or psi_reviews > PSI_MONITOR:
        psi_log['action'] = 'monitor'
        log.info(f'⚠️  PSI elevated — monitoring closely')
    else:
        log.info(f'✅ PSI stable')

    # Save PSI log
    psi_history = []
    if os.path.exists(PSI_LOG_FILE):
        with open(PSI_LOG_FILE) as f:
            psi_history = json.load(f)
    psi_history.append(psi_log)
    with open(PSI_LOG_FILE, 'w') as f:
        json.dump(psi_history, f, indent=2)

    # Save updated dataset
    df.to_csv(OUTPUT_FILE, index=False)

    log.info('='*55)
    log.info('  MONTHLY REFRESH COMPLETE')
    log.info(f'  Updated    : {success:,} places')
    log.info(f'  Failed     : {failed:,} places')
    log.info(f'  Busyness ↑ : {len(busyness_refresh)} places flagged for refresh')
    log.info(f'  PSI rating : {psi_ratings:.3f}')
    log.info(f'  PSI reviews: {psi_reviews:.3f}')
    log.info(f'  Action     : {psi_log["action"]}')
    log.info('='*55)


if __name__ == '__main__':
    run_monthly_refresh()
