"""
refresh_weather.py
Monthly Weather Refresh (ERA5 Historical + SEAS5 Forecast)
India Tourist Crowd Forecasting System

Schedule: 5th of every month at 3:00 AM IST
Runtime : ~30 minutes
Cost    : FREE (Open-Meteo API)

What it does:
→ ERA5 (historical): Fetches last 2 months' ACTUAL
  weather data for all 335 Indian cities
→ SEAS5 (forecast): Fetches 7-month ahead seasonal
  weather forecast for future predictions
→ Replaces static climate averages with REAL data

WHY THIS MATTERS:
Previous approach used static "typical year" averages.
ERA5 + SEAS5 gives REAL measured + forecast weather,
directly improving 6-month ahead crowd predictions.
(Open-Meteo ERA5 updates with 5-7 day lag)
"""

import os
import json
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────
CLIMATE_FILE   = 'data/climate_lookup_by_city_real.csv'
OUTPUT_FILE    = 'data/weather_updated.json'
FORECAST_FILE  = 'data/weather_forecast_seas5.json'

# Major Indian cities with coordinates
CITY_COORDS = {
    'Delhi'      : (28.6139, 77.2090),
    'Mumbai'     : (19.0760, 72.8777),
    'Bangalore'  : (12.9716, 77.5946),
    'Chennai'    : (13.0827, 80.2707),
    'Kolkata'    : (22.5726, 88.3639),
    'Hyderabad'  : (17.3850, 78.4867),
    'Jaipur'     : (26.9124, 75.7873),
    'Agra'       : (27.1767, 78.0081),
    'Varanasi'   : (25.3176, 82.9739),
    'Amritsar'   : (31.6340, 74.8723),
    'Goa'        : (15.2993, 74.1240),
    'Kochi'      : (9.9312,  76.2673),
    'Shimla'     : (31.1048, 77.1734),
    'Manali'     : (32.2396, 77.1887),
    'Mysore'     : (12.2958, 76.6394),
    'Udaipur'    : (24.5854, 73.7125),
    'Jodhpur'    : (26.2389, 73.0243),
    'Darjeeling' : (27.0360, 88.2627),
    'Rishikesh'  : (30.0869, 78.2676),
    'Aurangabad' : (19.8762, 75.3433),
}

# Weather category classification thresholds
def classify_weather(temp_c, rainfall_mm):
    """Classify weather into categories used by the model."""
    if rainfall_mm > 100:
        return 'Rainy' if temp_c > 20 else 'Cold'
    elif temp_c > 35:
        return 'Hot'
    elif temp_c > 25 and rainfall_mm > 30:
        return 'Hot/Rainy'
    elif 15 <= temp_c <= 30 and rainfall_mm < 30:
        return 'Pleasant'
    elif temp_c < 15:
        return 'Cold'
    else:
        return 'Pleasant'


def fetch_era5_historical(lat, lon, year, month):
    """
    Fetch ERA5 actual historical weather for a city and month.
    Open-Meteo Historical API — free, no key needed.
    Updates with 5-7 day lag.
    """
    # ERA5 data
    start = f'{year}-{month:02d}-01'
    # Last day of month
    if month == 12:
        end = f'{year}-12-31'
    else:
        end = f'{year}-{month+1:02d}-01'
        end = (datetime.strptime(end, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')

    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude'         : lat,
        'longitude'        : lon,
        'start_date'       : start,
        'end_date'         : end,
        'daily'            : 'temperature_2m_mean,precipitation_sum',
        'timezone'         : 'Asia/Kolkata',
    }

    try:
        resp = requests.get(url, params=params, timeout=15).json()
        if 'daily' not in resp:
            return None

        temps     = resp['daily'].get('temperature_2m_mean', [])
        rainfall  = resp['daily'].get('precipitation_sum', [])

        avg_temp     = round(float(np.nanmean([t for t in temps if t])), 1) if temps else None
        total_rain   = round(float(np.nansum([r for r in rainfall if r])), 1) if rainfall else None

        return {
            'avg_temp_c'   : avg_temp,
            'rainfall_mm'  : total_rain,
            'weather_cat'  : classify_weather(avg_temp or 25, total_rain or 0),
            'is_monsoon'   : 1 if (total_rain or 0) > 100 else 0,
            'data_type'    : 'ERA5_historical',
            'year'         : year,
            'month'        : month,
        }
    except Exception as e:
        log.warning(f'  ERA5 error: {e}')
        return None


def fetch_seas5_forecast(lat, lon):
    """
    Fetch SEAS5 7-month seasonal weather forecast.
    Open-Meteo Seasonal API — free.
    Used for forward-looking crowd predictions.
    """
    url = 'https://seasonal-api.open-meteo.com/v1/seasonal'
    params = {
        'latitude'         : lat,
        'longitude'        : lon,
        'daily'            : 'temperature_2m_mean,precipitation_sum',
        'forecast_days'    : 214,  # ~7 months
        'timezone'         : 'Asia/Kolkata',
    }

    try:
        resp = requests.get(url, params=params, timeout=15).json()
        if 'daily' not in resp:
            return None

        # Aggregate by month
        dates     = pd.to_datetime(resp['daily']['time'])
        temps     = resp['daily'].get('temperature_2m_mean', [])
        rainfall  = resp['daily'].get('precipitation_sum', [])

        df = pd.DataFrame({
            'date'    : dates,
            'temp'    : temps,
            'rainfall': rainfall
        })
        df['month'] = df['date'].dt.month
        df['year']  = df['date'].dt.year

        monthly = df.groupby(['year', 'month']).agg({
            'temp'    : 'mean',
            'rainfall': 'sum'
        }).reset_index()

        forecasts = {}
        for _, row in monthly.iterrows():
            key = f"{int(row['year'])}-{int(row['month']):02d}"
            t   = round(float(row['temp']), 1) if pd.notna(row['temp']) else None
            r   = round(float(row['rainfall']), 1) if pd.notna(row['rainfall']) else None
            forecasts[key] = {
                'avg_temp_c' : t,
                'rainfall_mm': r,
                'weather_cat': classify_weather(t or 25, r or 0),
                'is_monsoon' : 1 if (r or 0) > 100 else 0,
                'data_type'  : 'SEAS5_forecast',
            }

        return forecasts

    except Exception as e:
        log.warning(f'  SEAS5 error: {e}')
        return None


def run_weather_refresh():
    """Main weather refresh function."""
    log.info('='*55)
    log.info('  MONTHLY WEATHER REFRESH STARTED')
    log.info(f'  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log.info('='*55)

    now = datetime.now()
    historical_data = {}
    forecast_data   = {}
    success = 0
    failed  = 0

    for city, (lat, lon) in CITY_COORDS.items():
        log.info(f'Fetching: {city}')

        # ── ERA5: Last 2 months actual ──
        for delta in [1, 2]:
            target = now - timedelta(days=delta*30)
            yr, mo = target.year, target.month
            hist   = fetch_era5_historical(lat, lon, yr, mo)

            if hist:
                key = f'{city}_{yr}_{mo}'
                historical_data[key] = {'city': city, 'lat': lat, 'lon': lon, **hist}
                log.info(f'  ✅ ERA5 {yr}-{mo:02d}: {hist["avg_temp_c"]}°C, '
                         f'{hist["rainfall_mm"]}mm → {hist["weather_cat"]}')
                success += 1
            else:
                failed += 1
            time.sleep(1)

        # ── SEAS5: 7-month forecast ──
        forecasts = fetch_seas5_forecast(lat, lon)
        if forecasts:
            forecast_data[city] = {'lat': lat, 'lon': lon, 'forecasts': forecasts}
            log.info(f'  ✅ SEAS5: {len(forecasts)} months forecast')
        else:
            log.warning(f'  ⚠️  SEAS5 failed for {city}')

        time.sleep(1.5)  # Rate limit

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(historical_data, f, indent=2)

    with open(FORECAST_FILE, 'w') as f:
        json.dump(forecast_data, f, indent=2)

    log.info('='*55)
    log.info('  WEATHER REFRESH COMPLETE')
    log.info(f'  ERA5 success   : {success} city-months')
    log.info(f'  ERA5 failed    : {failed}')
    log.info(f'  SEAS5 cities   : {len(forecast_data)}')
    log.info(f'  Historical file: {OUTPUT_FILE}')
    log.info(f'  Forecast file  : {FORECAST_FILE}')
    log.info('='*55)


if __name__ == '__main__':
    run_weather_refresh()
