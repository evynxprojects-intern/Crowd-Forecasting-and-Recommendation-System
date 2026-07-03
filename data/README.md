# Data Directory

## Files in This Directory

| File | Size | Description |
|---|---|---|
| `sample_data.csv` | ~50KB | 100-row sample of the final dataset for quick testing |
| `asi_visitor_data.csv` | ~5KB | Real ASI government visitor statistics for 40 monuments |
| `calendar_lookup_2025_2027.csv` | ~2KB | Festival and holiday calendar (2025-2027) |

---

## Large Files (Not in GitHub — Available on Request)

The following large files are stored on Google Drive and available on request:

| File | Size | Description |
|---|---|---|
| `india_tourist_crowd_forecast_dataset.csv` | ~300MB | Full training dataset (450,540 rows × 151 cols) |
| `india_tourist_crowd_features.json` | ~5KB | Final 59 feature names for model training |
| `india_crowd_timeseries_v2.csv` | ~500MB | Intermediate timeseries (pre-encoding) |
| `india_crowd_enhanced_v6_FINAL.csv` | ~50MB | Raw enriched dataset (12,655 places × 95 cols) |
| `popular_times_real.json` | ~200MB | Real Google Popular Times data (busyness patterns) |
| `places_raw_checkpoint.json` | ~30MB | Raw Google Places API checkpoint |

**Contact:** [harshkumarsingh01@gmail.com] to request access to large files.

---

## Data Pipeline

```
Google Places API (335 cities)
        ↓
places_raw_checkpoint.json     ← 14,547 raw records
        ↓
india_places_clean.csv          ← 12,544 deduplicated
        ↓
popular_times_real.json         ← busyness enrichment
        ↓
india_crowd_enhanced_v6_FINAL.csv ← 12,655 places × 95 cols
        ↓
india_crowd_timeseries_v2.csv   ← 450,540 rows (place × month)
        ↓
india_tourist_crowd_forecast_dataset.csv  ← final training file
india_tourist_crowd_features.json          ← 59 selected features
```

---

## Real Data Sources

| Signal | Coverage | Source |
|---|---|---|
| Place names, coordinates, ratings, reviews | 100% | Google Places API |
| Festival / holiday calendar 2025-2027 | 100% | Official Indian calendars |
| Per-city monthly weather (temp, rainfall) | 97% | ERA5 via Open-Meteo API |
| Monthly search interest (Google Trends) | 81% | pytrends (free) |
| Opening hours (weekly off, open/close time) | 65.7% | Google Place Details API |
| Busyness patterns (Popular Times) | 9.7% | Google Popular Times API |

---

## ASI Validation Data

`asi_visitor_data.csv` contains **real official visitor statistics** from:
- India Tourism Data Compendium 2024 (Ministry of Tourism)
- Rajya Sabha responses (Archaeological Survey of India)
- Annual footfall for FY 2023-24

Used to validate model predictions against real-world crowd levels.
**Real-world accuracy achieved: 75%** (40 monuments tested).
