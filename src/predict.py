"""
predict.py
India Tourist Crowd Forecasting System
Prediction function for the trained model
"""

import pickle
import pandas as pd
import numpy as np
import json
import os

# ── Load model and features ────────────────────────
def load_model(model_path='models/crowd_model_v1.pkl'):
    """Load trained model, label encoder, and feature list."""
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['model'], bundle['label_encoder'], bundle['features']


def prepare_features(place_id, year, month, df, feature_list):
    """
    Extract features for a specific (place, year, month) combination.
    
    Args:
        place_id  : str  — place identifier
        year      : int  — prediction year (2025-2027)
        month     : int  — prediction month (1-12)
        df        : pd.DataFrame — full encoded dataset
        feature_list: list — features to use

    Returns:
        pd.DataFrame — single-row feature vector
    """
    row = df[
        (df['place_id'] == place_id) &
        (df['year'] == year) &
        (df['month'] == month)
    ]
    if row.empty:
        return None

    return row[feature_list].fillna(0)


def predict_crowd(place_id, year, month, df, model, label_encoder, features):
    """
    Predict crowd level for a place at a specific (year, month).

    Returns:
        dict with prediction, confidence, and explanation
    """
    X = prepare_features(place_id, year, month, df, features)
    if X is None:
        return {
            'error': f'Place {place_id} not found in dataset',
            'prediction': None,
            'confidence': None
        }

    # Predict
    pred_encoded   = model.predict(X)[0]
    pred_proba     = model.predict_proba(X)[0]
    prediction     = label_encoder.inverse_transform([pred_encoded])[0]
    confidence     = round(float(pred_proba.max()) * 100, 1)

    # Top contributing features (simple heuristic)
    top_features   = _get_top_reasons(X, features, prediction)

    return {
        'place_id'   : place_id,
        'year'       : year,
        'month'      : month,
        'prediction' : prediction,       # 'Low', 'Medium', or 'High'
        'confidence' : confidence,        # e.g. 87.3
        'reasons'    : top_features,      # top signals for this prediction
        'all_proba'  : {
            label_encoder.classes_[i]: round(float(pred_proba[i]) * 100, 1)
            for i in range(len(label_encoder.classes_))
        }
    }


def _get_top_reasons(X, features, prediction):
    """
    Return human-readable top reasons for the prediction.
    Based on feature values, not SHAP (lightweight version).
    """
    READABLE = {
        'has_major_festival'            : 'Major festival this month',
        'num_festivals_in_month'        : 'Multiple festivals in month',
        'days_to_nearest_festival'      : 'Close to a festival',
        'is_school_vacation'            : 'School vacation period',
        'is_long_weekend_month'         : 'Long weekend this month',
        'real_weather_category_Pleasant': 'Pleasant weather conditions',
        'real_weather_category_Cold'    : 'Cold/winter weather (peak season)',
        'real_weather_category_Rainy'   : 'Monsoon / rainy season',
        'season_Monsoon'                : 'Monsoon season',
        'season_Winter'                 : 'Peak winter tourist season',
        'season_Summer'                 : 'Summer holiday season',
        'search_trend_this_month'       : 'High search interest this month',
        'busyness_avg'                  : 'High historical footfall',
        'is_religious'                  : 'Religious site',
        'is_heritage'                   : 'Heritage / historical site',
        'is_beach'                      : 'Beach destination',
        'beach_x_winter'                : 'Beach in peak winter season',
        'heritage_x_festival'           : 'Heritage site during festival',
        'religious_x_festival'          : 'Religious site during festival',
    }

    reasons = []
    row     = X.iloc[0]
    for feat, label in READABLE.items():
        if feat in features:
            val = row.get(feat, 0)
            if val and val > 0:
                reasons.append(label)
        if len(reasons) >= 3:
            break

    return reasons if reasons else ['Seasonal crowd pattern']


# ── CLI Usage ──────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict crowd level for an Indian tourist place')
    parser.add_argument('--place_id', required=True, help='Place ID from dataset')
    parser.add_argument('--year',     type=int, required=True, help='Year (2025-2027)')
    parser.add_argument('--month',    type=int, required=True, help='Month (1-12)')
    parser.add_argument('--data',     default='data/india_tourist_crowd_forecast_dataset.csv')
    parser.add_argument('--model',    default='models/crowd_model_v1.pkl')
    args = parser.parse_args()

    print(f'Loading model from {args.model}...')
    model, le, features = load_model(args.model)

    print(f'Loading dataset from {args.data}...')
    df = pd.read_csv(args.data)

    result = predict_crowd(args.place_id, args.year, args.month, df, model, le, features)
    print(f'\n📍 Place    : {result[\"place_id\"]}')
    print(f'📅 Period   : {result[\"year\"]}-{result[\"month\"]:02d}')
    print(f'👥 Crowd    : {result[\"prediction\"]} ({result[\"confidence\"]}% confidence)')
    print(f'💡 Reasons  : {\" | \".join(result[\"reasons\"])}')
    print(f'📊 All proba: {result[\"all_proba\"]}')
