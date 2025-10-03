import joblib
import pandas as pd
from src.feature_engineering import generate_features  # Reuse gen

def load_model(model_path: str = 'outputs/stacking_ensemble.pkl'):
    """Load stacking model."""
    return joblib.load(model_path)

def predict_single(model, new_data: pd.DataFrame, features: list, scaler_path: str = 'outputs/scaler.pkl', pca_path: str = 'outputs/pca.pkl'):
    """Predict on new track (DF row)."""
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    
    # Gen features
    new_gen = generate_features(new_data[features])
    
    # Scale + PCA
    new_scaled = scaler.transform(new_gen)
    new_pca = pca.transform(new_scaled)
    
    # Predict
    prob = model.predict_proba(new_pca)[:, 1][0]
    pred = 1 if prob > 0.3 else 0  # Tweak thresh
    return {'prediction': pred, 'probability_high_pop': prob.round(3), 'will_be_popular': 'Yes' if pred == 1 else 'No'}

if __name__ == "__main__":
    model = load_model()
    features = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence', 
                'speechiness', 'instrumentalness', 'acousticness', 'mode', 'key', 
                'duration_ms', 'time_signature', 'release_year']
    
    # Example new track
    new_track = pd.DataFrame({
        'energy': [0.7], 'tempo': [120], 'danceability': [0.6], 'loudness': [-6], 
        'liveness': [0.1], 'valence': [0.5], 'speechiness': [0.05], 'instrumentalness': [0.01], 
        'acousticness': [0.2], 'mode': [1], 'key': [5], 'duration_ms': [200000], 
        'time_signature': [4], 'release_year': [2024]
    })
    
    result = predict_single(model, new_track, features)
    print("Prediction:", result)