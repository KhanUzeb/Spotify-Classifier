import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_clean(high_path: str, low_path: str):
    """Load CSVs, label, clean, parse dates."""
    high = pd.read_csv(high_path)
    low = pd.read_csv(low_path)
    high['popularity'] = 1
    low['popularity'] = 0
    df = pd.concat([high, low], ignore_index=True)
    
    # Parse year
    if 'track_album_release_date' in df.columns:
        df['release_year'] = pd.to_datetime(df['track_album_release_date'], errors='coerce').dt.year
    
    # Fill NaNs
    df = df.fillna(df.mean(numeric_only=True))
    
    # Genre dummies
    if 'playlist_genre' in df.columns:
        df = pd.get_dummies(df, columns=['playlist_genre'], prefix='genre')
    
    print(f"Cleaned shape: {df.shape}, Imbalance: {df['popularity'].value_counts(normalize=True)}")
    return df

def split_data(df: pd.DataFrame, features: list, test_size: float = 0.2, output_dir: str = 'outputs'):
    """Split X/y, save CSVs."""
    X = df[features]
    y = df['popularity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train).to_csv(f'{output_dir}/X_train_raw.csv', index=False)
    pd.DataFrame(X_test).to_csv(f'{output_dir}/X_test_raw.csv', index=False)
    pd.Series(y_train).to_csv(f'{output_dir}/y_train.csv', index=False)
    pd.Series(y_test).to_csv(f'{output_dir}/y_test.csv', index=False)
    
    print(f"Split saved: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    high_path = 'data/unprocessed/archive/high_popularity_spotify_data.csv'
    low_path = 'data/unprocessed/archive/low_popularity_spotify_data.csv'
    features = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence', 
                'speechiness', 'instrumentalness', 'acousticness', 'mode', 'key', 
                'duration_ms', 'time_signature', 'release_year']
    df = load_and_clean(high_path, low_path)
    split_data(df, features)