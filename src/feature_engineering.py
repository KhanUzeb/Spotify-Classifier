import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
import joblib
import os

def generate_features(X: pd.DataFrame) -> pd.DataFrame:
    """Basic gen: ratios, log, bins."""
    X_out = X.copy()
    X_out['energy_dance'] = X['energy'] * X['danceability']
    X_out['log_duration'] = np.log1p(X['duration_ms'])
    X_out['energy_bin'] = pd.cut(X['energy'], bins=3, labels=[0, 1, 2])
    return X_out

def process_features(X_train: pd.DataFrame, X_test: pd.DataFrame, output_dir: str = 'outputs'):
    """Gen + scale + PCA, save."""
    # Gen
    gen_tf = FunctionTransformer(generate_features)
    X_train_gen = gen_tf.fit_transform(X_train)
    X_test_gen = gen_tf.transform(X_test)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_gen)
    X_test_scaled = scaler.transform(X_test_gen)
    
    # PCA
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train_pca).to_csv(f'{output_dir}/X_train_pca.csv', index=False)
    pd.DataFrame(X_test_pca).to_csv(f'{output_dir}/X_test_pca.csv', index=False)
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    joblib.dump(pca, f'{output_dir}/pca.pkl')
    
    print(f"Processed: PCA shape {X_train_pca.shape}, Var {pca.explained_variance_ratio_.sum():.3f}")
    return X_train_pca, X_test_pca

if __name__ == "__main__":
    X_train = pd.read_csv('outputs/X_train_raw.csv')
    X_test = pd.read_csv('outputs/X_test_raw.csv')
    process_features(X_train, X_test)