# SPOTIFY POPULARITY CLASSIFIER

## ABOUT THE PROJECT

This project explores what makes a Spotify track pop off versus fade into the background. Using Spotify’s audio features—like danceability, energy, tempo, and valence—I built a machine learning pipeline to classify songs as high popularity (top-chart hits) or low popularity (hidden gems).

* PROBLEM STATEMENT:Binary classification—high pop (1) vs. low pop (0) for tracks from playlists like "Today's Top Hits" vs. "Rock Classics."

* Data:~4K tracks from two CSVs (high/low pop), features like energy, tempo, danceability DATA LINK:[text](https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset)

* Approach: EDA → Feature eng (ratios, log transforms, PCA) → Models (LR/RF/XGB with SMOTE) → Tuning (GridSearch/Optuna) → Ensemble (Stacking).

* Results: Stacking ensemble: 70.2% acc, 0.664 F1-macro, 0.754 AUC. Beats baseline (67% majority class) but capped by small data—room for 10K+ tracks!

## REPORTS

>For ALL EVALS,SCORE & FINAL DATA VISUALIZATION SEEK REPORTS FOLDER

## INSTALLATION AND WORK

### Clone the repo

```bash
git clone https://GitHub.com/KhanUzeb/Spotify-Classifier.git 
cd spotify-popularity-classifier
```

### Virtual Env Setup

```bash
python -m venv env
source env/bin/activate  # Mac/Linux
# or
env\Scripts\activate  # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Data: Drop high_popularity_spotify_data.csv and low_popularity_spotify_data.csv into /data/unprocessed/archive/

## OPTION 1:NOTEBOOKS

### OPEN JUPYTER

* 01_eda.ipynb: Explore data, spot imbalance (67% low-pop), outliers.

* 02_feature_engineering.ipynb: Gen features (energy_dance ratio), scale, PCA (95% var in 4 dims).

* 03_modeling.ipynb: SMOTE + tuning + stacking—watch F1 climb to 0.66.

* 04_evaluation.ipynb: Metrics, curves, 10-sample preds (80%+ with thresh tweak)

> Run jupyter notebook and dive in—cells build on each other.

## OPTION 2:SCRIPT

* PREPROCESS DATA

```bash
python src/data_preprocessing.py
```

* FEATURE ENGINEERING

```bash
python src/feature_engineering.py
```

* TRAIN MODELS

```bash
python src/models.py
```

* EVALUATE

```bash
python src/evaluate.py
```

* PREDICTION

```bash
python src/predict.py
```

Have fun!
Now it's all yours pls tell me how good and bad it is
