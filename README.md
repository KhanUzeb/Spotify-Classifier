# SPOTIFY POPULARITY CLASSIFIER

## QUICK SUMMARY

So first of all hi,so this is my first full fledge project so be earful,
if you find any flaws in the project pls criticize me regarding it as i am beginner so a full blown good project can't be hoped from me and if you find it good pls star it,now it's clear so lets get started

## ABOUT THE PROJECT

Ever wondered what makes a Spotify banger blow up? Is it the thumping bass, that earworm tempo, or just pure vibe? This project dives into that mystery using machine learning to predict if a track will be "high popularity" (top charts) or "low popularity" (hidden gem or classic) based on Spotify's audio features like energy, danceability, and valence.Built as my first ML project, it's a full end-to-end pipeline: from exploring the data to stacking ensembles that hit ~70% accuracy. No DL fireworks here—just solid classical ML tuned for imbalance (thanks, SMOTE!). Think of it as a jam session where trees and boosters riff off each other. Grab your headphones, and let's crank it up!

### P.S: 35% CODE IS WRITTEN BY GROK 4 & REST IS ALL MY WORK

* PROBLEM STATEMENT:Binary classification—high pop (1) vs. low pop (0) for tracks from playlists like "Today's Top Hits" vs. "Rock Classics."

* Data:~4K tracks from two CSVs (high/low pop), features like energy, tempo, danceability DATA LINK:[text](https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset)

* Approach: EDA → Feature eng (ratios, log transforms, PCA) → Models (LR/RF/XGB with SMOTE) → Tuning (GridSearch/Optuna) → Ensemble (Stacking).

* Results: Stacking ensemble: 70.2% acc, 0.664 F1-macro, 0.754 AUC. Beats baseline (67% majority class) but capped by small data—room for 10K+ tracks!

## INSTALLATION AND WORKING

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

### Data: Drop high_popularity_spotify_data.csv and low_popularity_spotify_data.csv into /data/

## OPTION 1:NOTEBOOKS

### OPEN JUPYTER

* 01_eda.ipynb: Explore data, spot imbalance (67% low-pop), outliers.

* 02_feature_selection.ipynb: ANOVA/RFE for top feats (danceability shines!).

* 03_prep_scaling.ipynb: Gen features (energy_dance ratio), scale, PCA (95% var in 4 dims).

* 04_modeling.ipynb: SMOTE + tuning + stacking—watch F1 climb to 0.66.

* 05_evaluation.ipynb: Metrics, curves, 10-sample preds (80%+ with thresh tweak)

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
