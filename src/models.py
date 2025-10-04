from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
import joblib
import os
import pandas as pd

def get_base_model(name: str):
    """Base model factory (for pipelines)."""
    if name == 'lr':
        return LogisticRegression(random_state=42, max_iter=1000)
    elif name == 'rf':
        return RandomForestClassifier(random_state=42)
    elif name == 'xgb':
        return XGBClassifier(random_state=42, eval_metric='logloss')
    raise ValueError("Unknown model")

def tune_lr(X_train, y_train, output_dir: str):
    """Tune LR with SMOTE."""
    lr_pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', get_base_model('lr'))
    ])
    param_grid = {'model__C': [0.01, 0.1, 1, 10, 100], 'model__solver': ['liblinear', 'lbfgs']}
    grid = GridSearchCV(lr_pipe, param_grid, cv=5, scoring='f1_macro',n_jobs=-1,verbose=1)
    grid.fit(X_train, y_train)
    joblib.dump(grid, f'{output_dir}/tuned_lr_smote.pkl')
    print(f"LR Best F1: {grid.best_score_:.3f}")
    return grid

def tune_rf(X_train, y_train, output_dir: str):
    """Tune RF with SMOTE."""
    rf_pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', get_base_model('rf'))
    ])
    param_grid = {'model__n_estimators': [100, 200, 300], 'model__max_depth': [None, 10, 20], 'model__min_samples_split': [2, 5]}
    grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring='f1_macro',n_jobs=-1,verbose=1)
    grid.fit(X_train, y_train)
    joblib.dump(grid, f'{output_dir}/tuned_rf_smote.pkl')
    print(f"RF Best F1: {grid.best_score_:.3f}")
    return grid

def tune_xgb(X_train, y_train, output_dir: str, n_trials: int = 100):
    """Optuna XGB with SMOTE."""
    scale_pos = len(y_train[y_train==0]) / len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
        }
        
        pipe = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', XGBClassifier(**params, random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos))
        ])
        
        return cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1_macro',n_jobs=-1,verbose=1).mean()
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=SuccessiveHalvingPruner())
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_pos
    xgb_pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', XGBClassifier(**best_params, random_state=42, eval_metric='logloss'))
    ])
    xgb_pipe.fit(X_train, y_train)
    joblib.dump(xgb_pipe, f'{output_dir}/tuned_xgb_smote_optuna.pkl')
    joblib.dump(study, f'{output_dir}/optuna_study.pkl')
    print(f"XGB Best F1: {study.best_value:.3f}")
    return xgb_pipe

def vote_ensemble(lr, rf, xgb, X_train, y_train, output_dir: str):
    """Voting ensemble."""
    lr_est = lr.named_steps['model']
    rf_est = rf.named_steps['model']
    xgb_est = xgb.named_steps['model']
    
    voting = VotingClassifier([
        ('lr', lr_est),
        ('rf', rf_est),
        ('xgb', xgb_est)
    ], voting='soft',weights=[1, 1, 2])
    
    voting.fit(X_train, y_train)
    joblib.dump(voting, f'{output_dir}/voting_ensemble.pkl')
    print("Voting saved")
    return voting

def create_stacking_ensemble(lr, rf, xgb, voting, X_train, y_train, output_dir: str):
    """Stacking with LR meta, including voting as base."""
    lr_est = lr.named_steps['model']
    rf_est = rf.named_steps['model']
    xgb_est = xgb.named_steps['model']
    voting_est = voting  # Voting is already fitted estimator
    
    stacking = StackingClassifier([
        ('lr', lr_est),
        ('rf', rf_est),
        ('xgb', xgb_est),
        ('voting', voting_est)  # Add prebuilt voting as base
    ], final_estimator=LogisticRegression(random_state=42), cv=5, n_jobs=-1, verbose=1)
    
    stacking.fit(X_train, y_train)
    joblib.dump(stacking, f'{output_dir}/stacking_ensemble.pkl')
    print("Stacking saved")
    return stacking

if __name__ == "__main__":
    X_train = pd.read_csv('outputs/X_train_pca.csv')
    y_train = pd.read_csv('outputs/y_train.csv').squeeze()
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    tuned_lr = tune_lr(X_train, y_train, output_dir)
    tuned_rf = tune_rf(X_train, y_train, output_dir)
    tuned_xgb = tune_xgb(X_train, y_train, output_dir)
    voting = vote_ensemble(tuned_lr, tuned_rf, tuned_xgb, X_train, y_train, output_dir)
    create_stacking_ensemble(tuned_lr, tuned_rf, tuned_xgb, voting, X_train, y_train, output_dir)