import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_eval(output_dir: str = 'outputs'):
    """Load models, compute metrics."""
    X_test = pd.read_csv(f'{output_dir}/X_test_pca.csv')
    y_test = pd.read_csv(f'{output_dir}/y_test.csv').squeeze()
    
    models = {
        'Tuned LR + SMOTE': joblib.load(f'{output_dir}/tuned_lr_smote.pkl'),
        'Tuned RF + SMOTE': joblib.load(f'{output_dir}/tuned_rf_smote.pkl'),
        'Tuned XGB + SMOTE/Optuna': joblib.load(f'{output_dir}/tuned_xgb_smote_optuna.pkl'),
        'Stacking Ensemble': joblib.load(f'{output_dir}/stacking_ensemble.pkl')
    }
    
    preds = {}
    for name, model in models.items():
        if hasattr(model, 'named_steps'):
            est = model.named_steps['model']
        else:
            est = model
        preds[name] = {
            'pred': est.predict(X_test),
            'proba': est.predict_proba(X_test)[:, 1]
        }
    
    # Metrics DF
    metrics_df = pd.DataFrame()
    for name, data in preds.items():
        y_pred = data['pred']
        y_proba = data['proba']
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_proba)
        acc = (y_pred == y_test).mean()
        metrics_row = pd.DataFrame({
            'F1 Macro': [report['macro avg']['f1-score']],
            'Precision Pos': [report['1']['precision']],
            'Recall Pos': [report['1']['recall']],
            'AUC': [auc],
            'Acc': [acc]
        }, index=[name])
        metrics_df = pd.concat([metrics_df, metrics_row])
    
    print("Metrics:\n", metrics_df.round(3))
    metrics_df.to_csv(f'{output_dir}/eval_summary.csv')
    
    # CM plots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (name, data) in enumerate(preds.items()):
        cm = confusion_matrix(y_test, data['pred'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(name)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cms.png')
    plt.show()
    
    # ROC/PR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, data in preds.items():
        fpr, tpr, _ = roc_curve(y_test, data['proba'])
        auc = roc_auc_score(y_test, data['proba'])
        ax1.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
        
        prec, rec, _ = precision_recall_curve(y_test, data['proba'])
        auc_pr = np.trapz(rec, prec)
        ax2.plot(rec, prec, label=f'{name} (AUC-PR={auc_pr:.3f})')
    
    ax1.plot([0,1], [0,1], 'k--'); ax1.legend(); ax1.set_title('ROC')
    ax2.legend(); ax2.set_title('PR'); ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/curves.png')
    plt.show()
    
    # 10 samples
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_test), 10, replace=False)
    X_sample = X_test.iloc[sample_idx]
    y_true_sample = y_test.iloc[sample_idx]
    
    # Use Stacking
    ens = models['Stacking Ensemble']
    if hasattr(ens, 'named_steps'):
        est = ens.named_steps['model']
    else:
        est = ens
    y_pred_sample = est.predict(X_sample)
    y_proba_sample = est.predict_proba(X_sample)[:, 1]
    y_pred_thresh = (y_proba_sample > 0.3).astype(int)
    
    results_df = pd.DataFrame({
        'True': y_true_sample,
        'Pred (0.5)': y_pred_sample,
        'Prob High': y_proba_sample.round(3),
        'Pred (0.3)': y_pred_thresh,
        'Correct (0.5)': (y_pred_sample == y_true_sample),
        'Correct (0.3)': (y_pred_thresh == y_true_sample)
    })
    print("\n10 Samples:\n", results_df)
    print(f"Sample Acc 0.5: {(y_pred_sample == y_true_sample).mean():.3f}")
    print(f"Sample Acc 0.3: {(y_pred_thresh == y_true_sample).mean():.3f}")
    
    results_df.to_csv(f'{output_dir}/sample_preds.csv', index=False)

if __name__ == "__main__":
    load_and_eval()