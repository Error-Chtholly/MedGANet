import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, average_precision_score,
                             precision_recall_curve, auc)
from pytorch_tabnet.tab_model import TabNetClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from scipy import interpolate
import torch

warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred, y_scores):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'auprc': average_precision_score(y_true, y_scores[:, 1])  # 只取正类概率
    }
    return metrics

def interpolate_pr_curve(precision, recall):
    """插值PR曲线到固定长度的点"""
    f = interpolate.interp1d(recall, precision, bounds_error=False, fill_value=(1.0, 0.0))
    new_recall = np.linspace(0, 1, 100)
    new_precision = f(new_recall)
    return new_precision, new_recall

def plot_pr_curve(y_true, y_scores, fold):
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_scores[:, 1])  # 只取正类概率
        auprc = auc(recall, precision)

        plt.figure()
        plt.plot(recall, precision, label=f'Fold {fold} (AUPRC = {auprc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (Fold {fold})')
        plt.legend()
        plt.savefig(f'pr_curve_fold{fold}.png')
        plt.close()
        return precision, recall
    except Exception as e:
        print(f"无法绘制Fold {fold}的PR曲线: {str(e)}")
        return None, None

def train_test_split(X, y, splits=10, batch_size=32):
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: {np.bincount(y)}")

    X = np.nan_to_num(X)
    y = np.nan_to_num(y).astype(int)

    k_fold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=2025)
    results = []
    best_model_info = {'val_score': -float('inf'), 'model': None}

    interp_precisions = []
    interp_recalls = np.linspace(0, 1, 100)

    for fold, (train_idx, test_idx) in enumerate(k_fold.split(X, y)):
        print(f'\n{"=" * 50}')
        print(f'Fold {fold + 1}/{splits}')
        print(f'{"=" * 50}')

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Use SMOTE for balancing the dataset
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        model = TabNetClassifier(
            n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=0,
            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
            mask_type='sparsemax', scheduler_params=dict(step_size=50, gamma=0.9),
            scheduler_fn=torch.optim.lr_scheduler.StepLR
        )

        model.fit(X_res, y_res, eval_set=[(X_test, y_test)], patience=50, batch_size=batch_size)

        y_scores = model.predict_proba(X_test)
        y_pred = np.argmax(y_scores, axis=1)
        metrics = calculate_metrics(y_test, y_pred, y_scores)

        precision, recall = plot_pr_curve(y_test, y_scores, fold + 1)

        if precision is not None and recall is not None:
            interp_precision, _ = interpolate_pr_curve(precision, recall)
            interp_precisions.append(interp_precision)

        results.append(metrics)

        if metrics['auprc'] > best_model_info['val_score']:
            best_model_info['val_score'] = metrics['auprc']
            best_model_info['model'] = model

        print(f'\nFold {fold + 1} Test Metrics:')
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"AUPRC: {metrics['auprc']:.4f}")

    # Save the best model
    best_model_info['model'].save_model('best_model.pkl')
    print(f"\nSaved best model with AUPRC {best_model_info['val_score']:.4f}")

    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in results]),
        'f1': np.mean([r['f1'] for r in results]),
        'precision': np.mean([r['precision'] for r in results]),
        'recall': np.mean([r['recall'] for r in results]),
        'auprc': np.mean([r['auprc'] for r in results])
    }

    std_metrics = {
        'accuracy': np.std([r['accuracy'] for r in results]),
        'f1': np.std([r['f1'] for r in results]),
        'precision': np.std([r['precision'] for r in results]),
        'recall': np.std([r['recall'] for r in results]),
        'auprc': np.std([r['auprc'] for r in results])
    }

    print('\n' + '=' * 50)
    print('Final Cross-Validation Results:')
    print('=' * 50)
    print(f"Average Accuracy: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"Average F1 Score: {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"Average Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"Average Recall: {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"Average AUPRC: {avg_metrics['auprc']:.4f} ± {std_metrics['auprc']:.4f}")

    if interp_precisions:
        plt.figure(figsize=(8, 6))
        mean_precision = np.mean(interp_precisions, axis=0)
        mean_auprc = auc(interp_recalls, mean_precision)

        for i, prec in enumerate(interp_precisions):
            plt.plot(interp_recalls, prec, alpha=0.3, label=f'Fold {i + 1}')

        plt.plot(interp_recalls, mean_precision, 'k-',
                 label=f'Mean (AUPRC = {mean_auprc:.2f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average Precision-Recall Curve')
        plt.legend()
        plt.savefig('average_pr_curve.png')
        plt.close()

# Load and preprocess data
data = pd.read_csv('preparations/heart_output.csv')

X = data.drop('HeartDisease', axis=1).values
y = data['HeartDisease'].values

X = X.astype(np.float32)
y = y.astype(np.int64)

train_test_split(X, y, splits=10, batch_size=32)