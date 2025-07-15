import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, average_precision_score,
                             precision_recall_curve, auc)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from scipy import interpolate
import warnings

warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred, y_scores):
    # Check for NaN values
    if np.isnan(y_scores).any():
        y_scores = np.nan_to_num(y_scores)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }

    # Calculate AUPRC
    try:
        metrics['auprc'] = average_precision_score(y_true, y_scores)
    except:
        print("无法计算AUPRC，使用默认值0")
        metrics['auprc'] = 0

    return metrics

def interpolate_pr_curve(precision, recall):
    """插值PR曲线到固定长度的点"""
    f = interpolate.interp1d(recall, precision, bounds_error=False, fill_value=(1.0, 0.0))
    new_recall = np.linspace(0, 1, 100)
    new_precision = f(new_recall)
    return new_precision, new_recall

def plot_pr_curve(y_true, y_scores, fold):
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auprc = auc(recall, precision)

        plt.figure()
        plt.plot(recall, precision, label=f'Fold {fold} (AUPRC = {auprc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(f'pr_curve_fold{fold}.png')
        plt.close()
        return precision, recall
    except Exception as e:
        print(f"无法绘制Fold {fold}的PR曲线: {str(e)}")
        return None, None

def train_test_split(X, y, splits=10, batch_size=32):
    # Check data
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: {np.bincount(y)}")

    # Handle possible NaN values
    X = np.nan_to_num(X)
    y = np.nan_to_num(y).astype(int)

    k_fold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=2025)
    results = []

    # Store PR curve data for all folds (interpolated)
    interp_precisions = []
    interp_recalls = np.linspace(0, 1, 100)  # Fixed 100 recall points

    for fold, (train_idx, test_idx) in enumerate(k_fold.split(X, y)):
        print(f'\n{"=" * 50}')
        print(f'Fold {fold + 1}/{splits}')
        print(f'{"=" * 50}')

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create RandomForest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        # Train model
        rf_model.fit(X_train, y_train)

        # Evaluate on test set
        y_scores = rf_model.predict_proba(X_test)[:, 1]  # Get the probability for class 1 (positive class)
        y_pred = (y_scores > 0.5).astype(int)
        metrics = calculate_metrics(y_test, y_pred, y_scores)

        # Plot and save PR curve for current fold
        precision, recall = plot_pr_curve(y_test, y_scores, fold + 1)

        # Interpolate PR curve to fixed length
        if precision is not None and recall is not None:
            interp_precision, _ = interpolate_pr_curve(precision, recall)
            interp_precisions.append(interp_precision)

        # Save results
        results.append(metrics)

        # Print current fold results
        print(f'\nFold {fold + 1} Test Metrics:')
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"AUPRC: {metrics['auprc']:.4f}")

    # Calculate and print average metrics
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

    # Plot average PR curve (using interpolated data)
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

# Load data
data = pd.read_csv('preparations/heart_output.csv')  # Replace with your actual file path

# Check data
print("数据前5行:")
print(data.head())
print("\n类别分布:")
print(data['HeartDisease'].value_counts())

# Separate features and target
X = data.drop('HeartDisease', axis=1).values
y = data['HeartDisease'].values

# Convert to numpy arrays
X = X.astype(np.float32)
y = y.astype(np.int64)

# Run training and evaluation
train_test_split(X, y, splits=10, batch_size=32)
