import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, average_precision_score,
                             precision_recall_curve, auc, roc_auc_score,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
from scipy import interpolate

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def calculate_metrics(y_true, y_pred, y_scores):
    if np.isnan(y_scores).any():
        y_scores = np.nan_to_num(y_scores)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics['auprc'] = average_precision_score(y_true, y_scores)
        metrics['auroc'] = roc_auc_score(y_true, y_scores)
    except:
        print("无法计算AUPRC/AUROC，使用默认值0")
        metrics['auprc'] = 0
        metrics['auroc'] = 0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

    return metrics


def interpolate_pr_curve(precision, recall):
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
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend()
        plt.savefig(f'pr_curve_fold{fold}.png')
        plt.close()
        return precision, recall
    except Exception as e:
        print(f"无法绘制Fold {fold}的PR曲线: {str(e)}")
        return None, None


def train_test_split(X, y, splits=10):
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: {np.bincount(y)}")

    try:
        feature_names = data.drop('HeartDisease', axis=1).columns.tolist()
    except:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

    X = np.nan_to_num(X)
    y = np.nan_to_num(y).astype(int)

    k_fold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=2025)
    results = []
    weights = []
    all_y_true = []
    all_y_pred = []
    all_y_scores = []

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

        model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=42)
        model.fit(X_train, y_train)

        fold_weights = {
            'intercept': model.intercept_[0],
            'coefficients': model.coef_[0]
        }
        weights.append(fold_weights)

        y_scores = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_scores.extend(y_scores)

        metrics = calculate_metrics(y_test, y_pred, y_scores)

        precision, recall = plot_pr_curve(y_test, y_scores, fold + 1)
        if precision is not None and recall is not None:
            interp_precision, _ = interpolate_pr_curve(precision, recall)
            interp_precisions.append(interp_precision)

        results.append(metrics)

        print(f'\nFold {fold + 1} 测试集指标:')
        print(f"准确度: {metrics['accuracy']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"特异性: {metrics['specificity']:.4f}")
        print(f"AUPRC: {metrics['auprc']:.4f}")
        print(f"AUROC: {metrics['auroc']:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, zero_division=0))

    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in results]),
        'f1': np.mean([r['f1'] for r in results]),
        'precision': np.mean([r['precision'] for r in results]),
        'recall': np.mean([r['recall'] for r in results]),
        'specificity': np.mean([r['specificity'] for r in results]),
        'auprc': np.mean([r['auprc'] for r in results]),
        'auroc': np.mean([r['auroc'] for r in results])
    }

    std_metrics = {
        'accuracy': np.std([r['accuracy'] for r in results]),
        'f1': np.std([r['f1'] for r in results]),
        'precision': np.std([r['precision'] for r in results]),
        'recall': np.std([r['recall'] for r in results]),
        'specificity': np.std([r['specificity'] for r in results]),
        'auprc': np.std([r['auprc'] for r in results]),
        'auroc': np.std([r['auroc'] for r in results])
    }

    print('\n' + '=' * 50)
    print('交叉验证最终结果:')
    print('=' * 50)
    print(f"平均准确度: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"平均F1分数: {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"平均精确率: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"平均召回率: {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"平均特异性: {avg_metrics['specificity']:.4f} ± {std_metrics['specificity']:.4f}")
    print(f"平均AUPRC: {avg_metrics['auprc']:.4f} ± {std_metrics['auprc']:.4f}")
    print(f"平均AUROC: {avg_metrics['auroc']:.4f} ± {std_metrics['auroc']:.4f}")

    print('\n整体分类报告:')
    print(classification_report(all_y_true, all_y_pred, zero_division=0))

    print('\n' + '=' * 50)
    print('各折模型权重:')
    print('=' * 50)
    for i, fold_weight in enumerate(weights):
        print(f'\nFold {i + 1} 权重:')
        print(f"截距项 (bias): {fold_weight['intercept']:.4f}")
        for name, coef in zip(feature_names, fold_weight['coefficients']):
            print(f"{name}: {coef:.4f}")

    avg_intercept = np.mean([w['intercept'] for w in weights])
    avg_coefficients = np.mean([w['coefficients'] for w in weights], axis=0)

    print('\n' + '=' * 50)
    print('跨折平均权重:')
    print('=' * 50)
    print(f"平均截距项: {avg_intercept:.4f}")
    for name, coef in zip(feature_names, avg_coefficients):
        print(f"{name}: {coef:.4f}")

    if interp_precisions:
        plt.figure(figsize=(8, 6))
        mean_precision = np.mean(interp_precisions, axis=0)
        mean_auprc = auc(interp_recalls, mean_precision)

        for i, prec in enumerate(interp_precisions):
            plt.plot(interp_recalls, prec, alpha=0.3, label=f'Fold {i + 1}')

        plt.plot(interp_recalls, mean_precision, 'k-',
                 label=f'平均 (AUPRC = {mean_auprc:.2f})', linewidth=2)
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('平均精确率-召回率曲线')
        plt.legend()
        plt.savefig('average_pr_curve.png')
        plt.close()


# 加载数据
data = pd.read_csv('preparations/heart_output.csv')

# 检查数据
print("数据前5行:")
print(data.head())
print("\n类别分布:")
print(data['HeartDisease'].value_counts())

# 分离特征和目标
X = data.drop('HeartDisease', axis=1).values
y = data['HeartDisease'].values

# 运行训练和评估
train_test_split(X, y, splits=10)