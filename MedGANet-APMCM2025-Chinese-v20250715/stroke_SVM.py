import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, average_precision_score,
                             precision_recall_curve, auc, confusion_matrix)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# 解决PR曲线长度不一致的问题
def interpolate_pr_curve(precision, recall, num_points=100):
    """将PR曲线插值到固定长度的点"""
    if len(precision) < 2 or len(recall) < 2:
        return np.linspace(0, 1, num_points), np.linspace(1, 0, num_points)

    # 确保recall是单调递增的
    sorted_indices = np.argsort(recall)
    recall = np.array(recall)[sorted_indices]
    precision = np.array(precision)[sorted_indices]

    # 插值
    f = interpolate.interp1d(recall, precision, bounds_error=False, fill_value=(precision[0], precision[-1]))
    new_recall = np.linspace(0, 1, num_points)
    new_precision = f(new_recall)
    return new_precision, new_recall


def calculate_metrics(y_true, y_pred, y_scores):
    """计算评估指标"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'auprc': average_precision_score(y_true, y_scores)
    }

    # 添加混淆矩阵信息
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'confusion_matrix': {
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
        },
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    })
    return metrics


def plot_pr_curve(y_true, y_scores, fold, save_path=None):
    """绘制PR曲线并返回插值后的数据"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)

    # 插值到固定长度
    interp_precision, interp_recall = interpolate_pr_curve(precision, recall)

    plt.figure()
    plt.plot(recall, precision, label=f'Fold {fold} (AUPRC = {auprc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve (Fold {fold})')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

    return interp_precision, interp_recall, auprc


def train_svm_model(X_train, y_train, X_val, y_val):
    """训练SVM模型"""
    svm = SVC(probability=True, random_state=42)

    # 使用网格搜索调参来寻找最优参数
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")

    # 使用最优参数训练SVM
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    return best_model


def cross_validate(X, y, n_splits=10):
    """执行交叉验证"""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    all_interp_precisions = []
    interp_recall = np.linspace(0, 1, 100)  # 固定100个recall点

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        print(f'\n{"=" * 40} Fold {fold}/{n_splits} {"=" * 40}')

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # SMOTE过采样
        smote = SMOTE(random_state=42, k_neighbors=min(5, sum(y_train == 1) - 1))
        X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

        # 训练SVM模型
        model = train_svm_model(X_res, y_res, X_test_scaled, y_test)

        # 预测
        y_scores = model.predict_proba(X_test_scaled)[:, 1]  # 取正类的预测概率
        y_pred = (y_scores >= 0.5).astype(int)

        # 计算指标
        metrics = calculate_metrics(y_test, y_pred, y_scores)
        results.append(metrics)

        # PR曲线
        interp_precision, _, _ = plot_pr_curve(
            y_test, y_scores, fold,
            save_path=f'pr_curve_fold{fold}.png'
        )
        all_interp_precisions.append(interp_precision)

        # 打印结果
        print(f"\nFold {fold} Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall/Sensitivity: {metrics['recall']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"AUPRC: {metrics['auprc']:.4f}")
        print(f"Confusion Matrix: {metrics['confusion_matrix']}")

    # 计算平均指标
    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in results]),
        'f1': np.mean([r['f1'] for r in results]),
        'precision': np.mean([r['precision'] for r in results]),
        'recall': np.mean([r['recall'] for r in results]),
        'specificity': np.mean([r['specificity'] for r in results]),
        'auprc': np.mean([r['auprc'] for r in results]),
    }

    std_metrics = {
        'accuracy': np.std([r['accuracy'] for r in results]),
        'f1': np.std([r['f1'] for r in results]),
        'precision': np.std([r['precision'] for r in results]),
        'recall': np.std([r['recall'] for r in results]),
        'specificity': np.std([r['specificity'] for r in results]),
        'auprc': np.std([r['auprc'] for r in results]),
    }

    # 绘制平均PR曲线
    if all_interp_precisions:
        mean_precision = np.mean(all_interp_precisions, axis=0)
        mean_auprc = auc(interp_recall, mean_precision)

        plt.figure(figsize=(10, 6))
        for i, prec in enumerate(all_interp_precisions, 1):
            plt.plot(interp_recall, prec, alpha=0.2, label=f'Fold {i}')

        plt.plot(interp_recall, mean_precision, 'r-',
                 linewidth=3, label=f'Mean (AUPRC = {mean_auprc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average Precision-Recall Curve')
        plt.legend()
        plt.savefig('average_pr_curve.png')
        plt.close()

    return avg_metrics, std_metrics


# 主程序
def main():
    # 加载数据
    data = pd.read_csv('preparations/stroke_output.csv')

    # 预处理
    categorical_cols = ['ever_married', 'work_type', 'smoking_status']
    data[categorical_cols] = data[categorical_cols].astype('category')
    X = data.drop('stroke', axis=1)
    y = data['stroke'].values

    # 独热编码
    X = pd.get_dummies(X, columns=categorical_cols)
    X = X.values.astype(np.float32)

    # 执行交叉验证
    print("Starting cross-validation...")
    avg_metrics, std_metrics = cross_validate(X, y, n_splits=10)

    # 打印最终结果
    print('\n' + '=' * 50)
    print('Final Cross-Validation Results:')
    print('=' * 50)
    print(f"Average Accuracy: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"Average F1 Score: {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"Average Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"Average Recall/Sensitivity: {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"Average Specificity: {avg_metrics['specificity']:.4f} ± {std_metrics['specificity']:.4f}")
    print(f"Average AUPRC: {avg_metrics['auprc']:.4f} ± {std_metrics['auprc']:.4f}")


if __name__ == '__main__':
    main()