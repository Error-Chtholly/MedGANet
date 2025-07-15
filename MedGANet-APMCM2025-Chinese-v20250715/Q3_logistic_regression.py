import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, average_precision_score,
                             roc_auc_score, confusion_matrix, classification_report)


# 1. 数据加载和预处理
def load_and_preprocess(filepath):
    # 加载数据
    data = pd.read_csv(filepath)

    # 显示数据信息
    print("=" * 50)
    print("数据概览:")
    print(f"总样本数: {len(data)}")
    print(f"特征数: {len(data.columns) - 1}")
    print("\n前5行数据:")
    print(data.head())
    print("\n标签分布:")
    print(data['Label'].value_counts(normalize=True))

    # 分离特征和标签
    X = data.drop('Label', axis=1)
    y = data['Label'].values

    return X, y


# 2. 评估指标计算
def calculate_metrics(y_true, y_pred, y_scores):
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
        metrics['auprc'] = 0
        metrics['auroc'] = 0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    return metrics


# 3. 模型训练和评估
def train_and_evaluate(X, y, n_splits=10):
    # 初始化结果存储
    results = []
    feature_names = X.columns.tolist()

    # 交叉验证
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2025)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        # 数据划分
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 模型训练
        model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear',
                                   random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)

        # 预测和评估
        y_pred = model.predict(X_test_scaled)
        y_scores = model.predict_proba(X_test_scaled)[:, 1]

        # 计算指标
        metrics = calculate_metrics(y_test, y_pred, y_scores)
        results.append(metrics)

        # 打印当前fold结果
        print(f"\nFold {fold + 1}/{n_splits} 结果:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")

    # 计算平均指标
    avg_metrics = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
    std_metrics = {k: np.std([r[k] for r in results]) for k in results[0].keys()}

    # 打印最终结果
    print("\n" + "=" * 50)
    print("交叉验证平均结果 (±标准差):")
    print("=" * 50)
    for metric in avg_metrics:
        print(f"{metric:12s}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")


# 4. 主执行流程
if __name__ == "__main__":
    # 数据路径
    data_path = 'preparations/Q3_data2.csv'

    # 加载和预处理数据
    X, y = load_and_preprocess(data_path)

    # 训练和评估模型
    train_and_evaluate(X, y, n_splits=10)