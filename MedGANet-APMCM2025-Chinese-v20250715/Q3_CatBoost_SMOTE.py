import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier  # 导入CatBoost
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, average_precision_score,
                             roc_auc_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline


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


# 3. 模型训练和评估（带SMOTE）
def train_and_evaluate_with_smote(X, y, n_splits=10):
    # 初始化结果存储
    results = []
    feature_names = X.columns.tolist()

    # 交叉验证
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2025)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        # 数据划分
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 创建SMOTE+标准化+CatBoost的pipeline
        pipeline = make_pipeline(
            StandardScaler(),
            SMOTE(random_state=42, k_neighbors=5),
            CatBoostClassifier(
                iterations=100,  # 树的数量
                learning_rate=0.1,  # 学习率
                depth=6,  # 树深度
                l2_leaf_reg=3,  # L2正则化系数
                random_seed=42,
                auto_class_weights='Balanced',  # 自动类别权重
                thread_count=-1,  # 使用所有CPU核心
                verbose=0  # 不输出训练日志
            )
        )

        # 训练模型
        pipeline.fit(X_train, y_train)

        # 预测和评估
        y_pred = pipeline.predict(X_test)
        y_scores = pipeline.predict_proba(X_test)[:, 1]  # 获取正类的概率

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

    print("\n" + "=" * 50)
    print("使用CatBoost（带SMOTE过采样）进行模型训练...")
    print("=" * 50)

    # 训练和评估模型
    train_and_evaluate_with_smote(X, y, n_splits=10)