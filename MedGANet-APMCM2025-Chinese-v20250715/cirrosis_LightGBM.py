import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, average_precision_score,
                             precision_recall_curve, auc, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
import warnings
from sklearn.metrics import confusion_matrix

# 忽略警告
warnings.filterwarnings('ignore', category=UserWarning)

# 设置字体为黑体，确保中文可见
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_metrics(y_true, y_pred, y_scores):
    """计算评估指标"""
    # 检查 NaN 值
    if np.isnan(y_scores).any():
        y_scores = np.nan_to_num(y_scores)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    # 计算 AUPRC 和 AUROC
    try:
        metrics['auprc'] = average_precision_score(y_true, y_scores, average='weighted')
    except:
        print("Could not calculate AUPRC, using default value 0")
        metrics['auprc'] = 0

    try:
        metrics['auroc'] = roc_auc_score(y_true, y_scores, multi_class='ovr', average='weighted')
    except:
        print("Could not calculate AUROC, using default value 0")
        metrics['auroc'] = 0

    return metrics


def interpolate_pr_curve(precision, recall):
    """对 PR 曲线进行插值"""
    f = interpolate.interp1d(recall, precision, bounds_error=False, fill_value=(1.0, 0.0))
    new_recall = np.linspace(0, 1, 100)
    new_precision = f(new_recall)
    return new_precision, new_recall


def plot_confusion_matrix(y_true, y_pred, fold, dpi=720):
    """绘制正方形混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # 百分比表示

    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(cm_percentage, annot=False, fmt='.2f', cmap='Blues', square=True, cbar=False,
                     linewidths=2, linecolor='black')

    # 在每个格子中显示个数和百分比
    for i in range(4):
        for j in range(4):
            # 判断字体颜色，深色背景用白色字体，浅色背景用黑色字体
            text_color = 'white' if cm_percentage[i, j] > 50 else 'black'

            # 将个数和百分比分行显示，个数在上，百分比在下
            ax.text(j + 0.5, i + 0.5, f'{cm[i, j]}\n({cm_percentage[i, j]:.2f}%)',
                    color=text_color, ha='center', va='center', fontsize=14, fontweight='bold')

    # 添加中文标签
    plt.xlabel('预测类别', fontsize=16, fontweight='bold')
    plt.ylabel('实际类别', fontsize=16, fontweight='bold')
    plt.xticks(ticks=np.arange(4) + 0.5, labels=np.arange(1, 5), fontsize=14, fontweight='bold')
    plt.yticks(ticks=np.arange(4) + 0.5, labels=np.arange(1, 5), fontsize=14, fontweight='bold')

    # 调整布局，减少空白边缘
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # 保存混淆矩阵图
    plt.savefig(f'CI_LG_best_fold_confusion_matrix_fold{fold}.png', dpi=dpi)
    plt.close()


def plot_pr_curve(y_true, y_scores, fold):
    """绘制 Precision-Recall 曲线"""
    try:
        # 多类别 PR 曲线 - 使用一对多的方式绘制每个类别
        precision = dict()
        recall = dict()
        auprc = dict()

        for i in range(4):  # 4 类别
            precision[i], recall[i], _ = precision_recall_curve(y_true == i, y_scores[:, i])
            auprc[i] = auc(recall[i], precision[i])

        plt.figure()
        for i in range(4):
            plt.plot(recall[i], precision[i], label=f'类别 {i} (AUPRC = {auprc[i]:.2f})')

        plt.xlabel('召回率', fontsize=14, fontweight='bold')
        plt.ylabel('精确率', fontsize=14, fontweight='bold')
        plt.title(f'Precision-Recall 曲线 (折 {fold})', fontsize=16, fontweight='bold')
        plt.legend()
        plt.savefig(f'pr_curve_fold{fold}.png')
        plt.close()
        return precision, recall
    except Exception as e:
        print(f"Could not plot PR curve for Fold {fold}: {str(e)}")
        return None, None


def train_test_split(X, y, splits=10, batch_size=32):
    """训练测试数据分割并进行交叉验证"""
    # 检查数据
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # 处理可能的 NaN 值
    X = np.nan_to_num(X)
    y = np.nan_to_num(y).astype(int)

    k_fold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=2025)
    results = []

    best_fold_metrics = None
    best_fold = -1

    for fold, (train_idx, test_idx) in enumerate(k_fold.split(X, y)):
        print(f'\n{"=" * 50}')
        print(f'Fold {fold + 1}/{splits}')
        print(f'{"=" * 50}')

        # 数据切分
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 创建 LightGBM 数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # LightGBM 参数
        params = {
            'objective': 'multiclass',
            'num_class': 4,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'random_state': 42
        }

        # 训练模型
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                       lgb.log_evaluation(period=50)]
        )

        # 预测
        y_pred = model.predict(X_test)
        y_pred_class = np.argmax(y_pred, axis=1)
        metrics = calculate_metrics(y_test, y_pred_class, y_pred)

        # 跟踪最佳折叠（根据 AUROC）
        if best_fold_metrics is None or metrics['auroc'] > best_fold_metrics['auroc']:
            best_fold_metrics = metrics
            best_fold = fold

        # 保存每一折的结果
        results.append(metrics)

        # 打印当前折的结果
        print(f'\nFold {fold + 1} Test Metrics:')
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"AUPRC: {metrics['auprc']:.4f}")
        print(f"AUROC: {metrics['auroc']:.4f}")

    # 计算并打印平均指标
    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in results]),
        'f1': np.mean([r['f1'] for r in results]),
        'precision': np.mean([r['precision'] for r in results]),
        'recall': np.mean([r['recall'] for r in results]),
        'auprc': np.mean([r['auprc'] for r in results]),
        'auroc': np.mean([r['auroc'] for r in results])
    }

    std_metrics = {
        'accuracy': np.std([r['accuracy'] for r in results]),
        'f1': np.std([r['f1'] for r in results]),
        'precision': np.std([r['precision'] for r in results]),
        'recall': np.std([r['recall'] for r in results]),
        'auprc': np.std([r['auprc'] for r in results]),
        'auroc': np.std([r['auroc'] for r in results])
    }

    print('\n' + '=' * 50)
    print('Final Cross-Validation Results:')
    print('=' * 50)
    print(f"Average Accuracy: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"Average F1 Score: {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"Average Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"Average Recall: {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"Average AUPRC: {avg_metrics['auprc']:.4f} ± {std_metrics['auprc']:.4f}")
    print(f"Average AUROC: {avg_metrics['auroc']:.4f} ± {std_metrics['auroc']:.4f}")

    # 绘制最佳折的混淆矩阵
    print(f"Plotting confusion matrix for best fold: {best_fold + 1}")
    plot_confusion_matrix(y_test, y_pred_class, best_fold + 1, dpi=720)


# 加载数据 - 修改为你的实际文件路径
data = pd.read_csv('preparations/cirrhosis_output.csv')  # 替换为你的实际文件路径

# 查看数据
print("First 5 rows of data:")
print(data.head())
print("\nClass distribution:")
print(data['Stage'].value_counts())

# 分割特征和标签
feature_cols = ['N_Days', 'Age', 'Bilirubin', 'Albumin', 'Copper', 'SGOT',
                'Tryglicerides', 'Platelets', 'Prothrombin']
X = data[feature_cols].values
y = data['Stage'].values - 1  # 转换标签为 0-3

# 转换为 numpy 数组
X = X.astype(np.float32)
y = y.astype(np.int64)

# 运行训练和评估
train_test_split(X, y, splits=10, batch_size=32)