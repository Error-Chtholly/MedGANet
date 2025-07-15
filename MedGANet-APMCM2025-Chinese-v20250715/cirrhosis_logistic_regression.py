import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix,
                             classification_report, roc_auc_score,
                             roc_curve, auc, cohen_kappa_score,
                             matthews_corrcoef, hamming_loss,
                             jaccard_score, log_loss,
                             balanced_accuracy_score)
import matplotlib.pyplot as plt
from itertools import cycle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def calculate_metrics(y_true, y_pred, y_scores):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'hamming_loss': hamming_loss(y_true, y_pred),
        'jaccard_macro': jaccard_score(y_true, y_pred, average='macro'),
        'jaccard_weighted': jaccard_score(y_true, y_pred, average='weighted'),
        'log_loss': log_loss(y_true, y_scores)
    }

    try:
        # 多分类的AUC需要指定multi_class和average参数
        if len(np.unique(y_true)) > 2:
            metrics['auroc_ovr'] = roc_auc_score(y_true, y_scores, multi_class='ovr', average='macro')
            metrics['auroc_ovo'] = roc_auc_score(y_true, y_scores, multi_class='ovo', average='macro')
        else:
            metrics['auroc'] = roc_auc_score(y_true, y_scores[:, 1])
    except Exception as e:
        print(f"无法计算AUROC: {str(e)}")
        metrics['auroc_ovr'] = 0
        metrics['auroc_ovo'] = 0

    # 计算每个类别的指标
    cm = confusion_matrix(y_true, y_pred)
    class_metrics = {}
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        class_metrics[f'class_{i + 1}_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        class_metrics[f'class_{i + 1}_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_metrics[f'class_{i + 1}_precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        class_metrics[f'class_{i + 1}_recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_metrics[f'class_{i + 1}_f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    metrics.update(class_metrics)

    return metrics


def plot_roc_curve(y_true, y_scores, fold, n_classes):
    try:
        # 为每个类别计算ROC曲线
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # 计算微平均ROC曲线
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # 计算宏平均ROC曲线
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.figure(figsize=(8, 6))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'类别 {i + 1} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title(f'Fold {fold} ROC曲线')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_fold{fold}.png')
        plt.close()

        return roc_auc
    except Exception as e:
        print(f"无法绘制Fold {fold}的ROC曲线: {str(e)}")
        return None


def train_test_split(X, y, splits=10):
    print(f"数据形状: X={X.shape}, y={y.shape}")

    # 确保y是整数类型
    y = y.astype(int)
    print(f"类别分布: {np.bincount(y)}")

    try:
        feature_names = data.drop('Stage', axis=1).columns.tolist()
    except:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

    X = np.nan_to_num(X)
    y = np.nan_to_num(y).astype(int)

    # 类别数
    n_classes = len(np.unique(y))

    k_fold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=2025)
    results = []
    weights = []
    all_y_true = []
    all_y_pred = []
    all_y_scores = []

    for fold, (train_idx, test_idx) in enumerate(k_fold.split(X, y)):
        print(f'\n{"=" * 50}')
        print(f'Fold {fold + 1}/{splits}')
        print(f'{"=" * 50}')

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear',
                                   random_state=42, multi_class='ovr')
        model.fit(X_train, y_train)

        fold_weights = {
            'intercept': model.intercept_,
            'coefficients': model.coef_
        }
        weights.append(fold_weights)

        y_scores = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_scores.extend(y_scores)

        metrics = calculate_metrics(y_test, y_pred, y_scores)
        roc_auc = plot_roc_curve(np.array(y_test), np.array(y_scores), fold + 1, n_classes)

        if roc_auc is not None:
            metrics.update({
                'auroc_micro': roc_auc['micro'],
                'auroc_macro': roc_auc['macro']
            })
            for i in range(n_classes):
                metrics[f'auroc_class{i + 1}'] = roc_auc[i]

        results.append(metrics)

        print(f'\nFold {fold + 1} 测试集指标:')
        print(f"准确度: {metrics['accuracy']:.4f}")
        print(f"平衡准确度: {metrics['balanced_accuracy']:.4f}")
        print(f"宏平均F1分数: {metrics['f1_macro']:.4f}")
        print(f"加权F1分数: {metrics['f1_weighted']:.4f}")
        print(f"宏平均精确率: {metrics['precision_macro']:.4f}")
        print(f"加权精确率: {metrics['precision_weighted']:.4f}")
        print(f"宏平均召回率: {metrics['recall_macro']:.4f}")
        print(f"加权召回率: {metrics['recall_weighted']:.4f}")
        print(f"Kappa系数: {metrics['kappa']:.4f}")
        print(f"马修斯相关系数(MCC): {metrics['mcc']:.4f}")
        print(f"汉明损失: {metrics['hamming_loss']:.4f}")
        print(f"宏平均杰卡德相似系数: {metrics['jaccard_macro']:.4f}")
        print(f"加权杰卡德相似系数: {metrics['jaccard_weighted']:.4f}")
        print(f"对数损失: {metrics['log_loss']:.4f}")
        print(f"OVR AUROC: {metrics['auroc_ovr']:.4f}")
        print(f"OVO AUROC: {metrics['auroc_ovo']:.4f}")
        if 'auroc_macro' in metrics:
            print(f"宏平均AUROC: {metrics['auroc_macro']:.4f}")

        for i in range(n_classes):
            print(f"\n类别 {i + 1} 指标:")
            print(f"精确率: {metrics[f'class_{i + 1}_precision']:.4f}")
            print(f"召回率: {metrics[f'class_{i + 1}_recall']:.4f}")
            print(f"F1分数: {metrics[f'class_{i + 1}_f1']:.4f}")
            print(f"特异性: {metrics[f'class_{i + 1}_specificity']:.4f}")
            print(f"敏感度: {metrics[f'class_{i + 1}_sensitivity']:.4f}")
            if f'auroc_class{i + 1}' in metrics:
                print(f"AUC: {metrics[f'auroc_class{i + 1}']:.4f}")

        print("\n分类报告:")
        print(classification_report(y_test, y_pred, zero_division=0))

    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in results]),
        'balanced_accuracy': np.mean([r['balanced_accuracy'] for r in results]),
        'f1_macro': np.mean([r['f1_macro'] for r in results]),
        'f1_weighted': np.mean([r['f1_weighted'] for r in results]),
        'precision_macro': np.mean([r['precision_macro'] for r in results]),
        'precision_weighted': np.mean([r['precision_weighted'] for r in results]),
        'recall_macro': np.mean([r['recall_macro'] for r in results]),
        'recall_weighted': np.mean([r['recall_weighted'] for r in results]),
        'kappa': np.mean([r['kappa'] for r in results]),
        'mcc': np.mean([r['mcc'] for r in results]),
        'hamming_loss': np.mean([r['hamming_loss'] for r in results]),
        'jaccard_macro': np.mean([r['jaccard_macro'] for r in results]),
        'jaccard_weighted': np.mean([r['jaccard_weighted'] for r in results]),
        'log_loss': np.mean([r['log_loss'] for r in results]),
        'auroc_ovr': np.mean([r['auroc_ovr'] for r in results]),
        'auroc_ovo': np.mean([r['auroc_ovo'] for r in results]),
    }

    if 'auroc_macro' in results[0]:
        avg_metrics['auroc_macro'] = np.mean([r['auroc_macro'] for r in results])

    for i in range(n_classes):
        avg_metrics[f'class_{i + 1}_specificity'] = np.mean([r[f'class_{i + 1}_specificity'] for r in results])
        avg_metrics[f'class_{i + 1}_sensitivity'] = np.mean([r[f'class_{i + 1}_sensitivity'] for r in results])
        avg_metrics[f'class_{i + 1}_precision'] = np.mean([r[f'class_{i + 1}_precision'] for r in results])
        avg_metrics[f'class_{i + 1}_recall'] = np.mean([r[f'class_{i + 1}_recall'] for r in results])
        avg_metrics[f'class_{i + 1}_f1'] = np.mean([r[f'class_{i + 1}_f1'] for r in results])
        if f'auroc_class{i + 1}' in results[0]:
            avg_metrics[f'auroc_class{i + 1}'] = np.mean([r[f'auroc_class{i + 1}'] for r in results])

    std_metrics = {
        'accuracy': np.std([r['accuracy'] for r in results]),
        'balanced_accuracy': np.std([r['balanced_accuracy'] for r in results]),
        'f1_macro': np.std([r['f1_macro'] for r in results]),
        'f1_weighted': np.std([r['f1_weighted'] for r in results]),
        'precision_macro': np.std([r['precision_macro'] for r in results]),
        'precision_weighted': np.std([r['precision_weighted'] for r in results]),
        'recall_macro': np.std([r['recall_macro'] for r in results]),
        'recall_weighted': np.std([r['recall_weighted'] for r in results]),
        'kappa': np.std([r['kappa'] for r in results]),
        'mcc': np.std([r['mcc'] for r in results]),
        'hamming_loss': np.std([r['hamming_loss'] for r in results]),
        'jaccard_macro': np.std([r['jaccard_macro'] for r in results]),
        'jaccard_weighted': np.std([r['jaccard_weighted'] for r in results]),
        'log_loss': np.std([r['log_loss'] for r in results]),
        'auroc_ovr': np.std([r['auroc_ovr'] for r in results]),
        'auroc_ovo': np.std([r['auroc_ovo'] for r in results]),
    }

    if 'auroc_macro' in results[0]:
        std_metrics['auroc_macro'] = np.std([r['auroc_macro'] for r in results])

    for i in range(n_classes):
        std_metrics[f'class_{i + 1}_specificity'] = np.std([r[f'class_{i + 1}_specificity'] for r in results])
        std_metrics[f'class_{i + 1}_sensitivity'] = np.std([r[f'class_{i + 1}_sensitivity'] for r in results])
        std_metrics[f'class_{i + 1}_precision'] = np.std([r[f'class_{i + 1}_precision'] for r in results])
        std_metrics[f'class_{i + 1}_recall'] = np.std([r[f'class_{i + 1}_recall'] for r in results])
        std_metrics[f'class_{i + 1}_f1'] = np.std([r[f'class_{i + 1}_f1'] for r in results])
        if f'auroc_class{i + 1}' in results[0]:
            std_metrics[f'auroc_class{i + 1}'] = np.std([r[f'auroc_class{i + 1}'] for r in results])

    print('\n' + '=' * 50)
    print('交叉验证最终结果:')
    print('=' * 50)
    print(f"平均准确度: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"平均平衡准确度: {avg_metrics['balanced_accuracy']:.4f} ± {std_metrics['balanced_accuracy']:.4f}")
    print(f"平均宏F1分数: {avg_metrics['f1_macro']:.4f} ± {std_metrics['f1_macro']:.4f}")
    print(f"平均加权F1分数: {avg_metrics['f1_weighted']:.4f} ± {std_metrics['f1_weighted']:.4f}")
    print(f"平均宏精确率: {avg_metrics['precision_macro']:.4f} ± {std_metrics['precision_macro']:.4f}")
    print(f"平均加权精确率: {avg_metrics['precision_weighted']:.4f} ± {std_metrics['precision_weighted']:.4f}")
    print(f"平均宏召回率: {avg_metrics['recall_macro']:.4f} ± {std_metrics['recall_macro']:.4f}")
    print(f"平均加权召回率: {avg_metrics['recall_weighted']:.4f} ± {std_metrics['recall_weighted']:.4f}")
    print(f"平均Kappa系数: {avg_metrics['kappa']:.4f} ± {std_metrics['kappa']:.4f}")
    print(f"平均马修斯相关系数(MCC): {avg_metrics['mcc']:.4f} ± {std_metrics['mcc']:.4f}")
    print(f"平均汉明损失: {avg_metrics['hamming_loss']:.4f} ± {std_metrics['hamming_loss']:.4f}")
    print(f"平均宏杰卡德相似系数: {avg_metrics['jaccard_macro']:.4f} ± {std_metrics['jaccard_macro']:.4f}")
    print(f"平均加权杰卡德相似系数: {avg_metrics['jaccard_weighted']:.4f} ± {std_metrics['jaccard_weighted']:.4f}")
    print(f"平均对数损失: {avg_metrics['log_loss']:.4f} ± {std_metrics['log_loss']:.4f}")
    print(f"平均OVR AUROC: {avg_metrics['auroc_ovr']:.4f} ± {std_metrics['auroc_ovr']:.4f}")
    print(f"平均OVO AUROC: {avg_metrics['auroc_ovo']:.4f} ± {std_metrics['auroc_ovo']:.4f}")
    if 'auroc_macro' in avg_metrics:
        print(f"平均宏AUROC: {avg_metrics['auroc_macro']:.4f} ± {std_metrics['auroc_macro']:.4f}")

    for i in range(n_classes):
        print(f"\n类别 {i + 1} 平均指标:")
        print(f"精确率: {avg_metrics[f'class_{i + 1}_precision']:.4f} ± {std_metrics[f'class_{i + 1}_precision']:.4f}")
        print(f"召回率: {avg_metrics[f'class_{i + 1}_recall']:.4f} ± {std_metrics[f'class_{i + 1}_recall']:.4f}")
        print(f"F1分数: {avg_metrics[f'class_{i + 1}_f1']:.4f} ± {std_metrics[f'class_{i + 1}_f1']:.4f}")
        print(
            f"特异性: {avg_metrics[f'class_{i + 1}_specificity']:.4f} ± {std_metrics[f'class_{i + 1}_specificity']:.4f}")
        print(
            f"敏感度: {avg_metrics[f'class_{i + 1}_sensitivity']:.4f} ± {std_metrics[f'class_{i + 1}_sensitivity']:.4f}")
        if f'auroc_class{i + 1}' in avg_metrics:
            print(f"AUC: {avg_metrics[f'auroc_class{i + 1}']:.4f} ± {std_metrics[f'auroc_class{i + 1}']:.4f}")

    print('\n整体分类报告:')
    print(classification_report(all_y_true, all_y_pred, zero_division=0))

    print('\n' + '=' * 50)
    print('各折模型权重:')
    print('=' * 50)
    for i, fold_weight in enumerate(weights):
        print(f'\nFold {i + 1} 权重:')
        for class_idx in range(n_classes):
            print(f"\n类别 {class_idx + 1}:")
            print(f"截距项 (bias): {fold_weight['intercept'][class_idx]:.4f}")
            for name, coef in zip(feature_names, fold_weight['coefficients'][class_idx]):
                print(f"{name}: {coef:.4f}")

    avg_intercept = np.mean([w['intercept'] for w in weights], axis=0)
    avg_coefficients = np.mean([w['coefficients'] for w in weights], axis=0)

    print('\n' + '=' * 50)
    print('跨折平均权重:')
    print('=' * 50)
    for class_idx in range(n_classes):
        print(f"\n类别 {class_idx + 1}:")
        print(f"平均截距项: {avg_intercept[class_idx]:.4f}")
        for name, coef in zip(feature_names, avg_coefficients[class_idx]):
            print(f"{name}: {coef:.4f}")


# 加载数据
data = pd.read_csv('preparations/cirrhosis_output.csv')  # 替换为您的实际文件路径

# 检查数据
print("数据前5行:")
print(data.head())
print("\n类别分布:")
print(data['Stage'].value_counts())

# 分离特征和目标
X = data.drop('Stage', axis=1).values
y = data['Stage'].values

# 将类别从1-4转换为0-3
y = (y - 1).astype(int)  # 确保转换为整数类型

# 运行训练和评估
train_test_split(X, y, splits=10)