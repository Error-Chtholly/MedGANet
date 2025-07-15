import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, average_precision_score,
                             precision_recall_curve, auc, roc_auc_score)
import copy
import matplotlib.pyplot as plt
from scipy import interpolate
import joblib  # 用于保存scaler


# TeLU激活函数
class TeLU(nn.Module):
    def __init__(self, alpha=0.15):
        super(TeLU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x >= 0, x, self.alpha * (torch.exp(x) - 1))


# 前馈神经网络
class FFNN(nn.Module):
    def __init__(self, input_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.telu1 = TeLU(alpha=0.15)
        self.fc2 = nn.Linear(32, 64)
        self.telu2 = TeLU(alpha=0.1)
        self.fc3 = nn.Linear(64, 4)  # 4个输出类别

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.telu1(x)
        x = self.fc2(x)
        x = self.telu2(x)
        x = self.fc3(x)
        return x


# 焦点损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def calculate_metrics(y_true, y_pred, y_scores):
    # 检查NaN值
    if np.isnan(y_scores).any():
        y_scores = np.nan_to_num(y_scores)

    # 初始化存储每个类别的指标
    class_metrics = []

    for class_idx in range(4):  # 4个类别
        # 确保y_true是numpy数组
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

        # 二分类指标计算
        y_true_class = (y_true_np == class_idx).astype(int)
        y_pred_class = (y_pred_np == class_idx).astype(int)
        y_scores_class = y_scores[:, class_idx]

        try:
            accuracy = accuracy_score(y_true_class, y_pred_class)
            f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
            precision = precision_score(y_true_class, y_pred_class, zero_division=0)
            recall = recall_score(y_true_class, y_pred_class, zero_division=0)

            # 处理AUPRC计算
            if len(np.unique(y_true_class)) > 1:
                auprc = average_precision_score(y_true_class, y_scores_class)
            else:
                auprc = 0.0

            # 处理AUROC计算
            if len(np.unique(y_true_class)) > 1:
                auroc = roc_auc_score(y_true_class, y_scores_class)
            else:
                auroc = 0.0

            class_metrics.append({
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auprc': auprc,
                'auroc': auroc
            })
        except Exception as e:
            print(f"计算类别{class_idx}指标时出错: {str(e)}")
            class_metrics.append({
                'accuracy': 0,
                'f1': 0,
                'precision': 0,
                'recall': 0,
                'auprc': 0,
                'auroc': 0
            })

    # 计算加权平均指标
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'auprc': average_precision_score(y_true, y_scores, average='weighted'),
        'auroc': roc_auc_score(y_true, y_scores, multi_class='ovr', average='weighted'),
        'class_metrics': class_metrics
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
        # 多分类PR曲线 - 使用每个类别的PR曲线
        precision = dict()
        recall = dict()
        auprc = dict()

        for i in range(4):  # 4个类别
            precision[i], recall[i], _ = precision_recall_curve(y_true == i, y_scores[:, i])
            auprc[i] = auc(recall[i], precision[i])

        plt.figure()
        for i in range(4):
            plt.plot(recall[i], precision[i], label=f'Class {i} (AUPRC = {auprc[i]:.2f})')

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


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=500, patience=20):
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        # 验证
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_scores = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                # 使用softmax获取概率
                scores = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(scores.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())

        # 检查NaN值
        if np.isnan(np.array(all_scores)).any():
            all_scores = np.nan_to_num(all_scores)

        # 计算指标
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        metrics = calculate_metrics(all_labels, all_preds, np.array(all_scores))

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f"Val Metrics - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
                  f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                  f"AUPRC: {metrics['auprc']:.4f}, AUROC: {metrics['auroc']:.4f}")

    # 加载最佳模型
    model.load_state_dict(best_model)
    return model, best_val_loss  # 返回最佳模型和对应的验证损失


def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            scores = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(scores.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    # 检查NaN值
    if np.isnan(np.array(all_scores)).any():
        all_scores = np.nan_to_num(all_scores)

    return all_labels, all_preds, np.array(all_scores)


def plot_class_metrics(metric_name, all_class_metrics, title, ylabel, filename):
    """绘制每个类别的特定指标"""
    plt.figure(figsize=(10, 6))
    class_data = [[] for _ in range(4)]  # 4个类别

    # 收集每个折的每个类别的指标
    for fold_metrics in all_class_metrics:
        for class_idx in range(4):
            class_data[class_idx].append(fold_metrics[class_idx][metric_name])

    # 绘制箱线图
    plt.boxplot(class_data, labels=[f'Class {i}' for i in range(4)])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()


def train_test_split(X, y, splits=10, epochs=500, batch_size=32, lr=0.001):
    # 检查数据
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: {np.bincount(y)}")

    # 处理可能的NaN值
    X = np.nan_to_num(X)
    y = np.nan_to_num(y).astype(int)

    k_fold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=2025)
    results = []
    all_class_metrics = []  # 存储所有折的每个类别的指标
    best_model_info = {'val_loss': float('inf'), 'model_state': None, 'fold': -1}

    # 存储所有折的PR曲线数据（插值后的）
    interp_precisions = [[] for _ in range(4)]  # 4个类别
    interp_recalls = np.linspace(0, 1, 100)  # 固定100个recall点

    for fold, (train_idx, test_idx) in enumerate(k_fold.split(X, y)):
        print(f'\n{"=" * 50}')
        print(f'Fold {fold + 1}/{splits}')
        print(f'{"=" * 50}')

        # 分割数据
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        # 创建数据集和数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # 初始化模型、损失函数和优化器
        model = FFNN(input_size=X.shape[1])
        criterion = FocalLoss(alpha=0.25, gamma=2)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练模型
        model, val_loss = train_model(model, train_loader, test_loader, criterion, optimizer,
                                      epochs=epochs, patience=20)

        # 检查是否为最佳模型
        if val_loss < best_model_info['val_loss']:
            best_model_info['val_loss'] = val_loss
            best_model_info['model_state'] = copy.deepcopy(model.state_dict())
            best_model_info['fold'] = fold + 1
            best_model_info['scaler'] = scaler  # 保存对应的scaler

        # 评估测试集
        y_true, y_pred, y_scores = evaluate_model(model, test_loader)
        metrics = calculate_metrics(y_true, y_pred, y_scores)
        all_class_metrics.append(metrics['class_metrics'])

        # 绘制并保存当前折的PR曲线
        precision, recall = plot_pr_curve(y_true, y_scores, fold + 1)

        # 插值PR曲线到固定长度
        if precision is not None and recall is not None:
            for i in range(4):
                interp_precision, _ = interpolate_pr_curve(precision[i], recall[i])
                interp_precisions[i].append(interp_precision)

        # 保存结果
        results.append(metrics)

        # 打印当前折的结果
        print(f'\nFold {fold + 1} Test Metrics:')
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"AUPRC: {metrics['auprc']:.4f}")
        print(f"AUROC: {metrics['auroc']:.4f}")

        # 打印每个类别的指标
        for class_idx in range(4):
            print(f"\nClass {class_idx} Metrics:")
            print(f"Accuracy: {metrics['class_metrics'][class_idx]['accuracy']:.4f}")
            print(f"F1 Score: {metrics['class_metrics'][class_idx]['f1']:.4f}")
            print(f"Precision: {metrics['class_metrics'][class_idx]['precision']:.4f}")
            print(f"Recall: {metrics['class_metrics'][class_idx]['recall']:.4f}")
            print(f"AUPRC: {metrics['class_metrics'][class_idx]['auprc']:.4f}")
            print(f"AUROC: {metrics['class_metrics'][class_idx]['auroc']:.4f}")

    # 保存最佳模型
    best_model = FFNN(input_size=X.shape[1])
    best_model.load_state_dict(best_model_info['model_state'])
    torch.save(best_model.state_dict(), 'cirrhosis_mlp.pth')

    # 保存对应的scaler
    joblib.dump(best_model_info['scaler'], 'scaler.pkl')

    print(
        f"\nSaved best model from fold {best_model_info['fold']} with val loss {best_model_info['val_loss']:.4f} as cirrhosis_mlp.pth")
    print("Saved corresponding scaler as scaler.pkl")

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

    # 计算每个类别的平均指标
    class_avg_metrics = []
    for class_idx in range(4):
        class_metrics = {
            'accuracy': np.mean([m[class_idx]['accuracy'] for m in all_class_metrics]),
            'f1': np.mean([m[class_idx]['f1'] for m in all_class_metrics]),
            'precision': np.mean([m[class_idx]['precision'] for m in all_class_metrics]),
            'recall': np.mean([m[class_idx]['recall'] for m in all_class_metrics]),
            'auprc': np.mean([m[class_idx]['auprc'] for m in all_class_metrics]),
            'auroc': np.mean([m[class_idx]['auroc'] for m in all_class_metrics])
        }
        class_avg_metrics.append(class_metrics)

    # 打印每个类别的平均指标
    for class_idx in range(4):
        print(f"\nClass {class_idx} Average Metrics:")
        print(f"Accuracy: {class_avg_metrics[class_idx]['accuracy']:.4f}")
        print(f"F1 Score: {class_avg_metrics[class_idx]['f1']:.4f}")
        print(f"Precision: {class_avg_metrics[class_idx]['precision']:.4f}")
        print(f"Recall: {class_avg_metrics[class_idx]['recall']:.4f}")
        print(f"AUPRC: {class_avg_metrics[class_idx]['auprc']:.4f}")
        print(f"AUROC: {class_avg_metrics[class_idx]['auroc']:.4f}")

    # 绘制平均PR曲线（使用插值后的数据）
    if interp_precisions[0]:
        plt.figure(figsize=(10, 8))
        for i in range(4):
            mean_precision = np.mean(interp_precisions[i], axis=0)
            mean_auprc = auc(interp_recalls, mean_precision)

            for prec in interp_precisions[i]:
                plt.plot(interp_recalls, prec, alpha=0.1, color=f'C{i}')

            plt.plot(interp_recalls, mean_precision, color=f'C{i}',
                     label=f'Class {i} (Mean AUPRC = {mean_auprc:.2f})', linewidth=2)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average Precision-Recall Curve by Class')
        plt.legend()
        plt.savefig('average_pr_curve_by_class.png')
        plt.close()

    # 绘制每个类别的各项指标
    metric_info = [
        ('accuracy', 'Accuracy by Class', 'Accuracy', 'class_accuracy.png'),
        ('f1', 'F1 Score by Class', 'F1 Score', 'class_f1.png'),
        ('precision', 'Precision by Class', 'Precision', 'class_precision.png'),
        ('recall', 'Recall by Class', 'Recall', 'class_recall.png'),
        ('auprc', 'AUPRC by Class', 'AUPRC', 'class_auprc.png'),
        ('auroc', 'AUROC by Class', 'AUROC', 'class_auroc.png')
    ]

    for metric, title, ylabel, filename in metric_info:
        plot_class_metrics(metric, all_class_metrics, title, ylabel, filename)


if __name__ == "__main__":
    # 加载数据
    data = pd.read_csv('preparations/cirrhosis_output.csv')  # 请替换为您的实际文件路径

    # 检查数据
    print("数据前5行:")
    print(data.head())
    print("\n类别分布:")
    print(data['Stage'].value_counts())

    # 分离特征和目标
    feature_cols = ['N_Days', 'Age', 'Bilirubin', 'Albumin', 'Copper', 'SGOT',
                    'Tryglicerides', 'Platelets', 'Prothrombin']
    X = data[feature_cols].values
    y = data['Stage'].values - 1  # 将类别转换为0-3

    # 转换为numpy数组
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    # 运行训练和评估
    train_test_split(X, y, splits=10, epochs=500, batch_size=32, lr=0.001)