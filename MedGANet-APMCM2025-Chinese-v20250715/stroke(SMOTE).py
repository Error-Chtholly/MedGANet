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
                             precision_recall_curve, auc)
from imblearn.over_sampling import SMOTE
import copy
import matplotlib.pyplot as plt


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
        self.fc3 = nn.Linear(64, 2)

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
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'auprc': average_precision_score(y_true, y_scores[:, 1])
    }
    return metrics


def plot_pr_curve(y_true, y_scores, fold):
    precision, recall, _ = precision_recall_curve(y_true, y_scores[:, 1])
    auprc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f'Fold {fold} (AUPRC = {auprc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(f'pr_curve_fold{fold}.png')
    plt.close()


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
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())

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
                  f"AUPRC: {metrics['auprc']:.4f}")

    # 加载最佳模型
    model.load_state_dict(best_model)
    return model


def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    return all_labels, all_preds, np.array(all_scores)


def train_test_split(X, y, splits=10, epochs=500, batch_size=512, lr=0.001):
    k_fold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=2025)
    results = []

    # 存储所有折的PR曲线数据
    all_precisions = []
    all_recalls = []

    for fold, (train_idx, test_idx) in enumerate(k_fold.split(X, y)):
        print(f'\n{"=" * 50}')
        print(f'Fold {fold + 1}/{splits}')
        print(f'{"=" * 50}')

        # 分割数据
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 1. 先标准化数据
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 2. 应用SMOTE过采样（仅对训练数据）
        # 检查少数类样本数量是否足够
        min_samples = sum(y_train == 1)
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1

        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train_res)
        y_train_tensor = torch.LongTensor(y_train_res)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
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

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        # 训练模型
        model = train_model(model, train_loader, test_loader, criterion, optimizer,
                            epochs=epochs, patience=20)

        # 评估测试集
        y_true, y_pred, y_scores = evaluate_model(model, test_loader)
        metrics = calculate_metrics(y_true, y_pred, y_scores)

        # 保存PR曲线数据
        precision, recall, _ = precision_recall_curve(y_true, y_scores[:, 1])
        all_precisions.append(precision)
        all_recalls.append(recall)

        # 绘制当前折的PR曲线
        plot_pr_curve(y_true, y_scores, fold + 1)

        # 保存结果
        results.append(metrics)

        # 打印当前折的结果
        print(f'\nFold {fold + 1} Test Metrics:')
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"AUPRC: {metrics['auprc']:.4f}")

    # 计算并打印平均指标
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

    # 绘制平均PR曲线
    plt.figure(figsize=(8, 6))
    for i in range(splits):
        plt.plot(all_recalls[i], all_precisions[i], alpha=0.3, label=f'Fold {i + 1}')

    mean_precision = np.mean([p for p in all_precisions], axis=0)
    mean_recall = np.mean([r for r in all_recalls], axis=0)
    mean_auprc = auc(mean_recall, mean_precision)

    plt.plot(mean_recall, mean_precision, 'k-',
             label=f'Mean (AUPRC = {mean_auprc:.2f})', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Average Precision-Recall Curve')
    plt.legend()
    plt.savefig('average_pr_curve.png')
    plt.close()


# 加载数据
data = pd.read_csv('preparations/stroke_output.csv')

# 预处理分类变量
categorical_cols = ['ever_married', 'work_type', 'smoking_status']
data[categorical_cols] = data[categorical_cols].astype('category')

# 分离特征和目标
X = data.drop('stroke', axis=1)
y = data['stroke'].values

# 对分类变量进行独热编码
X = pd.get_dummies(X, columns=categorical_cols)

# 转换为numpy数组
X = X.values.astype(np.float32)

# 运行训练和评估
train_test_split(X, y, splits=10, epochs=500, batch_size=512, lr=0.001)