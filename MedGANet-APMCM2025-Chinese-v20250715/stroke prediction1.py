import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import copy

# 检查并确保numpy可用
try:
    import numpy as np
except ImportError:
    raise RuntimeError("NumPy is required but not available. Please install it.")


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


# 训练模型函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=500, patience=20,
                model_save_path='best_model_stroke.pth'):
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
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)

        # 早停和模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # 加载最佳模型
    model.load_state_dict(best_model)
    return model


# 主训练和评估流程
def train_test_split(X, y, splits=10, epochs=500, batch_size=512, lr=0.001):
    k_fold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=2025)
    results = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(X, y)):
        print(f'\nFold {fold + 1}/{splits}')

        # 分割数据
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)

        # 创建数据集和数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 初始化模型、损失函数和优化器
        model = FFNN(input_size=X.shape[1])
        criterion = FocalLoss(alpha=0.25, gamma=2)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        # 训练模型
        model = train_model(model, train_loader, val_loader, criterion, optimizer,
                            epochs=epochs, patience=20,
                            model_save_path=f'best_model_fold{fold + 1}.pth')

        # 评估验证集
        model.eval()
        all_preds = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())

        val_acc = accuracy_score(y_val, all_preds)
        print(f'Fold {fold + 1} Validation Accuracy: {val_acc:.4f}')
        results.append(val_acc)

    # 打印最终结果
    print('\nCross-validation results:')
    for i, acc in enumerate(results):
        print(f'Fold {i + 1}: {acc:.4f}')
    print(f'Mean Accuracy: {np.mean(results):.4f} ± {np.std(results):.4f}')


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