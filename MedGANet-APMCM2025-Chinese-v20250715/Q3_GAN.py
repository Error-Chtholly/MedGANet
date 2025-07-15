import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, average_precision_score,
                             roc_auc_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from copy import deepcopy


# 自定义TeLU激活函数
class TeLU(nn.Module):
    def __init__(self, alpha=0.15):
        super(TeLU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x >= 0, x, self.alpha * (torch.exp(x) - 1))


# GAN生成器
class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.telu1 = TeLU(alpha=0.15)
        self.fc2 = nn.Linear(64, 32)
        self.telu2 = TeLU(alpha=0.1)
        self.fc3 = nn.Linear(32, input_size)  # 输出与原始特征相同维度

    def forward(self, x):
        x = self.fc1(x)
        x = self.telu1(x)
        x = self.fc2(x)
        x = self.telu2(x)
        x = self.fc3(x)
        return x


# GAN判别器
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.telu1 = TeLU(alpha=0.15)
        self.fc2 = nn.Linear(64, 32)
        self.telu2 = TeLU(alpha=0.1)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.telu1(x)
        x = self.fc2(x)
        x = self.telu2(x)
        x = self.fc3(x)
        return self.sigmoid(x)


# 分类器网络（原FFNN）
class FFNN(nn.Module):
    def __init__(self, input_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.telu1 = TeLU(alpha=0.15)
        self.dropout1 = nn.Dropout(0.15)

        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.telu2 = TeLU(alpha=0.1)
        self.dropout2 = nn.Dropout(0.15)

        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.telu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.telu2(x)
        x = self.dropout2(x)

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


# 早停类
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model = deepcopy(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = deepcopy(model)
            self.counter = 0


# GAN训练函数
def train_gan(generator, discriminator, minority_data, epochs=100, batch_size=32):
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    dataset = TensorDataset(torch.FloatTensor(minority_data))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for real_data in dataloader:
            real_data = real_data[0]
            batch_size = real_data.size(0)

            # 训练判别器
            d_optimizer.zero_grad()

            # 真实数据
            real_labels = torch.ones(batch_size, 1)
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)

            # 生成数据
            noise = torch.randn(batch_size, real_data.size(1))
            fake_data = generator(noise)
            fake_labels = torch.zeros(batch_size, 1)
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # 训练生成器
            g_optimizer.zero_grad()
            gen_labels = torch.ones(batch_size, 1)
            gen_output = discriminator(fake_data)
            g_loss = criterion(gen_output, gen_labels)
            g_loss.backward()
            g_optimizer.step()

    # 生成最终样本
    with torch.no_grad():
        noise = torch.randn(len(minority_data) * 2, minority_data.shape[1])  # 生成双倍样本
        generated_samples = generator(noise).numpy()

    return generated_samples


# 数据加载和预处理
def load_and_preprocess(filepath):
    data = pd.read_csv(filepath)

    print("=" * 50)
    print("数据概览:")
    print(f"总样本数: {len(data)}")
    print(f"特征数: {len(data.columns) - 1}")
    print("\n前5行数据:")
    print(data.head())
    print("\n标签分布:")
    print(data['Label'].value_counts(normalize=True))

    X = data.drop('Label', axis=1)
    y = data['Label'].values

    return X, y


# 评估指标计算
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


# 模型训练和评估（带GAN和SMOTE）
def train_and_evaluate_with_gan(X, y, n_splits=10, epochs=100, batch_size=32):
    results = []
    feature_names = X.columns.tolist()
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2025)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        # 数据划分
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 分离少数类样本用于GAN训练
        minority_data = X_train[y_train == 1].values
        scaler = StandardScaler()
        minority_scaled = scaler.fit_transform(minority_data)

        # 训练GAN生成新样本
        print(f"\nFold {fold + 1}/{n_splits} 训练GAN生成样本...")
        input_size = X_train.shape[1]
        generator = Generator(input_size)
        discriminator = Discriminator(input_size)
        generated_samples = train_gan(generator, discriminator, minority_scaled)

        # 反标准化生成的样本
        generated_samples = scaler.inverse_transform(generated_samples)
        generated_labels = np.ones(len(generated_samples))

        # 合并原始数据和GAN生成数据
        X_train_gan = np.vstack([X_train, generated_samples])
        y_train_gan = np.concatenate([y_train, generated_labels])

        # SMOTE过采样
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_res, y_train_res = smote.fit_resample(X_train_gan, y_train_gan)

        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)

        # 转换为PyTorch Dataset
        train_data = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.LongTensor(y_train_res)
        )
        test_data = TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.LongTensor(y_test)
        )

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        # 初始化分类器模型
        model = FFNN(input_size=input_size)
        criterion = FocalLoss(alpha=0.25, gamma=2)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)
        early_stopping = EarlyStopping(patience=10, delta=0.001)

        # 训练分类器
        print(f"Fold {fold + 1}/{n_splits} 训练分类器...")
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

            scheduler.step()

            # 验证过程
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

            # 早停检查
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                model = early_stopping.best_model
                break

        # 最终评估
        model.eval()
        all_preds = []
        all_probs = []
        all_true = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                probs = torch.softmax(outputs, dim=1)[:, 1]

                all_preds.extend(preds.numpy())
                all_probs.extend(probs.numpy())
                all_true.extend(labels.numpy())

        # 计算指标
        metrics = calculate_metrics(all_true, all_preds, all_probs)
        results.append(metrics)

        # 打印当前fold结果
        print(f"\nFold {fold + 1}/{n_splits} 结果:")
        print(classification_report(all_true, all_preds, zero_division=0))
        print(f"AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")
        print(f"最佳验证损失: {-early_stopping.best_score:.4f}")

    # 计算平均指标
    avg_metrics = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
    std_metrics = {k: np.std([r[k] for r in results]) for k in results[0].keys()}

    print("\n" + "=" * 50)
    print("交叉验证平均结果 (±标准差):")
    print("=" * 50)
    for metric in avg_metrics:
        print(f"{metric:12s}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")


# 主执行流程
if __name__ == "__main__":
    data_path = 'preparations/Q3_data2.csv'
    X, y = load_and_preprocess(data_path)

    print("\n" + "=" * 50)
    print("使用GAN+FFNN+FocalLoss（带SMOTE过采样）进行模型训练...")
    print("=" * 50)

    train_and_evaluate_with_gan(X, y, n_splits=10)