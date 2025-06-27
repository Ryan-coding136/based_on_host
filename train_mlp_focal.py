import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ==== 设定随机种子 ====
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==== 数据加载 ====
X_train = np.load("train_sequences.npy")
y_train = np.load("train_labels.npy")
X_test = np.load("test_sequences.npy")
y_test = np.load("test_labels.npy")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# ==== 模型定义 ====
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.layers(x)

# ==== Focal Loss ====
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            CE_loss = at * (1 - pt) ** self.gamma * CE_loss
        else:
            CE_loss = (1 - pt) ** self.gamma * CE_loss
        return CE_loss.mean()

# ==== 初始化 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

alpha = torch.tensor([1.0, 3.0]).to(device)
loss_fn = FocalLoss(alpha=alpha)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==== 训练 ====
best_auc = 0.0
for epoch in range(20):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    # 验证集评估
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs[:, 1])
            all_labels.extend(yb.numpy())

    auc_score = roc_auc_score(all_labels, all_probs)
    print(f"📈 Epoch {epoch+1} | AUC: {auc_score:.4f}")

    if auc_score > best_auc:
        best_auc = auc_score
        torch.save(model.state_dict(), "best_model.pt")
        print("✅ Saved new best model")

# ==== 评估并绘图 ====
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs[:, 1])
        all_labels.extend(yb.numpy())

preds = [1 if p > 0.5 else 0 for p in all_probs]
print("\nClassification Report:")
print(classification_report(all_labels, preds))

# 混淆矩阵
cm = confusion_matrix(all_labels, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_weighted_v2.png")
print("🗂 Confusion matrix saved")

# ROC曲线
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {best_auc:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_weighted_v2.png")
print("📈 ROC curve saved")
