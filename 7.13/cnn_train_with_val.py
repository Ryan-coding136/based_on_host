import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CNN model definition
class CNNClassifier(nn.Module):
    def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.flatten_dim = (input_size // 4) * 32
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# GradCAM class
class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=2, keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

# Load data
X_train = np.load("splits_val/X_train.npy")
y_train = np.load("splits_val/y_train.npy")
X_val = np.load("splits_val/X_val.npy")
y_val = np.load("splits_val/y_val.npy")
X_test = np.load("splits_val/X_test.npy")
y_test = np.load("splits_val/y_test.npy")

X_train = torch.tensor(X_train).float().unsqueeze(1)
y_train = torch.tensor(y_train).long()
X_val = torch.tensor(X_val).float().unsqueeze(1)
y_val = torch.tensor(y_val).long()
X_test = torch.tensor(X_test).float().unsqueeze(1)
y_test = torch.tensor(y_test).long()

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# Model, loss, optimizer
input_size = X_train.shape[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier(input_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training
best_val_loss = float('inf')
early_stop_counter = 0
train_log = []
save_dir = "cnn_outputs"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(1, 51):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * xb.size(0)
        correct += (preds.argmax(1) == yb).sum().item()
        total += xb.size(0)

    train_acc = correct / total
    train_loss /= total

    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)

            val_loss += loss.item() * xb.size(0)
            correct += (preds.argmax(1) == yb).sum().item()
            total += xb.size(0)

    val_acc = correct / total
    val_loss /= total
    scheduler.step(val_loss)

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    train_log.append([epoch, train_loss, val_loss, train_acc, val_acc])

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    else:
        early_stop_counter += 1
        if early_stop_counter >= 5:
            print("Early stopping triggered.")
            break

# Save log
log_df = pd.DataFrame(train_log, columns=["Epoch", "Train_Loss", "Val_Loss", "Train_Acc", "Val_Acc"])
log_df.to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

# Plot curves
plt.plot(log_df["Epoch"], log_df["Train_Loss"], label="Train Loss")
plt.plot(log_df["Epoch"], log_df["Val_Loss"], label="Val Loss")
plt.legend()
plt.savefig(os.path.join(save_dir, "loss_curve.png"))
plt.clf()

plt.plot(log_df["Epoch"], log_df["Train_Acc"], label="Train Acc")
plt.plot(log_df["Epoch"], log_df["Val_Acc"], label="Val Acc")
plt.legend()
plt.savefig(os.path.join(save_dir, "acc_curve.png"))
plt.clf()

# Evaluation
model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb)
        y_true.extend(yb.tolist())
        y_pred.extend(preds.argmax(1).cpu().tolist())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Avian", "Human"]))

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Avian", "Human"]).plot(cmap="Blues")
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
plt.clf()

# GradCAM example
sample_x = X_test[0].unsqueeze(0).to(device)
gradcam = GradCAM1D(model, model.conv1)
cam_result = gradcam.generate(sample_x)

plt.figure(figsize=(12, 4))
plt.plot(cam_result[0])
plt.title("Grad-CAM Heatmap (Sample 0)")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "gradcam_sample0.png"))
