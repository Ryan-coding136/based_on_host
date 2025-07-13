import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Load Data ---
# Assume user prepares a new sample of shape (1292,) in .npy format
X_input = np.load("new_sample.npy")  # shape: (1292,)
X_input = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0)  # shape: (1, 1292)

# --- Define Model Architecture (Same as Training) ---
class CNNClassifier(nn.Module):
    def __init__(self, input_dim=1292, hidden_dim=64):
        super(CNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# --- Load Model ---
model = CNNClassifier()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# --- Inference ---
with torch.no_grad():
    logits = model(X_input)
    prob = F.softmax(logits, dim=1)
    pred = torch.argmax(prob, dim=1).item()
    label = "Human" if pred == 1 else "Avian"
    confidence = prob[0, pred].item()

print(f"Prediction: {label} (Confidence: {confidence:.2f})")
