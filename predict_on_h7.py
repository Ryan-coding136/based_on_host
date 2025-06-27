import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class SimpleMLP(nn.Module):
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()


data = np.load("split7/H7_sequences.npy")
tensor_data = torch.tensor(data, dtype=torch.float32).to(device)


fasta_path = "split7/H7_sequences.fasta"
epi_ids = []
for record in SeqIO.parse(fasta_path, "fasta"):
    try:
        epi_id = record.id.split("|")[1]  # 通常格式为: A_Name|EPI_ISL_12345|...
    except IndexError:
        epi_id = "Unknown"
    epi_ids.append(epi_id)


if len(epi_ids) != len(data):
    raise ValueError(f"FASTA headers ({len(epi_ids)}) and embeddings ({len(data)}) count mismatch.")


with torch.no_grad():
    logits = model(tensor_data)
    probs = F.softmax(logits, dim=1)[:, 1]
    preds = (probs > 0.5).long().cpu().numpy()


df = pd.DataFrame({
    "Isolate_Id": epi_ids,
    "Predicted_Label": preds,
    "Binding_Probability": probs.cpu().numpy()
})
df.to_csv("H7_predictions.csv", index=False)
print("✅ Saved predictions to H7_predictions.csv")


try:
    label_df = pd.read_csv("H7_metadata_with_label.csv")
    merged = pd.merge(df, label_df, on="Isolate_Id")
    y_true = merged["binding_label"]
    y_pred = merged["Predicted_Label"]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-binding", "Binding"],
        yticklabels=["Non-binding", "Binding"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix: H7")
    plt.tight_layout()
    plt.savefig("confusion_matrix_H7.png")
    print("📊 Confusion matrix saved to: confusion_matrix_H7.png")
except Exception as e:
    print(f"⚠️ Could not generate confusion matrix: {e}")
