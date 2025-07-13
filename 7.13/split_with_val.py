import numpy as np
from sklearn.model_selection import train_test_split
import os


X = np.load("X.npy")
y = np.load("y.npy")


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


os.makedirs("splits_val", exist_ok=True)
np.save("splits_val/X_train.npy", X_train)
np.save("splits_val/X_val.npy", X_val)
np.save("splits_val/X_test.npy", X_test)
np.save("splits_val/y_train.npy", y_train)
np.save("splits_val/y_val.npy", y_val)
np.save("splits_val/y_test.npy", y_test)

print("✅ Done. Shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
