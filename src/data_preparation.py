import torch
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data():

    # =========================================================
    # 1. Chargement du dataset
    # =========================================================

    data = load_breast_cancer()

    X = data.data
    y = data.target

    # =========================================================
    # 2. Affichage d'informations
    # =========================================================

    print("Shape des features :", X.shape)
    print("Shape des labels :", y.shape)

    print("\nClasses :", data.target_names)

    # =========================================================
    # 3. Split Train + Temp
    # =========================================================

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # =========================================================
    # 4. Split Validation + Test
    # =========================================================

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # =========================================================
    # 5. Normalisation
    # =========================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_val = scaler.transform(X_val)

    X_test = scaler.transform(X_test)

    # =========================================================
    # 6. Conversion vers tensors PyTorch
    # =========================================================

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # =========================================================
    # 7. Device CPU/GPU
    # =========================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = X_train.to(device)
    X_val = X_val.to(device)
    X_test = X_test.to(device)

    y_train = y_train.to(device)
    y_val = y_val.to(device)
    y_test = y_test.to(device)

    print("\nDevice utilisé :", device)

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        device
    )