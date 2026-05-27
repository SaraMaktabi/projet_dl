import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_cnn(
    model,
    test_loader,
    device
):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(
                predicted.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # ============================================
    # Classification Report
    # ============================================

    print("\n===== CLASSIFICATION REPORT =====\n")

    print(
        classification_report(
            all_labels,
            all_preds
        )
    )

    # ============================================
    # Confusion Matrix
    # ============================================

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("True")

    plt.show()