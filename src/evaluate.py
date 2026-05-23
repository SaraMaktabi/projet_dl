import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test, device):

    # =====================================================
    # Chargement du meilleur modèle
    # =====================================================

    model.load_state_dict(
        torch.load(
            # "saved_models/best_sequential_model.pth",
            "saved_models/best_custom_model.pth",
            map_location=device
        )
    )

    model = model.to(device)

    # =====================================================
    # Mode évaluation
    # =====================================================

    model.eval()

    # =====================================================
    # Désactivation des gradients
    # =====================================================

    with torch.no_grad():

        outputs = model(X_test)

        predictions = (outputs >= 0.5).float()

    # =====================================================
    # Conversion numpy
    # =====================================================

    y_true = y_test.cpu().numpy()

    y_pred = predictions.cpu().numpy()

    # =====================================================
    # Métriques
    # =====================================================

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    # =====================================================
    # Affichage
    # =====================================================

    print("\n===== ÉVALUATION TEST =====\n")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    # =====================================================
    # Rapport classification
    # =====================================================

    print("\n===== CLASSIFICATION REPORT =====\n")

    print(classification_report(y_true, y_pred))

    # =====================================================
    # Matrice de confusion
    # =====================================================

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Matrice de confusion")

    plt.xlabel("Prédictions")

    plt.ylabel("Vraies classes")

    plt.show()