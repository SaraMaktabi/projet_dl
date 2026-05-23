import torch
import torch.nn as nn


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    epochs=100,
    learning_rate=0.001
):

    # =====================================================
    # Envoi du modèle sur le device
    # =====================================================

    model = model.to(device)

    # =====================================================
    # Fonction de perte
    # =====================================================

    criterion = nn.BCELoss()

    # =====================================================
    # Optimizer
    # =====================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # =====================================================
    # Variables de suivi
    # =====================================================

    best_val_loss = float("inf")

    # =====================================================
    # Boucle d'entraînement
    # =====================================================

    for epoch in range(epochs):

        # ==========================================
        # MODE TRAIN
        # ==========================================

        model.train()

        # ==========================================
        # Forward propagation
        # ==========================================

        outputs = model(X_train)

        # ==========================================
        # Calcul de la loss
        # ==========================================

        loss = criterion(outputs, y_train)

        # ==========================================
        # Reset gradients
        # ==========================================

        optimizer.zero_grad()

        # ==========================================
        # Backward propagation
        # ==========================================

        loss.backward()

        # ==========================================
        # Mise à jour des poids
        # ==========================================

        optimizer.step()

        # ==========================================
        # MODE EVALUATION
        # ==========================================

        model.eval()

        with torch.no_grad():

            val_outputs = model(X_val)

            val_loss = criterion(val_outputs, y_val)

            # ======================================
            # Accuracy train
            # ======================================

            train_predictions = (outputs >= 0.5).float()

            train_accuracy = (
                (train_predictions == y_train)
                .float()
                .mean()
            )

            # ======================================
            # Accuracy validation
            # ======================================

            val_predictions = (val_outputs >= 0.5).float()

            val_accuracy = (
                (val_predictions == y_val)
                .float()
                .mean()
            )

        # ==========================================
        # Sauvegarde du meilleur modèle
        # ==========================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                # "saved_models/best_sequential_model.pth"
                "saved_models/best_custom_model.pth"
            )

        # ==========================================
        # Affichage
        # ==========================================

        if (epoch + 1) % 10 == 0:

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"| Train Loss: {loss.item():.4f} "
                f"| Val Loss: {val_loss.item():.4f} "
                f"| Train Acc: {train_accuracy.item():.4f} "
                f"| Val Acc: {val_accuracy.item():.4f}"
            )

    print("\nEntraînement terminé.")

    return model