import torch
import torch.nn as nn
import torch.optim as optim


def train_cnn(
    model,
    train_loader,
    test_loader,
    device,
    epochs=10,
    learning_rate=0.001
):

    # ============================================
    # Device
    # ============================================

    model = model.to(device)

    # ============================================
    # Loss & Optimizer
    # ============================================

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # ============================================
    # Training Loop
    # ============================================

    for epoch in range(epochs):

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # =========================
            # Forward
            # =========================

            outputs = model(images)

            loss = criterion(outputs, labels)

            # =========================
            # Backprop
            # =========================

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # =========================
            # Stats
            # =========================

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # ============================================
        # Evaluation
        # ============================================

        model.eval()

        correct_test = 0
        total_test = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                total_test += labels.size(0)

                correct_test += (
                    predicted == labels
                ).sum().item()

        test_acc = 100 * correct_test / total_test

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"| Loss: {running_loss:.4f} "
            f"| Train Acc: {train_acc:.2f}% "
            f"| Test Acc: {test_acc:.2f}%"
        )

    print("\nEntraînement CNN terminé.")

    return model