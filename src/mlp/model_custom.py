import torch.nn as nn


class CustomMLP(nn.Module):

    def __init__(self):

        super(CustomMLP, self).__init__()

        # =================================================
        # Définition des couches
        # =================================================

        self.fc1 = nn.Linear(30, 64)

        self.fc2 = nn.Linear(64, 32)

        self.fc3 = nn.Linear(32, 1)

        # =================================================
        # Fonctions d'activation
        # =================================================

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    # =====================================================
    # Forward propagation
    # =====================================================

    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.relu(x)

        x = self.fc3(x)

        x = self.sigmoid(x)

        return x