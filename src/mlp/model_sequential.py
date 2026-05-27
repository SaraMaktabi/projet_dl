import torch.nn as nn


def create_sequential_model():

    model = nn.Sequential(

        # Couche 1
        nn.Linear(30, 64),
        nn.ReLU(),

        # Couche 2
        nn.Linear(64, 32),
        nn.ReLU(),

        # Couche de sortie
        nn.Linear(32, 1),
        nn.Sigmoid()

    )

    return model