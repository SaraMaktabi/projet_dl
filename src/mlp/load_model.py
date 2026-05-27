import torch
from mlp.model_sequential import create_sequential_model
from mlp.model_custom import CustomMLP


def load_model(model_type, device, path):

    # =====================================================
    # Choix du modèle
    # =====================================================

    if model_type == "sequential":
        model = create_sequential_model()

    elif model_type == "custom":
        model = CustomMLP()

    else:
        raise ValueError("Type de modèle invalide")

    # =====================================================
    # Chargement des poids
    # =====================================================

    model.load_state_dict(
        torch.load(path, map_location=device)
    )

    model = model.to(device)
    model.eval()

    print(f"\nModèle {model_type} chargé avec succès !")

    return model