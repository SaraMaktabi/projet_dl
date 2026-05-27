import torch
import matplotlib.pyplot as plt


def visualize_feature_maps(
    model,
    image,
    device
):

    model.eval()

    image = image.unsqueeze(0).to(device)

    # ============================================
    # Première couche convolutionnelle
    # ============================================

    with torch.no_grad():

        feature_maps = model.features[0](image)

    # ============================================
    # CPU
    # ============================================

    feature_maps = feature_maps.cpu()

    # ============================================
    # Affichage
    # ============================================

    num_maps = feature_maps.shape[1]

    plt.figure(figsize=(12, 8))

    for i in range(num_maps):

        plt.subplot(2, num_maps // 2, i + 1)

        plt.imshow(
            feature_maps[0, i],
            cmap="gray"
        )

        plt.axis("off")

        plt.title(f"Map {i+1}")

    plt.tight_layout()

    plt.show()