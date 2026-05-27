import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# transform = transforms.ToTensor()

# train_dataset = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=transform
# )

# test_dataset = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=transform
# )

# print("Train size :", len(train_dataset))
# print("Test size :", len(test_dataset))

# image, label = train_dataset[0]

# print("Image shape :", image.shape)
# print("Label :", label)




def load_data(batch_size=64):

    # ============================================
    # Transformations
    # ============================================

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

    # ============================================
    # Fashion-MNIST
    # ============================================

    train_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    # ============================================
    # DataLoaders
    # ============================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # ============================================
    # Device
    # ============================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    return train_loader, test_loader, device


train_loader, test_loader, device = load_data()

print("Device :", device)

for images, labels in train_loader:

    print("Images shape :", images.shape)

    print("Labels shape :", labels.shape)

    break