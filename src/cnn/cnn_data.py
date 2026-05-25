import torch
from torchvision import datasets
from torchvision import transforms


transform = transforms.ToTensor()

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

print("Train size :", len(train_dataset))
print("Test size :", len(test_dataset))

image, label = train_dataset[0]

print("Image shape :", image.shape)
print("Label :", label)