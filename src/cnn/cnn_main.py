from cnn_data import load_data
from cnn_model import SimpleCNN
from cnn_train import train_cnn
from cnn_evaluate import evaluate_cnn
from cnn_visualization import visualize_feature_maps
from mlp_image_model import ImageMLP

# ============================================
# Data
# ============================================

train_loader, test_loader, device = load_data()

print("Device :", device)

# ============================================
# Model
# ============================================

model = SimpleCNN()

print(model)

# ============================================
# Training
# ============================================

trained_model = train_cnn(
    model,
    train_loader,
    test_loader,
    device,
    epochs=10
)

evaluate_cnn(
    trained_model,
    test_loader,
    device
)

images, labels = next(iter(test_loader))

visualize_feature_maps(
    trained_model,
    images[2],
    device
)


print("\n===== MLP IMAGE =====\n")

mlp_model = ImageMLP()

trained_mlp = train_cnn(
    mlp_model,
    train_loader,
    test_loader,
    device,
    epochs=10
)

evaluate_cnn(
    trained_mlp,
    test_loader,
    device
)