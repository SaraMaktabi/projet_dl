from data_preparation import prepare_data
from model_sequential import create_sequential_model
from model_custom import CustomMLP
from train import train_model
from evaluate import evaluate_model
from load_model import load_model


# =========================================================
# 1. DATA
# =========================================================

X_train, X_val, X_test, y_train, y_val, y_test, device = prepare_data()

print("\nTrain :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)


# =========================================================
# 2. MODELE SEQUENTIAL
# =========================================================

print("\n================ SEQUENTIAL MODEL ================\n")

model = create_sequential_model()

print(model)


print("\nParamètres du modèle :\n")

for name, param in model.named_parameters():
    print("Nom :", name)
    print("Shape :", param.shape)
    print("-" * 40)

print("\nState Dict Keys :\n")

for key in model.state_dict():
    print(key)


# =========================================================
# 3. TRAIN SEQUENTIAL
# =========================================================

trained_model = train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    init_method="xavier"   #  changer: normal / constant / xavier
)


# =========================================================
# 4. EVALUATION SEQUENTIAL
# =========================================================

evaluate_model(
    trained_model,
    X_test,
    y_test,
    device
)


# =========================================================
# 5. RELOAD SEQUENTIAL (IMPORTANT POUR LE PROJET)
# =========================================================

loaded_model = load_model(
    model_type="sequential",
    device=device,
    path="saved_models/best_sequential_model.pth"
)

evaluate_model(
    loaded_model,
    X_test,
    y_test,
    device
)


# =========================================================
# 6. MODELE PERSONNALISE (nn.Module)
# =========================================================

print("\n================ CUSTOM MODEL ================\n")

custom_model = CustomMLP()

print(custom_model)


# =========================================================
# 7. TRAIN CUSTOM
# =========================================================

trained_custom_model = train_model(
    custom_model,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    init_method="xavier"
)


# =========================================================
# 8. EVALUATION CUSTOM
# =========================================================

evaluate_model(
    trained_custom_model,
    X_test,
    y_test,
    device
)


# =========================================================
# 9. RELOAD CUSTOM MODEL
# =========================================================

loaded_custom_model = load_model(
    model_type="custom",
    device=device,
    path="saved_models/best_custom_model.pth"
)

evaluate_model(
    loaded_custom_model,
    X_test,
    y_test,
    device
)