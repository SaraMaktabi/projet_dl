from data_preparation import prepare_data
from model_sequential import create_sequential_model
from train import train_model
from evaluate import evaluate_model
from model_custom import CustomMLP


X_train, X_val, X_test, y_train, y_val, y_test, device = prepare_data()

print("\nTrain :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)


model = create_sequential_model()

print("\nArchitecture du modèle :\n")
print(model)


print("\nParamètres du modèle :\n")

for name, param in model.named_parameters():

    print("Nom :", name)
    print("Shape :", param.shape)
    print("-" * 40)

print("\nState Dict :\n")

for key in model.state_dict():

    print(key)

# initialisation des poids

trained_model = train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    init_method="xavier"
)

evaluate_model(
    trained_model,
    X_test,
    y_test,
    device
)

#-----------------------------------------nn.module-----------------------------------------

print("\n===== MODÈLE PERSONNALISÉ =====\n")

custom_model = CustomMLP()

print(custom_model)

trained_custom_model = train_model(
    custom_model,
    X_train,
    y_train,
    X_val,
    y_val,
    device
)

evaluate_model(
    trained_custom_model,
    X_test,
    y_test,
    device
)