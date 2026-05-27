import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self):

        super(SimpleCNN, self).__init__()

        # ============================================
        # PARTIE CONVOLUTION
        # ============================================

        self.features = nn.Sequential(

            # Conv 1
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                # stride=2,
                # padding=0
                # modifer padding et stride pour etudier son impact
                padding=2
            ),

            nn.ReLU(),

            # conv 1x1----------
            nn.Conv2d(
                in_channels=6,
                out_channels=6,
                kernel_size=1
            ),

            nn.ReLU(),
            # -------------------

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # nn.AvgPool2d(
            #     kernel_size=2,
            #     stride=2
            # )

            # Conv 2
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )

            # nn.AvgPool2d(
            #     kernel_size=2,
            #     stride=2
            # )
        )

        # ============================================
        # PARTIE CLASSIFICATION
        # ============================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(16 * 5 * 5, 120),
            # mdifier la taille d'entrée du linear pour etudier son impact
            # nn.Linear(16 * 4 * 4, 120),

            nn.ReLU(),

            nn.Linear(120, 84),

            nn.ReLU(),

            nn.Linear(84, 10)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


from cnn_model import SimpleCNN
import torch


model = SimpleCNN()

print(model)

X = torch.randn(1, 1, 28, 28)

Y = model(X)

print("\nOutput shape :", Y.shape)