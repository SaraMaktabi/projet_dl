import torch.nn as nn


def initialize_weights(model, method="xavier"):

    for m in model.modules():

        if isinstance(m, nn.Linear):

            # =================================================
            # 1. Initialisation gaussienne
            # =================================================

            if method == "normal":

                nn.init.normal_(m.weight, mean=0.0, std=0.01)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # =================================================
            # 2. Initialisation constante
            # =================================================

            elif method == "constant":

                nn.init.constant_(m.weight, 0.5)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # =================================================
            # 3. Xavier (recommandée)
            # =================================================

            elif method == "xavier":

                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)