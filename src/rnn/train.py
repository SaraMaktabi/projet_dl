import torch
import torch.nn as nn


def train_model(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epochs=5,
    clip=1.0
):

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for source, target in loader:

            source = source.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            outputs = model(source)

            output_dim = outputs.shape[-1]

            outputs = outputs.reshape(
                -1,
                output_dim
            )

            target = target.reshape(-1)

            loss = criterion(
                outputs,
                target
            )

            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip
            )

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        perplexity = torch.exp(
            torch.tensor(avg_loss)
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"| Loss: {avg_loss:.4f} "
            f"| Perplexity: {perplexity:.4f}"
        )

    print("\nEntraînement terminé.")