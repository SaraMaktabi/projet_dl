from data_preparation import load_dataset, build_vocab_from_pairs, prepare_sequences
from dataset import TranslationDataset
from rnn_model import SimpleRNN
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from train import train_model



pairs = load_dataset("src/rnn/data/fra.txt")

eng_vocab, fra_vocab = build_vocab_from_pairs(pairs)

print("Taille vocab anglais :", eng_vocab.count)
print("Taille vocab français :", fra_vocab.count)

print("\nExemple tokenisation :")

sample = pairs[0][0]

print(sample)
print(eng_vocab.numericalize(sample))





eng_tensor, fra_tensor = prepare_sequences(
    pairs,
    eng_vocab,
    fra_vocab
)

print("Shape anglais :", eng_tensor.shape)
print("Shape français :", fra_tensor.shape)

print("\nExemple séquence :")
print(eng_tensor[0])


dataset = TranslationDataset(
    eng_tensor,
    fra_tensor
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

for source_batch, target_batch in loader:

    print("Source shape :", source_batch.shape)
    print("Target shape :", target_batch.shape)

    break

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device :", device)

model = SimpleRNN(
    vocab_size=eng_vocab.count,
    embedding_dim=128,
    hidden_dim=256,
    output_dim=fra_vocab.count
).to(device)

print(model)

for source_batch, target_batch in loader:

    source_batch = source_batch.to(device)

    outputs = model(source_batch)

    print("Output shape :", outputs.shape)

    break

criterion = nn.CrossEntropyLoss(
    ignore_index=fra_vocab.word2idx["<PAD>"]
)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

train_model(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epochs=10
)