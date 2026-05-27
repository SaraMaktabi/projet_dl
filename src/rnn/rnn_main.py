from data_preparation import load_dataset, build_vocab_from_pairs, prepare_sequences


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