import re
import torch
from vocabulary import Vocabulary

def normalize_text(text):

    text = text.lower().strip()

    text = re.sub(
        r"([.!?])",
        r" \1",
        text
    )

    text = re.sub(
        r"[^a-zA-ZÀ-ÿ.!?]+",
        r" ",
        text
    )

    return text


def load_dataset(path, num_samples=10000):

    pairs = []

    with open(
        path,
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    for line in lines[:num_samples]:

        parts = line.strip().split("\t")

        if len(parts) >= 2:

            eng = normalize_text(parts[0])

            fra = normalize_text(parts[1])

            pairs.append((eng, fra))

    return pairs

def build_vocab_from_pairs(pairs):

    eng_vocab = Vocabulary()
    fra_vocab = Vocabulary()

    eng_sentences = [p[0] for p in pairs]
    fra_sentences = [p[1] for p in pairs]

    eng_vocab.build_vocab(eng_sentences)
    fra_vocab.build_vocab(fra_sentences)

    return eng_vocab, fra_vocab


def add_tokens(indices, vocab):

    return (
        [vocab.word2idx["<SOS>"]] +
        indices +
        [vocab.word2idx["<EOS>"]]
    )




def pad_sequences(sequences, pad_value, max_len):

    padded = []

    for seq in sequences:

        if len(seq) < max_len:

            seq = seq + [pad_value] * (max_len - len(seq))

        else:

            seq = seq[:max_len]

        padded.append(seq)

    return torch.tensor(padded)


def prepare_sequences(pairs, eng_vocab, fra_vocab, max_len=20):

    eng_sequences = []
    fra_sequences = []

    for eng, fra in pairs:

        eng_idx = eng_vocab.encode_sentence(eng)
        fra_idx = fra_vocab.encode_sentence(fra)

        eng_idx = add_tokens(eng_idx, eng_vocab)
        fra_idx = add_tokens(fra_idx, fra_vocab)

        eng_sequences.append(eng_idx)
        fra_sequences.append(fra_idx)

    eng_padded = pad_sequences(
        eng_sequences,
        eng_vocab.word2idx["<PAD>"],
        max_len
    )

    fra_padded = pad_sequences(
        fra_sequences,
        fra_vocab.word2idx["<PAD>"],
        max_len
    )

    return eng_padded, fra_padded