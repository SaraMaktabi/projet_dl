class Vocabulary:

    def __init__(self):

        self.word2idx = {}

        self.idx2word = {}

        self.count = 0

        # tokens spéciaux
        self.add_word("<PAD>")
        self.add_word("<SOS>")
        self.add_word("<EOS>")
        self.add_word("<UNK>")

    def add_word(self, word):

        if word not in self.word2idx:

            self.word2idx[word] = self.count
            self.idx2word[self.count] = word
            self.count += 1

    def build_vocab(self, sentences):

        for sentence in sentences:

            for word in sentence.split():

                self.add_word(word)

    def numericalize(self, sentence):

        return [
            self.word2idx.get(word, self.word2idx["<UNK>"])
            for word in sentence.split()
        ]
    

    def add_special_tokens(self):

        self.pad_idx = self.word2idx["<PAD>"]
        self.sos_idx = self.word2idx["<SOS>"]
        self.eos_idx = self.word2idx["<EOS>"]
        self.unk_idx = self.word2idx["<UNK>"]

    
    def encode_sentence(self, sentence):

        tokens = sentence.split()

        indices = []

        for word in tokens:

            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.word2idx["<UNK>"])

        return indices