from typing import Tuple
from keras import layers
from keras import Model
from typing import List
import numpy as np
from tqdm import tqdm
import spacy
import os
import re
import json
import unicodedata
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def tokenise_text(
    abstracts: List[str], titles: List[str], seq_len: int = 512
) -> Tuple[List[List], List[List], Tokenizer]:
    tokeniser = Tokenizer(
        num_words=None,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{}~\t\n',
        lower=True,
        split=" ",
        char_level=False,
        oov_token="|UNK|",
        document_count=0,
    )

    tokeniser.fit_on_texts([abstracts, titles])

    seq_abstracts = tokeniser.texts_to_sequences(abstracts)
    seq_abstracts = pad_sequences(seq_abstracts, padding="post", maxlen=seq_len)

    seq_titles = tokeniser.texts_to_sequences(titles)
    seq_titles = pad_sequences(seq_titles, padding="post", maxlen=seq_len)

    return seq_abstracts, seq_titles, tokeniser


def roll_sequences(seqs: List[List[int]]):
    return np.roll(seqs, 1, axis=1)


def embed_sentence(text: str, nlp: spacy.Language):
    return [nlp(word).vector for word in text.split(" ")]


def embed_corpus(corpus: str, nlp: spacy.Language):
    return [embed_sentence(sample, nlp) for sample in corpus]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8", "ignore")
    text = text.lower()
    text = re.sub(r"[^a-zA-z0-9\s]", " ", text)
    return text


def load_data(scraped_dir: str = "./shared_data/data/scraped", n_docs: int = 5):
    docs = []
    for i, filename in enumerate(os.listdir(scraped_dir)):
        with open(os.path.join(scraped_dir, filename), "r") as f:
            doc = json.load(f)
        docs.append(doc)
        if (i + 1) == n_docs:
            break
    return docs


def yield_data(scraped_dir: str = "./shared_data/data/scraped"):
    for filename in os.listdir(scraped_dir):
        with open(os.path.join(scraped_dir, filename), "r") as f:
            doc = json.load(f)
        yield doc


def build_model(
    vocab_size: int,
    seq_len: int = 512,
    encoder_size: int = 128,
    decoder_size: int = 128,
):

    embedding = layers.Embedding(vocab_size, encoder_size, input_length=seq_len)

    # encoder
    encoder_input = layers.Input([seq_len])
    encoder_embedding = embedding(encoder_input)
    encoder_hidden = layers.LSTM(encoder_size, return_state=True)
    encoder_output, encoder_h, encoder_c = encoder_hidden(encoder_embedding)

    # decoder
    decoder_input = layers.Input([seq_len])
    decoder_embedding = embedding(decoder_input)
    decoder_hidden = layers.LSTM(decoder_size, return_sequences=True, return_state=True)
    decoder_output, _, _ = decoder_hidden(
        decoder_embedding, initial_state=[encoder_h, encoder_c]
    )
    decoder_dense = layers.Dense(1, activation="linear")
    decoder_prediction = decoder_dense(decoder_output)

    model = Model([encoder_input, decoder_input], decoder_prediction)

    return model, encoder_input, decoder_output


if __name__ == "__main__":

    scraped_dir = "./shared_data/data/scraped"
    n_docs = 10000
    seq_len = 500

    abstracts = []
    titles = []
    max_abstract_len = 0
    max_title_len = 0

    docs = load_data(scraped_dir, n_docs)

    for doc in tqdm(docs, total=n_docs, unit="doc"):
        abstract = doc["abstract"]
        title = doc["title"]
        abstracts.append(f"<SS> {abstract} <ES>")
        titles.append(f"|SS| {title} |ES|")
        max_abstract_len = max(max_abstract_len, len(abstract))
        max_title_len = max(max_title_len, len(title))

    print(f"max abstract length: {max_abstract_len}")
    print(f"max title length: {max_title_len}")

    seq_abstracts, seq_titles, tokeniser = tokenise_text(
        abstracts, titles, seq_len=seq_len
    )

    seq_titles_tplus1 = roll_sequences(seq_titles)

    print(seq_abstracts.shape)

    model, encoder_input, decoder_output = build_model(
        vocab_size=len(tokeniser.word_counts),
        encoder_size=32,
        decoder_size=32,
        seq_len=seq_len,
    )

    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()

    model.fit(x=[seq_abstracts, seq_titles], y=seq_titles_tplus1)
