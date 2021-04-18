from typing import Dict, Tuple
from tensorflow.keras import layers
from tensorflow.keras import Model
from typing import List
import numpy as np
from tqdm import tqdm
import os
import re
import json
import unicodedata
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tokenise_text(
    abstracts: List[str],
    titles: List[str],
    seq_len: int = 512,
    special_tokens: Dict[str, str] = {"start": "|SS|", "end": "|ES|", "OOV": "|UNK|"},
) -> Tuple[List[List], List[List], Tokenizer]:
    tokeniser = Tokenizer(
        num_words=None,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{}~\t\n',
        lower=False,
        split=" ",
        char_level=False,
        # oov_token=special_tokens["OOV"],
        document_count=0,
    )

    tokeniser.fit_on_texts([*abstracts, *titles])

    seq_abstracts = tokeniser.texts_to_sequences(abstracts)
    seq_abstracts = pad_sequences(seq_abstracts, padding="post", maxlen=seq_len)

    seq_titles = tokeniser.texts_to_sequences(titles)
    seq_titles = pad_sequences(seq_titles, padding="post", maxlen=seq_len)

    return seq_abstracts, seq_titles, tokeniser


def roll_sequences(seqs: List[List[int]]):
    return np.roll(seqs, 1, axis=1)


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
    seq_len: int = 500,
    encoder_size: int = 128,
    decoder_size: int = 128,
):

    embedding = layers.Embedding(vocab_size, encoder_size, input_length=seq_len)

    # training model
    encoder_input = layers.Input([seq_len])
    encoder_embedding = embedding(encoder_input)
    encoder_hidden = layers.LSTM(encoder_size, return_state=True)
    encoder_output, encoder_h, encoder_c = encoder_hidden(encoder_embedding)

    decoder_input = layers.Input([seq_len])
    decoder_embedding = embedding(decoder_input)
    decoder_hidden = layers.LSTM(decoder_size, return_sequences=True, return_state=True)
    decoder_output, _, _ = decoder_hidden(
        decoder_embedding, initial_state=[encoder_h, encoder_c]
    )
    decoder_dense = layers.Dense(1, activation="linear")
    decoder_prediction = decoder_dense(decoder_output)

    # infer model
    decoder_input_h = layers.Input((encoder_size,))
    decoder_input_c = layers.Input((encoder_size,))
    infer_output, decoder_output_h, decoder_output_c = decoder_hidden(
        decoder_embedding, initial_state=[decoder_input_h, decoder_input_c]
    )
    infer_prediction = decoder_dense(infer_output)

    model = Model([encoder_input, decoder_input], decoder_prediction)
    encoder_model = Model(encoder_input, [encoder_h, encoder_c])
    decoder_model = Model(
        [decoder_input, decoder_input_h, decoder_input_c],
        [infer_prediction, decoder_output_h, decoder_output_c],
    )
    return model, encoder_model, decoder_model


def decode_sequence(
    input_text: str,
    tokeniser: Tokenizer,
    encoder_model: Model,
    decoder_model: Model,
    seq_len: int,
    special_tokens: Dict[str, str] = {"start": "|SS|", "end": "|ES|", "OOV": "|UNK|"},
):
    # convert the sequence to tokens
    input_seq = tokeniser.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, padding="post", maxlen=seq_len)

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, seq_len))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = tokeniser.word_index[special_tokens["start"]]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    i = 0
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq, *states_value])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, :])
        next_word = tokeniser.index_word.get(sampled_token_index, special_tokens["OOV"])
        decoded_sentence.append(next_word)

        # Exit condition: either hit max length
        # or find stop character.
        if next_word == special_tokens["end"] or len(decoded_sentence) >= seq_len:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq[0, i] = sampled_token_index

        # Update states
        states_value = [h, c]

        i += 1

    return " ".join(decoded_sentence)


if __name__ == "__main__":

    scraped_dir = "./shared_data/data/scraped"
    n_docs = 100000
    seq_len = 150
    special_tokens = {"start": "|SS|", "end": "|ES|", "OOV": "|UNK|"}

    abstracts = []
    titles = []
    max_abstract_len = 0
    max_title_len = 0

    docs = load_data(scraped_dir, n_docs)

    for doc in tqdm(docs, total=n_docs, unit="doc"):
        abstract = doc["abstract"]
        title = doc["title"]
        abstracts.append(
            f"{special_tokens['start']} {clean_text(abstract)} {special_tokens['end']}"
        )
        titles.append(
            f"{special_tokens['start']} {clean_text(title)} {special_tokens['end']}"
        )
        max_abstract_len = max(max_abstract_len, len(abstract))
        max_title_len = max(max_title_len, len(title))

    print(f"max abstract length: {max_abstract_len}")
    print(f"max title length: {max_title_len}")

    seq_abstracts, seq_titles, tokeniser = tokenise_text(
        abstracts=abstracts,
        titles=titles,
        seq_len=seq_len,
        special_tokens=special_tokens,
    )

    seq_titles_tplus1 = roll_sequences(seq_titles)

    model, encoder_model, decoder_model = build_model(
        vocab_size=len(tokeniser.word_index) + 1,
        encoder_size=32,
        decoder_size=32,
        seq_len=seq_len,
    )

    model.compile(optimizer="rmsprop", loss="mae", metrics=["accuracy"])

    model.summary()
    encoder_model.summary()
    decoder_model.summary()

    model.fit(
        x=[seq_abstracts, seq_titles],
        y=seq_titles_tplus1,
        epochs=1,
        batch_size=256,
    )

    # abstract from: https://arxiv.org/abs/2104.07257
    # title is: A Novel Neuron Model of Visual Processor
    test_input = "Simulating and imitating the neuronal network of humans or mammals is a popular topic that has been explored for many years in the fields of pattern recognition and computer vision. Inspired by neuronal conduction characteristics in the primary visual cortex of cats, pulse-coupled neural networks (PCNNs) can exhibit synchronous oscillation behavior, which can process digital images without training. However, according to the study of single cells in the cat primary visual cortex, when a neuron is stimulated by an external periodic signal, the interspike-interval (ISI) distributions represent a multimodal distribution. This phenomenon cannot be explained by all PCNN models. By analyzing the working mechanism of the PCNN, we present a novel neuron model of the primary visual cortex consisting of a continuous-coupled neural network (CCNN). Our model inherited the threshold exponential decay and synchronous pulse oscillation property of the original PCNN model, and it can exhibit chaotic behavior consistent with the testing results of cat primary visual cortex neurons. Therefore, our CCNN model is closer to real visual neural networks. For image segmentation tasks, the algorithm based on CCNN model has better performance than the state-of-art of visual cortex neural network model. The strength of our approach is that it helps neurophysiologists further understand how the primary visual cortex works and can be used to quantitatively predict the temporal-spatial behavior of real neural networks. CCNN may also inspire engineers to create brain-inspired deep learning networks for artificial intelligence purposes. "
    test_output = decode_sequence(
        test_input,
        tokeniser=tokeniser,
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        seq_len=seq_len,
        special_tokens=special_tokens,
    )
    print(test_output)
