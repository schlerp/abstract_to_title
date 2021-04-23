from typing import List, Dict, Tuple, Union, Any
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.initializers import Constant
import numpy as np
from tqdm import tqdm
import os
import re
import json
import unicodedata
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adadelta


def tokenise_text(
    abstracts: List[str],
    titles: List[str],
    seq_len: int = 512,
    special_tokens: Dict[str, str] = {"start": "|SS|", "end": "|ES|", "OOV": "|UNK|"},
    vocab_size: str = None
) -> Tuple[List[List], List[List], Tokenizer]:
    tokeniser = Tokenizer(
        num_words=vocab_size,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{}~\t\n',
        lower=False,
        split=" ",
        char_level=False,
        oov_token=special_tokens["OOV"],
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
    embedding_matrix: Any = None
):
    if embedding_matrix is not None:
        embedding = layers.Embedding(vocab_size, encoder_size, embeddings_initializer=Constant(embedding_matrix), trainable=False, input_length=seq_len)
    else:
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
        sampled_token_index = int(max(output_tokens[0, i, 0], 0))
        next_word = tokeniser.index_word.get(sampled_token_index, "")
        decoded_sentence.append(next_word)

        # Exit condition: either hit max length
        # or find stop character.
        if next_word == special_tokens["end"] or len(decoded_sentence) >= (seq_len-1):
            stop_condition = True

        i += 1

        target_seq[0, i] = sampled_token_index

        # Update states
        states_value = [h, c]

    return " ".join(decoded_sentence)


def prepare_embeddings(file_path: str = "./shared_data/embeddings/glove.6B.100d.txt"):
    embeddings_index = {}
    with open(file_path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))
    return embeddings_index


def create_embedding_matrix(embeddings_index: Dict, vocab_size: int, embedding_size: int, tokeniser: Tokenizer):
    hits = 0
    misses = 0
    print(len(tokeniser.word_index))

    # Prepare embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_size))
    for word, i in tokeniser.word_index.items():
        if i >= vocab_size:
            break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix

    
if __name__ == "__main__":

    scraped_dir = "./shared_data/data/scraped"
    n_docs = 500_000
    seq_len = 150
    hidden_size = 100
    epochs = 100
    batch_size = 2500
    vocab_size = 400_000
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
        vocab_size=vocab_size,
    )

    seq_titles_tplus1 = roll_sequences(seq_titles)
#     seq_titles_tplus1 = to_categorical(seq_titles_tplus1, num_classes=vocab_size)
    
    
    embeddings_index = prepare_embeddings()
    
    embedding_matrix = create_embedding_matrix(embeddings_index=embeddings_index, vocab_size=vocab_size, embedding_size=hidden_size, tokeniser=tokeniser)

    model, encoder_model, decoder_model = build_model(
        vocab_size=vocab_size,
        encoder_size=hidden_size,
        decoder_size=hidden_size,
        seq_len=seq_len,
        embedding_matrix=embedding_matrix
    )

    opt = Adadelta(
        learning_rate=1.0, rho=0.95, epsilon=1e-06, name='Adadelta',
    )
    
    model.compile(optimizer=opt, loss="mse")

    model.summary()
    encoder_model.summary()
    decoder_model.summary()
    
#     checkpoint_cb = ModelCheckpoint(
#         filepath, 
#         monitor='val_loss', 
#         verbose=0, 
#         save_best_only=true,
#         save_weights_only=False, 
#         mode='auto', 
#         save_freq='epoch',
#     )

    for i in range(1, 100):
        start_epoch = max(epochs*(i-1), 1)
        finish_epoch = start_epoch+epochs
        model.fit(
            x=[seq_abstracts, seq_titles],
            y=seq_titles_tplus1,
            epochs=finish_epoch,
            batch_size=batch_size,
            validation_split=0.1,
            validation_freq=10,
            initial_epoch=start_epoch
        )

        test_input = docs[random.randint(0, n_docs)]['abstract']
        test_output = decode_sequence(
            test_input,
            tokeniser=tokeniser,
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            seq_len=seq_len,
            special_tokens=special_tokens,
        )
        print(test_input)
        print(test_output)

        model.save(
            os.path.join(
                "shared_data/models/",
                f"{n_docs}_{seq_len}_{hidden_size}_{epochs*i}_{batch_size}.model",
            )
        )

        encoder_model.save(
            os.path.join(
                "shared_data/models/",
                f"{n_docs}_{seq_len}_{hidden_size}_{epochs*i}_{batch_size}.encoder",
            )
        )

        decoder_model.save(
            os.path.join(
                "shared_data/models/",
                f"{n_docs}_{seq_len}_{hidden_size}_{epochs*i}_{batch_size}.decoder",
            )
        )
