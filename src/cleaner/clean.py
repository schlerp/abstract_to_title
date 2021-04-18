from typing import List
import os
import re
import json
import unicodedata
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def tokenise_text(abstracts: List[str], titles: List[str], max_len: int = 512):
    tokeniser = Tokenizer(
        num_words=None,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=" ",
        char_level=False,
        oov_token=None,
        document_count=0,
    )

    tokeniser.fit_on_texts([*abstracts, *titles])

    seq_abstracts = tokeniser.texts_to_sequences(abstracts)
    seq_abstracts = pad_sequences(seq_abstracts, padding="post")

    seq_titles = tokeniser.texts_to_sequences(titles)
    seq_titles = pad_sequences(seq_titles, padding="post")

    return seq_abstracts, seq_titles, tokeniser


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
        if i > n_docs:
            break
    return docs


if __name__ == "__main__":
    docs = load_data()

    abstracts = []
    titles = []
    for doc in docs:
        abstracts.append(clean_text(doc["abstract"]))
        titles.append(clean_text(doc["title"]))

    seq_abstracts, seq_titles, tokeniser = tokenise_text(abstracts, titles)

    print(seq_titles.shape)

    print(tokeniser.sequences_to_texts(seq_abstracts))