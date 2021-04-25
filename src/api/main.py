import fastapi
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import pydantic
from typing import Dict
from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


# load models
with open(f"./shared_data/models/tokeniser_1000000.json") as f:
    tokeniser = tokenizer_from_json(f.read())
    print("loaded tokeniser")
encoder = load_model("./shared_data/models/1000000_150_100_1000_1000.encoder")
print("loaded encoder")
decoder = load_model("./shared_data/models/1000000_150_100_1000_1000.decoder")
print("loaded decoder")

# infer function
def infer(
    input_text: str,
    tokeniser: Tokenizer,
    encoder_model: Model,
    decoder_model: Model,
    seq_len: int = 150,
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
        if next_word == special_tokens["end"] or len(decoded_sentence) >= (seq_len - 1):
            stop_condition = True

        i += 1

        target_seq[0, i] = sampled_token_index

        # Update states
        states_value = [h, c]

    return " ".join(decoded_sentence)


# request schema
class TitleRequest(pydantic.BaseModel):
    abstract: str
    max_len: int


# create the API
description = "An API for predicting titles of research papers using the abstract and a seq2seq model."
app = fastapi.FastAPI(
    title="Abstacts to Titles",
    description=description,
    version="0.0.9",
)


# define the endpoint
@app.post("/get_title_from_abstract")
async def get_title_from_abstract(title_req: TitleRequest):
    try:
        title = infer(
            input_text=title_req.abstract,
            tokeniser=tokeniser,
            encoder_model=encoder,
            decoder_model=decoder,
        )
        title_list = title.split(" ")
        title = " ".join(
            title_list[0 : title_req.max_len]
            if len(title_list) > title_req.max_len
            else title_list
        )
    except:
        raise HTTPException(
            500,
            "There was an internal error... I probably should have put more effort into this API.",
        )
    return JSONResponse({"title": title})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8321)
