import torch
import torch.nn.functional as F
from tokenizer import *
from config import *
from model import GPT
from tokenizer import encode, decode, END_ID
from setup import save_dir


device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(
    enc.n_vocab,
    EMBEDDING_DIM,
    BLOCK_SIZE,
    NUM_HEAD,
    NUM_LAYER
).to(device)


checkpoint = torch.load(
    f"{save_dir}/gpt_512_best.pt",
    map_location="cpu"
)

state = checkpoint["model"]

clean_state = {}

for key, value in state.items():
    key = key.replace("_orig_mod.", "")
    clean_state[key] = value

model.load_state_dict(
    clean_state,
    strict=True
)

model.eval()


def generate(
    prompt,
    max_tokens=100,
    temperature=1.0,
    top_k=50
):

    text = f"<|user|>{prompt}<|assistant|>"

    ids = encode(text)

    x = torch.tensor(
        ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():

        for _ in range(max_tokens):

            x_cond = x[:, -model.block_size:]

            logits, _ = model(x_cond)

            logits = logits[:, -1, :]

            logits = logits / temperature

            if top_k is not None:

                values, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
                )

                logits[
                    logits < values[:, [-1]]
                ] = float("-inf")

            probs = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            x = torch.cat(
                (x, next_token),
                dim=1
            )

            if next_token.item() == END_ID:
                break

    return decode(x.squeeze(0))