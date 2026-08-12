import tiktoken
import torch


special_token = {
    "<|user|>" : 200019,
    "<|assistant|>" : 200020,
    "<|end|>" : 200021
}

enc = tiktoken.get_encoding("o200k_base")

enc = tiktoken.Encoding(
    name="o200k_base",
    pat_str=enc._pat_str,
    mergeable_ranks=enc._mergeable_ranks,
    special_tokens={
        **enc._special_tokens,
        **special_token
    }
)


USER_ID = enc.encode("<|user|>", allowed_special={"<|user|>"})[0]
ASSISTNAT_ID = enc.encode("<|assistant|>", allowed_special={"<|assistant|>"})[0]
END_ID = enc.encode("<|end|>", allowed_special={"<|end|>"})[0]


EOT_ID = enc.eot_token


def encode(text : str):
    return enc.encode(
        text, 
        allowed_special={
            "<|user|>",
            "<|assistant|>",
            "<|end|>"
        }
    )


def decode(ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
        
    return enc.decode(ids)

