from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    e: int
    h: int
    n: int

def get_gpt():
    pass
