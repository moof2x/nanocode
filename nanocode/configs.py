"""Pre-defined GPTConfig model sizes."""
from nanocode.gpt import GPTConfig

# GPT-2 1.3B
d24 = GPTConfig(n_layer=24, n_embed=2048, n_head=8, n_kv_head=8, vocab_size=32768, sequence_len=4096)
# nanochat's d20 model, ~560m
d20 = GPTConfig(n_layer=20, n_embed=1280, n_head=5, n_kv_head=5, vocab_size=32768, sequence_len=2048)
# this is roughly the GPT2-small hyper-params but adjusted for head_dim=256 to better feed the 256x256 tpuv6e systolic arrays
# you'd end must closer to the original 117M param count without tied weights
d12 = GPTConfig(n_layer=12, n_embed=768, n_head=3, n_kv_head=3, vocab_size=32768, sequence_len=1024) # ~162M
d6 = GPTConfig(n_layer=6, n_embed=384, n_head=3, n_kv_head=3, vocab_size=32768, sequence_len=512) # ~23M
d3 = GPTConfig(n_layer=3, n_embed=192, n_head=2, n_kv_head=2, vocab_size=8000, sequence_len=256) # 4M, but at this size the model is dominated by the embedding and classifier weights

CONFIGS = {"d3": d3, "d6": d6, "d12": d12, "d20": d20, "d24": d24}
